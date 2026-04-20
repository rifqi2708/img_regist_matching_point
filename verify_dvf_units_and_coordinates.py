#!/usr/bin/env python3
"""Verify whether Elastix DVF is in physical units and validate image/DVF coordinate systems.

This script runs two checks:
1) Synthetic known-shift test:
   - Creates a synthetic fixed image.
   - Creates a moving image by shifting fixed by known voxel offsets.
   - Runs Elastix registration + Transformix DVF generation.
   - Compares DVF vector against expected shift in mm and vox.
   - Concludes whether DVF is likely physical-domain (mm) or voxel-domain.

2) Real-data coordinate consistency test (if files exist):
   - Loads Test, Retest, and DVF from your dataset.
   - Compares ITK (LPS) and nibabel (RAS) physical coordinates.
   - Verifies that converting DVF vector LPS->RAS aligns ITK mapping with nibabel mapping.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import itk
import nibabel as nib
import numpy as np


# Default real-data paths for this workspace.
FIXED_REAL = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_009/QUADRA_HC_009_Test_CT-AC.nii.gz"
)
MOVING_REAL = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_009/QUADRA_HC_009_Retest_CT-AC.nii.gz"
)
DVF_REAL = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/outputs/quadra_hc_009/dvf.nii.gz"
)


@dataclass
class SyntheticResult:
    observed_dvf_xyz: np.ndarray
    expected_mm_xyz: np.ndarray
    expected_vox_xyz: np.ndarray
    dist_to_mm: float
    dist_to_vox: float
    likely_units: str


def _build_translation_parameter_object() -> itk.ParameterObject:
    """Create a robust translation-only registration setup for synthetic verification."""
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap("translation", 2)
    pm["MaximumNumberOfIterations"] = ["96"]
    pm["NumberOfSpatialSamples"] = ["3000"]
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    pm["WriteResultImage"] = ["false"]
    po.AddParameterMap(pm)
    return po


def _compute_dvf(moving_image: itk.Image, transform_parameter_object: itk.ParameterObject) -> itk.Image:
    """Generate DVF in-memory while isolating temporary side files."""
    with tempfile.TemporaryDirectory(prefix="verify_dvf_") as td:
        out_dir = Path(td)
        tf = itk.TransformixFilter.New(moving_image)
        tf.SetTransformParameterObject(transform_parameter_object)
        tf.SetComputeDeformationField(True)
        tf.SetLogToConsole(False)
        tf.SetOutputDirectory(str(out_dir))

        previous_cwd = Path.cwd()
        os.chdir(out_dir)
        try:
            tf.Update()
        finally:
            os.chdir(previous_cwd)

        # Remove any side files from Transformix.
        for side_name in ("deformationField.nii.gz", "deformationField.nii"):
            side_file = out_dir / side_name
            if side_file.exists():
                side_file.unlink()

        return tf.GetOutputDeformationField()


def run_synthetic_unit_test() -> SyntheticResult:
    """Run a known-shift synthetic test to infer DVF units."""
    # Synthetic settings.
    size_xyz = (48, 44, 32)  # (x, y, z)
    spacing_xyz = np.array([2.0, 3.0, 4.0], dtype=np.float64)  # mm
    shift_vox_xyz = np.array([5.0, 6.0, 7.0], dtype=np.float64)

    # ITK array layout is (z, y, x), so build in that order.
    fixed_zyx = np.zeros((size_xyz[2], size_xyz[1], size_xyz[0]), dtype=np.float32)
    fixed_zyx[9:22, 11:24, 12:27] = 1.0  # simple bright cuboid

    # Create moving by known roll in z,y,x corresponding to shift xyz.
    moving_zyx = np.roll(
        fixed_zyx,
        shift=(int(shift_vox_xyz[2]), int(shift_vox_xyz[1]), int(shift_vox_xyz[0])),
        axis=(0, 1, 2),
    )

    fixed = itk.image_from_array(fixed_zyx)
    moving = itk.image_from_array(moving_zyx)
    fixed.SetSpacing(tuple(spacing_xyz))
    moving.SetSpacing(tuple(spacing_xyz))

    parameter_object = _build_translation_parameter_object()
    _, transform_parameter_object = itk.elastix_registration_method(
        fixed, moving, parameter_object=parameter_object, log_to_console=False
    )
    dvf_image = _compute_dvf(moving, transform_parameter_object)

    # Query center of the bright cuboid in fixed image.
    query_xyz = np.array([19, 17, 15], dtype=np.int64)
    observed_dvf_xyz = np.array(
        dvf_image.GetPixel((int(query_xyz[0]), int(query_xyz[1]), int(query_xyz[2]))), dtype=np.float64
    )

    expected_mm_xyz = shift_vox_xyz * spacing_xyz
    expected_vox_xyz = shift_vox_xyz

    dist_to_mm = float(np.linalg.norm(observed_dvf_xyz - expected_mm_xyz))
    dist_to_vox = float(np.linalg.norm(observed_dvf_xyz - expected_vox_xyz))
    likely_units = "physical (mm)" if dist_to_mm < dist_to_vox else "voxel"

    return SyntheticResult(
        observed_dvf_xyz=observed_dvf_xyz,
        expected_mm_xyz=expected_mm_xyz,
        expected_vox_xyz=expected_vox_xyz,
        dist_to_mm=dist_to_mm,
        dist_to_vox=dist_to_vox,
        likely_units=likely_units,
    )


def lps_to_ras_point(p_lps: np.ndarray) -> np.ndarray:
    """Convert point from LPS to RAS coordinates."""
    return np.array([-p_lps[0], -p_lps[1], p_lps[2]], dtype=np.float64)


def lps_to_ras_vector(v_lps: np.ndarray) -> np.ndarray:
    """Convert vector from LPS to RAS coordinates."""
    return np.array([-v_lps[0], -v_lps[1], v_lps[2]], dtype=np.float64)


def run_real_coordinate_test(fixed_path: Path, moving_path: Path, dvf_path: Path) -> None:
    """Check image and DVF coordinate consistency on real files."""
    fixed_itk = itk.imread(str(fixed_path), itk.F)
    moving_itk = itk.imread(str(moving_path), itk.F)
    dvf_itk = itk.imread(str(dvf_path))

    fixed_nib = nib.load(str(fixed_path))
    moving_nib = nib.load(str(moving_path))
    inv_moving_affine = np.linalg.inv(moving_nib.affine)

    size = fixed_itk.GetLargestPossibleRegion().GetSize()
    query_xyz = np.array([int(size[0] // 2), int(size[1] // 2), int(size[2] // 2)], dtype=np.int64)

    # ITK fixed physical point is LPS.
    idx = itk.Index[3]()
    idx[0], idx[1], idx[2] = int(query_xyz[0]), int(query_xyz[1]), int(query_xyz[2])
    p_fixed_lps = np.array(fixed_itk.TransformIndexToPhysicalPoint(idx), dtype=np.float64)

    # nibabel affine output is RAS.
    p_fixed_ras_from_nib = nib.affines.apply_affine(fixed_nib.affine, query_xyz.astype(np.float64))
    p_fixed_ras_from_itk = lps_to_ras_point(p_fixed_lps)
    image_coord_diff = float(np.linalg.norm(p_fixed_ras_from_nib - p_fixed_ras_from_itk))

    # DVF vector from ITK is interpreted in LPS.
    v_lps = np.array(
        dvf_itk.GetPixel((int(query_xyz[0]), int(query_xyz[1]), int(query_xyz[2]))), dtype=np.float64
    )

    # ITK mapping path (all in LPS).
    p_moving_lps = p_fixed_lps + v_lps
    idx_moving_itk = moving_itk.TransformPhysicalPointToContinuousIndex(tuple(p_moving_lps))
    idx_moving_itk_xyz = np.array([idx_moving_itk[0], idx_moving_itk[1], idx_moving_itk[2]], dtype=np.float64)

    # nib path with proper LPS->RAS vector conversion.
    v_ras = lps_to_ras_vector(v_lps)
    p_moving_ras = p_fixed_ras_from_nib + v_ras
    idx_moving_nib_xyz = nib.affines.apply_affine(inv_moving_affine, p_moving_ras)

    # nib path without conversion (intentionally wrong for comparison).
    p_moving_ras_wrong = p_fixed_ras_from_nib + v_lps
    idx_moving_nib_wrong_xyz = nib.affines.apply_affine(inv_moving_affine, p_moving_ras_wrong)

    mapping_diff_with_conversion = float(np.linalg.norm(idx_moving_itk_xyz - idx_moving_nib_xyz))
    mapping_diff_without_conversion = float(np.linalg.norm(idx_moving_itk_xyz - idx_moving_nib_wrong_xyz))

    print("\n[Real-data coordinate test]")
    print(f"Query voxel (x,y,z): {tuple(int(v) for v in query_xyz)}")
    print(
        "Image coordinate check |Ras(nib) - Ras(from ITK LPS)|: "
        f"{image_coord_diff:.6f} (should be ~0)"
    )
    print(
        "Mapping index diff with LPS->RAS DVF conversion: "
        f"{mapping_diff_with_conversion:.6f} (should be ~0)"
    )
    print(
        "Mapping index diff WITHOUT DVF conversion: "
        f"{mapping_diff_without_conversion:.6f} (usually much larger)"
    )

    if image_coord_diff < 1e-4:
        print("Image coordinate conclusion: ITK uses LPS, nibabel affine is RAS (consistent after conversion).")
    else:
        print("Image coordinate conclusion: unexpected mismatch, check headers/directions.")

    if mapping_diff_with_conversion < mapping_diff_without_conversion:
        print("DVF coordinate conclusion: DVF vectors behave as LPS physical vectors.")
    else:
        print("DVF coordinate conclusion: unexpected behavior, re-check registration/DVF source.")


def main() -> None:
    """Execute synthetic DVF unit test and optional real-data coordinate test."""
    print("[Synthetic DVF unit test]")
    synthetic = run_synthetic_unit_test()
    print(f"Observed DVF vector (xyz): {np.array2string(synthetic.observed_dvf_xyz, precision=4)}")
    print(f"Expected if physical mm:  {np.array2string(synthetic.expected_mm_xyz, precision=4)}")
    print(f"Expected if voxel units:  {np.array2string(synthetic.expected_vox_xyz, precision=4)}")
    print(f"Distance to mm expectation:   {synthetic.dist_to_mm:.6f}")
    print(f"Distance to voxel expectation:{synthetic.dist_to_vox:.6f}")
    print(f"Likely DVF units: {synthetic.likely_units}")

    if FIXED_REAL.exists() and MOVING_REAL.exists() and DVF_REAL.exists():
        run_real_coordinate_test(FIXED_REAL, MOVING_REAL, DVF_REAL)
    else:
        print("\n[Real-data coordinate test]")
        print("Skipped because one or more default files are missing:")
        print(f"  fixed:  {FIXED_REAL}")
        print(f"  moving: {MOVING_REAL}")
        print(f"  dvf:    {DVF_REAL}")


if __name__ == "__main__":
    main()
