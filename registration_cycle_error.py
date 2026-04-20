#!/usr/bin/env python3  # Tell the system to run this file with Python 3.

from pathlib import Path  # Use Path objects for clear and safe file paths.
import tempfile  # Create a temporary folder for Transformix outputs.
from time import perf_counter  # High-resolution timer for step/runtime monitoring.

import itk  # ITK + Elastix tools for registration and DVF generation.
import numpy as np  # Numerical array tools for masks and coordinates.

# Set the fixed image (Test) path used as the registration reference image.
FIXED_PATH = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_009/QUADRA_HC_009_Test_CT-AC.nii.gz"
)
# Set the moving image (Retest) path that will be aligned to the fixed image.
MOVING_PATH = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_009/QUADRA_HC_009_Retest_CT-AC.nii.gz"
)
# Set the threshold used to remove air/background and keep body voxels.
FOREGROUND_THRESHOLD = -900.0


def _print_step_time(step_name: str, start_time: float) -> float:
    """Print elapsed seconds for one step and return a new step start timestamp."""
    elapsed = perf_counter() - start_time
    print(f"[time] {step_name}: {elapsed:.3f}s")
    return perf_counter()


def _as_string_list(value: str | int | float | bool | list | tuple) -> list[str]:
    """Convert scalar/list values into Elastix-style list[str] parameters."""
    values = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in values:
        if isinstance(item, bool):
            out.append("true" if item else "false")
        else:
            out.append(str(item))
    return out


def _apply_overrides(parameter_map: dict, overrides: dict[str, str | int | float | bool | list]) -> dict:
    """Apply Python overrides into a single Elastix parameter map dictionary."""
    for key, value in overrides.items():
        parameter_map[key] = _as_string_list(value)
    return parameter_map


def build_parameter_object() -> itk.ParameterObject:
    """Build rigid + b-spline registration parameters."""
    parameter_object = itk.ParameterObject.New()

    rigid_map = parameter_object.GetDefaultParameterMap("rigid")
    rigid_overrides = {
        "Transform": "EulerTransform",
        "Metric": "AdvancedMattesMutualInformation",
        "Optimizer": "AdaptiveStochasticGradientDescent",
        "ImageSampler": "RandomCoordinate",
        "NumberOfResolutions": 3,
        "ImagePyramidSchedule": [4, 4, 4, 2, 2, 2, 1, 1, 1],
        "MaximumNumberOfIterations": 500,
        "NumberOfSpatialSamples": 5000,
        "AutomaticTransformInitialization": True,
        "AutomaticScalesEstimation": True,
        "AutomaticParameterEstimation": True,
        "UseAdaptiveStepSizes": True,
        "ASGDParameterEstimationMethod": "OriginalButSigmoidToDefault",
        "NumberOfHistogramBins": 32,
        "FixedLimitRangeRatio": 0.0,
        "MovingLimitRangeRatio": 0.0,
        "FixedKernelBSplineOrder": 1,
        "MovingKernelBSplineOrder": 3,
        "UseFastAndLowMemoryVersion": True,
        "UseDirectionCosines": True,
        "ErodeMask": False,
        "NewSamplesEveryIteration": True,
        "BSplineInterpolationOrder": 1,
        "FinalBSplineInterpolationOrder": 3,
        "HowToCombineTransforms": "Compose",
        "DefaultPixelValue": -1024,
        "WriteResultImage": False,
    }
    rigid_map = _apply_overrides(rigid_map, rigid_overrides)

    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_overrides = {
        "Transform": "BSplineTransform",
        "Metric": "AdvancedMattesMutualInformation",
        "Optimizer": "AdaptiveStochasticGradientDescent",
        "ImageSampler": "RandomCoordinate",
        "NumberOfResolutions": 3,
        "ImagePyramidSchedule": [4, 4, 4, 2, 2, 2, 1, 1, 1],
        "MaximumNumberOfIterations": 500,
        "NumberOfSpatialSamples": 5000,
        "FinalGridSpacingInPhysicalUnits": [10.0, 10.0, 10.0],
        "GridSpacingSchedule": [4.0, 2.0, 1.0],
        "AutomaticParameterEstimation": True,
        "UseAdaptiveStepSizes": True,
        "ASGDParameterEstimationMethod": "OriginalButSigmoidToDefault",
        "NumberOfHistogramBins": 32,
        "FixedLimitRangeRatio": 0.0,
        "MovingLimitRangeRatio": 0.0,
        "FixedKernelBSplineOrder": 1,
        "MovingKernelBSplineOrder": 3,
        "UseFastAndLowMemoryVersion": True,
        "UseDirectionCosines": True,
        "ErodeMask": False,
        "NewSamplesEveryIteration": True,
        "UseRandomSampleRegion": False,
        "ExactMetricSampleGridSpacing": 2,
        "BSplineInterpolationOrder": 1,
        "FinalBSplineInterpolationOrder": 3,
        "HowToCombineTransforms": "Compose",
        "DefaultPixelValue": -1024,
        "WriteResultImage": False,
    }
    bspline_map = _apply_overrides(bspline_map, bspline_overrides)

    parameter_object.AddParameterMap(rigid_map)
    parameter_object.AddParameterMap(bspline_map)
    return parameter_object


def run_registration_and_compute_dvf(fixed_image: itk.Image, moving_image: itk.Image) -> itk.Image:
    """Run registration and return an in-memory DVF that maps fixed points to moving points."""
    parameter_object = build_parameter_object()
    _, transform_parameter_object = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    with tempfile.TemporaryDirectory(prefix="minimal_dvf_") as temp_dir:
        transformix_filter = itk.TransformixFilter.New(moving_image)
        transformix_filter.SetTransformParameterObject(transform_parameter_object)
        transformix_filter.SetComputeDeformationField(True)
        transformix_filter.SetLogToConsole(False)
        transformix_filter.SetOutputDirectory(temp_dir)
        transformix_filter.Update()
        dvf_image = transformix_filter.GetOutputDeformationField()

        for side_name in ("deformationField.nii.gz", "deformationField.nii"):
            side_path = Path(temp_dir) / side_name
            if side_path.exists():
                side_path.unlink()

    return dvf_image


def is_in_bounds(index_xyz: np.ndarray, shape_xyz: tuple[int, int, int]) -> bool:
    """Return True when integer xyz index is inside image bounds."""
    return (
        0 <= int(index_xyz[0]) < int(shape_xyz[0])
        and 0 <= int(index_xyz[1]) < int(shape_xyz[1])
        and 0 <= int(index_xyz[2]) < int(shape_xyz[2])
    )


def index_xyz_to_physical(image: itk.Image, index_xyz: np.ndarray) -> np.ndarray:
    """Convert integer voxel xyz index to physical xyz coordinates (mm)."""
    itk_index = itk.Index[3]()
    itk_index[0] = int(index_xyz[0])
    itk_index[1] = int(index_xyz[1])
    itk_index[2] = int(index_xyz[2])
    return np.array(image.TransformIndexToPhysicalPoint(itk_index), dtype=np.float64)


def map_point_with_dvf(
    query_xyz: np.ndarray, fixed_image: itk.Image, moving_image: itk.Image, dvf_image: itk.Image
) -> dict:
    """Map one fixed-image point to the moving image using DVF in ITK physical space."""
    query_xyz = np.asarray(query_xyz, dtype=np.int64)
    query_physical = index_xyz_to_physical(fixed_image, query_xyz)

    dvf_vector = np.array(
        dvf_image.GetPixel((int(query_xyz[0]), int(query_xyz[1]), int(query_xyz[2]))),
        dtype=np.float64,
    )
    matched_physical = query_physical + dvf_vector

    matched_cont_index = moving_image.TransformPhysicalPointToContinuousIndex(tuple(matched_physical))
    matched_cont_xyz = np.array(
        [matched_cont_index[0], matched_cont_index[1], matched_cont_index[2]],
        dtype=np.float64,
    )
    matched_round_xyz = np.rint(matched_cont_xyz).astype(np.int64)

    moving_size = moving_image.GetLargestPossibleRegion().GetSize()
    moving_shape_xyz = (int(moving_size[0]), int(moving_size[1]), int(moving_size[2]))
    in_bounds = is_in_bounds(matched_round_xyz, moving_shape_xyz)

    return {
        "query_xyz": query_xyz,
        "query_physical": query_physical,
        "dvf_vector": dvf_vector,
        "matched_physical": matched_physical,
        "matched_cont_xyz": matched_cont_xyz,
        "matched_round_xyz": matched_round_xyz,
        "in_bounds": in_bounds,
    }


def select_one_foreground_query_point(
    fixed_image: itk.Image, threshold_hu: float
) -> tuple[np.ndarray, float]:
    """Pick one deterministic foreground point in Test image."""
    fixed_array = itk.array_view_from_image(fixed_image)  # (z, y, x)
    foreground_mask = fixed_array > threshold_hu
    if not np.any(foreground_mask):
        raise RuntimeError("No foreground voxel found above threshold in Test image.")

    query_zyx = np.argwhere(foreground_mask)[0].astype(np.int64)
    query_xyz = np.array([query_zyx[2], query_zyx[1], query_zyx[0]], dtype=np.int64)
    query_hu = float(fixed_array[int(query_zyx[0]), int(query_zyx[1]), int(query_zyx[2])])
    return query_xyz, query_hu


def main() -> None:  # Define one simple main function to keep the script minimal.
    total_start = perf_counter()
    step_start = total_start

    print("Loading fixed (Test) and moving (Retest) images...")
    fixed_image = itk.imread(str(FIXED_PATH), itk.F)
    moving_image = itk.imread(str(MOVING_PATH), itk.F)
    step_start = _print_step_time("Load images", step_start)

    print("Running forward registration and DVF (Test -> Retest)...")
    forward_dvf = run_registration_and_compute_dvf(fixed_image, moving_image)
    step_start = _print_step_time("Forward registration + DVF", step_start)

    print("Running backward registration and DVF (Retest -> Test)...")
    backward_dvf = run_registration_and_compute_dvf(moving_image, fixed_image)
    step_start = _print_step_time("Backward registration + DVF", step_start)

    print("Selecting one foreground query point in Test image...")
    query_test_xyz, query_hu = select_one_foreground_query_point(fixed_image, FOREGROUND_THRESHOLD)
    step_start = _print_step_time("Select foreground query point", step_start)

    print("Forward mapping: Test query -> Retest point...")
    forward_match = map_point_with_dvf(query_test_xyz, fixed_image, moving_image, forward_dvf)
    if not bool(forward_match["in_bounds"]):
        raise RuntimeError(
            "Forward-matched Retest point is out of bounds; cannot continue cycle mapping."
        )
    step_start = _print_step_time("Forward point mapping", step_start)

    cycle_query_retest_xyz = np.asarray(forward_match["matched_round_xyz"], dtype=np.int64)

    print("Backward mapping: Retest cycle query -> Test point...")
    backward_match = map_point_with_dvf(cycle_query_retest_xyz, moving_image, fixed_image, backward_dvf)
    if not bool(backward_match["in_bounds"]):
        raise RuntimeError(
            "Backward-matched Test cycle point is out of bounds; cannot compute cycle error."
        )
    step_start = _print_step_time("Backward point mapping", step_start)

    cycle_test_xyz = np.asarray(backward_match["matched_round_xyz"], dtype=np.int64)
    cycle_delta_xyz = cycle_test_xyz.astype(np.int64) - query_test_xyz.astype(np.int64)
    voxel_error = float(np.linalg.norm(cycle_delta_xyz.astype(np.float64)))

    original_test_physical = index_xyz_to_physical(fixed_image, query_test_xyz)
    cycle_test_physical = index_xyz_to_physical(fixed_image, cycle_test_xyz)
    mm_error = float(np.linalg.norm(cycle_test_physical - original_test_physical))
    _print_step_time("Cycle error computation", step_start)

    print("")
    print("=== Result ===")
    print(f"Query voxel in Test (x, y, z): {tuple(int(v) for v in query_test_xyz)}")
    print(f"Query voxel intensity (HU): {query_hu:.2f}")
    print(
        "Forward query physical in Test (mm): "
        f"{np.array2string(np.asarray(forward_match['query_physical']), precision=3)}"
    )
    print(
        "Forward DVF vector at query (mm): "
        f"{np.array2string(np.asarray(forward_match['dvf_vector']), precision=3)}"
    )
    print(
        "Forward matched continuous voxel in Retest (x, y, z): "
        f"{np.array2string(np.asarray(forward_match['matched_cont_xyz']), precision=3)}"
    )
    print(
        "Forward matched rounded voxel in Retest (x, y, z): "
        f"{tuple(int(v) for v in np.asarray(forward_match['matched_round_xyz']))}"
    )
    print(f"Forward matched voxel in bounds: {bool(forward_match['in_bounds'])}")
    print(f"Cycle query voxel in Retest (x, y, z): {tuple(int(v) for v in cycle_query_retest_xyz)}")
    print(
        "Backward query physical in Retest (mm): "
        f"{np.array2string(np.asarray(backward_match['query_physical']), precision=3)}"
    )
    print(
        "Backward DVF vector at cycle query (mm): "
        f"{np.array2string(np.asarray(backward_match['dvf_vector']), precision=3)}"
    )
    print(
        "Backward matched continuous voxel in Test (x, y, z): "
        f"{np.array2string(np.asarray(backward_match['matched_cont_xyz']), precision=3)}"
    )
    print(
        "Backward matched rounded voxel in Test (x, y, z): "
        f"{tuple(int(v) for v in np.asarray(backward_match['matched_round_xyz']))}"
    )
    print(f"Backward matched voxel in bounds: {bool(backward_match['in_bounds'])}")
    print(f"Cycle delta in Test (dx, dy, dz): {tuple(int(v) for v in cycle_delta_xyz)}")
    print(f"Cycle voxel error: {voxel_error:.6f}")
    print(f"Cycle mm error: {mm_error:.6f}")
    print(f"[time] Total runtime: {perf_counter() - total_start:.3f}s")


if __name__ == "__main__":  # Run main only when this file is executed directly.
    main()  # Start the minimal registration + DVF + point-matching pipeline.



# parameter

# def build_parameter_object() -> itk.ParameterObject:
#     """Build rigid + b-spline registration parameters."""
#     parameter_object = itk.ParameterObject.New()
#     rigid_map = parameter_object.GetDefaultParameterMap("rigid")
#     bspline_map = parameter_object.GetDefaultParameterMap("bspline")
#     rigid_map["AutomaticTransformInitialization"] = ["true"]
#     rigid_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
#     rigid_map["WriteResultImage"] = ["false"]
#     bspline_map["WriteResultImage"] = ["false"]
#     parameter_object.AddParameterMap(rigid_map)
#     parameter_object.AddParameterMap(bspline_map)
#     return parameter_object

# result

# Loading fixed (Test) and moving (Retest) images...
# Running forward registration and DVF (Test -> Retest)...
# Running backward registration and DVF (Retest -> Test)...
# Selecting one foreground query point in Test image...
# Forward mapping: Test query -> Retest point...
# Backward mapping: Retest cycle query -> Test point...

# === Result ===
# Query voxel in Test (x, y, z): (247, 122, 0)
# Query voxel intensity (HU): -884.00
# Forward query physical in Test (mm): [ -12.949   48.379 -887.   ]
# Forward DVF vector at query (mm): [   7.35     4.695 1297.625]
# Forward matched continuous voxel in Retest (x, y, z): [ 2.518e+02  1.025e+02 -1.874e-01]
# Forward matched rounded voxel in Retest (x, y, z): (252, 103, 0)
# Forward matched voxel in bounds: True
# Cycle query voxel in Retest (x, y, z): (252, 103, 0)
# Backward query physical in Retest (mm): [ -5.332  52.324 411.   ]
# Backward DVF vector at cycle query (mm): [   -8.743     5.94  -1297.727]
# Backward matched continuous voxel in Test (x, y, z): [2.463e+02 1.155e+02 1.363e-01]
# Backward matched rounded voxel in Test (x, y, z): (246, 116, 0)
# Backward matched voxel in bounds: True
# Cycle delta in Test (dx, dy, dz): (-1, -6, 0)
# Cycle voxel error: 6.082763
# Cycle mm error: 9.266709
