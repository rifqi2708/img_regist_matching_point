#!/usr/bin/env python3
"""Single-pair Elastix registration: rigid + B-spline, export DVF + comparison PNG."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from time import perf_counter

import itk
import numpy as np
from PIL import Image, ImageDraw


# -----------------------------
# In-code configuration (no CLI)
# -----------------------------
SUBJECT_DIR = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_016"
)
FIXED_NAME = "QUADRA_HC_016_Test_CT-AC.nii.gz"
MOVING_NAME = "QUADRA_HC_016_Retest_CT-AC.nii.gz"
OUTPUT_DIR = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/outputs/quadra_hc_016"
)
SLICE_INDEX = None  # use center slice when None


def _print_step_time(step_name: str, start_time: float) -> float:
    """Print elapsed seconds for one step and return a new step start timestamp."""
    elapsed = perf_counter() - start_time
    print(f"[time] {step_name}: {elapsed:.3f}s", flush=True)
    return perf_counter()


def _as_string_list(value: str | int | float | bool | list | tuple) -> list[str]:
    """Convert a scalar/list value into Elastix-style list[str] parameters."""
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]

    out = []
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
    """Build a 2-stage registration setup: rigid followed by B-spline deformable."""
    parameter_object = itk.ParameterObject.New()

    rigid_map = parameter_object.GetDefaultParameterMap("rigid")
    rigid_overrides = {
        "NumberOfResolutions": 4,
        "MaximumNumberOfIterations": 256,
        "NumberOfSpatialSamples": 8192,
        "ImageSampler": "RandomCoordinate",
        "NewSamplesEveryIteration": "true",
        "AutomaticTransformInitialization": "true",
        "AutomaticTransformInitializationMethod": "GeometricalCenter",
        "WriteResultImage": "false",
        "ResultImageFormat": "nii.gz",
        "DefaultPixelValue": -1024,
    }
    rigid_map = _apply_overrides(rigid_map, rigid_overrides)

    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_overrides = {
         "NumberOfResolutions": 4,
        "MaximumNumberOfIterations": 256,
        "NumberOfSpatialSamples": 8192,
        "ImageSampler": "RandomCoordinate",
        "NewSamplesEveryIteration": "true",
        "FinalGridSpacingInPhysicalUnits": 32.0,
        "WriteResultImage": "false",
        "ResultImageFormat": "nii.gz",
        "DefaultPixelValue": -1024,
    }
    bspline_map = _apply_overrides(bspline_map, bspline_overrides)

    parameter_object.AddParameterMap(rigid_map)
    parameter_object.AddParameterMap(bspline_map)
    return parameter_object


class _Pushd:
    """Temporarily change current working directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._previous = Path.cwd()

    def __enter__(self) -> None:
        os.chdir(self._path)

    def __exit__(self, exc_type, exc, tb) -> None:
        os.chdir(self._previous)


def run_registration(fixed_image: itk.Image, moving_image: itk.Image) -> itk.ParameterObject:
    """Run rigid + B-spline registration and return the transform parameter object."""
    parameter_object = build_parameter_object()
    _, transform_parameter_object = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )
    return transform_parameter_object


def compute_dvf_image(
    moving_image: itk.Image, transform_parameter_object: itk.ParameterObject, work_dir: Path | None
) -> itk.Image:
    """Compute DVF with Transformix and prevent deformationField side-file leakage."""
    if work_dir is None:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="dvf_work_")
        output_dir = Path(temp_dir_ctx.__enter__())
        close_temp_dir = temp_dir_ctx.__exit__
    else:
        output_dir = Path(work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        close_temp_dir = None

    try:
        transformix_filter = itk.TransformixFilter.New(moving_image)
        transformix_filter.SetTransformParameterObject(transform_parameter_object)
        transformix_filter.SetComputeDeformationField(True)
        transformix_filter.SetLogToConsole(False)
        transformix_filter.SetOutputDirectory(str(output_dir))

        with _Pushd(output_dir):
            transformix_filter.Update()

        dvf_image = transformix_filter.GetOutputDeformationField()

        # Transformix can auto-write deformationField.nii(.gz). Remove it so callers control outputs.
        for auto_name in ("deformationField.nii.gz", "deformationField.nii"):
            auto_path = output_dir / auto_name
            if auto_path.exists():
                auto_path.unlink()

        return dvf_image
    finally:
        if close_temp_dir is not None:
            close_temp_dir(None, None, None)


def generate_dvf_from_paths(fixed_path: Path, moving_path: Path, save_path: Path | None = None) -> itk.Image:
    """Generate DVF from image paths; optionally save it, always return in-memory DVF image."""
    fixed_image = itk.imread(str(fixed_path), itk.F)
    moving_image = itk.imread(str(moving_path), itk.F)
    transform_parameter_object = run_registration(fixed_image, moving_image)

    dvf_work_dir = save_path.parent if save_path is not None else None
    dvf_image = compute_dvf_image(moving_image, transform_parameter_object, dvf_work_dir)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        itk.imwrite(dvf_image, str(save_path))

    return dvf_image


def resample_moving_to_fixed_grid(
    fixed_image: itk.Image, moving_image: itk.Image, default_pixel_value: float = -1024.0
) -> itk.Image:
    """Put the unregistered moving image on fixed geometry for direct visual comparison."""
    identity = itk.IdentityTransform[itk.D, 3].New()
    return itk.resample_image_filter(
        moving_image,
        transform=identity,
        use_reference_image=True,
        reference_image=fixed_image,
        default_pixel_value=default_pixel_value,
    )


def normalize_slice(slice_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize a 2D slice to [0, 1] with clipping for stable visualization."""
    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype=np.float32)
    normalized = (slice_2d - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def save_registration_comparison(
    fixed_image: itk.Image,
    unregistered_image: itk.Image,
    deformable_image: itk.Image,
    output_path: Path,
    slice_index: int | None,
) -> None:
    """Save a 2-panel PNG overlay: unregistered vs deformable, both against fixed."""
    fixed_np = itk.array_view_from_image(fixed_image)
    unregistered_np = itk.array_view_from_image(unregistered_image)
    deformable_np = itk.array_view_from_image(deformable_image)

    z_size = fixed_np.shape[0]
    z = z_size // 2 if slice_index is None else int(slice_index)
    z = int(np.clip(z, 0, z_size - 1))

    # Match prior plot orientation (origin='lower').
    fixed_slice = np.flipud(fixed_np[z])
    unregistered_slice = np.flipud(unregistered_np[z])
    deformable_slice = np.flipud(deformable_np[z])

    # Use one global display window for all panels (subsampled for lower memory pressure).
    fixed_sample = fixed_np[::4, ::4, ::4]
    low = float(np.percentile(fixed_sample, 1.0))
    high = float(np.percentile(fixed_sample, 99.0))

    fixed_norm = normalize_slice(fixed_slice, low, high)
    unregistered_norm = normalize_slice(unregistered_slice, low, high)
    deformable_norm = normalize_slice(deformable_slice, low, high)

    panels = [
        ("Unregistered", unregistered_norm),
        ("Deformable (B-spline)", deformable_norm),
    ]

    def build_overlay_panel(moving_panel: np.ndarray, alpha: float = 0.35) -> Image.Image:
        """Create one RGB panel by blending red moving intensity over gray fixed image."""
        base_rgb = np.stack([fixed_norm, fixed_norm, fixed_norm], axis=-1)
        alpha_map = np.clip(alpha * moving_panel, 0.0, 1.0)
        red = np.zeros_like(base_rgb)
        red[..., 0] = 1.0
        overlay = base_rgb * (1.0 - alpha_map[..., None]) + red * alpha_map[..., None]
        panel_u8 = (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(panel_u8)

    panel_images = [build_overlay_panel(panel) for _, panel in panels]
    panel_w, panel_h = panel_images[0].size

    pad = 16
    gap = 12
    header_h = 40
    title_h = 24
    panel_count = len(panel_images)
    canvas_w = pad * 2 + panel_w * panel_count + gap * (panel_count - 1)
    canvas_h = pad * 2 + header_h + title_h + panel_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, pad), "Fixed (gray) vs Moving/Registered (red)", fill=(255, 255, 255))
    draw.text((canvas_w - 120, pad), f"z={z}", fill=(220, 220, 220))

    for i, ((title, _), panel_img) in enumerate(zip(panels, panel_images)):
        x = pad + i * (panel_w + gap)
        y = pad + header_h
        draw.text((x, y), title, fill=(255, 255, 255))
        canvas.paste(panel_img, (x, y + title_h))

    canvas.save(output_path)


def main() -> None:
    """Run the full pipeline and write only registration_comparison.png and dvf.nii.gz."""
    total_start = perf_counter()
    step_start = total_start

    fixed_path = SUBJECT_DIR / FIXED_NAME
    moving_path = SUBJECT_DIR / MOVING_NAME
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_path}")
    if not moving_path.exists():
        raise FileNotFoundError(f"Moving image not found: {moving_path}")
    step_start = _print_step_time("Path validation", step_start)

    print("Loading images...", flush=True)
    fixed_image = itk.imread(str(fixed_path), itk.F)
    moving_image = itk.imread(str(moving_path), itk.F)
    step_start = _print_step_time("Load images", step_start)

    print("Running Elastix registration...", flush=True)
    transform_parameter_object = run_registration(fixed_image, moving_image)
    step_start = _print_step_time("Elastix registration", step_start)

    print("Computing full deformable registered image for visualization...", flush=True)
    deformable_image = itk.transformix_filter(
        moving_image,
        transform_parameter_object=transform_parameter_object,
        log_to_console=False,
    )
    step_start = _print_step_time("Compute deformable image", step_start)

    print("Resampling unregistered moving image to fixed grid for comparison...", flush=True)
    unregistered_on_fixed = resample_moving_to_fixed_grid(fixed_image, moving_image)
    step_start = _print_step_time("Resample moving to fixed grid", step_start)

    comparison_path = OUTPUT_DIR / "registration_comparison.png"
    print(f"Saving comparison figure: {comparison_path}", flush=True)
    save_registration_comparison(
        fixed_image=fixed_image,
        unregistered_image=unregistered_on_fixed,
        deformable_image=deformable_image,
        output_path=comparison_path,
        slice_index=SLICE_INDEX,
    )
    step_start = _print_step_time("Save comparison figure", step_start)

    print("Computing DVF from full (rigid + bspline) transform...", flush=True)
    dvf_image = compute_dvf_image(
        moving_image=moving_image,
        transform_parameter_object=transform_parameter_object,
        work_dir=OUTPUT_DIR,
    )
    step_start = _print_step_time("Compute DVF", step_start)

    dvf_path = OUTPUT_DIR / "dvf.nii.gz"
    print(f"Saving DVF: {dvf_path}", flush=True)
    itk.imwrite(dvf_image, str(dvf_path))
    _print_step_time("Save DVF", step_start)

    print("Done.", flush=True)
    print(f"  - {comparison_path}", flush=True)
    print(f"  - {dvf_path}", flush=True)
    print(f"[time] Total runtime: {perf_counter() - total_start:.3f}s", flush=True)


if __name__ == "__main__":
    main()
