#!/usr/bin/env python3
"""Analyze registration cycle-error CSV outputs and render selected cases."""

from __future__ import annotations

import csv
import glob
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itk

try:
    from coord_space_utils import COORD_SPACE_RAW_ITK, resolve_subject_images
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "coord_space_utils":
        raise
    from new_pipeline.coord_space_utils import COORD_SPACE_RAW_ITK, resolve_subject_images


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent

for _p in (str(PROJECT_ROOT), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# In-script arguments (edit these values as needed).
CSV_PATH = "outputs/registration_cycle_error/cycle_points_*.csv"
DATASET_ROOT = "data/quadra_dataset_cropped"
OUTPUT_DIR = ""  # Empty means "<csv_dir>/<csv_stem>_analysis".
IS_MRI = False
DRY_RUN = False
PROGRESS_EVERY = 5
DVF_QUIVER_STRIDE = 24
DVF_DPI = 180
OVERLAY_ALPHA = 0.35


REGISTRATION_PARAMETER_OVERRIDES = {
    "rigid": {
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
    },
    "bspline": {
        "NumberOfResolutions": 4,
        "MaximumNumberOfIterations": 256,
        "NumberOfSpatialSamples": 8192,
        "ImageSampler": "RandomCoordinate",
        "NewSamplesEveryIteration": "true",
        "FinalGridSpacingInPhysicalUnits": 32.0,
        "WriteResultImage": "false",
        "ResultImageFormat": "nii.gz",
        "DefaultPixelValue": -1024,
    },
}


REQUIRED_COLUMNS = (
    "idx",
    "mask_name",
    "pt1_x",
    "pt1_y",
    "pt1_z",
    "pt2_x",
    "pt2_y",
    "pt2_z",
    "pt1_back_x",
    "pt1_back_y",
    "pt1_back_z",
    "voxel_error",
    "mm_error",
    "score_12",
    "score_21",
)


def _format_seconds(seconds):
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes}m{rem:04.1f}s"


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def resolve_project_path(path_like):
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_csv_path(csv_path_pattern):
    pattern_path = Path(csv_path_pattern).expanduser()
    if not pattern_path.is_absolute():
        pattern_path = PROJECT_ROOT / pattern_path
    pattern_str = str(pattern_path)

    matches = [Path(p).resolve() for p in glob.glob(pattern_str)]
    if matches:
        return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))

    direct_path = pattern_path.resolve()
    if direct_path.is_file():
        return direct_path
    raise FileNotFoundError(f"No CSV matched path/pattern: {csv_path_pattern}")


def resolve_output_dir(csv_path):
    if OUTPUT_DIR:
        return resolve_project_path(OUTPUT_DIR)
    return csv_path.with_name(f"{csv_path.stem}_analysis")


def extract_subject_id_and_organ(mask_name):
    if not mask_name:
        raise ValueError("mask_name is empty")
    parts = str(mask_name).split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"mask_name must look like '<subject>/<mask_file>', got: {mask_name}")
    subject_id = parts[0]
    organ = strip_nii_suffix(Path(parts[1]).name)
    if not organ:
        raise ValueError(f"Unable to parse organ from mask_name: {mask_name}")
    return subject_id, organ


def parse_required_int(row, key):
    return int(str(row.get(key, "")).strip())


def parse_required_float(row, key):
    return float(str(row.get(key, "")).strip())


def read_point_xyz(raw_row, prefix):
    return np.array(
        [
            parse_required_int(raw_row, f"{prefix}_x"),
            parse_required_int(raw_row, f"{prefix}_y"),
            parse_required_int(raw_row, f"{prefix}_z"),
        ],
        dtype=int,
    )


def _normalize_coord_space(raw_row):
    if "coord_space" not in raw_row:
        return COORD_SPACE_RAW_ITK
    coord_space = str(raw_row.get("coord_space", "")).strip()
    if coord_space != COORD_SPACE_RAW_ITK:
        raise ValueError(
            f"Unsupported coord_space={coord_space!r}. Expected {COORD_SPACE_RAW_ITK!r}."
        )
    return coord_space


def load_cycle_rows(csv_path):
    rows = []
    skipped = []

    with csv_path.open("r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        header = reader.fieldnames or []
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in header]
        if missing_cols:
            raise ValueError(f"Missing required CSV columns: {missing_cols}. Found: {header}")
        has_subject_id = "subject_id" in header

        for row_number, raw in enumerate(reader, start=2):
            try:
                subject_id_from_mask, organ = extract_subject_id_and_organ(raw.get("mask_name", ""))
                coord_space = _normalize_coord_space(raw)
                csv_subject_id = str(raw.get("subject_id", "")).strip()
                if has_subject_id and csv_subject_id and csv_subject_id != subject_id_from_mask:
                    raise ValueError(
                        f"subject_id {csv_subject_id!r} does not match mask_name subject {subject_id_from_mask!r}"
                    )

                pt1 = read_point_xyz(raw, "pt1")
                pt2 = read_point_xyz(raw, "pt2")
                pt1_back = read_point_xyz(raw, "pt1_back")
                delta = pt1_back.astype(int) - pt1.astype(int)
                norm_sq = int(np.sum(delta.astype(np.int64) ** 2))

                rows.append(
                    {
                        "idx": parse_required_int(raw, "idx"),
                        "mask_name": str(raw.get("mask_name", "")),
                        "subject_id": subject_id_from_mask,
                        "organ": organ,
                        "coord_space": coord_space,
                        "pt1_x": int(pt1[0]),
                        "pt1_y": int(pt1[1]),
                        "pt1_z": int(pt1[2]),
                        "pt2_x": int(pt2[0]),
                        "pt2_y": int(pt2[1]),
                        "pt2_z": int(pt2[2]),
                        "pt1_back_x": int(pt1_back[0]),
                        "pt1_back_y": int(pt1_back[1]),
                        "pt1_back_z": int(pt1_back[2]),
                        "dx": int(delta[0]),
                        "dy": int(delta[1]),
                        "dz": int(delta[2]),
                        "norm_sq": norm_sq,
                        "voxel_error": parse_required_float(raw, "voxel_error"),
                        "mm_error": parse_required_float(raw, "mm_error"),
                        "score_12": parse_required_float(raw, "score_12"),
                        "score_21": parse_required_float(raw, "score_21"),
                        "source_row_number": row_number,
                    }
                )
            except Exception as exc:
                skipped.append(
                    {
                        "scope": "row_parse",
                        "row_number": row_number,
                        "idx": raw.get("idx", ""),
                        "mask_name": raw.get("mask_name", ""),
                        "subject_id": "",
                        "organ": "",
                        "reason": f"parse_error: {exc}",
                    }
                )

    if not rows:
        raise RuntimeError(f"No valid rows loaded from CSV: {csv_path}")
    return rows, skipped


def _register_selection(selected_map, row, reason):
    idx = int(row["idx"])
    entry = selected_map.get(idx)
    if entry is None:
        entry = {
            "row": row,
            "reasons": [],
            "selected_by_highest": False,
            "selected_by_median": False,
            "selected_by_zero": False,
            "selected_zero_exact": False,
            "selected_zero_nearest": False,
        }
        selected_map[idx] = entry

    if reason not in entry["reasons"]:
        entry["reasons"].append(reason)
    if reason == "highest_mm":
        entry["selected_by_highest"] = True
    if reason == "median_mm":
        entry["selected_by_median"] = True
    if reason in ("zero_exact", "zero_nearest"):
        entry["selected_by_zero"] = True
    if reason == "zero_exact":
        entry["selected_zero_exact"] = True
    if reason == "zero_nearest":
        entry["selected_zero_nearest"] = True


def select_cases(rows):
    organ_to_rows = defaultdict(list)
    for row in rows:
        organ_to_rows[row["organ"]].append(row)

    selected_map = {}
    summary_rows = []

    for organ in sorted(organ_to_rows.keys()):
        organ_rows = organ_to_rows[organ]
        organ_rows_by_mm = sorted(organ_rows, key=lambda r: (r["mm_error"], r["idx"]))
        highest_row = sorted(organ_rows, key=lambda r: (-r["mm_error"], r["idx"]))[0]
        median_row = organ_rows_by_mm[(len(organ_rows_by_mm) - 1) // 2]

        zero_exact_rows = [row for row in organ_rows_by_mm if int(row["norm_sq"]) == 0]
        if zero_exact_rows:
            zero_row = zero_exact_rows[0]
            zero_reason = "zero_exact"
            zero_selection_kind = "exact"
        else:
            zero_row = min(organ_rows, key=lambda r: (r["mm_error"], r["norm_sq"], r["idx"]))
            zero_reason = "zero_nearest"
            zero_selection_kind = "nearest"

        _register_selection(selected_map, highest_row, "highest_mm")
        _register_selection(selected_map, median_row, "median_mm")
        _register_selection(selected_map, zero_row, zero_reason)

        selected_entries_for_organ = [
            entry for entry in selected_map.values() if entry["row"]["organ"] == organ
        ]
        summary_rows.append(
            {
                "organ": organ,
                "total_rows": len(organ_rows),
                "exact_zero_rows": len(zero_exact_rows),
                "selected_unique": len(selected_entries_for_organ),
                "selected_by_highest": sum(
                    1 for entry in selected_entries_for_organ if entry["selected_by_highest"]
                ),
                "selected_by_median": sum(
                    1 for entry in selected_entries_for_organ if entry["selected_by_median"]
                ),
                "selected_by_zero": sum(
                    1 for entry in selected_entries_for_organ if entry["selected_by_zero"]
                ),
                "zero_selection_kind": zero_selection_kind,
                "highest_idx": int(highest_row["idx"]),
                "median_idx": int(median_row["idx"]),
                "zero_idx": int(zero_row["idx"]),
                "min_mm_error": float(organ_rows_by_mm[0]["mm_error"]),
                "median_mm_error": float(median_row["mm_error"]),
                "max_mm_error": float(highest_row["mm_error"]),
                "rendered_images": 0,
            }
        )

    selected_entries = sorted(
        selected_map.values(),
        key=lambda entry: (entry["row"]["organ"], entry["row"]["subject_id"], entry["row"]["idx"]),
    )
    return selected_entries, summary_rows


def _itk_array_yxz(itk_image):
    array_zyx = np.asarray(itk.array_view_from_image(itk_image))
    if array_zyx.ndim != 3:
        raise ValueError(f"Expected 3D ITK image array, got shape {array_zyx.shape}")
    return np.transpose(array_zyx, (1, 2, 0))


def load_image_context(image_path):
    itk_image = itk.imread(str(image_path), itk.F)
    img_yxz = np.array(_itk_array_yxz(itk_image), copy=True).astype(np.float32, copy=False)
    itk_size = itk_image.GetLargestPossibleRegion().GetSize()
    return {
        "image_path": str(image_path),
        "itk_image": itk_image,
        "img_yxz": img_yxz,
        "itk_shape_xyz": (int(itk_size[0]), int(itk_size[1]), int(itk_size[2])),
    }


def _as_string_list(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for item in values:
        if isinstance(item, bool):
            out.append("true" if item else "false")
        else:
            out.append(str(item))
    return out


def _apply_overrides(parameter_map, overrides):
    for key, value in overrides.items():
        parameter_map[key] = _as_string_list(value)
    return parameter_map


def build_parameter_object():
    parameter_object = itk.ParameterObject.New()
    rigid_map = parameter_object.GetDefaultParameterMap("rigid")
    rigid_map = _apply_overrides(rigid_map, REGISTRATION_PARAMETER_OVERRIDES["rigid"])
    bspline_map = parameter_object.GetDefaultParameterMap("bspline")
    bspline_map = _apply_overrides(bspline_map, REGISTRATION_PARAMETER_OVERRIDES["bspline"])
    parameter_object.AddParameterMap(rigid_map)
    parameter_object.AddParameterMap(bspline_map)
    return parameter_object


def run_registration_artifacts(fixed_image, moving_image):
    parameter_object = build_parameter_object()
    _, transform_parameter_object = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    warped_image = itk.transformix_filter(
        moving_image,
        transform_parameter_object=transform_parameter_object,
        log_to_console=False,
    )
    warped_yxz = np.array(_itk_array_yxz(warped_image), copy=True).astype(np.float32, copy=False)

    with tempfile.TemporaryDirectory(prefix="analysis_registration_dvf_") as temp_dir:
        transformix_filter = itk.TransformixFilter.New(moving_image)
        transformix_filter.SetTransformParameterObject(transform_parameter_object)
        transformix_filter.SetComputeDeformationField(True)
        transformix_filter.SetLogToConsole(False)
        transformix_filter.SetOutputDirectory(temp_dir)
        transformix_filter.Update()
        dvf_image = transformix_filter.GetOutputDeformationField()
        dvf_array_zyx = np.array(itk.array_from_image(dvf_image), copy=True)

    return {
        "warped_img_yxz": warped_yxz,
        "dvf_array_zyx": dvf_array_zyx,
    }


def load_subject_registration_data_cached(subject_id, dataset_root, cache):
    if subject_id in cache:
        cached = cache[subject_id]
        if isinstance(cached, Exception):
            raise cached
        return cached

    try:
        subject_images = resolve_subject_images(str(dataset_root), subject_id)
        test_ctx = load_image_context(subject_images["test"])
        retest_ctx = load_image_context(subject_images["retest"])
        forward_artifacts = run_registration_artifacts(test_ctx["itk_image"], retest_ctx["itk_image"])
        backward_artifacts = run_registration_artifacts(retest_ctx["itk_image"], test_ctx["itk_image"])
        cache[subject_id] = {
            "test_ctx": test_ctx,
            "retest_ctx": retest_ctx,
            "forward": forward_artifacts,
            "backward": backward_artifacts,
        }
        return cache[subject_id]
    except Exception as exc:
        cache[subject_id] = exc
        raise


def sanitize_filename_text(text):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("-")
    joined = "".join(safe).strip("-")
    if not joined:
        joined = "na"
    return joined[:140]


def _normalize_volume_for_display(img3d, is_mri=False):
    img3d = np.asarray(img3d, dtype=np.float32)
    if is_mri:
        low = float(np.min(img3d))
        high = float(np.max(img3d))
    else:
        low, high = -100.0, 200.0
    if high <= low:
        return np.zeros_like(img3d, dtype=np.float32)
    img3d = np.clip(img3d, low, high)
    return ((img3d - low) / (high - low)).astype(np.float32)


def _slice_plane_with_point(volume_yxz, point_xyz, plane):
    x = int(point_xyz[0])
    y = int(point_xyz[1])
    z = int(point_xyz[2])

    sy, sx, sz = volume_yxz.shape
    x = int(np.clip(x, 0, sx - 1))
    y = int(np.clip(y, 0, sy - 1))
    z = int(np.clip(z, 0, sz - 1))

    if plane == "axial":
        sl = volume_yxz[:, :, z]
        px, py = x, y
    elif plane == "coronal":
        sl = volume_yxz[y, :, :].T
        sl = sl[::-1, :]
        px, py = x, (sz - 1 - z)
    elif plane == "sagittal":
        sl = volume_yxz[:, x, :].T
        sl = sl[::-1, :]
        px, py = y, (sz - 1 - z)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return sl, (px, py)


def _draw_marker(ax, xy, color, label=None, coord_text=None):
    ax.plot(
        float(xy[0]),
        float(xy[1]),
        "+",
        markerfacecolor="none",
        markeredgecolor=color,
        markersize=8,
        markeredgewidth=1.5,
        label=label,
    )
    if coord_text:
        ax.annotate(
            coord_text,
            xy=(float(xy[0]), float(xy[1])),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.35, "pad": 1.0},
        )


def _coord_text(point_xyz):
    point_xyz = np.asarray(point_xyz, dtype=int)
    return f"({int(point_xyz[0])}, {int(point_xyz[1])}, {int(point_xyz[2])})"


def _normalize_map_for_display(map2d):
    map2d = np.asarray(map2d, dtype=np.float32)
    if map2d.size == 0:
        return map2d
    low = float(np.percentile(map2d, 1))
    high = float(np.percentile(map2d, 99))
    if high <= low:
        low = float(np.min(map2d))
        high = float(np.max(map2d))
    if high <= low:
        return np.zeros_like(map2d, dtype=np.float32)
    map2d = np.clip(map2d, low, high)
    return (map2d - low) / (high - low)


def _save_figure(fig, out_path, dpi=150):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
    return str(out_path.resolve())


def save_cycle_visualization_multiplane(query_img, target_img, result, out_dir, file_stem, is_mri=False, dpi=150):
    query_norm = _normalize_volume_for_display(query_img, is_mri=is_mri)
    target_norm = _normalize_volume_for_display(target_img, is_mri=is_mri)
    plane_paths = {}

    for plane in ("axial", "sagittal", "coronal"):
        q_slice, qxy = _slice_plane_with_point(query_norm, result["pt1"], plane)
        t_slice, txy = _slice_plane_with_point(target_norm, result["pt2"], plane)
        qb_slice, qxy_back = _slice_plane_with_point(query_norm, result["pt1_back"], plane)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].set_title(f"{plane.capitalize()} Query")
        ax[0].imshow(q_slice, cmap="gray")
        _draw_marker(ax[0], qxy, color="lime", coord_text=_coord_text(result["pt1"]))

        ax[1].set_title(f"{plane.capitalize()} Target")
        ax[1].imshow(t_slice, cmap="gray")
        _draw_marker(ax[1], txy, color="deepskyblue", coord_text=_coord_text(result["pt2"]))

        ax[2].set_title(f"{plane.capitalize()} Query + Cycle")
        ax[2].imshow(qb_slice, cmap="gray")
        _draw_marker(ax[2], qxy, color="lime", label="query", coord_text=_coord_text(result["pt1"]))
        _draw_marker(
            ax[2],
            qxy_back,
            color="orange",
            label="cycle",
            coord_text=_coord_text(result["pt1_back"]),
        )
        ax[2].legend(loc="upper right", fontsize=8, framealpha=0.8)

        for axis in ax.ravel():
            axis.set_xticks([])
            axis.set_yticks([])

        fig.suptitle(
            f"voxel_err={result['voxel_error']:.4f}, mm_err={result['mm_error']:.4f}, "
            f"score_12={result['score_12']:.6f}, score_21={result['score_21']:.6f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plane_paths[plane] = _save_figure(fig, out_dir / f"{file_stem}_{plane}.png", dpi=dpi)

    return plane_paths


def _build_overlay_rgb(fixed_slice, moving_slice, alpha):
    base_rgb = np.stack([fixed_slice, fixed_slice, fixed_slice], axis=-1)
    alpha_map = np.clip(alpha * moving_slice, 0.0, 1.0)
    red = np.zeros_like(base_rgb)
    red[..., 0] = 1.0
    overlay = base_rgb * (1.0 - alpha_map[..., None]) + red * alpha_map[..., None]
    return np.clip(overlay, 0.0, 1.0)


def save_registered_visualization_multiplane(
    fixed_img_yxz,
    moving_img_yxz,
    warped_img_yxz,
    fixed_point_xyz,
    moving_point_xyz,
    out_dir,
    file_stem,
    direction_label,
    fixed_label,
    moving_label,
    is_mri=False,
    dpi=150,
):
    fixed_norm = _normalize_volume_for_display(fixed_img_yxz, is_mri=is_mri)
    moving_norm = _normalize_volume_for_display(moving_img_yxz, is_mri=is_mri)
    warped_norm = _normalize_volume_for_display(warped_img_yxz, is_mri=is_mri)
    plane_paths = {}

    for plane in ("axial", "sagittal", "coronal"):
        fixed_slice, fixed_xy = _slice_plane_with_point(fixed_norm, fixed_point_xyz, plane)
        moving_slice, moving_xy = _slice_plane_with_point(moving_norm, moving_point_xyz, plane)
        warped_slice, warped_xy = _slice_plane_with_point(warped_norm, fixed_point_xyz, plane)
        overlay_rgb = _build_overlay_rgb(fixed_slice, warped_slice, alpha=OVERLAY_ALPHA)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        ax[0].set_title(f"{plane.capitalize()} {fixed_label} (fixed)")
        ax[0].imshow(fixed_slice, cmap="gray")
        _draw_marker(ax[0], fixed_xy, color="lime", coord_text=_coord_text(fixed_point_xyz))

        ax[1].set_title(f"{plane.capitalize()} {moving_label} (original)")
        ax[1].imshow(moving_slice, cmap="gray")
        _draw_marker(ax[1], moving_xy, color="deepskyblue", coord_text=_coord_text(moving_point_xyz))

        ax[2].set_title(f"{plane.capitalize()} Registered Overlay")
        ax[2].imshow(overlay_rgb)
        _draw_marker(ax[2], warped_xy, color="lime", coord_text=_coord_text(fixed_point_xyz))

        for axis in ax.ravel():
            axis.set_xticks([])
            axis.set_yticks([])

        fig.suptitle(
            f"{direction_label}: fixed(gray) + warped moving(red)",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plane_paths[plane] = _save_figure(fig, out_dir / f"{file_stem}_{plane}.png", dpi=dpi)

    return plane_paths


def _index_xyz_to_physical(image, index_xyz):
    itk_index = itk.Index[3]()
    itk_index[0] = int(index_xyz[0])
    itk_index[1] = int(index_xyz[1])
    itk_index[2] = int(index_xyz[2])
    return np.array(image.TransformIndexToPhysicalPoint(itk_index), dtype=np.float64)


def _physical_to_cont_index(image, point_xyz):
    cont_index = image.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(point_xyz, dtype=float)))
    return np.array([cont_index[0], cont_index[1], cont_index[2]], dtype=np.float64)


def _vector_mm_to_index_delta(image, index_xyz, vector_mm_xyz):
    base_phys = _index_xyz_to_physical(image, index_xyz)
    base_cont = _physical_to_cont_index(image, base_phys)
    moved_cont = _physical_to_cont_index(image, base_phys + np.asarray(vector_mm_xyz, dtype=np.float64))
    return moved_cont - base_cont


def _get_vector_slice_display(dvf_array_zyx, point_xyz, plane):
    x = int(point_xyz[0])
    y = int(point_xyz[1])
    z = int(point_xyz[2])
    sz, sy, sx = dvf_array_zyx.shape[:3]
    x = int(np.clip(x, 0, sx - 1))
    y = int(np.clip(y, 0, sy - 1))
    z = int(np.clip(z, 0, sz - 1))

    if plane == "axial":
        vector_slice = dvf_array_zyx[z, :, :, :]
        mag_slice = np.linalg.norm(vector_slice, axis=-1)
    elif plane == "coronal":
        vector_slice = dvf_array_zyx[:, y, :, :][::-1, :, :]
        mag_slice = np.linalg.norm(vector_slice, axis=-1)
    elif plane == "sagittal":
        vector_slice = dvf_array_zyx[:, :, x, :][::-1, :, :]
        mag_slice = np.linalg.norm(vector_slice, axis=-1)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return vector_slice.astype(np.float32, copy=False), mag_slice.astype(np.float32, copy=False)


def _quiver_components_for_plane(image, point_xyz, dvf_array_zyx, plane, stride):
    x0 = int(point_xyz[0])
    y0 = int(point_xyz[1])
    z0 = int(point_xyz[2])
    shape_xyz = (
        int(dvf_array_zyx.shape[2]),
        int(dvf_array_zyx.shape[1]),
        int(dvf_array_zyx.shape[0]),
    )
    x0 = int(np.clip(x0, 0, shape_xyz[0] - 1))
    y0 = int(np.clip(y0, 0, shape_xyz[1] - 1))
    z0 = int(np.clip(z0, 0, shape_xyz[2] - 1))

    xs = []
    ys = []
    us = []
    vs = []

    if plane == "axial":
        h = shape_xyz[1]
        w = shape_xyz[0]
        for py in range(0, h, stride):
            for px in range(0, w, stride):
                vector_mm = np.asarray(dvf_array_zyx[z0, py, px], dtype=np.float64)
                delta_idx = _vector_mm_to_index_delta(image, np.array([px, py, z0]), vector_mm)
                xs.append(px)
                ys.append(py)
                us.append(float(delta_idx[0]))
                vs.append(float(delta_idx[1]))
    elif plane == "coronal":
        h = shape_xyz[2]
        w = shape_xyz[0]
        for py in range(0, h, stride):
            z = shape_xyz[2] - 1 - py
            for px in range(0, w, stride):
                vector_mm = np.asarray(dvf_array_zyx[z, y0, px], dtype=np.float64)
                delta_idx = _vector_mm_to_index_delta(image, np.array([px, y0, z]), vector_mm)
                xs.append(px)
                ys.append(py)
                us.append(float(delta_idx[0]))
                vs.append(float(-delta_idx[2]))
    elif plane == "sagittal":
        h = shape_xyz[2]
        w = shape_xyz[1]
        for py in range(0, h, stride):
            z = shape_xyz[2] - 1 - py
            for px in range(0, w, stride):
                vector_mm = np.asarray(dvf_array_zyx[z, px, x0], dtype=np.float64)
                delta_idx = _vector_mm_to_index_delta(image, np.array([x0, px, z]), vector_mm)
                xs.append(px)
                ys.append(py)
                us.append(float(delta_idx[1]))
                vs.append(float(-delta_idx[2]))
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return np.asarray(xs), np.asarray(ys), np.asarray(us), np.asarray(vs)


def save_dvf_visualization_multiplane(
    fixed_img_yxz,
    fixed_itk_image,
    dvf_array_zyx,
    point_xyz,
    out_dir,
    file_stem,
    direction_label,
    is_mri=False,
    dpi=180,
):
    fixed_norm = _normalize_volume_for_display(fixed_img_yxz, is_mri=is_mri)
    plane_paths = {}
    stride = max(1, int(DVF_QUIVER_STRIDE))

    for plane in ("axial", "sagittal", "coronal"):
        fixed_slice, point_xy = _slice_plane_with_point(fixed_norm, point_xyz, plane)
        _, mag_slice = _get_vector_slice_display(dvf_array_zyx, point_xyz, plane)
        mag_disp = _normalize_map_for_display(mag_slice)
        grid_x, grid_y, u, v = _quiver_components_for_plane(
            fixed_itk_image,
            point_xyz,
            dvf_array_zyx,
            plane,
            stride=stride,
        )

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0), constrained_layout=True)
        ax.set_title(f"{plane.capitalize()} {direction_label} DVF")
        ax.imshow(fixed_slice, cmap="gray")
        ax.imshow(mag_disp, cmap="magma", alpha=0.34)
        if grid_x.size:
            ax.quiver(
                grid_x,
                grid_y,
                u,
                v,
                color="cyan",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
            )
        _draw_marker(ax, point_xy, color="white", coord_text=_coord_text(point_xyz))
        ax.set_xticks([])
        ax.set_yticks([])
        plane_paths[plane] = _save_figure(fig, out_dir / f"{file_stem}_{plane}.png", dpi=dpi)

    return plane_paths


def render_selected_cases(selected_entries, output_dir, dataset_root):
    skipped = []
    rendered_by_organ = defaultdict(int)
    registration_cache = {}
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    total_cases = int(len(selected_entries))
    render_start_time = time.time()
    rendered_success_count = 0

    print(f"Rendering visualization cases: {total_cases}")

    progress_every = max(1, int(PROGRESS_EVERY))
    for case_idx, entry in enumerate(selected_entries, start=1):
        row = entry["row"]
        subject_id = row["subject_id"]
        organ = row["organ"]

        try:
            subject_data = load_subject_registration_data_cached(
                subject_id=subject_id,
                dataset_root=dataset_root,
                cache=registration_cache,
            )

            result = {
                "pt1": np.array([row["pt1_x"], row["pt1_y"], row["pt1_z"]], dtype=int),
                "pt2": np.array([row["pt2_x"], row["pt2_y"], row["pt2_z"]], dtype=int),
                "pt1_back": np.array(
                    [row["pt1_back_x"], row["pt1_back_y"], row["pt1_back_z"]],
                    dtype=int,
                ),
                "score_12": float(row["score_12"]),
                "score_21": float(row["score_21"]),
                "voxel_error": float(row["voxel_error"]),
                "mm_error": float(row["mm_error"]),
            }

            reason_tag = sanitize_filename_text("+".join(sorted(entry["reasons"])))
            case_tag = (
                f"{sanitize_filename_text(subject_id)}__{sanitize_filename_text(organ)}__"
                f"idx{int(row['idx']):06d}__{reason_tag}__"
                f"mm{float(row['mm_error']):.3f}__vox{float(row['voxel_error']):.3f}"
            )
            organ_dir = images_dir / sanitize_filename_text(organ)
            case_dir = (organ_dir / case_tag).resolve()
            case_dir.mkdir(parents=True, exist_ok=True)

            cycle_paths = save_cycle_visualization_multiplane(
                query_img=subject_data["test_ctx"]["img_yxz"],
                target_img=subject_data["retest_ctx"]["img_yxz"],
                result=result,
                out_dir=case_dir,
                file_stem="cycle_points",
                is_mri=IS_MRI,
                dpi=150,
            )
            forward_registered_paths = save_registered_visualization_multiplane(
                fixed_img_yxz=subject_data["test_ctx"]["img_yxz"],
                moving_img_yxz=subject_data["retest_ctx"]["img_yxz"],
                warped_img_yxz=subject_data["forward"]["warped_img_yxz"],
                fixed_point_xyz=result["pt1"],
                moving_point_xyz=result["pt2"],
                out_dir=case_dir,
                file_stem="forward_registered",
                direction_label="Forward registration (Test <- Retest)",
                fixed_label="Test",
                moving_label="Retest",
                is_mri=IS_MRI,
                dpi=150,
            )
            backward_registered_paths = save_registered_visualization_multiplane(
                fixed_img_yxz=subject_data["retest_ctx"]["img_yxz"],
                moving_img_yxz=subject_data["test_ctx"]["img_yxz"],
                warped_img_yxz=subject_data["backward"]["warped_img_yxz"],
                fixed_point_xyz=result["pt2"],
                moving_point_xyz=result["pt1_back"],
                out_dir=case_dir,
                file_stem="backward_registered",
                direction_label="Backward registration (Retest <- Test)",
                fixed_label="Retest",
                moving_label="Test",
                is_mri=IS_MRI,
                dpi=150,
            )
            forward_dvf_paths = save_dvf_visualization_multiplane(
                fixed_img_yxz=subject_data["test_ctx"]["img_yxz"],
                fixed_itk_image=subject_data["test_ctx"]["itk_image"],
                dvf_array_zyx=subject_data["forward"]["dvf_array_zyx"],
                point_xyz=result["pt1"],
                out_dir=case_dir,
                file_stem="forward_dvf",
                direction_label="Forward",
                is_mri=IS_MRI,
                dpi=DVF_DPI,
            )
            backward_dvf_paths = save_dvf_visualization_multiplane(
                fixed_img_yxz=subject_data["retest_ctx"]["img_yxz"],
                fixed_itk_image=subject_data["retest_ctx"]["itk_image"],
                dvf_array_zyx=subject_data["backward"]["dvf_array_zyx"],
                point_xyz=result["pt2"],
                out_dir=case_dir,
                file_stem="backward_dvf",
                direction_label="Backward",
                is_mri=IS_MRI,
                dpi=DVF_DPI,
            )

            entry["case_dir"] = str(case_dir)
            entry["image_path"] = cycle_paths.get("axial", "")
            entry["cycle_axial_path"] = cycle_paths.get("axial", "")
            entry["cycle_sagittal_path"] = cycle_paths.get("sagittal", "")
            entry["cycle_coronal_path"] = cycle_paths.get("coronal", "")
            entry["forward_registered_axial_path"] = forward_registered_paths.get("axial", "")
            entry["forward_registered_sagittal_path"] = forward_registered_paths.get("sagittal", "")
            entry["forward_registered_coronal_path"] = forward_registered_paths.get("coronal", "")
            entry["backward_registered_axial_path"] = backward_registered_paths.get("axial", "")
            entry["backward_registered_sagittal_path"] = backward_registered_paths.get("sagittal", "")
            entry["backward_registered_coronal_path"] = backward_registered_paths.get("coronal", "")
            entry["forward_dvf_axial_path"] = forward_dvf_paths.get("axial", "")
            entry["forward_dvf_sagittal_path"] = forward_dvf_paths.get("sagittal", "")
            entry["forward_dvf_coronal_path"] = forward_dvf_paths.get("coronal", "")
            entry["backward_dvf_axial_path"] = backward_dvf_paths.get("axial", "")
            entry["backward_dvf_sagittal_path"] = backward_dvf_paths.get("sagittal", "")
            entry["backward_dvf_coronal_path"] = backward_dvf_paths.get("coronal", "")
            entry["render_note"] = ""

            rendered_by_organ[organ] += 1
            rendered_success_count += 1
        except Exception as exc:
            skipped.append(
                {
                    "scope": "render",
                    "row_number": row["source_row_number"],
                    "idx": row["idx"],
                    "mask_name": row["mask_name"],
                    "subject_id": subject_id,
                    "organ": organ,
                    "reason": f"render_error: {exc!s}",
                }
            )

        should_report = (case_idx == 1) or (case_idx % progress_every == 0) or (case_idx == total_cases)
        if should_report:
            elapsed = time.time() - render_start_time
            speed = case_idx / elapsed if elapsed > 0 else 0.0
            remain = total_cases - case_idx
            eta_sec = (remain / speed) if speed > 0 else 0.0
            pct = (100.0 * case_idx / total_cases) if total_cases > 0 else 100.0
            print(
                f"[render] {case_idx}/{total_cases} ({pct:.1f}%) "
                f"ok={rendered_success_count} skipped={len(skipped)} "
                f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta_sec)}"
            )

    return rendered_by_organ, skipped


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_selected_rows_for_export(selected_entries):
    rows = []
    for entry in selected_entries:
        row = entry["row"]
        rows.append(
            {
                "idx": row["idx"],
                "mask_name": row["mask_name"],
                "subject_id": row["subject_id"],
                "organ": row["organ"],
                "coord_space": row["coord_space"],
                "pt1_x": row["pt1_x"],
                "pt1_y": row["pt1_y"],
                "pt1_z": row["pt1_z"],
                "pt2_x": row["pt2_x"],
                "pt2_y": row["pt2_y"],
                "pt2_z": row["pt2_z"],
                "pt1_back_x": row["pt1_back_x"],
                "pt1_back_y": row["pt1_back_y"],
                "pt1_back_z": row["pt1_back_z"],
                "dx": row["dx"],
                "dy": row["dy"],
                "dz": row["dz"],
                "norm_sq": row["norm_sq"],
                "voxel_error": row["voxel_error"],
                "mm_error": row["mm_error"],
                "score_12": row["score_12"],
                "score_21": row["score_21"],
                "selection_reason": "|".join(sorted(entry["reasons"])),
                "selected_by_highest": int(entry["selected_by_highest"]),
                "selected_by_median": int(entry["selected_by_median"]),
                "selected_by_zero": int(entry["selected_by_zero"]),
                "selected_zero_exact": int(entry["selected_zero_exact"]),
                "selected_zero_nearest": int(entry["selected_zero_nearest"]),
                "case_dir": entry.get("case_dir", ""),
                "image_path": entry.get("image_path", ""),
                "cycle_axial_path": entry.get("cycle_axial_path", ""),
                "cycle_sagittal_path": entry.get("cycle_sagittal_path", ""),
                "cycle_coronal_path": entry.get("cycle_coronal_path", ""),
                "forward_registered_axial_path": entry.get("forward_registered_axial_path", ""),
                "forward_registered_sagittal_path": entry.get("forward_registered_sagittal_path", ""),
                "forward_registered_coronal_path": entry.get("forward_registered_coronal_path", ""),
                "backward_registered_axial_path": entry.get("backward_registered_axial_path", ""),
                "backward_registered_sagittal_path": entry.get("backward_registered_sagittal_path", ""),
                "backward_registered_coronal_path": entry.get("backward_registered_coronal_path", ""),
                "forward_dvf_axial_path": entry.get("forward_dvf_axial_path", ""),
                "forward_dvf_sagittal_path": entry.get("forward_dvf_sagittal_path", ""),
                "forward_dvf_coronal_path": entry.get("forward_dvf_coronal_path", ""),
                "backward_dvf_axial_path": entry.get("backward_dvf_axial_path", ""),
                "backward_dvf_sagittal_path": entry.get("backward_dvf_sagittal_path", ""),
                "backward_dvf_coronal_path": entry.get("backward_dvf_coronal_path", ""),
                "render_note": entry.get("render_note", ""),
                "source_row_number": row["source_row_number"],
            }
        )
    return rows


def main():
    csv_path = resolve_csv_path(CSV_PATH)
    output_dir = resolve_output_dir(csv_path)
    dataset_root = resolve_project_path(DATASET_ROOT)

    print(f"CSV path: {csv_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print("Selection policy: highest mm_error, median mm_error, zero-error-or-nearest per organ")
    print(f"Dry run: {DRY_RUN}")

    all_rows, skipped_parse = load_cycle_rows(csv_path)
    selected_entries, summary_rows = select_cases(all_rows)
    skipped_rows = list(skipped_parse)

    print(f"Loaded valid rows: {len(all_rows)}")
    print(f"Selected unique rows: {len(selected_entries)}")
    print(f"Rows skipped during parse: {len(skipped_parse)}")

    if DRY_RUN:
        print("Dry run enabled: skipping image rendering.")
        rendered_by_organ = defaultdict(int)
        skipped_render = []
    else:
        rendered_by_organ, skipped_render = render_selected_cases(
            selected_entries=selected_entries,
            output_dir=output_dir,
            dataset_root=dataset_root,
        )
        skipped_rows.extend(skipped_render)
        print(f"Rendered images: {sum(rendered_by_organ.values())}")
        print(f"Rows skipped during render: {len(skipped_render)}")

    for summary in summary_rows:
        summary["rendered_images"] = int(rendered_by_organ.get(summary["organ"], 0))

    selected_rows_for_export = build_selected_rows_for_export(selected_entries)
    summary_rows_for_export = sorted(summary_rows, key=lambda row: row["organ"])
    skipped_rows_for_export = sorted(
        skipped_rows,
        key=lambda row: (row.get("scope", ""), int(row.get("row_number", 0) or 0)),
    )

    selected_csv = output_dir / "selected_cases.csv"
    summary_csv = output_dir / "selection_summary.csv"
    skipped_csv = output_dir / "skipped_cases.csv"

    write_csv(
        selected_csv,
        [
            "idx",
            "mask_name",
            "subject_id",
            "organ",
            "coord_space",
            "pt1_x",
            "pt1_y",
            "pt1_z",
            "pt2_x",
            "pt2_y",
            "pt2_z",
            "pt1_back_x",
            "pt1_back_y",
            "pt1_back_z",
            "dx",
            "dy",
            "dz",
            "norm_sq",
            "voxel_error",
            "mm_error",
            "score_12",
            "score_21",
            "selection_reason",
            "selected_by_highest",
            "selected_by_median",
            "selected_by_zero",
            "selected_zero_exact",
            "selected_zero_nearest",
            "case_dir",
            "image_path",
            "cycle_axial_path",
            "cycle_sagittal_path",
            "cycle_coronal_path",
            "forward_registered_axial_path",
            "forward_registered_sagittal_path",
            "forward_registered_coronal_path",
            "backward_registered_axial_path",
            "backward_registered_sagittal_path",
            "backward_registered_coronal_path",
            "forward_dvf_axial_path",
            "forward_dvf_sagittal_path",
            "forward_dvf_coronal_path",
            "backward_dvf_axial_path",
            "backward_dvf_sagittal_path",
            "backward_dvf_coronal_path",
            "render_note",
            "source_row_number",
        ],
        selected_rows_for_export,
    )
    write_csv(
        summary_csv,
        [
            "organ",
            "total_rows",
            "exact_zero_rows",
            "selected_unique",
            "selected_by_highest",
            "selected_by_median",
            "selected_by_zero",
            "zero_selection_kind",
            "highest_idx",
            "median_idx",
            "zero_idx",
            "min_mm_error",
            "median_mm_error",
            "max_mm_error",
            "rendered_images",
        ],
        summary_rows_for_export,
    )
    write_csv(
        skipped_csv,
        ["scope", "row_number", "idx", "mask_name", "subject_id", "organ", "reason"],
        skipped_rows_for_export,
    )

    print(f"selected cases csv saved: {selected_csv}")
    print(f"selection summary csv saved: {summary_csv}")
    print(f"skipped cases csv saved: {skipped_csv}")
    if not DRY_RUN:
        print(f"images saved under: {output_dir / 'images'}")


if __name__ == "__main__":
    main()
