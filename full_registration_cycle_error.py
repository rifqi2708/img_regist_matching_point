#!/usr/bin/env python3
"""Full-cycle error pipeline using registration + DVF point matching."""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from datetime import datetime

import itk
import numpy as np

# Keep matplotlib fully headless and writable in this environment.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig_cycle_error")
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    from rd_cycle_error_helper import (
        print_summary,
        sample_random_mask_points,
        validate_fixed_point,
        validate_mask_file,
        validate_origin_mask,
        validate_sampled_points_inside_mask,
        visualize_cycle_result,
        write_points_csv_with_mask,
        write_summary_with_mask_labels_csv,
    )
except ImportError:
    from tools.rd_cycle_error_helper import (
        print_summary,
        sample_random_mask_points,
        validate_fixed_point,
        validate_mask_file,
        validate_origin_mask,
        validate_sampled_points_inside_mask,
        visualize_cycle_result,
        write_points_csv_with_mask,
        write_summary_with_mask_labels_csv,
    )


DATASET_ROOT = "quadra_cropped_eval"
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
MASKS_ROOT = os.path.join(DATASET_ROOT, "masks")
OUTPUT_DIR = "outputs/registration_cycle_error"

POINT_MODE = "random"
FIXED_POINT = None
NUM_POINTS_PER_MASK = 100
SEED = 0
IS_MRI = False
VISUALIZE = False
VIZ_SHOW = False
VIZ_SAVE = True
EXPORT_CSV = True
VIZ_LAYOUT = (2, 2)

MAX_RANDOM_RETRY_FACTOR = 50
MIN_RANDOM_RETRY_ATTEMPTS = 100


class PointMappingError(RuntimeError):
    """Raised when a point cannot be mapped safely through DVF."""


def is_nifti_file(name):
    return isinstance(name, str) and (name.endswith(".nii.gz") or name.endswith(".nii"))


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def list_subject_ids(images_root):
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images root not found: {images_root}")
    subjects = []
    for name in sorted(os.listdir(images_root)):
        path = os.path.join(images_root, name)
        if os.path.isdir(path):
            subjects.append(name)
    if not subjects:
        raise RuntimeError(f"No subject directories found under: {images_root}")
    return subjects


def resolve_subject_pair(subject_id, images_root, masks_root):
    subject_image_dir = os.path.join(images_root, subject_id)
    image_files = []
    for name in sorted(os.listdir(subject_image_dir)):
        path = os.path.join(subject_image_dir, name)
        if os.path.isfile(path) and is_nifti_file(name):
            image_files.append(name)

    test_files = [name for name in image_files if "_Test_" in name]
    retest_files = [name for name in image_files if "_Retest_" in name]
    if len(test_files) != 1 or len(retest_files) != 1:
        raise RuntimeError(
            f"Expected one Test and one Retest image in '{subject_image_dir}', "
            f"got Test={len(test_files)} Retest={len(retest_files)}."
        )

    test_name = test_files[0]
    retest_name = retest_files[0]
    test_stem = strip_nii_suffix(test_name)
    mask1_dir = os.path.join(masks_root, subject_id, test_stem)
    if not os.path.isdir(mask1_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask1_dir}")

    return {
        "subject_id": subject_id,
        "im1_file": os.path.join(subject_image_dir, test_name),
        "im2_file": os.path.join(subject_image_dir, retest_name),
        "mask1_dir": mask1_dir,
    }


def list_mask_files(mask_dir):
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    files = {}
    for name in sorted(os.listdir(mask_dir)):
        path = os.path.join(mask_dir, name)
        if not os.path.isfile(path):
            continue
        if not is_nifti_file(name):
            continue
        files[name] = path
    if not files:
        raise RuntimeError(f"No .nii/.nii.gz mask files found in directory: {mask_dir}")
    return files


def _itk_array_yxz(itk_image):
    # ITK array view is z,y,x; helper functions and visualizer use y,x,z.
    array_zyx = np.asarray(itk.array_view_from_image(itk_image))
    if array_zyx.ndim != 3:
        raise ValueError(f"Expected 3D ITK image array, got shape {array_zyx.shape}")
    return np.transpose(array_zyx, (1, 2, 0))


def load_mask_array(im_file, mask_file, is_mri=False, ref_image=None, mask_name="mask"):
    del im_file, is_mri  # kept in signature for interface parity
    validate_mask_file(mask_file, mask_name)
    mask_itk = itk.imread(str(mask_file))
    mask_array = _itk_array_yxz(mask_itk)
    mask_array = validate_origin_mask(mask_array, mask_array, mask_name)
    if ref_image is not None and tuple(mask_array.shape) != tuple(np.asarray(ref_image).shape):
        raise ValueError(
            f"{mask_name} shape {mask_array.shape} does not match reference image shape {np.asarray(ref_image).shape}."
        )
    return mask_array


def _as_string_list(value: str | int | float | bool | list | tuple) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for item in values:
        if isinstance(item, bool):
            out.append("true" if item else "false")
        else:
            out.append(str(item))
    return out


def _apply_overrides(parameter_map: dict, overrides: dict[str, str | int | float | bool | list]) -> dict:
    for key, value in overrides.items():
        parameter_map[key] = _as_string_list(value)
    return parameter_map


def build_parameter_object() -> itk.ParameterObject:
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


def run_registration_and_compute_dvf(fixed_image: itk.Image, moving_image: itk.Image) -> itk.Image:
    parameter_object = build_parameter_object()
    _, transform_parameter_object = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    with tempfile.TemporaryDirectory(prefix="cycle_dvf_") as temp_dir:
        transformix_filter = itk.TransformixFilter.New(moving_image)
        transformix_filter.SetTransformParameterObject(transform_parameter_object)
        transformix_filter.SetComputeDeformationField(True)
        transformix_filter.SetLogToConsole(False)
        transformix_filter.SetOutputDirectory(temp_dir)
        transformix_filter.Update()
        dvf_image = transformix_filter.GetOutputDeformationField()

    return dvf_image


def _safe_itk_cont_index(image: itk.Image, point_physical_xyz: np.ndarray) -> np.ndarray:
    try:
        cont_index = image.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(point_physical_xyz, dtype=float)))
    except Exception as exc:
        raise PointMappingError(f"Failed physical->index conversion in ITK image: {exc}") from exc
    return np.array([cont_index[0], cont_index[1], cont_index[2]], dtype=np.float64)


def _is_in_bounds(index_xyz: np.ndarray, shape_xyz: tuple[int, int, int]) -> bool:
    return (
        0 <= int(index_xyz[0]) < int(shape_xyz[0])
        and 0 <= int(index_xyz[1]) < int(shape_xyz[1])
        and 0 <= int(index_xyz[2]) < int(shape_xyz[2])
    )


def _index_xyz_to_physical(image: itk.Image, index_xyz: np.ndarray) -> np.ndarray:
    itk_index = itk.Index[3]()
    itk_index[0] = int(index_xyz[0])
    itk_index[1] = int(index_xyz[1])
    itk_index[2] = int(index_xyz[2])
    return np.array(image.TransformIndexToPhysicalPoint(itk_index), dtype=np.float64)


def load_image_context(im_file, is_mri=False):
    del is_mri
    itk_image = itk.imread(str(im_file), itk.F)
    img_yxz = _itk_array_yxz(itk_image).astype(np.float32, copy=False)

    itk_size = itk_image.GetLargestPossibleRegion().GetSize()
    itk_shape_xyz = (int(itk_size[0]), int(itk_size[1]), int(itk_size[2]))
    read_shape_xyz = (int(img_yxz.shape[1]), int(img_yxz.shape[0]), int(img_yxz.shape[2]))

    spacing_xyz = itk_image.GetSpacing()
    spacing_yxz = (float(spacing_xyz[1]), float(spacing_xyz[0]), float(spacing_xyz[2]))
    origin_info = {
        "img": img_yxz,
        "shape": (1, int(img_yxz.shape[0]), int(img_yxz.shape[1]), int(img_yxz.shape[2])),
        "spacing": spacing_yxz,
    }

    return {
        "im_file": im_file,
        "img": origin_info,
        "itk_image": itk_image,
        "itk_shape_xyz": itk_shape_xyz,
        "read_shape_xyz": read_shape_xyz,
    }


def map_point_with_dvf(point_query_xyz, query_ctx, key_ctx, dvf_image):
    point_query_xyz = np.asarray(point_query_xyz, dtype=np.int64)
    if not _is_in_bounds(point_query_xyz, query_ctx["itk_shape_xyz"]):
        raise PointMappingError(
            f"Query index {point_query_xyz.tolist()} out of bounds for shape {query_ctx['itk_shape_xyz']}."
        )

    query_phys = _index_xyz_to_physical(query_ctx["itk_image"], point_query_xyz)
    dvf_vector = np.array(
        dvf_image.GetPixel((int(point_query_xyz[0]), int(point_query_xyz[1]), int(point_query_xyz[2]))),
        dtype=np.float64,
    )
    if dvf_vector.shape != (3,):
        raise PointMappingError(f"Unexpected DVF vector shape at query: {dvf_vector.shape}")

    matched_phys = query_phys + dvf_vector
    matched_cont_itk = _safe_itk_cont_index(key_ctx["itk_image"], matched_phys)
    matched_round_itk = np.rint(matched_cont_itk).astype(np.int64)
    if not _is_in_bounds(matched_round_itk, key_ctx["itk_shape_xyz"]):
        raise PointMappingError(
            f"Matched index {matched_round_itk.tolist()} out of bounds for shape {key_ctx['itk_shape_xyz']}."
        )

    return {
        "matched_point_xyz": matched_round_itk.astype(np.int64),
        "query_phys": query_phys,
        "dvf_vector": dvf_vector,
        "matched_phys": matched_phys,
        "matched_cont_itk": matched_cont_itk,
        "matched_round_itk": matched_round_itk,
    }


def compute_cycle_for_point(point_1, ctx_12, ctx_21, forward_dvf, backward_dvf):
    point_1 = np.asarray(point_1, dtype=np.int64)

    map_12 = map_point_with_dvf(point_1, ctx_12, ctx_21, forward_dvf)
    point_2 = np.asarray(map_12["matched_point_xyz"], dtype=np.int64)

    map_21 = map_point_with_dvf(point_2, ctx_21, ctx_12, backward_dvf)
    point_1_back = np.asarray(map_21["matched_point_xyz"], dtype=np.int64)

    delta = point_1_back.astype(np.float64) - point_1.astype(np.float64)
    voxel_error = float(np.linalg.norm(delta))

    point_1_phys = _index_xyz_to_physical(ctx_12["itk_image"], point_1)
    point_1_back_phys = _index_xyz_to_physical(ctx_12["itk_image"], point_1_back)
    mm_error = float(np.linalg.norm(point_1_back_phys - point_1_phys))

    nan_score = float("nan")
    return {
        "pt1": point_1,
        "pt2": point_2,
        "pt1_back": point_1_back,
        "score_12": nan_score,
        "score_21": nan_score,
        "voxel_error": voxel_error,
        "mm_error": mm_error,
    }


def _validate_viz_layout(viz_layout):
    if tuple(viz_layout) != (2, 2):
        raise ValueError(f"Only 2x2 layout is supported, got {viz_layout}")


def run_cycle_pair(
    subject_id,
    im1_file,
    im2_file,
    mask1_dir,
    point_mode=POINT_MODE,
    fixed_point=FIXED_POINT,
    num_points_per_mask=NUM_POINTS_PER_MASK,
    seed=SEED,
    is_mri=IS_MRI,
    visualize=VISUALIZE,
    viz_show=VIZ_SHOW,
    viz_save=VIZ_SAVE,
    viz_dir=OUTPUT_DIR,
    viz_layout=VIZ_LAYOUT,
):
    if point_mode not in ("random", "fixed"):
        raise ValueError("point_mode must be either 'random' or 'fixed'")
    if point_mode == "random" and num_points_per_mask < 1:
        raise ValueError("num_points_per_mask must be >= 1 when point_mode='random'")
    _validate_viz_layout(viz_layout)

    for im_path in (im1_file, im2_file):
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")

    if point_mode == "random":
        mask_map_1 = list_mask_files(mask1_dir)
        mask_items = sorted(mask_map_1.items())
    else:
        if fixed_point is None:
            raise ValueError("fixed_point must be provided when point_mode='fixed'")
        mask_items = [("fixed_point", None)]

    if visualize and viz_save:
        os.makedirs(viz_dir, exist_ok=True)

    time_start = time.time()
    ctx1 = load_image_context(im1_file, is_mri=is_mri)
    ctx2 = load_image_context(im2_file, is_mri=is_mri)
    time_after_load = time.time()
    print(f"[{subject_id}] image context loading time: {time_after_load - time_start:.3f}s")

    forward_dvf = run_registration_and_compute_dvf(ctx1["itk_image"], ctx2["itk_image"])
    time_after_forward = time.time()
    print(f"[{subject_id}] forward registration+DVF time: {time_after_forward - time_after_load:.3f}s")

    backward_dvf = run_registration_and_compute_dvf(ctx2["itk_image"], ctx1["itk_image"])
    time_after_backward = time.time()
    print(f"[{subject_id}] backward registration+DVF time: {time_after_backward - time_after_forward:.3f}s")

    all_results = []
    per_mask_results = {}

    for mask_idx, (mask_file_name, mask1_path) in enumerate(mask_items):
        mask_label = f"{subject_id}/{mask_file_name}"
        print(f"[{subject_id}] Processing mask: {mask_file_name}")

        if point_mode == "fixed":
            candidate_points = np.asarray([validate_fixed_point(fixed_point, ctx1["img"])], dtype=np.int64)
            max_attempts = 1
        else:
            mask1_array = load_mask_array(
                im1_file,
                mask1_path,
                is_mri=is_mri,
                ref_image=ctx1["img"]["img"],
                mask_name=f"{subject_id}:{mask_file_name}",
            )
            max_attempts = max(
                int(num_points_per_mask) * int(MAX_RANDOM_RETRY_FACTOR),
                int(MIN_RANDOM_RETRY_ATTEMPTS),
            )
            candidate_points = None

        mask_results = []
        attempts = 0
        batch_id = 0
        last_error = None

        while len(mask_results) < (1 if point_mode == "fixed" else int(num_points_per_mask)):
            if attempts >= max_attempts:
                break

            if point_mode == "fixed":
                points_batch = candidate_points
            else:
                remaining = int(num_points_per_mask) - len(mask_results)
                points_batch = sample_random_mask_points(
                    mask1_array,
                    num_points=remaining,
                    seed=int(seed) + int(mask_idx) * 100000 + int(batch_id),
                )
                validate_sampled_points_inside_mask(points_batch, mask1_array, f"{subject_id}:{mask_file_name}")
                batch_id += 1

            for point in points_batch:
                if attempts >= max_attempts:
                    break
                attempts += 1
                try:
                    result = compute_cycle_for_point(
                        point_1=point,
                        ctx_12=ctx1,
                        ctx_21=ctx2,
                        forward_dvf=forward_dvf,
                        backward_dvf=backward_dvf,
                    )
                except PointMappingError as exc:
                    last_error = exc
                    if point_mode == "fixed":
                        raise
                    continue

                result["mask_name"] = mask_label
                result["subject_id"] = subject_id
                mask_results.append(result)
                all_results.append(result)

                if visualize:
                    save_path = None
                    if viz_save:
                        pt1 = result["pt1"]
                        pt2 = result["pt2"]
                        pt1_back = result["pt1_back"]
                        safe_mask = strip_nii_suffix(mask_file_name)
                        save_name = (
                            f"{subject_id}_{safe_mask}_cycle_{len(mask_results)-1:03d}_"
                            f"q_{pt1[0]}_{pt1[1]}_{pt1[2]}_"
                            f"m_{pt2[0]}_{pt2[1]}_{pt2[2]}_"
                            f"c_{pt1_back[0]}_{pt1_back[1]}_{pt1_back[2]}.png"
                        )
                        save_path = os.path.join(viz_dir, save_name)

                    visualize_cycle_result(
                        query_img=ctx1["img"]["img"],
                        target_img=ctx2["img"]["img"],
                        result=result,
                        out_path=save_path,
                        show=viz_show,
                        is_mri=is_mri,
                        viz_layout=viz_layout,
                    )

                if point_mode == "fixed":
                    break
                if len(mask_results) >= int(num_points_per_mask):
                    break

            if point_mode == "fixed":
                break

        required_points = 1 if point_mode == "fixed" else int(num_points_per_mask)
        if len(mask_results) < required_points:
            details = (
                f"Only collected {len(mask_results)}/{required_points} valid mapped points "
                f"after {attempts} attempts for mask {mask_label}."
            )
            if last_error is not None:
                details += f" Last mapping error: {last_error}"
            raise RuntimeError(details)

        per_mask_results[mask_label] = mask_results

    del forward_dvf
    del backward_dvf
    del ctx1
    del ctx2
    gc.collect()

    if not all_results:
        raise RuntimeError(f"No cycle results were produced for subject '{subject_id}'.")

    return all_results, per_mask_results


def run_dataset_cycle(
    dataset_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
    point_mode=POINT_MODE,
    fixed_point=FIXED_POINT,
    num_points_per_mask=NUM_POINTS_PER_MASK,
    seed=SEED,
    is_mri=IS_MRI,
    visualize=VISUALIZE,
    viz_show=VIZ_SHOW,
    viz_save=VIZ_SAVE,
    export_csv=EXPORT_CSV,
    viz_layout=VIZ_LAYOUT,
):
    images_root = os.path.join(dataset_root, "images")
    masks_root = os.path.join(dataset_root, "masks")

    if export_csv or (visualize and viz_save):
        os.makedirs(output_dir, exist_ok=True)

    subject_ids = list_subject_ids(images_root)
    print(f"Found {len(subject_ids)} subjects under '{images_root}'.")

    all_results = []
    per_mask_aggregate = {}
    failed_subjects = []
    processed_subjects = 0

    for subject_idx, subject_id in enumerate(subject_ids, start=1):
        print(f"\n[{subject_idx:03d}/{len(subject_ids):03d}] Subject: {subject_id}")
        try:
            pair = resolve_subject_pair(subject_id, images_root, masks_root)
            pair_results, pair_per_mask = run_cycle_pair(
                subject_id=pair["subject_id"],
                im1_file=pair["im1_file"],
                im2_file=pair["im2_file"],
                mask1_dir=pair["mask1_dir"],
                point_mode=point_mode,
                fixed_point=fixed_point,
                num_points_per_mask=num_points_per_mask,
                seed=seed,
                is_mri=is_mri,
                visualize=visualize,
                viz_show=viz_show,
                viz_save=viz_save,
                viz_dir=output_dir,
                viz_layout=viz_layout,
            )
            all_results.extend(pair_results)
            for mask_name, mask_results in pair_per_mask.items():
                per_mask_aggregate.setdefault(mask_name, []).extend(mask_results)
            processed_subjects += 1
        except Exception as exc:
            failed_subjects.append((subject_id, str(exc)))
            print(f"WARNING: skipping subject '{subject_id}' due to error: {exc}")
            continue

    gc.collect()

    if not all_results:
        raise RuntimeError("No cycle results were produced for the dataset.")

    print("\nDataset summary across all masks and subjects")
    global_voxel_stats, global_mm_stats = print_summary(all_results)

    per_mask_rows = []
    for mask_name in sorted(per_mask_aggregate.keys()):
        voxel_stats, mm_stats = print_summary(per_mask_aggregate[mask_name])
        per_mask_rows.append({"mask_name": mask_name, "voxel_stats": voxel_stats, "mm_stats": mm_stats})

    if export_csv:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv_path = os.path.join(output_dir, f"cycle_summary_{run_stamp}.csv")
        points_csv_path = os.path.join(output_dir, f"cycle_points_{run_stamp}.csv")
        write_summary_with_mask_labels_csv(
            per_mask_rows,
            summary_csv_path,
            global_voxel_stats=global_voxel_stats,
            global_mm_stats=global_mm_stats,
            all_masks_label="ALL_MASKS",
        )
        write_points_csv_with_mask(all_results, points_csv_path)
        print(f"summary csv saved: {summary_csv_path}")
        print(f"points csv saved: {points_csv_path}")

    print("\nRun complete")
    print(f"Processed subjects: {processed_subjects}")
    print(f"Failed subjects: {len(failed_subjects)}")
    if failed_subjects:
        for subject_id, reason in failed_subjects:
            print(f"  - {subject_id}: {reason}")


if __name__ == "__main__":
    try:
        run_dataset_cycle(
            dataset_root=DATASET_ROOT,
            output_dir=OUTPUT_DIR,
            point_mode=POINT_MODE,
            fixed_point=FIXED_POINT,
            num_points_per_mask=NUM_POINTS_PER_MASK,
            seed=SEED,
            is_mri=IS_MRI,
            visualize=VISUALIZE,
            viz_show=VIZ_SHOW,
            viz_save=VIZ_SAVE,
            export_csv=EXPORT_CSV,
            viz_layout=VIZ_LAYOUT,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
