#!/usr/bin/env python3
"""Generate DVF in-memory and use it for one Test->Retest point match."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from run_single_pair_elastix import generate_dvf_from_paths


# -----------------------------
# In-code configuration (no CLI)
# -----------------------------
SUBJECT_DIR = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_009"
)
TEST_PATH = SUBJECT_DIR / "QUADRA_HC_009_Test_CT-AC.nii.gz"
RETEST_PATH = SUBJECT_DIR / "QUADRA_HC_009_Retest_CT-AC.nii.gz"
OUTPUT_PNG = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/outputs/quadra_hc_009/point_match_visualization.png"
)

HU_THRESHOLD = -900.0
RANDOM_SEED = 42
MAX_ATTEMPTS = 200_000


def normalize_slice(slice_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize a 2D image to [0, 1] for consistent display."""
    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype=np.float32)
    normalized = (slice_2d - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def lps_vector_to_ras(vector_lps: np.ndarray) -> np.ndarray:
    """Convert ITK/LPS displacement components to Nibabel/RAS components."""
    return np.array([-vector_lps[0], -vector_lps[1], vector_lps[2]], dtype=np.float64)


def is_in_bounds(index_xyz: np.ndarray, shape_xyz: tuple[int, int, int]) -> bool:
    """Return True if an integer voxel index is inside image bounds."""
    return (
        0 <= index_xyz[0] < shape_xyz[0]
        and 0 <= index_xyz[1] < shape_xyz[1]
        and 0 <= index_xyz[2] < shape_xyz[2]
    )


def select_and_match_point(
    test_img: nib.Nifti1Image,
    retest_img: nib.Nifti1Image,
    dvf_image,
    threshold_hu: float,
    seed: int,
    max_attempts: int,
) -> dict:
    """Sample a valid Test point and map it to Retest with DVF in physical space."""
    test_shape = tuple(int(v) for v in test_img.shape[:3])
    retest_shape = tuple(int(v) for v in retest_img.shape[:3])

    dvf_size = dvf_image.GetLargestPossibleRegion().GetSize()
    dvf_shape = (int(dvf_size[0]), int(dvf_size[1]), int(dvf_size[2]))
    if dvf_shape != test_shape:
        raise ValueError(f"DVF shape {dvf_shape} does not match Test image shape {test_shape}.")

    test_data = test_img.dataobj
    inv_retest_affine = np.linalg.inv(retest_img.affine)
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        query_vox = np.array(
            [
                int(rng.integers(0, test_shape[0])),
                int(rng.integers(0, test_shape[1])),
                int(rng.integers(0, test_shape[2])),
            ],
            dtype=np.int64,
        )

        query_value = float(test_data[query_vox[0], query_vox[1], query_vox[2]])
        if query_value <= threshold_hu:
            continue

        # DVF comes from ITK and is in LPS physical components at Test-grid voxel.
        dvf_lps = np.asarray(
            dvf_image.GetPixel((int(query_vox[0]), int(query_vox[1]), int(query_vox[2]))),
            dtype=np.float64,
        )
        dvf_ras = lps_vector_to_ras(dvf_lps)

        query_mm = nib.affines.apply_affine(test_img.affine, query_vox.astype(np.float64))
        matched_mm = query_mm + dvf_ras
        matched_vox_float = nib.affines.apply_affine(inv_retest_affine, matched_mm)
        matched_vox_round = np.rint(matched_vox_float).astype(np.int64)

        if not is_in_bounds(matched_vox_round, retest_shape):
            continue

        return {
            "query_vox": query_vox,
            "query_value_hu": query_value,
            "query_mm": query_mm,
            "dvf_lps_mm": dvf_lps,
            "dvf_ras_mm": dvf_ras,
            "matched_mm": matched_mm,
            "matched_vox_float": matched_vox_float,
            "matched_vox_round": matched_vox_round,
        }

    raise RuntimeError(
        f"Failed to find a valid foreground Test point mapping in {max_attempts} attempts."
    )


def draw_point_marker(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a point marker (circle + crosshair) at pixel coordinates."""
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=color, width=2)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=color, width=2)


def save_match_visualization(
    test_img: nib.Nifti1Image, retest_img: nib.Nifti1Image, match: dict, output_png: Path
) -> None:
    """Save side-by-side Test/Retest slices with query and matched points highlighted."""
    qx, qy, qz = (int(v) for v in match["query_vox"])
    mx, my, mz = (int(v) for v in match["matched_vox_round"])

    # Transpose to map voxel (x, y) to draw coords (x, y) in PIL.
    test_slice = np.asarray(test_img.dataobj[:, :, qz], dtype=np.float32).T
    retest_slice = np.asarray(retest_img.dataobj[:, :, mz], dtype=np.float32).T

    # Shared window from subsampled Test volume.
    test_sample = np.asarray(test_img.dataobj[::4, ::4, ::4], dtype=np.float32)
    low = float(np.percentile(test_sample, 1.0))
    high = float(np.percentile(test_sample, 99.0))

    test_norm = normalize_slice(test_slice, low, high)
    retest_norm = normalize_slice(retest_slice, low, high)

    test_rgb = (np.stack([test_norm, test_norm, test_norm], axis=-1) * 255.0).astype(np.uint8)
    retest_rgb = (np.stack([retest_norm, retest_norm, retest_norm], axis=-1) * 255.0).astype(np.uint8)

    test_panel = Image.fromarray(test_rgb)
    retest_panel = Image.fromarray(retest_rgb)

    draw_test = ImageDraw.Draw(test_panel)
    draw_retest = ImageDraw.Draw(retest_panel)
    draw_point_marker(draw_test, qx, qy, (0, 255, 0))
    draw_point_marker(draw_retest, mx, my, (255, 165, 0))

    panel_w, panel_h = test_panel.size
    pad = 16
    gap = 14
    title_h = 36
    canvas_w = pad * 2 + panel_w * 2 + gap
    canvas_h = pad * 2 + title_h + panel_h + 56
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw_canvas = ImageDraw.Draw(canvas)

    x_left = pad
    x_right = pad + panel_w + gap
    y_img = pad + title_h

    draw_canvas.text((x_left, pad), f"Test (query) z={qz}", fill=(255, 255, 255))
    draw_canvas.text((x_right, pad), f"Retest (match) z={mz}", fill=(255, 255, 255))

    canvas.paste(test_panel, (x_left, y_img))
    canvas.paste(retest_panel, (x_right, y_img))

    footer_y = y_img + panel_h + 10
    draw_canvas.text(
        (pad, footer_y),
        f"Query voxel: ({qx}, {qy}, {qz}) | Matched voxel: ({mx}, {my}, {mz})",
        fill=(230, 230, 230),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def main() -> None:
    """Generate DVF in-memory, find one matched point, and save visualization."""
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {TEST_PATH}")
    if not RETEST_PATH.exists():
        raise FileNotFoundError(f"Retest image not found: {RETEST_PATH}")

    print("Generating DVF in-memory from Test/Retest registration...", flush=True)
    dvf_image = generate_dvf_from_paths(TEST_PATH, RETEST_PATH, save_path=None)

    print("Loading Test and Retest images with nibabel...", flush=True)
    test_img = nib.load(str(TEST_PATH))
    retest_img = nib.load(str(RETEST_PATH))

    print("Selecting one foreground query point and matching with DVF...", flush=True)
    match = select_and_match_point(
        test_img=test_img,
        retest_img=retest_img,
        dvf_image=dvf_image,
        threshold_hu=HU_THRESHOLD,
        seed=RANDOM_SEED,
        max_attempts=MAX_ATTEMPTS,
    )

    print(
        f"Query voxel (Test): {tuple(int(v) for v in match['query_vox'])} "
        f"HU={match['query_value_hu']:.2f}",
        flush=True,
    )
    print(
        f"Query physical mm (RAS): {np.array2string(match['query_mm'], precision=3)}",
        flush=True,
    )
    print(
        f"DVF at query (LPS mm): {np.array2string(match['dvf_lps_mm'], precision=3)}",
        flush=True,
    )
    print(
        f"Matched voxel float (Retest): {np.array2string(match['matched_vox_float'], precision=3)}",
        flush=True,
    )
    print(
        f"Matched voxel rounded (Retest): {tuple(int(v) for v in match['matched_vox_round'])}",
        flush=True,
    )
    print(
        f"Matched physical mm (RAS): {np.array2string(match['matched_mm'], precision=3)}",
        flush=True,
    )

    print(f"Saving point match visualization: {OUTPUT_PNG}", flush=True)
    save_match_visualization(
        test_img=test_img,
        retest_img=retest_img,
        match=match,
        output_png=OUTPUT_PNG,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
