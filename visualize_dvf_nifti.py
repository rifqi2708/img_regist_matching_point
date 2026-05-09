#!/usr/bin/env python3
"""Create quick-look slice visualizations for a DVF NIfTI file."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# -----------------------------
# In-code configuration
# -----------------------------
DVF_PATH = Path("/Users/rifqiab2708/Documents/img_regist_matching_point/outputs/quadra_hc_016/dvf.nii.gz")
FIXED_IMAGE_PATH = Path(
    "/Users/rifqiab2708/Documents/img_regist_matching_point/quadra_cropped_eval/images/quadra_hc_016/QUADRA_HC_016_Test_CT-AC.nii.gz"
)
OUTPUT_DIR = Path("/Users/rifqiab2708/Documents/img_regist_matching_point/outputs/quadra_hc_016")
OUTPUT_STEM = "dvf_visualization"
QUIVER_STRIDE = 24
QUIVER_SCALE = 1.0
HEATMAP_ALPHA = 0.34


def squeeze_vector_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 4 and shape[-1] == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    if len(shape) == 5 and shape[3] == 1 and shape[4] == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    raise ValueError(f"Unsupported DVF shape {shape}; expected (X,Y,Z,3) or (X,Y,Z,1,3).")


def get_vector_slice(dataobj, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        vector_slice = np.asarray(dataobj[index, :, :, 0, :] if dataobj.ndim == 5 else dataobj[index, :, :, :])
    elif axis == 1:
        vector_slice = np.asarray(dataobj[:, index, :, 0, :] if dataobj.ndim == 5 else dataobj[:, index, :, :])
    elif axis == 2:
        vector_slice = np.asarray(dataobj[:, :, index, 0, :] if dataobj.ndim == 5 else dataobj[:, :, index, :])
    else:
        raise ValueError(f"Invalid axis {axis}")
    return vector_slice.astype(np.float32, copy=False)


def lps_to_ras(vectors_lps: np.ndarray) -> np.ndarray:
    vectors_ras = vectors_lps.copy()
    vectors_ras[..., 0] *= -1.0
    vectors_ras[..., 1] *= -1.0
    return vectors_ras


def ras_mm_to_voxel_vectors(vectors_ras: np.ndarray, affine: np.ndarray) -> np.ndarray:
    rotation_scale = affine[:3, :3]
    inv_rotation_scale = np.linalg.inv(rotation_scale)
    return np.einsum("ij,...j->...i", inv_rotation_scale, vectors_ras, optimize=True)


def magnitude_image(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=-1)


def percentile_limits(values: np.ndarray) -> tuple[float, float]:
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def normalize_slice(slice_2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(slice_2d, dtype=np.float32)
    normalized = (slice_2d - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def sample_scalar_plane(
    image_data: np.ndarray,
    image_affine: np.ndarray,
    dvf_affine: np.ndarray,
    shape_xyz: tuple[int, int, int],
    axis: int,
    index: int,
) -> np.ndarray:
    nx, ny, nz = shape_xyz
    if axis == 2:
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        coords = np.stack([ii, jj, np.full_like(ii, index)], axis=-1)
    elif axis == 1:
        ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        coords = np.stack([ii, np.full_like(ii, index), kk], axis=-1)
    elif axis == 0:
        jj, kk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
        coords = np.stack([np.full_like(jj, index), jj, kk], axis=-1)
    else:
        raise ValueError(f"Invalid axis {axis}")

    world = nib.affines.apply_affine(dvf_affine, coords.reshape(-1, 3))
    inv_image_affine = np.linalg.inv(image_affine)
    image_ijk = nib.affines.apply_affine(inv_image_affine, world)
    image_ijk = np.rint(image_ijk).astype(np.int64)

    sampled = np.full((coords.shape[0] * coords.shape[1],), np.nan, dtype=np.float32)
    valid = (
        (image_ijk[:, 0] >= 0)
        & (image_ijk[:, 0] < image_data.shape[0])
        & (image_ijk[:, 1] >= 0)
        & (image_ijk[:, 1] < image_data.shape[1])
        & (image_ijk[:, 2] >= 0)
        & (image_ijk[:, 2] < image_data.shape[2])
    )
    sampled[valid] = image_data[
        image_ijk[valid, 0],
        image_ijk[valid, 1],
        image_ijk[valid, 2],
    ]
    return sampled.reshape(coords.shape[0], coords.shape[1])


def add_plane(
    ax: plt.Axes,
    fixed_slice: np.ndarray | None,
    voxel_vectors: np.ndarray,
    title: str,
    component_x: int,
    component_y: int,
    stride: int,
    scale: float,
    heatmap_alpha: float,
    x_label: str,
    y_label: str,
    fixed_window: tuple[float, float],
) -> None:
    mag = magnitude_image(voxel_vectors).T
    vmin, vmax = percentile_limits(mag)
    if fixed_slice is not None:
        fixed_norm = normalize_slice(fixed_slice.T, fixed_window[0], fixed_window[1])
        ax.imshow(fixed_norm, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    ax.imshow(mag, cmap="magma", origin="lower", vmin=vmin, vmax=vmax, alpha=heatmap_alpha)

    field = voxel_vectors.transpose(1, 0, 2)
    ys = np.arange(0, field.shape[0], stride)
    xs = np.arange(0, field.shape[1], stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    u = field[ys[:, None], xs[None, :], component_x] * scale
    v = field[ys[:, None], xs[None, :], component_y] * scale
    ax.quiver(grid_x, grid_y, u, v, color="cyan", angles="xy", scale_units="xy", scale=1.0, width=0.003)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def infer_output_base(dvf_path: Path) -> Path:
    base = dvf_path.name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return dvf_path.with_name(base)


def save_single_plane(
    output: Path,
    fixed_slice: np.ndarray | None,
    voxel_vectors: np.ndarray,
    title: str,
    component_x: int,
    component_y: int,
    stride: int,
    scale: float,
    heatmap_alpha: float,
    x_label: str,
    y_label: str,
    fixed_window: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    add_plane(
        ax=ax,
        fixed_slice=fixed_slice,
        voxel_vectors=voxel_vectors,
        title=title,
        component_x=component_x,
        component_y=component_y,
        stride=stride,
        scale=scale,
        heatmap_alpha=heatmap_alpha,
        x_label=x_label,
        y_label=y_label,
        fixed_window=fixed_window,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    if not DVF_PATH.exists():
        raise FileNotFoundError(f"DVF file not found: {DVF_PATH}")
    if not FIXED_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Fixed image not found: {FIXED_IMAGE_PATH}")

    dvf_img = nib.load(str(DVF_PATH))
    shape_xyz = squeeze_vector_shape(tuple(int(v) for v in dvf_img.shape))
    dataobj = dvf_img.dataobj

    fixed_img = nib.load(str(FIXED_IMAGE_PATH))
    fixed_data = np.asarray(fixed_img.dataobj, dtype=np.float32)
    fixed_sample = fixed_data[::4, ::4, ::4]
    fixed_window = (
        float(np.nanpercentile(fixed_sample, 1.0)),
        float(np.nanpercentile(fixed_sample, 99.0)),
    )

    center = tuple(v // 2 for v in shape_xyz)
    axial = get_vector_slice(dataobj, axis=2, index=center[2])
    coronal = get_vector_slice(dataobj, axis=1, index=center[1])
    sagittal = get_vector_slice(dataobj, axis=0, index=center[0])

    axial_vox = ras_mm_to_voxel_vectors(lps_to_ras(axial), dvf_img.affine)
    coronal_vox = ras_mm_to_voxel_vectors(lps_to_ras(coronal), dvf_img.affine)
    sagittal_vox = ras_mm_to_voxel_vectors(lps_to_ras(sagittal), dvf_img.affine)

    axial_fixed = sample_scalar_plane(fixed_data, fixed_img.affine, dvf_img.affine, shape_xyz, axis=2, index=center[2])
    coronal_fixed = sample_scalar_plane(fixed_data, fixed_img.affine, dvf_img.affine, shape_xyz, axis=1, index=center[1])
    sagittal_fixed = sample_scalar_plane(fixed_data, fixed_img.affine, dvf_img.affine, shape_xyz, axis=0, index=center[0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_base = OUTPUT_DIR / f"{OUTPUT_STEM}_overlay"
    dvf_base = OUTPUT_DIR / f"{OUTPUT_STEM}_dvf_only"

    axial_overlay_output = overlay_base.with_name(f"{overlay_base.name}_axial.png")
    coronal_overlay_output = overlay_base.with_name(f"{overlay_base.name}_coronal.png")
    sagittal_overlay_output = overlay_base.with_name(f"{overlay_base.name}_sagittal.png")

    axial_dvf_output = dvf_base.with_name(f"{dvf_base.name}_axial.png")
    coronal_dvf_output = dvf_base.with_name(f"{dvf_base.name}_coronal.png")
    sagittal_dvf_output = dvf_base.with_name(f"{dvf_base.name}_sagittal.png")

    save_single_plane(
        output=axial_overlay_output,
        fixed_slice=axial_fixed,
        voxel_vectors=axial_vox,
        title=f"Axial z={center[2]} | overlay | fixed={FIXED_IMAGE_PATH.stem}",
        component_x=0,
        component_y=1,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=HEATMAP_ALPHA,
        x_label="voxel x",
        y_label="voxel y",
        fixed_window=fixed_window,
    )
    save_single_plane(
        output=coronal_overlay_output,
        fixed_slice=coronal_fixed,
        voxel_vectors=coronal_vox,
        title=f"Coronal y={center[1]} | overlay | fixed={FIXED_IMAGE_PATH.stem}",
        component_x=0,
        component_y=2,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=HEATMAP_ALPHA,
        x_label="voxel x",
        y_label="voxel z",
        fixed_window=fixed_window,
    )
    save_single_plane(
        output=sagittal_overlay_output,
        fixed_slice=sagittal_fixed,
        voxel_vectors=sagittal_vox,
        title=f"Sagittal x={center[0]} | overlay | fixed={FIXED_IMAGE_PATH.stem}",
        component_x=1,
        component_y=2,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=HEATMAP_ALPHA,
        x_label="voxel y",
        y_label="voxel z",
        fixed_window=fixed_window,
    )

    save_single_plane(
        output=axial_dvf_output,
        fixed_slice=None,
        voxel_vectors=axial_vox,
        title=f"Axial z={center[2]} | DVF only",
        component_x=0,
        component_y=1,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=1.0,
        x_label="voxel x",
        y_label="voxel y",
        fixed_window=fixed_window,
    )
    save_single_plane(
        output=coronal_dvf_output,
        fixed_slice=None,
        voxel_vectors=coronal_vox,
        title=f"Coronal y={center[1]} | DVF only",
        component_x=0,
        component_y=2,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=1.0,
        x_label="voxel x",
        y_label="voxel z",
        fixed_window=fixed_window,
    )
    save_single_plane(
        output=sagittal_dvf_output,
        fixed_slice=None,
        voxel_vectors=sagittal_vox,
        title=f"Sagittal x={center[0]} | DVF only",
        component_x=1,
        component_y=2,
        stride=QUIVER_STRIDE,
        scale=QUIVER_SCALE,
        heatmap_alpha=1.0,
        x_label="voxel y",
        y_label="voxel z",
        fixed_window=fixed_window,
    )

    print(f"Saved axial overlay visualization to: {axial_overlay_output}")
    print(f"Saved coronal overlay visualization to: {coronal_overlay_output}")
    print(f"Saved sagittal overlay visualization to: {sagittal_overlay_output}")
    print(f"Saved axial DVF-only visualization to: {axial_dvf_output}")
    print(f"Saved coronal DVF-only visualization to: {coronal_dvf_output}")
    print(f"Saved sagittal DVF-only visualization to: {sagittal_dvf_output}")
    print(f"Detected DVF grid shape: {shape_xyz}")
    print(f"Using fixed image: {FIXED_IMAGE_PATH}")
    print(f"Affine diagonal spacing/orientation: {np.diag(dvf_img.affine[:3, :3])}")


if __name__ == "__main__":
    main()
