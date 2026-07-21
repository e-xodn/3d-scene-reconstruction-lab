"""Visualization and point-cloud export helpers for the two-view reconstruction pipeline.

Uses matplotlib only (headless "Agg" backend); OpenCV is intentionally not
used here — this repository restricts OpenCV usage to image I/O and
feature detection/matching (see :mod:`reconstruction.features.matching`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3D projection)

from reconstruction.geometry.epipolar import epipolar_lines_in_image2  # noqa: E402

__all__ = [
    "save_match_visualization",
    "save_epipolar_visualization",
    "save_sparse_reconstruction_visualization",
    "write_colored_ply",
]


def _clip_line_to_image(line: np.ndarray, width: float, height: float) -> np.ndarray | None:
    """Clip a line a*u + b*v + c = 0 to the image rectangle [0, width] x [0, height].

    Returns a (2, 2) array of two distinct boundary points forming a
    segment inside the image, or None if the line does not cross the
    rectangle in two distinct points.
    """
    a, b, c = line
    candidates: list[tuple[float, float]] = []

    if abs(b) > 1e-9:
        for u in (0.0, float(width)):
            v = -(a * u + c) / b
            if -1e-6 <= v <= height + 1e-6:
                candidates.append((u, float(np.clip(v, 0.0, height))))
    if abs(a) > 1e-9:
        for v in (0.0, float(height)):
            u = -(b * v + c) / a
            if -1e-6 <= u <= width + 1e-6:
                candidates.append((float(np.clip(u, 0.0, width)), v))

    if len(candidates) < 2:
        return None

    unique: list[tuple[float, float]] = []
    for point in candidates:
        if not any(np.allclose(point, existing, atol=1e-6) for existing in unique):
            unique.append(point)
    if len(unique) < 2:
        return None

    unique_arr = np.array(unique)
    best_pair = None
    best_distance = -1.0
    for i in range(len(unique_arr)):
        for j in range(i + 1, len(unique_arr)):
            distance = float(np.linalg.norm(unique_arr[i] - unique_arr[j]))
            if distance > best_distance:
                best_distance = distance
                best_pair = (unique_arr[i], unique_arr[j])

    if best_pair is None or best_distance < 1e-6:
        return None
    return np.array(best_pair)


def _plot_camera_orientation(ax, center: np.ndarray, forward: np.ndarray, length: float, color: str) -> None:
    """Draw a short line from a camera center along its forward (+Z) direction."""
    end = center + length * forward
    ax.plot(
        [center[0], end[0]], [center[1], end[1]], [center[2], end[2]], color=color, linewidth=2
    )


def save_match_visualization(
    image1_gray: np.ndarray,
    image2_gray: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    distances: np.ndarray,
    output_path: str | Path,
    max_matches: int = 100,
    title: str = "",
) -> None:
    """Save a side-by-side visualization of matched keypoints.

    Args:
        image1_gray: Grayscale image 1, shape (H1, W1).
        image2_gray: Grayscale image 2, shape (H2, W2).
        points1: Matched pixel coordinates in image 1, shape (N, 2).
        points2: Matched pixel coordinates in image 2, shape (N, 2).
        distances: Descriptor distance per match, shape (N,); used to pick
            the ``max_matches`` smallest-distance matches to draw.
        output_path: Destination PNG path (parent directory created if
            needed).
        max_matches: Maximum number of match lines to draw.
        title: Figure title; a default is used if empty.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if points1.shape != points2.shape or points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(
            f"points1 and points2 must both have shape (N, 2) and match, got "
            f"{points1.shape} and {points2.shape}."
        )
    if distances.shape[0] != points1.shape[0]:
        raise ValueError(
            f"distances must have length N={points1.shape[0]}, got {distances.shape[0]}."
        )

    height = max(image1_gray.shape[0], image2_gray.shape[0])
    width1 = image1_gray.shape[1]
    width2 = image2_gray.shape[1]
    canvas = np.zeros((height, width1 + width2), dtype=np.uint8)
    canvas[: image1_gray.shape[0], :width1] = image1_gray
    canvas[: image2_gray.shape[0], width1:] = image2_gray

    order = np.argsort(distances)[:max_matches]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    for idx in order:
        u1, v1 = points1[idx]
        u2, v2 = points2[idx]
        ax.plot([u1, u2 + width1], [v1, v2], linewidth=0.5, alpha=0.6)
        ax.scatter([u1, u2 + width1], [v1, v2], s=4, c="tab:red")

    ax.axis("off")
    ax.set_title(title or f"{len(order)} of {points1.shape[0]} matches shown")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_epipolar_visualization(
    image2_gray: np.ndarray,
    points1_subset: np.ndarray,
    points2_subset: np.ndarray,
    fundamental: np.ndarray,
    output_path: str | Path,
    title: str = "",
) -> None:
    """Save a visualization of epipolar lines l2 = F x1 drawn over image 2.

    Args:
        image2_gray: Grayscale image 2, shape (H, W); used as the plot
            background, not modified.
        points1_subset: Selected points in image 1, shape (M, 2), whose
            epipolar lines are computed and drawn
            (:func:`reconstruction.geometry.epipolar.epipolar_lines_in_image2`).
        points2_subset: The corresponding points in image 2, shape (M, 2),
            plotted as markers.
        fundamental: Fundamental matrix, shape (3, 3).
        output_path: Destination PNG path.
        title: Figure title.

    Raises:
        ValueError: If shapes are inconsistent (see
            :func:`reconstruction.geometry.epipolar.epipolar_lines_in_image2`
            for fundamental/point validation).
    """
    points1_subset = np.asarray(points1_subset, dtype=np.float64)
    points2_subset = np.asarray(points2_subset, dtype=np.float64)
    if points1_subset.shape != points2_subset.shape:
        raise ValueError(
            "points1_subset and points2_subset must have the same shape, got "
            f"{points1_subset.shape} and {points2_subset.shape}."
        )

    height, width = image2_gray.shape[:2]
    lines2 = epipolar_lines_in_image2(points1_subset, fundamental)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image2_gray, cmap="gray", vmin=0, vmax=255)

    line_label_used = False
    for line in lines2:
        segment = _clip_line_to_image(line, width, height)
        if segment is not None:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                linewidth=1.0,
                alpha=0.8,
                label=None if line_label_used else "epipolar lines l2 = F x1",
            )
            line_label_used = True

    ax.scatter(
        points2_subset[:, 0],
        points2_subset[:, 1],
        s=25,
        c="tab:red",
        edgecolors="black",
        label="inlier points (image 2)",
    )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image convention: v increases downward
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sparse_reconstruction_visualization(
    points_3d: np.ndarray,
    rotation_camera2_camera1: np.ndarray,
    translation_camera2_camera1: np.ndarray,
    filtered_point_count: int,
    median_reprojection_error_px: float,
    positive_depth_ratio: float,
    output_path: str | Path,
    max_points: int = 5000,
    seed: int = 0,
) -> None:
    """Save a 3D scatter visualization of the filtered sparse reconstruction.

    Shows the filtered 3D points, both camera centers, and a short line
    indicating each camera's forward (+Z) viewing direction. Camera 1 is
    at the origin (the reconstruction reference frame); camera 2's center
    in the camera 1 frame is ``C2 = -R_21.T @ t_21``. Axis limits use the
    2nd/98th percentile of the plotted data so a few outlier points do not
    dominate the scale.

    Args:
        points_3d: Filtered 3D points (camera 1 frame), shape (N, 3).
        rotation_camera2_camera1: Selected R_21, shape (3, 3).
        translation_camera2_camera1: Selected t_21 (unit-norm direction),
            shape (3,).
        filtered_point_count: Number of points after filtering (for the
            title).
        median_reprojection_error_px: Median combined reprojection error,
            in pixels (for the title).
        positive_depth_ratio: Positive-depth ratio of the selected pose
            (for the title).
        output_path: Destination PNG path.
        max_points: If ``points_3d`` has more rows than this, a random
            subset is plotted (for rendering performance).
        seed: Random seed for subsampling, for reproducibility.

    Raises:
        ValueError: If ``points_3d`` does not have shape (N, 3).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    rotation_camera2_camera1 = np.asarray(rotation_camera2_camera1, dtype=np.float64)
    translation_camera2_camera1 = np.asarray(translation_camera2_camera1, dtype=np.float64)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}.")

    if points_3d.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(points_3d.shape[0], size=max_points, replace=False)
        display_points = points_3d[subset_indices]
    else:
        display_points = points_3d

    camera1_center = np.zeros(3)
    camera2_center = -rotation_camera2_camera1.T @ translation_camera2_camera1
    baseline = float(np.linalg.norm(camera2_center - camera1_center))
    axis_length = max(baseline * 0.5, 1e-6)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if display_points.shape[0] > 0:
        ax.scatter(
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            s=2,
            alpha=0.6,
            label="sparse points",
        )
    ax.scatter(*camera1_center, s=90, marker="^", c="tab:red", label="camera 1 center")
    ax.scatter(*camera2_center, s=90, marker="^", c="tab:green", label="camera 2 center")

    _plot_camera_orientation(ax, camera1_center, np.array([0.0, 0.0, 1.0]), axis_length, "tab:red")
    forward2 = rotation_camera2_camera1.T @ np.array([0.0, 0.0, 1.0])
    _plot_camera_orientation(ax, camera2_center, forward2, axis_length, "tab:green")

    bounds_points = np.vstack(
        [display_points, camera1_center[None, :], camera2_center[None, :]]
        if display_points.shape[0] > 0
        else [camera1_center[None, :], camera2_center[None, :]]
    )
    lower = np.percentile(bounds_points, 2, axis=0)
    upper = np.percentile(bounds_points, 98, axis=0)
    padding = np.maximum((upper - lower) * 0.1, 1e-6)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_zlim(lower[2] - padding[2], upper[2] + padding[2])

    ax.set_xlabel("X (camera 1 frame)")
    ax.set_ylabel("Y (camera 1 frame)")
    ax.set_zlabel("Z (camera 1 frame)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Filtered points: {filtered_point_count} | "
        f"Median reprojection error: {median_reprojection_error_px:.3f} px | "
        f"Positive-depth ratio: {positive_depth_ratio:.3f}\n"
        "Note: reconstruction scale is arbitrary (no absolute metric scale).",
        fontsize=9,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_colored_ply(
    points_3d: np.ndarray,
    colors_rgb: np.ndarray,
    output_path: str | Path,
) -> None:
    """Write an ASCII PLY point cloud with per-point RGB color.

    Args:
        points_3d: Points, shape (N, 3).
        colors_rgb: RGB colors in [0, 255], shape (N, 3) (rounded and
            clipped to uint8 on write).
        output_path: Destination .ply path (parent directory created if
            needed).

    Raises:
        ValueError: If shapes are inconsistent.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    colors_rgb = np.asarray(colors_rgb, dtype=np.float64)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}.")
    if colors_rgb.shape != points_3d.shape:
        raise ValueError(
            f"colors_rgb must have shape {points_3d.shape}, got {colors_rgb.shape}."
        )

    colors_uint8 = np.clip(np.round(colors_rgb), 0, 255).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_points = points_3d.shape[0]
    with output_path.open("w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        for i in range(num_points):
            x, y, z = points_3d[i]
            r, g, b = colors_uint8[i]
            ply_file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
