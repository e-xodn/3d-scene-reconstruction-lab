#!/usr/bin/env python3
"""Demonstrate synthetic two-view epipolar geometry.

Generates a synthetic calibrated two-camera scene, produces clean and
noisy pixel correspondences, computes the ground-truth essential and
fundamental matrices from the known relative pose, estimates the
fundamental matrix from correspondences with the normalized eight-point
algorithm, and reports epipolar-geometry error metrics. Saves a
visualization of epipolar lines in image 2.

Scene convention: camera 1's frame is treated as the world frame
(``T_camera1_world = I``); camera 2's pose relative to camera 1 is
``T_camera2_camera1 = [R_21 | t_21]``, so
``X_camera2 = R_21 @ X_camera1 + t_21``.

Usage:
    python scripts/demo_epipolar_geometry.py \\
        --config configs/two_view/synthetic_epipolar.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconstruction.cameras.pinhole import (  # noqa: E402
    create_intrinsic_matrix,
    project_points,
)
from reconstruction.geometry.epipolar import (  # noqa: E402
    algebraic_epipolar_residuals,
    canonicalize_fundamental_matrix,
    epipolar_lines_in_image2,
    essential_from_pose,
    estimate_fundamental_matrix,
    fundamental_from_pose,
    point_to_epipolar_line_distances,
    sampson_distances,
)
from reconstruction.geometry.transforms import make_transform, transform_points  # noqa: E402


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or has invalid values."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Fetch ``key`` from ``mapping`` or raise a clear ConfigError."""
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"Missing required config key '{key}' in '{context}'.")
    return mapping[key]


def _validate_camera_config(camera: dict[str, Any], name: str) -> None:
    """Validate an image_width/image_height/intrinsics camera config block."""
    width = _require(camera, "image_width", name)
    height = _require(camera, "image_height", name)
    if not isinstance(width, int) or width <= 0:
        raise ConfigError(f"{name}.image_width must be a positive integer, got {width}.")
    if not isinstance(height, int) or height <= 0:
        raise ConfigError(f"{name}.image_height must be a positive integer, got {height}.")

    intrinsics = _require(camera, "intrinsics", name)
    for key in ("fx", "fy", "cx", "cy"):
        _require(intrinsics, key, f"{name}.intrinsics")
    if intrinsics["fx"] <= 0:
        raise ConfigError(f"{name}.intrinsics.fx must be > 0, got {intrinsics['fx']}.")
    if intrinsics["fy"] <= 0:
        raise ConfigError(f"{name}.intrinsics.fy must be > 0, got {intrinsics['fy']}.")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the synthetic epipolar-geometry YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        ConfigError: If the file is missing required keys or values are of
            the wrong type / out of range.
    """
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError(f"Config file '{config_path}' must contain a YAML mapping.")

    camera1 = _require(config, "camera1", str(config_path))
    camera2 = _require(config, "camera2", str(config_path))
    _validate_camera_config(camera1, "camera1")
    _validate_camera_config(camera2, "camera2")

    relative_pose = _require(config, "relative_pose", str(config_path))
    rotation_degrees = _require(relative_pose, "rotation_degrees", "relative_pose")
    translation = _require(relative_pose, "translation", "relative_pose")
    for key in ("x", "y", "z"):
        _require(rotation_degrees, key, "relative_pose.rotation_degrees")
        _require(translation, key, "relative_pose.translation")
    translation_norm = float(
        np.linalg.norm([translation["x"], translation["y"], translation["z"]])
    )
    if translation_norm <= 0.0:
        raise ConfigError(
            "relative_pose.translation must have nonzero norm (zero baseline "
            "is undefined for epipolar geometry)."
        )

    scene = _require(config, "scene", str(config_path))
    for key in (
        "seed",
        "num_points",
        "x_range",
        "y_range",
        "z_range",
        "noise_std_px",
        "max_sampling_iterations",
    ):
        _require(scene, key, "scene")
    if scene["num_points"] < 8:
        raise ConfigError(f"scene.num_points must be >= 8, got {scene['num_points']}.")
    if scene["z_range"][0] >= scene["z_range"][1]:
        raise ConfigError("scene.z_range must satisfy lower < upper.")
    if scene["noise_std_px"] < 0:
        raise ConfigError(
            f"scene.noise_std_px must be >= 0, got {scene['noise_std_px']}."
        )
    if scene["max_sampling_iterations"] <= 0:
        raise ConfigError("scene.max_sampling_iterations must be > 0.")

    visualization = _require(config, "visualization", str(config_path))
    num_correspondences = _require(visualization, "num_correspondences", "visualization")
    _require(visualization, "figure_path", "visualization")
    if not (1 <= num_correspondences <= scene["num_points"]):
        raise ConfigError(
            "visualization.num_correspondences must be between 1 and "
            f"scene.num_points ({scene['num_points']}), got "
            f"{num_correspondences}."
        )

    return config


def rotation_matrix_from_euler_degrees(angles_degrees: dict[str, float]) -> np.ndarray:
    """Build a rotation matrix from Euler angles given in degrees.

    Rotation order convention: ``R = Rz @ Ry @ Rx`` (intrinsic rotations
    applied X first, then Y, then Z, composed as shown).

    Args:
        angles_degrees: Mapping with keys "x", "y", "z" giving rotation
            angles in degrees about each axis.

    Returns:
        Rotation matrix, shape (3, 3).
    """
    angle_x = np.radians(angles_degrees["x"])
    angle_y = np.radians(angles_degrees["y"])
    angle_z = np.radians(angles_degrees["z"])

    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

    return rotation_z @ rotation_y @ rotation_x


def generate_synthetic_correspondences(
    rng: np.random.Generator,
    intrinsic1: np.ndarray,
    intrinsic2: np.ndarray,
    image_size1: tuple[int, int],
    image_size2: tuple[int, int],
    rotation_camera2_camera1: np.ndarray,
    translation_camera2_camera1: np.ndarray,
    num_points: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    max_sampling_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rejection-sample 3D points visible (positive depth, in-bounds) in both cameras.

    Points are drawn uniformly at random from the given camera-1-frame box
    and kept only if their projections are valid (positive depth and inside
    the image) in both camera 1 and camera 2.

    Args:
        rng: Random generator used for all sampling.
        intrinsic1: Camera 1 intrinsic matrix, shape (3, 3).
        intrinsic2: Camera 2 intrinsic matrix, shape (3, 3).
        image_size1: Camera 1 (width, height) in pixels.
        image_size2: Camera 2 (width, height) in pixels.
        rotation_camera2_camera1: Rotation R_21, shape (3, 3).
        translation_camera2_camera1: Translation t_21, shape (3,).
        num_points: Number of valid correspondences to collect.
        x_range: (min, max) for the point x-coordinate in camera 1 frame.
        y_range: (min, max) for the point y-coordinate in camera 1 frame.
        z_range: (min, max) for the point z-coordinate (depth) in camera 1
            frame.
        max_sampling_iterations: Maximum total number of candidate points to
            draw before giving up.

    Returns:
        A tuple ``(points_camera1, pixels1, pixels2)``:
            points_camera1: shape (num_points, 3), 3D points in camera 1 /
                world frame.
            pixels1: shape (num_points, 2), clean projections in camera 1.
            pixels2: shape (num_points, 2), clean projections in camera 2.

    Raises:
        RuntimeError: If fewer than ``num_points`` valid correspondences are
            found within ``max_sampling_iterations`` candidate draws.
    """
    transform_camera2_camera1 = make_transform(
        rotation_camera2_camera1, translation_camera2_camera1
    )

    collected_points: list[np.ndarray] = []
    collected_pixels1: list[np.ndarray] = []
    collected_pixels2: list[np.ndarray] = []
    total_collected = 0
    total_sampled = 0
    batch_size = max(num_points * 4, 64)

    while total_collected < num_points and total_sampled < max_sampling_iterations:
        current_batch = min(batch_size, max_sampling_iterations - total_sampled)
        candidates = np.stack(
            [
                rng.uniform(x_range[0], x_range[1], size=current_batch),
                rng.uniform(y_range[0], y_range[1], size=current_batch),
                rng.uniform(z_range[0], z_range[1], size=current_batch),
            ],
            axis=1,
        )
        total_sampled += current_batch

        pixels1, _, valid1 = project_points(candidates, intrinsic1, image_size=image_size1)
        points_camera2 = transform_points(candidates, transform_camera2_camera1)
        pixels2, _, valid2 = project_points(points_camera2, intrinsic2, image_size=image_size2)

        valid = valid1 & valid2
        num_valid = int(np.sum(valid))
        if num_valid > 0:
            collected_points.append(candidates[valid])
            collected_pixels1.append(pixels1[valid])
            collected_pixels2.append(pixels2[valid])
            total_collected += num_valid

    if total_collected < num_points:
        raise RuntimeError(
            f"Could not sample {num_points} correspondences visible in both "
            f"cameras within {max_sampling_iterations} sampling iterations "
            f"(found {total_collected})."
        )

    points_camera1 = np.concatenate(collected_points, axis=0)[:num_points]
    pixels1 = np.concatenate(collected_pixels1, axis=0)[:num_points]
    pixels2 = np.concatenate(collected_pixels2, axis=0)[:num_points]
    return points_camera1, pixels1, pixels2


def scale_invariant_fundamental_error(
    fundamental_estimated: np.ndarray,
    fundamental_ground_truth: np.ndarray,
) -> float:
    """Compare two fundamental matrices up to their inherent scale/sign ambiguity.

    Args:
        fundamental_estimated: Estimated fundamental matrix, shape (3, 3).
        fundamental_ground_truth: Ground-truth fundamental matrix, shape
            (3, 3).

    Returns:
        ``min(||Fc_est - Fc_gt||_F, ||Fc_est + Fc_gt||_F)`` where ``Fc_*``
        are the canonicalized (unit Frobenius norm, fixed sign) matrices.
    """
    canonical_estimated = canonicalize_fundamental_matrix(fundamental_estimated)
    canonical_ground_truth = canonicalize_fundamental_matrix(fundamental_ground_truth)
    return float(
        min(
            np.linalg.norm(canonical_estimated - canonical_ground_truth),
            np.linalg.norm(canonical_estimated + canonical_ground_truth),
        )
    )


def _clip_line_to_image(
    line: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray | None:
    """Clip a line a*u + b*v + c = 0 to the image rectangle [0, width] x [0, height].

    Args:
        line: Line coefficients [a, b, c].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Array of shape (2, 2) with two distinct boundary points forming a
        segment inside the image, or None if the line does not cross the
        image rectangle in two distinct points.
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


def plot_epipolar_geometry(
    points2_noisy: np.ndarray,
    selected_indices: np.ndarray,
    lines2_selected: np.ndarray,
    image_width: int,
    image_height: int,
    principal_point2: tuple[float, float],
    num_correspondences: int,
    median_sampson_distance: float,
    scale_invariant_f_error: float,
    output_path: Path,
) -> None:
    """Save a visualization of epipolar lines and noisy correspondences in image 2.

    Args:
        points2_noisy: All noisy points in image 2, shape (N, 2).
        selected_indices: Indices into ``points2_noisy`` chosen to display
            epipolar lines for.
        lines2_selected: Epipolar lines l2 = F x1 for the selected
            correspondences, shape (len(selected_indices), 3).
        image_width: Image 2 width in pixels.
        image_height: Image 2 height in pixels.
        principal_point2: (cx, cy) of camera 2.
        num_correspondences: Total number of correspondences (for the title).
        median_sampson_distance: Median Sampson distance for noisy
            correspondences (for the title).
        scale_invariant_f_error: Scale-invariant fundamental matrix error
            for the noisy estimate (for the title).
        output_path: Destination path for the saved figure (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        points2_noisy[:, 0],
        points2_noisy[:, 1],
        s=10,
        alpha=0.5,
        label="noisy points (image 2)",
    )
    ax.scatter(
        points2_noisy[selected_indices, 0],
        points2_noisy[selected_indices, 1],
        s=45,
        edgecolors="black",
        label="selected correspondences",
    )

    line_label_used = False
    for line in lines2_selected:
        segment = _clip_line_to_image(line, image_width, image_height)
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
        [principal_point2[0]],
        [principal_point2[1]],
        marker="x",
        s=100,
        label="principal point",
    )

    boundary = plt.Rectangle(
        (0, 0),
        image_width,
        image_height,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        label="image boundary",
    )
    ax.add_patch(boundary)

    ax.set_xlim(-0.05 * image_width, 1.05 * image_width)
    ax.set_ylim(1.05 * image_height, -0.05 * image_height)  # v increases downward
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Correspondences: {num_correspondences} | "
        f"Median Sampson dist (noisy): {median_sampson_distance:.3e} | "
        f"Scale-invariant F error (noisy): {scale_invariant_f_error:.3e}",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the synthetic epipolar geometry demonstration end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the synthetic epipolar-geometry YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    camera1_cfg = config["camera1"]
    camera2_cfg = config["camera2"]
    intrinsics1_cfg = camera1_cfg["intrinsics"]
    intrinsics2_cfg = camera2_cfg["intrinsics"]
    relative_pose_cfg = config["relative_pose"]
    scene_cfg = config["scene"]
    visualization_cfg = config["visualization"]

    image_size1 = (int(camera1_cfg["image_width"]), int(camera1_cfg["image_height"]))
    image_size2 = (int(camera2_cfg["image_width"]), int(camera2_cfg["image_height"]))

    intrinsic1 = create_intrinsic_matrix(
        fx=float(intrinsics1_cfg["fx"]),
        fy=float(intrinsics1_cfg["fy"]),
        cx=float(intrinsics1_cfg["cx"]),
        cy=float(intrinsics1_cfg["cy"]),
    )
    intrinsic2 = create_intrinsic_matrix(
        fx=float(intrinsics2_cfg["fx"]),
        fy=float(intrinsics2_cfg["fy"]),
        cx=float(intrinsics2_cfg["cx"]),
        cy=float(intrinsics2_cfg["cy"]),
    )

    rotation_camera2_camera1 = rotation_matrix_from_euler_degrees(
        relative_pose_cfg["rotation_degrees"]
    )
    translation_camera2_camera1 = np.array(
        [
            relative_pose_cfg["translation"]["x"],
            relative_pose_cfg["translation"]["y"],
            relative_pose_cfg["translation"]["z"],
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(int(scene_cfg["seed"]))

    _, pixels1_clean, pixels2_clean = generate_synthetic_correspondences(
        rng=rng,
        intrinsic1=intrinsic1,
        intrinsic2=intrinsic2,
        image_size1=image_size1,
        image_size2=image_size2,
        rotation_camera2_camera1=rotation_camera2_camera1,
        translation_camera2_camera1=translation_camera2_camera1,
        num_points=int(scene_cfg["num_points"]),
        x_range=tuple(scene_cfg["x_range"]),
        y_range=tuple(scene_cfg["y_range"]),
        z_range=tuple(scene_cfg["z_range"]),
        max_sampling_iterations=int(scene_cfg["max_sampling_iterations"]),
    )
    num_points = pixels1_clean.shape[0]

    noise_std_px = float(scene_cfg["noise_std_px"])
    if noise_std_px > 0:
        pixels1_noisy = pixels1_clean + rng.normal(0.0, noise_std_px, size=pixels1_clean.shape)
        pixels2_noisy = pixels2_clean + rng.normal(0.0, noise_std_px, size=pixels2_clean.shape)
    else:
        pixels1_noisy = pixels1_clean.copy()
        pixels2_noisy = pixels2_clean.copy()

    essential_gt = essential_from_pose(rotation_camera2_camera1, translation_camera2_camera1)
    fundamental_gt = fundamental_from_pose(
        intrinsic1, intrinsic2, rotation_camera2_camera1, translation_camera2_camera1
    )

    fundamental_clean = estimate_fundamental_matrix(pixels1_clean, pixels2_clean)
    fundamental_noisy = estimate_fundamental_matrix(pixels1_noisy, pixels2_noisy)

    clean_residuals = algebraic_epipolar_residuals(pixels1_clean, pixels2_clean, fundamental_clean)
    clean_sampson = sampson_distances(pixels1_clean, pixels2_clean, fundamental_clean)
    clean_f_error = scale_invariant_fundamental_error(fundamental_clean, fundamental_gt)

    noisy_residuals = algebraic_epipolar_residuals(pixels1_noisy, pixels2_noisy, fundamental_noisy)
    noisy_sampson = sampson_distances(pixels1_noisy, pixels2_noisy, fundamental_noisy)
    noisy_lines2 = epipolar_lines_in_image2(pixels1_noisy, fundamental_noisy)
    noisy_point_line_distances = point_to_epipolar_line_distances(pixels2_noisy, noisy_lines2)
    noisy_f_error = scale_invariant_fundamental_error(fundamental_noisy, fundamental_gt)

    num_correspondences = int(visualization_cfg["num_correspondences"])
    selected_indices = np.sort(
        rng.choice(num_points, size=num_correspondences, replace=False)
    )
    lines2_selected = noisy_lines2[selected_indices]

    figure_path = Path(visualization_cfg["figure_path"])
    plot_epipolar_geometry(
        points2_noisy=pixels2_noisy,
        selected_indices=selected_indices,
        lines2_selected=lines2_selected,
        image_width=image_size2[0],
        image_height=image_size2[1],
        principal_point2=(float(intrinsics2_cfg["cx"]), float(intrinsics2_cfg["cy"])),
        num_correspondences=num_points,
        median_sampson_distance=float(np.median(noisy_sampson)),
        scale_invariant_f_error=noisy_f_error,
        output_path=figure_path,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("Camera 1 intrinsic matrix:")
    print(intrinsic1)
    print()
    print("Camera 2 intrinsic matrix:")
    print(intrinsic2)
    print()
    print("Relative rotation R_21:")
    print(rotation_camera2_camera1)
    print()
    print("Relative translation t_21:")
    print(translation_camera2_camera1)
    print()
    print("Ground-truth essential matrix:")
    print(essential_gt)
    print()
    print("Ground-truth fundamental matrix:")
    print(fundamental_gt)
    print()
    print("Estimated fundamental matrix from clean correspondences:")
    print(fundamental_clean)
    print()
    print("Estimated fundamental matrix from noisy correspondences:")
    print(fundamental_noisy)
    print()
    print(f"Number of correspondences: {num_points}")
    print()
    print("Clean correspondences:")
    print(f"Mean absolute algebraic residual: {np.mean(np.abs(clean_residuals)):.3e}")
    print(f"Median Sampson distance: {np.median(clean_sampson):.3e}")
    print(f"Maximum Sampson distance: {np.max(clean_sampson):.3e}")
    print(f"Scale-invariant F error: {clean_f_error:.3e}")
    print()
    print("Noisy correspondences:")
    print(f"Mean absolute algebraic residual: {np.mean(np.abs(noisy_residuals)):.3e}")
    print(f"Median Sampson distance: {np.median(noisy_sampson):.3e}")
    print(f"Maximum Sampson distance: {np.max(noisy_sampson):.3e}")
    print(f"Median point-to-line distance in image 2: {np.median(noisy_point_line_distances):.3e}")
    print(f"Scale-invariant F error: {noisy_f_error:.3e}")
    print()
    print("Saved figure:")
    print(figure_path)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
