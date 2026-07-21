#!/usr/bin/env python3
"""Demonstrate pinhole camera projection and unprojection.

Loads a YAML scene/camera configuration, generates synthetic 3D points
(some inside the image, some outside the image boundary, some behind the
camera), projects them onto the image plane, unprojects the valid pixels
back into 3D, measures the round-trip reconstruction error, and saves a
visualization of the projected points.

Usage:
    python scripts/demo_camera_geometry.py \\
        --config configs/camera_geometry/example.yaml
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
    unproject_pixels,
)


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or has invalid values."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Fetch ``key`` from ``mapping`` or raise a clear ConfigError."""
    if key not in mapping:
        raise ConfigError(f"Missing required config key '{key}' in '{context}'.")
    return mapping[key]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the camera-geometry demo YAML configuration.

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

    camera = _require(config, "camera", str(config_path))
    intrinsics = _require(camera, "intrinsics", "camera")
    for key in ("image_width", "image_height"):
        _require(camera, key, "camera")
    for key in ("fx", "fy", "cx", "cy"):
        _require(intrinsics, key, "camera.intrinsics")
    if camera["image_width"] <= 0 or camera["image_height"] <= 0:
        raise ConfigError("camera.image_width and camera.image_height must be > 0.")

    scene = _require(config, "scene", str(config_path))
    for key in (
        "seed",
        "num_valid_points",
        "num_outside_points",
        "num_behind_points",
        "min_depth",
        "max_depth",
    ):
        _require(scene, key, "scene")
    if scene["min_depth"] <= 0 or scene["max_depth"] <= scene["min_depth"]:
        raise ConfigError(
            "scene.min_depth must be > 0 and scene.max_depth must be > min_depth."
        )

    output = _require(config, "output", str(config_path))
    _require(output, "figure_path", "output")

    return config


def generate_synthetic_points(
    rng: np.random.Generator,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    num_valid_points: int,
    num_outside_points: int,
    num_behind_points: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Generate synthetic camera-frame 3D points for the demo.

    Produces three groups of points, concatenated into a single array:
      - ``num_valid_points`` points that project strictly inside the image
        with positive depth.
      - ``num_outside_points`` points with positive depth that project
        outside the image boundary.
      - ``num_behind_points`` points with negative depth (behind the
        camera).

    Args:
        rng: Random generator used for all sampling.
        intrinsic: Camera intrinsic matrix, shape (3, 3).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        num_valid_points: Number of in-bounds, positive-depth points.
        num_outside_points: Number of out-of-bounds, positive-depth points.
        num_behind_points: Number of behind-camera points.
        min_depth: Minimum depth magnitude used for sampling.
        max_depth: Maximum depth magnitude used for sampling.

    Returns:
        Camera-frame 3D points, shape
        (num_valid_points + num_outside_points + num_behind_points, 3).
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    margin = 20.0
    u_valid = rng.uniform(margin, image_width - margin, size=num_valid_points)
    v_valid = rng.uniform(margin, image_height - margin, size=num_valid_points)
    z_valid = rng.uniform(min_depth, max_depth, size=num_valid_points)
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    points_valid = np.stack([x_valid, y_valid, z_valid], axis=1)

    outside_offset = 100.0
    side = rng.integers(0, 2, size=num_outside_points)
    u_outside = np.where(
        side == 0,
        rng.uniform(-image_width, -outside_offset, size=num_outside_points),
        rng.uniform(image_width + outside_offset, 2 * image_width, size=num_outside_points),
    )
    v_outside = rng.uniform(0.0, image_height, size=num_outside_points)
    z_outside = rng.uniform(min_depth, max_depth, size=num_outside_points)
    x_outside = (u_outside - cx) * z_outside / fx
    y_outside = (v_outside - cy) * z_outside / fy
    points_outside = np.stack([x_outside, y_outside, z_outside], axis=1)

    x_behind = rng.uniform(-2.0, 2.0, size=num_behind_points)
    y_behind = rng.uniform(-2.0, 2.0, size=num_behind_points)
    z_behind = -rng.uniform(min_depth, max_depth, size=num_behind_points)
    points_behind = np.stack([x_behind, y_behind, z_behind], axis=1)

    return np.concatenate([points_valid, points_outside, points_behind], axis=0)


def plot_projection(
    pixels: np.ndarray,
    valid: np.ndarray,
    image_width: int,
    image_height: int,
    principal_point: tuple[float, float],
    mean_error: float,
    output_path: Path,
) -> None:
    """Save a visualization of the projected points to ``output_path``.

    Args:
        pixels: Projected pixel coordinates, shape (N, 2), NaN for invalid.
        valid: Boolean validity mask, shape (N,).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        principal_point: (cx, cy) principal point coordinates.
        mean_error: Mean round-trip reconstruction error, for the title.
        output_path: Destination path for the saved figure (PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_pixels = pixels[valid]
    ax.scatter(
        valid_pixels[:, 0],
        valid_pixels[:, 1],
        s=15,
        c="tab:blue",
        label="valid projected points",
    )
    ax.scatter(
        [principal_point[0]],
        [principal_point[1]],
        marker="x",
        s=100,
        c="tab:red",
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
        f"Valid points: {int(valid.sum())} | "
        f"Mean round-trip error: {mean_error:.3e}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the camera geometry demonstration end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the camera-geometry YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    camera_cfg = config["camera"]
    intrinsics_cfg = camera_cfg["intrinsics"]
    scene_cfg = config["scene"]
    output_cfg = config["output"]

    image_width = int(camera_cfg["image_width"])
    image_height = int(camera_cfg["image_height"])

    intrinsic = create_intrinsic_matrix(
        fx=float(intrinsics_cfg["fx"]),
        fy=float(intrinsics_cfg["fy"]),
        cx=float(intrinsics_cfg["cx"]),
        cy=float(intrinsics_cfg["cy"]),
    )

    rng = np.random.default_rng(int(scene_cfg["seed"]))

    num_valid_points = int(scene_cfg["num_valid_points"])
    num_outside_points = int(scene_cfg["num_outside_points"])
    num_behind_points = int(scene_cfg["num_behind_points"])

    points_camera = generate_synthetic_points(
        rng=rng,
        intrinsic=intrinsic,
        image_width=image_width,
        image_height=image_height,
        num_valid_points=num_valid_points,
        num_outside_points=num_outside_points,
        num_behind_points=num_behind_points,
        min_depth=float(scene_cfg["min_depth"]),
        max_depth=float(scene_cfg["max_depth"]),
    )
    total_points = points_camera.shape[0]

    pixels, depth, valid = project_points(
        points_camera, intrinsic, image_size=(image_width, image_height)
    )
    positive_depth_count = int(np.sum(depth > 0))
    inside_image_count = int(np.sum(valid))
    invalid_count = total_points - inside_image_count

    valid_pixels = pixels[valid]
    valid_depth = depth[valid]
    restored_points = unproject_pixels(valid_pixels, valid_depth, intrinsic)

    original_valid_points = points_camera[valid]
    errors = np.linalg.norm(restored_points - original_valid_points, axis=1)
    mean_error = float(np.mean(errors)) if errors.size else float("nan")
    max_error = float(np.max(errors)) if errors.size else float("nan")

    figure_path = Path(output_cfg["figure_path"])
    plot_projection(
        pixels=pixels,
        valid=valid,
        image_width=image_width,
        image_height=image_height,
        principal_point=(float(intrinsics_cfg["cx"]), float(intrinsics_cfg["cy"])),
        mean_error=mean_error,
        output_path=figure_path,
    )

    print("Camera intrinsic matrix:")
    print(intrinsic)
    print()
    print(f"Total points: {total_points}")
    print(f"Positive-depth points: {positive_depth_count}")
    print(f"Points inside image: {inside_image_count}")
    print(f"Invalid points: {invalid_count}")
    print(f"Mean round-trip error: {mean_error:.1e}")
    print(f"Max round-trip error: {max_error:.1e}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)
