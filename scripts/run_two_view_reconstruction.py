#!/usr/bin/env python3
"""Two-view sparse reconstruction from a real image pair.

Pipeline: feature detection (SIFT/ORB) -> descriptor matching -> Lowe ratio
test -> optional mutual consistency check -> custom RANSAC fundamental
matrix estimation (normalized eight-point + Sampson distance) -> essential
matrix -> essential matrix decomposition into four pose candidates ->
cheirality test -> DLT triangulation -> reprojection-error filtering ->
sparse 3D point cloud (with color) and diagnostic visualizations.

OpenCV is used only for image I/O and SIFT/ORB feature
detection/matching (see reconstruction.features.matching); RANSAC,
essential-matrix decomposition, cheirality, and triangulation are custom
NumPy implementations (reconstruction.geometry.robust/pose/triangulation).

Scale ambiguity: the recovered translation direction and the triangulated
point cloud have no absolute metric scale (see
reconstruction.geometry.pose module docstring). This is standard for
two-view reconstruction from a single image pair.

Usage:
    python scripts/run_two_view_reconstruction.py \\
        --config configs/two_view/real_pair.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reconstruction.cameras.pinhole import create_intrinsic_matrix  # noqa: E402
from reconstruction.evaluation.reprojection import (  # noqa: E402
    two_view_reprojection_errors,
)
from reconstruction.features.matching import (  # noqa: E402
    detect_and_describe,
    load_grayscale_image,
    matched_keypoint_coordinates,
    mutual_ratio_matches,
    ratio_test_matches,
)
from reconstruction.geometry.pose import (  # noqa: E402
    decompose_essential_matrix,
    essential_from_fundamental,
    select_pose_by_cheirality,
)
from reconstruction.geometry.robust import estimate_fundamental_matrix_ransac  # noqa: E402
from reconstruction.geometry.triangulation import (  # noqa: E402
    camera_depths,
    projection_matrix,
    triangulate_points_dlt,
)
from reconstruction.visualization.two_view import (  # noqa: E402
    save_epipolar_visualization,
    save_match_visualization,
    save_sparse_reconstruction_visualization,
    write_colored_ply,
)


class ConfigError(ValueError):
    """Raised when the YAML configuration is missing or has invalid values."""


class PipelineFailure(RuntimeError):
    """Raised when the reconstruction pipeline cannot proceed past a stage."""


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
    """Load and validate the two-view real-pair YAML configuration.

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

    images = _require(config, "images", str(config_path))
    _require(images, "image1_path", "images")
    _require(images, "image2_path", "images")

    camera1 = _require(config, "camera1", str(config_path))
    camera2 = _require(config, "camera2", str(config_path))
    _validate_camera_config(camera1, "camera1")
    _validate_camera_config(camera2, "camera2")

    features = _require(config, "features", str(config_path))
    for key in ("method", "max_features", "ratio_threshold", "mutual_check"):
        _require(features, key, "features")
    if features["method"].lower() not in ("sift", "orb"):
        raise ConfigError(f"features.method must be 'sift' or 'orb', got {features['method']}.")
    if features["max_features"] <= 0:
        raise ConfigError("features.max_features must be > 0.")
    if not (0.0 < features["ratio_threshold"] < 1.0):
        raise ConfigError("features.ratio_threshold must be in (0, 1).")

    ransac = _require(config, "ransac", str(config_path))
    for key in ("threshold_px", "max_iterations", "confidence", "seed", "min_inliers"):
        _require(ransac, key, "ransac")
    if ransac["min_inliers"] < 8:
        raise ConfigError("ransac.min_inliers must be >= 8.")

    reconstruction = _require(config, "reconstruction", str(config_path))
    for key in (
        "min_positive_depth",
        "min_positive_depth_ratio",
        "max_reprojection_error_px",
        "max_points_for_pose_selection",
    ):
        _require(reconstruction, key, "reconstruction")
    if not (0.0 < reconstruction["min_positive_depth_ratio"] <= 1.0):
        raise ConfigError("reconstruction.min_positive_depth_ratio must be in (0, 1].")
    if reconstruction["max_reprojection_error_px"] <= 0:
        raise ConfigError("reconstruction.max_reprojection_error_px must be > 0.")

    visualization = _require(config, "visualization", str(config_path))
    for key in ("max_raw_matches", "max_inlier_matches", "num_epipolar_lines", "max_3d_points"):
        _require(visualization, key, "visualization")

    output = _require(config, "output", str(config_path))
    for key in (
        "raw_matches_path",
        "inlier_matches_path",
        "epipolar_path",
        "reconstruction_path",
        "pointcloud_path",
        "metrics_path",
    ):
        _require(output, key, "output")

    return config


def _load_color_image(path: Path) -> np.ndarray:
    """Load a BGR color image with OpenCV, raising clear errors on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image (unsupported format or corrupt file): {path}")
    return image


def _check_image_size(image: np.ndarray, expected_width: int, expected_height: int, name: str) -> None:
    """Raise ValueError if ``image``'s (H, W) does not match the configured size."""
    height, width = image.shape[:2]
    if width != expected_width or height != expected_height:
        raise ValueError(
            f"{name}: actual image size {width}x{height} does not match config "
            f"size {expected_width}x{expected_height}. Update the config's "
            "image_width/image_height (and rescale intrinsics accordingly if "
            "you resize the image) rather than letting the pipeline proceed "
            "silently with mismatched intrinsics."
        )


def _sample_colors(image_bgr: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sample RGB colors from a BGR image at pixel coordinates.

    Args:
        image_bgr: BGR image, shape (H, W, 3).
        pixels: Pixel coordinates (u, v), shape (N, 2).

    Returns:
        RGB colors in [0, 255], shape (N, 3), dtype float64.
    """
    height, width = image_bgr.shape[:2]
    u = np.clip(np.round(pixels[:, 0]).astype(np.int64), 0, width - 1)
    v = np.clip(np.round(pixels[:, 1]).astype(np.int64), 0, height - 1)
    bgr = image_bgr[v, u]
    return bgr[:, ::-1].astype(np.float64)


def _json_safe(value: Any) -> Any:
    """Recursively convert a metrics value into a JSON-safe form (NaN/Inf -> null)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main() -> None:
    """Run the two-view sparse reconstruction pipeline end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the real-pair YAML configuration."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    images_cfg = config["images"]
    camera1_cfg = config["camera1"]
    camera2_cfg = config["camera2"]
    features_cfg = config["features"]
    ransac_cfg = config["ransac"]
    reconstruction_cfg = config["reconstruction"]
    visualization_cfg = config["visualization"]
    output_cfg = config["output"]

    image1_path = Path(images_cfg["image1_path"])
    image2_path = Path(images_cfg["image2_path"])

    print("=== Loading images ===")
    image1_color = _load_color_image(image1_path)
    image2_color = _load_color_image(image2_path)
    image1_gray = load_grayscale_image(image1_path)
    image2_gray = load_grayscale_image(image2_path)

    _check_image_size(image1_gray, camera1_cfg["image_width"], camera1_cfg["image_height"], "image1")
    _check_image_size(image2_gray, camera2_cfg["image_width"], camera2_cfg["image_height"], "image2")
    print(f"Image 1: {image1_path} ({image1_gray.shape[1]}x{image1_gray.shape[0]})")
    print(f"Image 2: {image2_path} ({image2_gray.shape[1]}x{image2_gray.shape[0]})")

    intrinsic1 = create_intrinsic_matrix(**{k: float(v) for k, v in camera1_cfg["intrinsics"].items()})
    intrinsic2 = create_intrinsic_matrix(**{k: float(v) for k, v in camera2_cfg["intrinsics"].items()})
    intrinsics_source1 = camera1_cfg.get("intrinsics_source", "unspecified")
    intrinsics_source2 = camera2_cfg.get("intrinsics_source", "unspecified")

    print("\n=== Feature detection ===")
    method = features_cfg["method"].lower()
    max_features = int(features_cfg["max_features"])
    ratio_threshold = float(features_cfg["ratio_threshold"])
    mutual_check = bool(features_cfg["mutual_check"])

    keypoints1, descriptors1 = detect_and_describe(image1_gray, method=method, max_features=max_features)
    keypoints2, descriptors2 = detect_and_describe(image2_gray, method=method, max_features=max_features)
    print(f"Image 1 keypoints: {len(keypoints1)}")
    print(f"Image 2 keypoints: {len(keypoints2)}")

    forward_matches = ratio_test_matches(descriptors1, descriptors2, method, ratio_threshold)
    mutual_matches = mutual_ratio_matches(descriptors1, descriptors2, method, ratio_threshold)
    forward_ratio_match_count = len(forward_matches)
    mutual_match_count = len(mutual_matches)
    matches = mutual_matches if mutual_check else forward_matches
    print(f"Ratio-test matches: {forward_ratio_match_count}")
    print(f"Mutual matches: {mutual_match_count}")
    print(f"Matches used downstream ({'mutual' if mutual_check else 'forward'}): {len(matches)}")

    if len(matches) < 8:
        raise PipelineFailure(
            f"Only {len(matches)} ratio-test matches were found (< 8 required "
            "for eight-point estimation). Capture an image pair with more "
            "overlap, less motion blur, and a scene with more texture."
        )

    points1_all, points2_all = matched_keypoint_coordinates(keypoints1, keypoints2, matches)
    match_distances = np.array([m.distance for m in matches], dtype=np.float64)

    raw_matches_path = Path(output_cfg["raw_matches_path"])
    save_match_visualization(
        image1_gray,
        image2_gray,
        points1_all,
        points2_all,
        match_distances,
        raw_matches_path,
        max_matches=int(visualization_cfg["max_raw_matches"]),
        title=f"Raw matches ({method.upper()}, ratio<{ratio_threshold})",
    )

    print("\n=== Robust fundamental matrix (RANSAC) ===")
    try:
        fundamental, inlier_mask, ransac_stats = estimate_fundamental_matrix_ransac(
            points1_all,
            points2_all,
            threshold_px=float(ransac_cfg["threshold_px"]),
            max_iterations=int(ransac_cfg["max_iterations"]),
            confidence=float(ransac_cfg["confidence"]),
            seed=int(ransac_cfg["seed"]),
            min_inliers=int(ransac_cfg["min_inliers"]),
        )
    except RuntimeError as exc:
        raise PipelineFailure(str(exc)) from exc

    points1_inliers = points1_all[inlier_mask]
    points2_inliers = points2_all[inlier_mask]
    inlier_count = ransac_stats["final_inlier_count"]
    print(f"RANSAC inliers: {inlier_count} / {len(matches)}")
    print(f"RANSAC inlier ratio: {ransac_stats['inlier_ratio']:.3f}")
    print(f"Median inlier Sampson distance: {ransac_stats['median_inlier_sampson_distance']:.3f} px^2")

    inlier_matches_path = Path(output_cfg["inlier_matches_path"])
    save_match_visualization(
        image1_gray,
        image2_gray,
        points1_inliers,
        points2_inliers,
        match_distances[inlier_mask],
        inlier_matches_path,
        max_matches=int(visualization_cfg["max_inlier_matches"]),
        title=f"RANSAC inliers ({inlier_count}/{len(matches)})",
    )

    num_epipolar_lines = min(int(visualization_cfg["num_epipolar_lines"]), inlier_count)
    epipolar_rng = np.random.default_rng(int(ransac_cfg["seed"]))
    epipolar_indices = epipolar_rng.choice(inlier_count, size=num_epipolar_lines, replace=False)
    epipolar_path = Path(output_cfg["epipolar_path"])
    save_epipolar_visualization(
        image2_gray,
        points1_inliers[epipolar_indices],
        points2_inliers[epipolar_indices],
        fundamental,
        epipolar_path,
        title=f"Epipolar lines l2=Fx1 ({num_epipolar_lines} of {inlier_count} inliers)",
    )

    print("\n=== Essential matrix and pose recovery ===")
    essential = essential_from_fundamental(fundamental, intrinsic1, intrinsic2)
    pose_candidates = decompose_essential_matrix(essential)

    max_pose_points = int(reconstruction_cfg["max_points_for_pose_selection"])
    if inlier_count > max_pose_points:
        pose_rng = np.random.default_rng(int(ransac_cfg["seed"]))
        pose_subset = pose_rng.choice(inlier_count, size=max_pose_points, replace=False)
    else:
        pose_subset = np.arange(inlier_count)

    try:
        rotation, translation, _, _, positive_depth_counts = select_pose_by_cheirality(
            pose_candidates,
            points1_inliers[pose_subset],
            points2_inliers[pose_subset],
            intrinsic1,
            intrinsic2,
            min_depth=float(reconstruction_cfg["min_positive_depth"]),
            min_positive_ratio=float(reconstruction_cfg["min_positive_depth_ratio"]),
        )
    except RuntimeError as exc:
        raise PipelineFailure(str(exc)) from exc

    selected_pose_index = next(
        i for i, (r, t) in enumerate(pose_candidates) if r is rotation and t is translation
    )
    print(f"Selected pose candidate: {selected_pose_index}")
    print(f"Positive-depth counts per candidate: {positive_depth_counts}")

    print("\n=== Triangulation (all RANSAC inliers) ===")
    projection1 = projection_matrix(intrinsic1, np.eye(3), np.zeros(3))
    projection2 = projection_matrix(intrinsic2, rotation, translation)
    points_3d_all = triangulate_points_dlt(points1_inliers, points2_inliers, projection1, projection2)
    depth1_all, depth2_all = camera_depths(points_3d_all, rotation, translation)

    finite_mask = np.all(np.isfinite(points_3d_all), axis=1)
    positive_depth_mask = (
        finite_mask & (depth1_all > reconstruction_cfg["min_positive_depth"])
        & (depth2_all > reconstruction_cfg["min_positive_depth"])
    )
    positive_depth_count = int(np.sum(positive_depth_mask))
    positive_depth_ratio = positive_depth_count / inlier_count if inlier_count else 0.0
    print(f"Positive-depth points: {positive_depth_count} / {inlier_count}")
    print(f"Positive-depth ratio: {positive_depth_ratio:.3f}")

    error1, error2, combined_error = two_view_reprojection_errors(
        points_3d_all, points1_inliers, points2_inliers, projection1, projection2
    )
    triangulated_point_count = points_3d_all.shape[0]

    filter_mask = (
        finite_mask
        & (depth1_all > reconstruction_cfg["min_positive_depth"])
        & (depth2_all > reconstruction_cfg["min_positive_depth"])
        & (combined_error <= reconstruction_cfg["max_reprojection_error_px"])
    )
    filtered_points_3d = points_3d_all[filter_mask]
    filtered_point_count = int(np.sum(filter_mask))
    print(f"\nTriangulated points: {triangulated_point_count}")
    print(f"Filtered points: {filtered_point_count}")

    if filtered_point_count < 8:
        raise PipelineFailure(
            f"Only {filtered_point_count} points passed reprojection-error "
            "filtering (< 8). Check camera intrinsics accuracy, RANSAC "
            "threshold, and image overlap/baseline; a very small filtered "
            "count usually indicates inaccurate intrinsics or a degenerate "
            "pose."
        )

    finite_error1 = error1[np.isfinite(error1)]
    finite_error2 = error2[np.isfinite(error2)]
    finite_combined = combined_error[np.isfinite(combined_error)]

    print(f"\nMean reprojection error image 1: {np.mean(finite_error1):.3f} px")
    print(f"Median reprojection error image 1: {np.median(finite_error1):.3f} px")
    print(f"Mean reprojection error image 2: {np.mean(finite_error2):.3f} px")
    print(f"Median reprojection error image 2: {np.median(finite_error2):.3f} px")
    print(f"Median combined reprojection error: {np.median(finite_combined):.3f} px")

    camera2_center = -rotation.T @ translation

    print("\nRelative rotation R_21:")
    print(rotation)
    print("\nRelative translation direction t_21 (unit norm):")
    print(translation)
    print("\nCamera 2 center in camera 1 frame:")
    print(camera2_center)
    print("\nWarning: translation and 3D reconstruction have arbitrary scale.")

    reconstruction_path = Path(output_cfg["reconstruction_path"])
    save_sparse_reconstruction_visualization(
        filtered_points_3d,
        rotation,
        translation,
        filtered_point_count,
        float(np.median(combined_error[filter_mask])),
        positive_depth_ratio,
        reconstruction_path,
        max_points=int(visualization_cfg["max_3d_points"]),
        seed=int(ransac_cfg["seed"]),
    )

    filtered_pixels1 = points1_inliers[filter_mask]
    colors = _sample_colors(image1_color, filtered_pixels1)
    pointcloud_path = Path(output_cfg["pointcloud_path"])
    write_colored_ply(filtered_points_3d, colors, pointcloud_path)

    metrics: dict[str, Any] = {
        "image1_keypoints": len(keypoints1),
        "image2_keypoints": len(keypoints2),
        "forward_ratio_match_count": forward_ratio_match_count,
        "mutual_match_count": mutual_match_count,
        "ransac_inlier_count": inlier_count,
        "ransac_inlier_ratio": ransac_stats["inlier_ratio"],
        "ransac_attempted_iterations": ransac_stats["attempted_iterations"],
        "ransac_valid_models": ransac_stats["valid_models"],
        "ransac_skipped_degenerate_samples": ransac_stats["skipped_degenerate_samples"],
        "median_inlier_sampson_distance": ransac_stats["median_inlier_sampson_distance"],
        "mean_inlier_sampson_distance": ransac_stats["mean_inlier_sampson_distance"],
        "pose_candidate_positive_depth_counts": positive_depth_counts,
        "selected_pose_index": selected_pose_index,
        "positive_depth_count": positive_depth_count,
        "positive_depth_ratio": positive_depth_ratio,
        "triangulated_point_count": triangulated_point_count,
        "filtered_point_count": filtered_point_count,
        "mean_reprojection_error_image1_px": float(np.mean(finite_error1)),
        "median_reprojection_error_image1_px": float(np.median(finite_error1)),
        "max_reprojection_error_image1_px": float(np.max(finite_error1)),
        "mean_reprojection_error_image2_px": float(np.mean(finite_error2)),
        "median_reprojection_error_image2_px": float(np.median(finite_error2)),
        "max_reprojection_error_image2_px": float(np.max(finite_error2)),
        "mean_combined_reprojection_error_px": float(np.mean(finite_combined)),
        "median_combined_reprojection_error_px": float(np.median(finite_combined)),
        "max_combined_reprojection_error_px": float(np.max(finite_combined)),
        "rotation_camera2_camera1": rotation.tolist(),
        "translation_direction_camera2_camera1": translation.tolist(),
        "camera2_center_in_camera1_frame": camera2_center.tolist(),
        "intrinsics_source_camera1": intrinsics_source1,
        "intrinsics_source_camera2": intrinsics_source2,
    }

    metrics_path = Path(output_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    print("\nSaved:")
    print(f"  {raw_matches_path}")
    print(f"  {inlier_matches_path}")
    print(f"  {epipolar_path}")
    print(f"  {reconstruction_path}")
    print(f"  {pointcloud_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, PipelineFailure, FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
