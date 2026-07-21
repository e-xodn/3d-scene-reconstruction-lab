"""Tests for reconstruction.geometry.robust (fundamental matrix RANSAC).

Uses a deterministic synthetic two-view scene (known K1, K2, R_21, t_21)
with Gaussian pixel noise on true correspondences plus injected random
outlier correspondences, built as a minimal local test helper (following
the M2A synthetic-correspondence approach in tests/test_epipolar.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.cameras.pinhole import create_intrinsic_matrix, project_points
from reconstruction.geometry.epipolar import (
    canonicalize_fundamental_matrix,
    fundamental_from_pose,
)
from reconstruction.geometry.robust import estimate_fundamental_matrix_ransac
from reconstruction.geometry.transforms import make_transform, transform_points

IMAGE_SIZE = (640, 480)


def _make_synthetic_scene(
    seed: int = 0,
    num_inliers: int = 150,
    num_outliers: int = 50,
    noise_std_px: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic correspondence set with known-inlier noise + outliers.

    Returns:
        points1, points2: shape (num_inliers + num_outliers, 2), shuffled.
        true_inlier_mask: shape (N,), boolean (True for the noisy true
            correspondences, False for injected outliers).
        fundamental_gt: ground-truth fundamental matrix, shape (3, 3).
        intrinsic1, intrinsic2, rotation_21: for callers that need them.
    """
    rng = np.random.default_rng(seed)
    intrinsic1 = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    intrinsic2 = create_intrinsic_matrix(fx=510.0, fy=505.0, cx=315.0, cy=245.0)

    angle = np.radians(8.0)
    rotation_21 = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    translation_21 = np.array([-0.4, 0.05, 0.1])
    transform_21 = make_transform(rotation_21, translation_21)

    oversample = num_inliers * 3
    points_camera1 = np.stack(
        [
            rng.uniform(-2.0, 2.0, size=oversample),
            rng.uniform(-1.5, 1.5, size=oversample),
            rng.uniform(5.0, 12.0, size=oversample),
        ],
        axis=1,
    )
    points_camera2 = transform_points(points_camera1, transform_21)
    pixels1, _, valid1 = project_points(points_camera1, intrinsic1, image_size=IMAGE_SIZE)
    pixels2, _, valid2 = project_points(points_camera2, intrinsic2, image_size=IMAGE_SIZE)
    valid = valid1 & valid2
    assert np.sum(valid) >= num_inliers, "synthetic fixture undersampled valid points"

    pixels1 = pixels1[valid][:num_inliers]
    pixels2 = pixels2[valid][:num_inliers]
    pixels1_noisy = pixels1 + rng.normal(0.0, noise_std_px, size=pixels1.shape)
    pixels2_noisy = pixels2 + rng.normal(0.0, noise_std_px, size=pixels2.shape)

    outliers1 = rng.uniform([0.0, 0.0], [IMAGE_SIZE[0], IMAGE_SIZE[1]], size=(num_outliers, 2))
    outliers2 = rng.uniform([0.0, 0.0], [IMAGE_SIZE[0], IMAGE_SIZE[1]], size=(num_outliers, 2))

    points1 = np.concatenate([pixels1_noisy, outliers1], axis=0)
    points2 = np.concatenate([pixels2_noisy, outliers2], axis=0)
    true_inlier_mask = np.concatenate(
        [np.ones(num_inliers, dtype=bool), np.zeros(num_outliers, dtype=bool)]
    )

    order = rng.permutation(points1.shape[0])
    points1 = points1[order]
    points2 = points2[order]
    true_inlier_mask = true_inlier_mask[order]

    fundamental_gt = fundamental_from_pose(intrinsic1, intrinsic2, rotation_21, translation_21)

    return points1, points2, true_inlier_mask, fundamental_gt, intrinsic1, intrinsic2, rotation_21


def test_ransac_reproducible_with_fixed_seed() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=1)

    fundamental_a, mask_a, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)
    fundamental_b, mask_b, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)

    np.testing.assert_array_equal(fundamental_a, fundamental_b)
    np.testing.assert_array_equal(mask_a, mask_b)


def test_ransac_estimated_fundamental_rank_is_2() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=2)
    fundamental, _, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)
    assert np.linalg.matrix_rank(fundamental, tol=1e-8) == 2


def test_ransac_recovers_sufficient_true_inliers() -> None:
    points1, points2, true_inlier_mask, _, _, _, _ = _make_synthetic_scene(seed=3)
    _, inlier_mask, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)

    recall = np.sum(inlier_mask & true_inlier_mask) / np.sum(true_inlier_mask)
    assert recall >= 0.80


def test_ransac_inlier_precision_is_high() -> None:
    points1, points2, true_inlier_mask, _, _, _, _ = _make_synthetic_scene(seed=4)
    _, inlier_mask, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)

    precision = np.sum(inlier_mask & true_inlier_mask) / np.sum(inlier_mask)
    assert precision >= 0.90


def test_ransac_fundamental_matches_ground_truth_within_scale_ambiguity() -> None:
    points1, points2, _, fundamental_gt, _, _, _ = _make_synthetic_scene(seed=5)
    fundamental_est, _, _ = estimate_fundamental_matrix_ransac(points1, points2, seed=42)

    canonical_est = canonicalize_fundamental_matrix(fundamental_est)
    canonical_gt = canonicalize_fundamental_matrix(fundamental_gt)
    error = min(
        np.linalg.norm(canonical_est - canonical_gt),
        np.linalg.norm(canonical_est + canonical_gt),
    )
    assert error < 0.05


def test_ransac_statistics_contents() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=6)
    _, inlier_mask, statistics = estimate_fundamental_matrix_ransac(
        points1, points2, threshold_px=1.5, seed=42
    )

    expected_keys = {
        "attempted_iterations",
        "valid_models",
        "skipped_degenerate_samples",
        "initial_match_count",
        "final_inlier_count",
        "inlier_ratio",
        "median_inlier_sampson_distance",
        "mean_inlier_sampson_distance",
        "threshold_px",
    }
    assert expected_keys.issubset(statistics.keys())
    assert statistics["initial_match_count"] == points1.shape[0]
    assert statistics["final_inlier_count"] == int(np.sum(inlier_mask))
    assert statistics["threshold_px"] == 1.5


def test_ransac_all_outliers_raises_runtime_error() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(
        seed=7, num_inliers=0, num_outliers=60
    )
    with pytest.raises(RuntimeError):
        estimate_fundamental_matrix_ransac(points1, points2, seed=42, min_inliers=20)


def test_ransac_too_few_points_raises_value_error() -> None:
    points1 = np.random.default_rng(0).uniform(0, 640, size=(7, 2))
    points2 = np.random.default_rng(1).uniform(0, 480, size=(7, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix_ransac(points1, points2)


def test_ransac_invalid_threshold_raises_value_error() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=8, num_inliers=20, num_outliers=0)
    with pytest.raises(ValueError):
        estimate_fundamental_matrix_ransac(points1, points2, threshold_px=0.0)


def test_ransac_invalid_min_inliers_raises_value_error() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=9, num_inliers=20, num_outliers=0)
    with pytest.raises(ValueError):
        estimate_fundamental_matrix_ransac(points1, points2, min_inliers=5)


def test_ransac_does_not_mutate_inputs() -> None:
    points1, points2, _, _, _, _, _ = _make_synthetic_scene(seed=10)
    points1_copy = points1.copy()
    points2_copy = points2.copy()

    estimate_fundamental_matrix_ransac(points1, points2, seed=42)

    np.testing.assert_array_equal(points1, points1_copy)
    np.testing.assert_array_equal(points2, points2_copy)
