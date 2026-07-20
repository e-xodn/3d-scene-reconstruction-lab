"""Tests for reconstruction.geometry.epipolar."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.cameras.pinhole import create_intrinsic_matrix, project_points
from reconstruction.geometry.epipolar import (
    algebraic_epipolar_residuals,
    canonicalize_fundamental_matrix,
    enforce_rank2,
    epipolar_lines_in_image1,
    epipolar_lines_in_image2,
    essential_from_pose,
    estimate_fundamental_matrix,
    from_homogeneous,
    fundamental_from_pose,
    normalize_points_2d,
    point_to_epipolar_line_distances,
    sampson_distances,
    skew_symmetric,
    to_homogeneous,
)
from reconstruction.geometry.transforms import make_transform, transform_points


def _synthetic_two_view_correspondences(
    num_points: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a well-conditioned synthetic two-view correspondence set for tests.

    Returns:
        pixels1, pixels2, intrinsic1, intrinsic2, rotation_camera2_camera1,
        translation_camera2_camera1.
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

    points_camera1 = np.stack(
        [
            rng.uniform(-2.0, 2.0, size=num_points),
            rng.uniform(-1.5, 1.5, size=num_points),
            rng.uniform(5.0, 12.0, size=num_points),
        ],
        axis=1,
    )
    transform_21 = make_transform(rotation_21, translation_21)
    points_camera2 = transform_points(points_camera1, transform_21)

    pixels1, _, valid1 = project_points(points_camera1, intrinsic1, image_size=(640, 480))
    pixels2, _, valid2 = project_points(points_camera2, intrinsic2, image_size=(640, 480))
    valid = valid1 & valid2
    assert np.sum(valid) >= 8, "synthetic fixture must produce at least 8 correspondences"

    return pixels1[valid], pixels2[valid], intrinsic1, intrinsic2, rotation_21, translation_21


# ---------------------------------------------------------------------------
# Homogeneous coordinate tests
# ---------------------------------------------------------------------------


def test_to_homogeneous_shape_and_values() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = to_homogeneous(points)
    expected = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]])
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
    assert result.dtype == np.float64


def test_to_homogeneous_empty_input() -> None:
    points = np.empty((0, 2))
    result = to_homogeneous(points)
    assert result.shape == (0, 3)


def test_from_homogeneous_round_trip() -> None:
    points = np.array([[1.5, -2.5], [100.0, 200.0]])
    points_h = to_homogeneous(points)
    restored = from_homogeneous(points_h)
    np.testing.assert_allclose(restored, points, atol=1e-12, rtol=1e-12)


def test_from_homogeneous_zero_scale_raises() -> None:
    points_h = np.array([[1.0, 2.0, 0.0]])
    with pytest.raises(ValueError):
        from_homogeneous(points_h)


def test_to_homogeneous_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        to_homogeneous(np.zeros((5, 3)))


def test_from_homogeneous_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        from_homogeneous(np.zeros((5, 2)))


def test_to_homogeneous_does_not_mutate_input() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    points_copy = points.copy()
    to_homogeneous(points)
    np.testing.assert_array_equal(points, points_copy)


def test_from_homogeneous_does_not_mutate_input() -> None:
    points_h = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 2.0]])
    points_h_copy = points_h.copy()
    from_homogeneous(points_h)
    np.testing.assert_array_equal(points_h, points_h_copy)


# ---------------------------------------------------------------------------
# Skew-symmetric tests
# ---------------------------------------------------------------------------


def test_skew_symmetric_is_skew_symmetric() -> None:
    vector = np.array([1.0, -2.0, 3.0])
    matrix = skew_symmetric(vector)
    np.testing.assert_allclose(matrix.T, -matrix, atol=1e-12, rtol=1e-12)


def test_skew_symmetric_matches_cross_product() -> None:
    rng = np.random.default_rng(0)
    v = rng.uniform(-5, 5, size=3)
    w = rng.uniform(-5, 5, size=3)
    matrix = skew_symmetric(v)
    np.testing.assert_allclose(matrix @ w, np.cross(v, w), atol=1e-12, rtol=1e-12)


def test_skew_symmetric_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        skew_symmetric(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Hartley normalization tests
# ---------------------------------------------------------------------------


def test_normalize_points_2d_centroid_near_zero() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-50, 50, size=(20, 2)) + np.array([300.0, 200.0])
    normalized, _ = normalize_points_2d(points)
    np.testing.assert_allclose(
        normalized.mean(axis=0), np.zeros(2), atol=1e-10, rtol=1e-10
    )


def test_normalize_points_2d_mean_radial_distance_is_sqrt2() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform(-50, 50, size=(20, 2)) + np.array([300.0, 200.0])
    normalized, _ = normalize_points_2d(points)
    mean_distance = np.mean(np.linalg.norm(normalized, axis=1))
    assert np.isclose(mean_distance, np.sqrt(2.0), atol=1e-10, rtol=1e-10)


def test_normalize_points_2d_transform_shape() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    _, transform = normalize_points_2d(points)
    assert transform.shape == (3, 3)
    np.testing.assert_allclose(
        transform[2, :], np.array([0.0, 0.0, 1.0]), atol=1e-12, rtol=1e-12
    )


def test_normalize_points_2d_identical_points_raises() -> None:
    points = np.tile(np.array([[5.0, 5.0]]), (5, 1))
    with pytest.raises(ValueError):
        normalize_points_2d(points)


def test_normalize_points_2d_allows_collinear_points() -> None:
    t = np.linspace(0, 1, 10)
    points = np.stack([10.0 + 100.0 * t, 20.0 + 50.0 * t], axis=1)
    normalized, transform = normalize_points_2d(points)
    assert normalized.shape == points.shape
    assert transform.shape == (3, 3)


def test_estimate_fundamental_matrix_rejects_collinear_even_though_normalize_allows() -> None:
    t = np.linspace(0, 1, 10)
    points1 = np.stack([10.0 + 100.0 * t, 20.0 + 50.0 * t], axis=1)
    points2 = np.random.default_rng(3).uniform(0, 640, size=(10, 2))

    # normalize_points_2d does not check collinearity, so this must succeed.
    normalize_points_2d(points1)

    # estimate_fundamental_matrix must reject the same points as degenerate.
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


# ---------------------------------------------------------------------------
# Rank-2 tests
# ---------------------------------------------------------------------------


def test_enforce_rank2_smallest_singular_value_near_zero() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.uniform(-1, 1, size=(3, 3))
    result = enforce_rank2(matrix)
    singular_values = np.linalg.svd(result, compute_uv=False)
    assert singular_values[-1] < 1e-10


def test_enforce_rank2_matrix_rank_is_2() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.uniform(-1, 1, size=(3, 3))
    result = enforce_rank2(matrix)
    assert np.linalg.matrix_rank(result, tol=1e-8) == 2


def test_enforce_rank2_does_not_mutate_input() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.uniform(-1, 1, size=(3, 3))
    matrix_copy = matrix.copy()
    enforce_rank2(matrix)
    np.testing.assert_array_equal(matrix, matrix_copy)


# ---------------------------------------------------------------------------
# Ground-truth geometry tests
# ---------------------------------------------------------------------------


def test_essential_from_pose_matches_definition() -> None:
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])
    essential = essential_from_pose(rotation, translation)
    expected = skew_symmetric(translation) @ rotation
    np.testing.assert_allclose(essential, expected, atol=1e-12, rtol=1e-12)
    assert np.linalg.matrix_rank(essential, tol=1e-10) == 2


def test_essential_from_pose_zero_translation_raises() -> None:
    with pytest.raises(ValueError):
        essential_from_pose(np.eye(3), np.zeros(3))


def test_fundamental_from_pose_matches_known_matrix_up_to_scale() -> None:
    intrinsic = create_intrinsic_matrix(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])

    fundamental = fundamental_from_pose(intrinsic, intrinsic, rotation, translation)
    essential = essential_from_pose(rotation, translation)
    expected = canonicalize_fundamental_matrix(essential)  # K = I, so F == E up to scale.

    np.testing.assert_allclose(fundamental, expected, atol=1e-10, rtol=1e-10)
    assert np.isclose(np.linalg.norm(fundamental), 1.0, atol=1e-10)
    assert np.linalg.matrix_rank(fundamental, tol=1e-8) == 2


def test_fundamental_from_pose_singular_intrinsic_raises() -> None:
    singular_intrinsic = np.array(
        [
            [500.0, 500.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 0.0],
        ]
    )
    valid_intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        fundamental_from_pose(singular_intrinsic, valid_intrinsic, rotation, translation)


# ---------------------------------------------------------------------------
# Eight-point algorithm tests
# ---------------------------------------------------------------------------


def test_estimate_fundamental_matrix_design_matrix_row_convention() -> None:
    """The design-matrix row order must satisfy row @ vec(F) == x2^T F x1
    when F is flattened in the same row-major order used to reshape the
    SVD solution back into a 3x3 matrix."""
    fundamental = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    u1, v1 = 0.3, -0.7
    u2, v2 = 1.1, 0.2

    row = np.array([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0])
    f_vec = fundamental.reshape(-1)

    x1 = np.array([u1, v1, 1.0])
    x2 = np.array([u2, v2, 1.0])
    expected = x2 @ fundamental @ x1

    np.testing.assert_allclose(row @ f_vec, expected, atol=1e-10, rtol=1e-10)


def test_estimate_fundamental_matrix_recovers_noiseless_geometry() -> None:
    pixels1, pixels2, intrinsic1, intrinsic2, rotation_21, translation_21 = (
        _synthetic_two_view_correspondences(num_points=60, seed=1)
    )
    fundamental_gt = fundamental_from_pose(intrinsic1, intrinsic2, rotation_21, translation_21)
    fundamental_est = estimate_fundamental_matrix(pixels1, pixels2)

    assert np.linalg.matrix_rank(fundamental_est, tol=1e-8) == 2

    residuals = algebraic_epipolar_residuals(pixels1, pixels2, fundamental_est)
    assert np.mean(np.abs(residuals)) < 1e-8

    error = min(
        np.linalg.norm(fundamental_est - fundamental_gt),
        np.linalg.norm(fundamental_est + fundamental_gt),
    )
    assert error < 1e-6


# ---------------------------------------------------------------------------
# Epipolar line tests
# ---------------------------------------------------------------------------


def test_epipolar_lines_in_image2_matches_definition() -> None:
    fundamental = np.array(
        [
            [0.0, -0.001, 0.1],
            [0.001, 0.0, -0.2],
            [-0.1, 0.2, 1.0],
        ]
    )
    points1 = np.array([[100.0, 50.0], [200.0, 300.0]])

    lines = epipolar_lines_in_image2(points1, fundamental)

    points1_h = np.concatenate([points1, np.ones((2, 1))], axis=1)
    expected = points1_h @ fundamental.T
    expected_normalized = expected / np.linalg.norm(expected[:, :2], axis=1, keepdims=True)

    np.testing.assert_allclose(lines, expected_normalized, atol=1e-10, rtol=1e-10)


def test_epipolar_lines_in_image1_matches_definition() -> None:
    fundamental = np.array(
        [
            [0.0, -0.001, 0.1],
            [0.001, 0.0, -0.2],
            [-0.1, 0.2, 1.0],
        ]
    )
    points2 = np.array([[400.0, 120.0], [50.0, 60.0]])

    lines = epipolar_lines_in_image1(points2, fundamental)

    points2_h = np.concatenate([points2, np.ones((2, 1))], axis=1)
    expected = points2_h @ fundamental
    expected_normalized = expected / np.linalg.norm(expected[:, :2], axis=1, keepdims=True)

    np.testing.assert_allclose(lines, expected_normalized, atol=1e-10, rtol=1e-10)


def test_epipolar_lines_normalized_to_unit_norm() -> None:
    fundamental = np.eye(3) * 2.0
    points1 = np.array([[10.0, 20.0], [30.0, -5.0]])
    lines = epipolar_lines_in_image2(points1, fundamental)
    norms = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-10, rtol=1e-10)


def test_point_to_epipolar_line_distances_matches_known_geometry() -> None:
    lines = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -2.0]])
    points = np.array([[8.0, 3.0], [10.0, 10.0]])
    distances = point_to_epipolar_line_distances(points, lines)
    np.testing.assert_allclose(distances, np.array([3.0, 8.0]), atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Sampson distance tests
# ---------------------------------------------------------------------------


def test_sampson_distance_near_zero_for_perfect_correspondence() -> None:
    pixels1, pixels2, intrinsic1, intrinsic2, rotation_21, translation_21 = (
        _synthetic_two_view_correspondences(num_points=30, seed=2)
    )
    fundamental_gt = fundamental_from_pose(intrinsic1, intrinsic2, rotation_21, translation_21)
    distances = sampson_distances(pixels1, pixels2, fundamental_gt)
    assert np.all(distances >= 0.0)
    assert np.max(distances) < 1e-6


def test_sampson_distance_increases_after_pixel_perturbation() -> None:
    pixels1, pixels2, intrinsic1, intrinsic2, rotation_21, translation_21 = (
        _synthetic_two_view_correspondences(num_points=30, seed=3)
    )
    fundamental_gt = fundamental_from_pose(intrinsic1, intrinsic2, rotation_21, translation_21)
    baseline = sampson_distances(pixels1, pixels2, fundamental_gt)

    rng = np.random.default_rng(4)
    perturbed_pixels2 = pixels2 + rng.normal(0.0, 3.0, size=pixels2.shape)
    perturbed = sampson_distances(pixels1, perturbed_pixels2, fundamental_gt)

    assert np.all(perturbed > baseline)


def test_sampson_distance_non_negative() -> None:
    pixels1, pixels2, intrinsic1, intrinsic2, rotation_21, translation_21 = (
        _synthetic_two_view_correspondences(num_points=25, seed=5)
    )
    fundamental_gt = fundamental_from_pose(intrinsic1, intrinsic2, rotation_21, translation_21)
    rng = np.random.default_rng(6)
    noisy_pixels2 = pixels2 + rng.normal(0.0, 2.0, size=pixels2.shape)
    distances = sampson_distances(pixels1, noisy_pixels2, fundamental_gt)
    assert np.all(distances >= 0.0)


def test_sampson_distance_degenerate_denominator_raises() -> None:
    fundamental = np.zeros((3, 3))
    points1 = np.array([[10.0, 20.0]])
    points2 = np.array([[15.0, 25.0]])
    with pytest.raises(ValueError):
        sampson_distances(points1, points2, fundamental)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_estimate_fundamental_matrix_too_few_points_raises() -> None:
    points1 = np.random.default_rng(0).uniform(0, 100, size=(7, 2))
    points2 = np.random.default_rng(1).uniform(0, 100, size=(7, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_length_mismatch_raises() -> None:
    points1 = np.random.default_rng(0).uniform(0, 100, size=(10, 2))
    points2 = np.random.default_rng(1).uniform(0, 100, size=(9, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_invalid_shape_raises() -> None:
    points1 = np.random.default_rng(0).uniform(0, 100, size=(10, 3))
    points2 = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_nan_raises() -> None:
    points1 = np.random.default_rng(0).uniform(0, 100, size=(10, 2))
    points2 = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
    points2[0, 0] = np.nan
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_identical_points_raises() -> None:
    points1 = np.tile(np.array([[50.0, 60.0]]), (10, 1))
    points2 = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_collinear_points_raises() -> None:
    t = np.linspace(0, 1, 10)
    points1 = np.stack([10.0 + 100.0 * t, 20.0 + 50.0 * t], axis=1)
    points2 = np.random.default_rng(1).uniform(0, 100, size=(10, 2))
    with pytest.raises(ValueError):
        estimate_fundamental_matrix(points1, points2)


def test_estimate_fundamental_matrix_does_not_mutate_inputs() -> None:
    pixels1, pixels2, _, _, _, _ = _synthetic_two_view_correspondences(num_points=20, seed=7)
    pixels1_copy = pixels1.copy()
    pixels2_copy = pixels2.copy()

    estimate_fundamental_matrix(pixels1, pixels2)

    np.testing.assert_array_equal(pixels1, pixels1_copy)
    np.testing.assert_array_equal(pixels2, pixels2_copy)
