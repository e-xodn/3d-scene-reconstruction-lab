"""Tests for reconstruction.geometry.pose (essential matrix, decomposition, cheirality)."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.cameras.pinhole import create_intrinsic_matrix, project_points
from reconstruction.geometry.epipolar import essential_from_pose, fundamental_from_pose
from reconstruction.geometry.pose import (
    decompose_essential_matrix,
    enforce_essential_constraints,
    essential_from_fundamental,
    select_pose_by_cheirality,
)
from reconstruction.geometry.transforms import make_transform, transform_points

IMAGE_SIZE = (640, 480)


def _synthetic_noiseless_scene(
    num_points: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic noiseless synthetic two-view scene for pose-recovery tests.

    Returns:
        pixels1, pixels2, intrinsic1, intrinsic2, rotation_21,
        translation_21_unit (unit-norm ground-truth translation direction).
    """
    rng = np.random.default_rng(seed)
    intrinsic1 = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    intrinsic2 = create_intrinsic_matrix(fx=510.0, fy=505.0, cx=315.0, cy=245.0)

    angle = np.radians(10.0)
    rotation_21 = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    translation_21 = np.array([-0.5, 0.05, 0.1])
    translation_21_unit = translation_21 / np.linalg.norm(translation_21)
    transform_21 = make_transform(rotation_21, translation_21)

    points_camera1 = np.stack(
        [
            rng.uniform(-2.0, 2.0, size=num_points),
            rng.uniform(-1.5, 1.5, size=num_points),
            rng.uniform(5.0, 12.0, size=num_points),
        ],
        axis=1,
    )
    points_camera2 = transform_points(points_camera1, transform_21)
    pixels1, _, valid1 = project_points(points_camera1, intrinsic1, image_size=IMAGE_SIZE)
    pixels2, _, valid2 = project_points(points_camera2, intrinsic2, image_size=IMAGE_SIZE)
    valid = valid1 & valid2
    assert np.sum(valid) >= 8

    return (
        pixels1[valid],
        pixels2[valid],
        intrinsic1,
        intrinsic2,
        rotation_21,
        translation_21_unit,
    )


def _rotation_angle_error_degrees(rotation_est: np.ndarray, rotation_gt: np.ndarray) -> float:
    rotation_error = rotation_est @ rotation_gt.T
    cos_angle = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _direction_angle_error_degrees(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    cos_angle = np.clip(
        np.dot(t_est, t_gt) / (np.linalg.norm(t_est) * np.linalg.norm(t_gt)), -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# essential_from_fundamental
# ---------------------------------------------------------------------------


def test_essential_from_fundamental_matches_ground_truth_up_to_sign() -> None:
    intrinsic1 = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    intrinsic2 = create_intrinsic_matrix(fx=510.0, fy=505.0, cx=315.0, cy=245.0)
    angle = np.radians(12.0)
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    translation = np.array([0.6, 0.0, 0.0])
    translation_unit = translation / np.linalg.norm(translation)

    fundamental = fundamental_from_pose(intrinsic1, intrinsic2, rotation, translation)
    essential = essential_from_fundamental(fundamental, intrinsic1, intrinsic2)
    essential_gt = essential_from_pose(rotation, translation_unit)  # singular values already [1, 1, 0]

    error = min(
        np.linalg.norm(essential - essential_gt),
        np.linalg.norm(essential + essential_gt),
    )
    assert error < 1e-6
    assert np.linalg.matrix_rank(essential, tol=1e-8) == 2


def test_essential_from_fundamental_singular_intrinsic_raises() -> None:
    singular_intrinsic = np.array(
        [[500.0, 500.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 0.0]]
    )
    valid_intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    fundamental = np.eye(3)
    with pytest.raises(ValueError):
        essential_from_fundamental(fundamental, singular_intrinsic, valid_intrinsic)


# ---------------------------------------------------------------------------
# enforce_essential_constraints
# ---------------------------------------------------------------------------


def test_enforce_essential_constraints_singular_values() -> None:
    rng = np.random.default_rng(0)
    noisy_matrix = rng.uniform(-1, 1, size=(3, 3))
    corrected = enforce_essential_constraints(noisy_matrix)

    singular_values = np.linalg.svd(corrected, compute_uv=False)
    assert np.isclose(singular_values[0], singular_values[1], atol=1e-9)
    assert singular_values[2] < 1e-9
    assert np.linalg.matrix_rank(corrected, tol=1e-8) == 2


def test_enforce_essential_constraints_zero_matrix_raises() -> None:
    with pytest.raises(ValueError):
        enforce_essential_constraints(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# decompose_essential_matrix
# ---------------------------------------------------------------------------


def test_decompose_essential_matrix_returns_four_candidates() -> None:
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)
    assert len(candidates) == 4


def test_decompose_essential_matrix_rotations_are_valid() -> None:
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)

    for rotation, _ in candidates:
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


def test_decompose_essential_matrix_translations_have_unit_norm() -> None:
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)

    for _, translation in candidates:
        assert np.isclose(np.linalg.norm(translation), 1.0, atol=1e-8)


def test_decompose_essential_matrix_candidates_do_not_share_references() -> None:
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)

    rotation0, translation0 = candidates[0]
    rotation0_copy = rotation0.copy()
    translation0_copy = translation0.copy()
    rotation0 += 100.0  # mutate in place
    translation0 += 100.0

    # other candidates using the "same" rotation/translation must be unaffected
    for rotation, translation in candidates[1:]:
        assert not np.allclose(rotation, rotation0)
        assert not np.allclose(translation, translation0)
    np.testing.assert_array_equal(candidates[0][0], rotation0_copy + 100.0)
    np.testing.assert_array_equal(candidates[0][1], translation0_copy + 100.0)


# ---------------------------------------------------------------------------
# select_pose_by_cheirality
# ---------------------------------------------------------------------------


def test_select_pose_by_cheirality_recovers_ground_truth_pose() -> None:
    pixels1, pixels2, intrinsic1, intrinsic2, rotation_gt, translation_gt_unit = (
        _synthetic_noiseless_scene(seed=1)
    )
    essential = essential_from_pose(rotation_gt, translation_gt_unit)
    candidates = decompose_essential_matrix(essential)

    rotation_est, translation_est, points_3d, positive_mask, counts = select_pose_by_cheirality(
        candidates, pixels1, pixels2, intrinsic1, intrinsic2
    )

    assert len(counts) == 4
    assert np.sum(positive_mask) == max(counts)

    rotation_error_deg = _rotation_angle_error_degrees(rotation_est, rotation_gt)
    translation_error_deg = _direction_angle_error_degrees(translation_est, translation_gt_unit)

    assert rotation_error_deg < 1e-4
    assert translation_error_deg < 1e-4
    assert np.all(np.isfinite(points_3d[positive_mask]))


def test_select_pose_by_cheirality_empty_candidates_raises() -> None:
    points1 = np.zeros((10, 2))
    points2 = np.zeros((10, 2))
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    with pytest.raises(ValueError):
        select_pose_by_cheirality([], points1, points2, intrinsic, intrinsic)


def test_select_pose_by_cheirality_shape_mismatch_raises() -> None:
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)
    points1 = np.zeros((10, 2))
    points2 = np.zeros((9, 2))
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    with pytest.raises(ValueError):
        select_pose_by_cheirality(candidates, points1, points2, intrinsic, intrinsic)


def test_select_pose_by_cheirality_all_candidates_fail_raises_runtime_error() -> None:
    # Points that are not consistent with any of the candidate poses: pure
    # noise correspondences give a low positive-depth ratio for every
    # candidate.
    rng = np.random.default_rng(2)
    essential = essential_from_pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    candidates = decompose_essential_matrix(essential)
    points1 = rng.uniform(0, 640, size=(30, 2))
    points2 = rng.uniform(0, 480, size=(30, 2))
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    with pytest.raises(RuntimeError):
        select_pose_by_cheirality(
            candidates, points1, points2, intrinsic, intrinsic, min_positive_ratio=0.99
        )
