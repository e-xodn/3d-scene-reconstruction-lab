"""Tests for reconstruction.evaluation.reprojection."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.cameras.pinhole import create_intrinsic_matrix
from reconstruction.evaluation.reprojection import (
    project_with_projection_matrix,
    reprojection_errors,
    two_view_reprojection_errors,
)
from reconstruction.geometry.triangulation import projection_matrix

INTRINSIC = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
PROJECTION1 = projection_matrix(INTRINSIC, np.eye(3), np.zeros(3))
PROJECTION2 = projection_matrix(INTRINSIC, np.eye(3), np.array([1.0, 0.0, 0.0]))


def test_project_with_projection_matrix_known_point() -> None:
    points_3d = np.array([[0.0, 0.0, 5.0]])
    pixels, valid = project_with_projection_matrix(points_3d, PROJECTION1)

    np.testing.assert_allclose(pixels, [[320.0, 240.0]], atol=1e-9)
    assert valid[0]


def test_project_with_projection_matrix_invalid_points_are_flagged() -> None:
    points_3d = np.array([[0.0, 0.0, 5.0], [np.nan, 0.0, 5.0]])
    pixels, valid = project_with_projection_matrix(points_3d, PROJECTION1)

    np.testing.assert_array_equal(valid, [True, False])
    assert np.all(np.isnan(pixels[1]))


def test_project_with_projection_matrix_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        project_with_projection_matrix(np.zeros((5, 2)), PROJECTION1)


def test_reprojection_errors_zero_for_perfect_observation() -> None:
    points_3d = np.array([[0.0, 0.0, 5.0], [1.0, -1.0, 8.0]])
    pixels, valid = project_with_projection_matrix(points_3d, PROJECTION1)
    assert np.all(valid)

    errors = reprojection_errors(points_3d, pixels, PROJECTION1)
    np.testing.assert_allclose(errors, np.zeros(2), atol=1e-9)


def test_reprojection_errors_increase_with_pixel_perturbation() -> None:
    points_3d = np.array([[0.0, 0.0, 5.0]])
    pixels, _ = project_with_projection_matrix(points_3d, PROJECTION1)
    perturbed = pixels + np.array([[3.0, 4.0]])

    errors = reprojection_errors(points_3d, perturbed, PROJECTION1)
    np.testing.assert_allclose(errors, [5.0], atol=1e-9)  # 3-4-5 triangle


def test_reprojection_errors_invalid_projection_is_inf() -> None:
    points_3d = np.array([[np.nan, 0.0, 5.0]])
    observed = np.array([[320.0, 240.0]])
    errors = reprojection_errors(points_3d, observed, PROJECTION1)
    assert np.isinf(errors[0])


def test_reprojection_errors_shape_mismatch_raises() -> None:
    points_3d = np.zeros((5, 3))
    observed = np.zeros((4, 2))
    with pytest.raises(ValueError):
        reprojection_errors(points_3d, observed, PROJECTION1)


def test_reprojection_errors_nan_observed_raises() -> None:
    points_3d = np.zeros((2, 3))
    observed = np.array([[1.0, 2.0], [np.nan, 4.0]])
    with pytest.raises(ValueError):
        reprojection_errors(points_3d, observed, PROJECTION1)


def test_two_view_reprojection_errors_definition() -> None:
    points_3d = np.array([[0.2, -0.1, 6.0], [0.5, 0.3, 7.0]])
    pixels1, valid1 = project_with_projection_matrix(points_3d, PROJECTION1)
    pixels2, valid2 = project_with_projection_matrix(points_3d, PROJECTION2)
    assert np.all(valid1) and np.all(valid2)

    observed1 = pixels1 + np.array([[1.0, 0.0], [0.0, 2.0]])
    observed2 = pixels2 + np.array([[0.0, 3.0], [4.0, 0.0]])

    error1, error2, combined = two_view_reprojection_errors(
        points_3d, observed1, observed2, PROJECTION1, PROJECTION2
    )

    np.testing.assert_allclose(error1, [1.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(error2, [3.0, 4.0], atol=1e-9)
    expected_combined = np.sqrt((error1**2 + error2**2) / 2.0)
    np.testing.assert_allclose(combined, expected_combined, atol=1e-12)


def test_project_with_projection_matrix_does_not_mutate_inputs() -> None:
    points_3d = np.array([[0.0, 0.0, 5.0]])
    points_3d_copy = points_3d.copy()
    projection_copy = PROJECTION1.copy()

    project_with_projection_matrix(points_3d, PROJECTION1)

    np.testing.assert_array_equal(points_3d, points_3d_copy)
    np.testing.assert_array_equal(PROJECTION1, projection_copy)
