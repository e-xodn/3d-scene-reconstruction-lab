"""Tests for reconstruction.geometry.transforms."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.geometry.transforms import (
    compose_transforms,
    invert_transform,
    make_transform,
    transform_points,
)

ATOL = 1e-9
RTOL = 1e-9


def _rotation_z(angle_rad: float) -> np.ndarray:
    """A rotation matrix about the z-axis, for use as test fixtures."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_identity_transform_does_not_change_points() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-5, 5, size=(10, 3))
    identity = make_transform(np.eye(3), np.zeros(3))

    result = transform_points(points, identity)

    np.testing.assert_allclose(result, points, atol=ATOL, rtol=RTOL)


def test_translation_applied_correctly() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    translation = np.array([10.0, -5.0, 2.0])
    transform = make_transform(np.eye(3), translation)

    result = transform_points(points, transform)

    expected = points + translation
    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_rotation_applied_correctly() -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    rotation = _rotation_z(np.pi / 2)
    transform = make_transform(rotation, np.zeros(3))

    result = transform_points(points, transform)

    expected = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_transform_then_inverse_restores_original_points() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, size=(20, 3))
    rotation = _rotation_z(0.7)
    translation = np.array([3.0, -1.0, 4.0])
    transform = make_transform(rotation, translation)
    inverse = invert_transform(transform)

    transformed = transform_points(points, transform)
    restored = transform_points(transformed, inverse)

    np.testing.assert_allclose(restored, points, atol=ATOL, rtol=RTOL)


def test_transform_composed_with_inverse_is_identity() -> None:
    rotation = _rotation_z(1.234)
    translation = np.array([5.0, 6.0, 7.0])
    transform = make_transform(rotation, translation)
    inverse = invert_transform(transform)

    identity = np.eye(4)
    np.testing.assert_allclose(inverse @ transform, identity, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(transform @ inverse, identity, atol=ATOL, rtol=RTOL)


def test_compose_transforms_matches_sequential_application() -> None:
    rng = np.random.default_rng(2)
    points = rng.uniform(-5, 5, size=(15, 3))

    t_b_a = make_transform(_rotation_z(0.3), np.array([1.0, 0.0, 0.0]))
    t_c_b = make_transform(_rotation_z(-0.8), np.array([0.0, 2.0, 0.0]))

    t_c_a = compose_transforms(t_c_b, t_b_a)

    direct = transform_points(points, t_c_a)
    sequential = transform_points(transform_points(points, t_b_a), t_c_b)

    np.testing.assert_allclose(direct, sequential, atol=ATOL, rtol=RTOL)


def test_transform_points_handles_empty_array() -> None:
    points = np.empty((0, 3))
    transform = make_transform(np.eye(3), np.zeros(3))

    result = transform_points(points, transform)

    assert result.shape == (0, 3)


def test_transform_points_invalid_point_shape_raises() -> None:
    points = np.zeros((5, 2))
    transform = make_transform(np.eye(3), np.zeros(3))

    with pytest.raises(ValueError):
        transform_points(points, transform)


def test_transform_points_invalid_transform_shape_raises() -> None:
    points = np.zeros((5, 3))
    bad_transform = np.eye(3)

    with pytest.raises(ValueError):
        transform_points(points, bad_transform)


def test_make_transform_non_rotation_matrix_raises() -> None:
    not_a_rotation = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        make_transform(not_a_rotation, np.zeros(3))


def test_make_transform_nan_raises() -> None:
    rotation = np.eye(3)
    translation = np.array([np.nan, 0.0, 0.0])

    with pytest.raises(ValueError):
        make_transform(rotation, translation)


def test_transform_points_does_not_mutate_inputs() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    points_copy = points.copy()
    rotation = _rotation_z(0.5)
    translation = np.array([1.0, 1.0, 1.0])
    transform = make_transform(rotation, translation)
    transform_copy = transform.copy()

    transform_points(points, transform)

    np.testing.assert_array_equal(points, points_copy)
    np.testing.assert_array_equal(transform, transform_copy)


def test_invert_transform_does_not_mutate_input() -> None:
    transform = make_transform(_rotation_z(0.2), np.array([1.0, 2.0, 3.0]))
    transform_copy = transform.copy()

    invert_transform(transform)

    np.testing.assert_array_equal(transform, transform_copy)


def test_compose_transforms_invalid_shape_raises() -> None:
    valid = make_transform(np.eye(3), np.zeros(3))
    invalid = np.eye(3)

    with pytest.raises(ValueError):
        compose_transforms(valid, invalid)
