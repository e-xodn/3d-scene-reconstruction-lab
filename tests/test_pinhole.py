"""Tests for reconstruction.cameras.pinhole."""

from __future__ import annotations

import numpy as np
import pytest

from reconstruction.cameras.pinhole import (
    create_intrinsic_matrix,
    project_points,
    unproject_pixels,
)

ATOL = 1e-9
RTOL = 1e-9


def _example_intrinsic() -> np.ndarray:
    return create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def test_create_intrinsic_matrix_values() -> None:
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=600.0, cx=320.0, cy=240.0)

    expected = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(intrinsic, expected, atol=ATOL, rtol=RTOL)
    assert intrinsic.dtype == np.float64


def test_create_intrinsic_matrix_nonpositive_fx_raises() -> None:
    with pytest.raises(ValueError):
        create_intrinsic_matrix(fx=0.0, fy=500.0, cx=320.0, cy=240.0)


def test_create_intrinsic_matrix_nonpositive_fy_raises() -> None:
    with pytest.raises(ValueError):
        create_intrinsic_matrix(fx=500.0, fy=-1.0, cx=320.0, cy=240.0)


def test_optical_axis_point_projects_to_principal_point() -> None:
    intrinsic = _example_intrinsic()
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 100.0]])

    pixels, depth, valid = project_points(points, intrinsic)

    expected_pixels = np.array([[320.0, 240.0], [320.0, 240.0]])
    np.testing.assert_allclose(pixels, expected_pixels, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(depth, np.array([5.0, 100.0]), atol=ATOL, rtol=RTOL)
    assert np.all(valid)


def test_known_point_projection_matches_equation() -> None:
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    points = np.array([[1.0, 2.0, 4.0]])

    pixels, depth, valid = project_points(points, intrinsic)

    expected_u = 500.0 * 1.0 / 4.0 + 320.0
    expected_v = 500.0 * 2.0 / 4.0 + 240.0
    np.testing.assert_allclose(
        pixels, np.array([[expected_u, expected_v]]), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(depth, np.array([4.0]), atol=ATOL, rtol=RTOL)
    assert valid[0]


def test_zero_or_negative_depth_is_invalid() -> None:
    intrinsic = _example_intrinsic()
    points = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, -5.0], [1.0, 1.0, 5.0]])

    pixels, depth, valid = project_points(points, intrinsic)

    np.testing.assert_array_equal(valid, np.array([False, False, True]))
    assert np.all(np.isnan(pixels[0]))
    assert np.all(np.isnan(pixels[1]))
    assert not np.any(np.isnan(pixels[2]))


def test_point_outside_image_bounds_is_invalid_when_image_size_given() -> None:
    intrinsic = _example_intrinsic()
    # Far off to the side: projects way outside a 640x480 image.
    points = np.array([[1000.0, 0.0, 1.0], [0.0, 0.0, 5.0]])

    pixels, depth, valid = project_points(
        points, intrinsic, image_size=(640, 480)
    )

    np.testing.assert_array_equal(valid, np.array([False, True]))
    assert np.all(np.isnan(pixels[0]))


def test_positive_depth_valid_regardless_of_bounds_when_no_image_size() -> None:
    intrinsic = _example_intrinsic()
    points = np.array([[1000.0, 0.0, 1.0], [0.0, 0.0, 5.0]])

    pixels, depth, valid = project_points(points, intrinsic, image_size=None)

    assert np.all(valid)
    assert not np.any(np.isnan(pixels))


def test_project_unproject_round_trip() -> None:
    rng = np.random.default_rng(42)
    intrinsic = _example_intrinsic()
    n = 50
    xy = rng.uniform(-2.0, 2.0, size=(n, 2))
    z = rng.uniform(1.0, 20.0, size=(n, 1))
    points_camera = np.hstack([xy, z])

    pixels, depth, valid = project_points(points_camera, intrinsic)
    assert np.all(valid)

    restored = unproject_pixels(pixels, depth, intrinsic)

    np.testing.assert_allclose(restored, points_camera, atol=1e-9, rtol=1e-9)


def test_unproject_pixels_length_mismatch_raises() -> None:
    intrinsic = _example_intrinsic()
    pixels = np.zeros((3, 2))
    depth = np.ones(4)

    with pytest.raises(ValueError):
        unproject_pixels(pixels, depth, intrinsic)


def test_unproject_pixels_nonpositive_depth_raises() -> None:
    intrinsic = _example_intrinsic()
    pixels = np.array([[320.0, 240.0], [100.0, 100.0]])
    depth = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        unproject_pixels(pixels, depth, intrinsic)


def test_project_points_does_not_mutate_inputs() -> None:
    intrinsic = _example_intrinsic()
    intrinsic_copy = intrinsic.copy()
    points = np.array([[1.0, 1.0, 5.0], [2.0, -1.0, 3.0]])
    points_copy = points.copy()

    project_points(points, intrinsic)

    np.testing.assert_array_equal(points, points_copy)
    np.testing.assert_array_equal(intrinsic, intrinsic_copy)


def test_unproject_pixels_does_not_mutate_inputs() -> None:
    intrinsic = _example_intrinsic()
    pixels = np.array([[320.0, 240.0], [100.0, 100.0]])
    pixels_copy = pixels.copy()
    depth = np.array([5.0, 3.0])
    depth_copy = depth.copy()

    unproject_pixels(pixels, depth, intrinsic)

    np.testing.assert_array_equal(pixels, pixels_copy)
    np.testing.assert_array_equal(depth, depth_copy)
