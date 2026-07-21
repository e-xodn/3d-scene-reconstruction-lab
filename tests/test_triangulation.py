"""Tests for reconstruction.geometry.triangulation (DLT triangulation, depths)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reconstruction.cameras.pinhole import create_intrinsic_matrix, project_points
from reconstruction.geometry.transforms import make_transform, transform_points
from reconstruction.geometry.triangulation import (
    camera_depths,
    projection_matrix,
    triangulate_point_dlt,
    triangulate_points_dlt,
)

IMAGE_SIZE = (640, 480)


def _known_scene(num_points: int = 20, seed: int = 0):
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])

    projection1 = projection_matrix(intrinsic, np.eye(3), np.zeros(3))
    projection2 = projection_matrix(intrinsic, rotation, translation)

    rng = np.random.default_rng(seed)
    points_3d_gt = np.stack(
        [
            rng.uniform(-2.0, 2.0, size=num_points),
            rng.uniform(-2.0, 2.0, size=num_points),
            rng.uniform(5.0, 15.0, size=num_points),
        ],
        axis=1,
    )
    pixels1, _, valid1 = project_points(points_3d_gt, intrinsic, image_size=IMAGE_SIZE)
    points_camera2 = transform_points(points_3d_gt, make_transform(rotation, translation))
    pixels2, _, valid2 = project_points(points_camera2, intrinsic, image_size=IMAGE_SIZE)
    valid = valid1 & valid2
    assert np.sum(valid) >= 8

    return (
        pixels1[valid],
        pixels2[valid],
        points_3d_gt[valid],
        projection1,
        projection2,
        rotation,
        translation,
    )


# ---------------------------------------------------------------------------
# projection_matrix
# ---------------------------------------------------------------------------


def test_projection_matrix_shape_and_values() -> None:
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    projection = projection_matrix(intrinsic, np.eye(3), np.zeros(3))
    assert projection.shape == (3, 4)
    np.testing.assert_allclose(projection[:, :3], intrinsic, atol=1e-12)
    np.testing.assert_allclose(projection[:, 3], np.zeros(3), atol=1e-12)


def test_projection_matrix_invalid_rotation_shape_raises() -> None:
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    with pytest.raises(ValueError):
        projection_matrix(intrinsic, np.eye(2), np.zeros(3))


# ---------------------------------------------------------------------------
# triangulate_points_dlt / triangulate_point_dlt
# ---------------------------------------------------------------------------


def test_triangulate_points_dlt_matches_ground_truth() -> None:
    pixels1, pixels2, points_3d_gt, projection1, projection2, _, _ = _known_scene()
    points_3d_est = triangulate_points_dlt(pixels1, pixels2, projection1, projection2)

    assert points_3d_est.shape == points_3d_gt.shape
    np.testing.assert_allclose(points_3d_est, points_3d_gt, atol=1e-6, rtol=1e-6)


def test_triangulate_points_dlt_output_shape() -> None:
    pixels1, pixels2, _, projection1, projection2, _, _ = _known_scene(num_points=15)
    points_3d = triangulate_points_dlt(pixels1, pixels2, projection1, projection2)
    assert points_3d.shape == (pixels1.shape[0], 3)


def test_camera_depths_are_positive_for_valid_triangulation() -> None:
    pixels1, pixels2, _, projection1, projection2, rotation, translation = _known_scene()
    points_3d = triangulate_points_dlt(pixels1, pixels2, projection1, projection2)
    depth1, depth2 = camera_depths(points_3d, rotation, translation)

    assert np.all(depth1 > 0)
    assert np.all(depth2 > 0)


def test_triangulate_points_dlt_empty_input() -> None:
    projection = np.zeros((3, 4))
    points_3d = triangulate_points_dlt(
        np.empty((0, 2)), np.empty((0, 2)), projection, projection
    )
    assert points_3d.shape == (0, 3)


def test_triangulate_points_dlt_length_mismatch_raises() -> None:
    projection = np.eye(3, 4)
    points1 = np.zeros((5, 2))
    points2 = np.zeros((4, 2))
    with pytest.raises(ValueError):
        triangulate_points_dlt(points1, points2, projection, projection)


def test_triangulate_points_dlt_invalid_projection_shape_raises() -> None:
    points1 = np.zeros((5, 2))
    points2 = np.zeros((5, 2))
    bad_projection = np.zeros((3, 3))
    with pytest.raises(ValueError):
        triangulate_points_dlt(points1, points2, bad_projection, bad_projection)


def test_triangulate_point_dlt_large_epsilon_flags_degenerate() -> None:
    """A very large epsilon forces any normally-finite homogeneous scale to
    be treated as degenerate, exercising the point-at-infinity / degenerate
    ray handling path deterministically."""
    intrinsic = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    projection1 = projection_matrix(intrinsic, np.eye(3), np.zeros(3))
    projection2 = projection_matrix(intrinsic, np.eye(3), np.array([1.0, 0.0, 0.0]))
    point1 = np.array([340.0, 250.0])
    point2 = np.array([300.0, 250.0])

    result = triangulate_point_dlt(point1, point2, projection1, projection2, epsilon=1e6)
    assert np.all(np.isnan(result))


def test_triangulate_points_dlt_does_not_mutate_inputs() -> None:
    pixels1, pixels2, _, projection1, projection2, _, _ = _known_scene()
    pixels1_copy = pixels1.copy()
    pixels2_copy = pixels2.copy()
    projection1_copy = projection1.copy()
    projection2_copy = projection2.copy()

    triangulate_points_dlt(pixels1, pixels2, projection1, projection2)

    np.testing.assert_array_equal(pixels1, pixels1_copy)
    np.testing.assert_array_equal(pixels2, pixels2_copy)
    np.testing.assert_array_equal(projection1, projection1_copy)
    np.testing.assert_array_equal(projection2, projection2_copy)


def test_camera_depths_invalid_rotation_shape_raises() -> None:
    points_camera1 = np.zeros((5, 3))
    with pytest.raises(ValueError):
        camera_depths(points_camera1, np.eye(2), np.zeros(3))


def test_triangulation_module_does_not_use_cv2_triangulate_points() -> None:
    """Guard against accidentally reintroducing cv2.triangulatePoints."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "reconstruction"
        / "geometry"
        / "triangulation.py"
    )
    source = module_path.read_text()
    assert "triangulatePoints" not in source
    assert "cv2" not in source
