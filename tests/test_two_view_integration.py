"""End-to-end synthetic integration test for the M2B two-view math pipeline.

Exercises the full geometry pipeline — RANSAC fundamental matrix, essential
matrix, pose decomposition, cheirality selection, DLT triangulation, and
reprojection evaluation — on a synthetic scene with known ground truth,
pixel noise, and injected outliers. This does *not* cover feature
detection/matching (see tests/test_matching.py for that in isolation); it
validates that the downstream geometry modules compose correctly.
"""

from __future__ import annotations

import numpy as np

from reconstruction.cameras.pinhole import create_intrinsic_matrix, project_points
from reconstruction.evaluation.reprojection import two_view_reprojection_errors
from reconstruction.geometry.pose import (
    decompose_essential_matrix,
    essential_from_fundamental,
    select_pose_by_cheirality,
)
from reconstruction.geometry.robust import estimate_fundamental_matrix_ransac
from reconstruction.geometry.transforms import make_transform, transform_points
from reconstruction.geometry.triangulation import projection_matrix, triangulate_points_dlt

IMAGE_SIZE = (640, 480)


def _rotation_angle_error_degrees(rotation_est: np.ndarray, rotation_gt: np.ndarray) -> float:
    rotation_error = rotation_est @ rotation_gt.T
    cos_angle = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _direction_angle_error_degrees(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    cos_angle = np.clip(
        np.dot(t_est, t_gt) / (np.linalg.norm(t_est) * np.linalg.norm(t_gt)), -1.0, 1.0
    )
    return float(np.degrees(np.arccos(cos_angle)))


def test_full_synthetic_two_view_pipeline() -> None:
    rng = np.random.default_rng(123)

    intrinsic1 = create_intrinsic_matrix(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    intrinsic2 = create_intrinsic_matrix(fx=510.0, fy=505.0, cx=315.0, cy=245.0)

    angle = np.radians(9.0)
    rotation_gt = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    translation_gt = np.array([-0.5, 0.06, 0.08])
    translation_gt_unit = translation_gt / np.linalg.norm(translation_gt)
    transform_gt = make_transform(rotation_gt, translation_gt)

    num_inliers = 150
    num_outliers = 50
    oversample = num_inliers * 3
    points_camera1 = np.stack(
        [
            rng.uniform(-2.0, 2.0, size=oversample),
            rng.uniform(-1.5, 1.5, size=oversample),
            rng.uniform(5.0, 12.0, size=oversample),
        ],
        axis=1,
    )
    points_camera2 = transform_points(points_camera1, transform_gt)
    pixels1_clean, _, valid1 = project_points(points_camera1, intrinsic1, image_size=IMAGE_SIZE)
    pixels2_clean, _, valid2 = project_points(points_camera2, intrinsic2, image_size=IMAGE_SIZE)
    valid = valid1 & valid2
    assert np.sum(valid) >= num_inliers

    pixels1_clean = pixels1_clean[valid][:num_inliers]
    pixels2_clean = pixels2_clean[valid][:num_inliers]
    pixels1_noisy = pixels1_clean + rng.normal(0.0, 0.5, size=pixels1_clean.shape)
    pixels2_noisy = pixels2_clean + rng.normal(0.0, 0.5, size=pixels2_clean.shape)

    outliers1 = rng.uniform([0.0, 0.0], [IMAGE_SIZE[0], IMAGE_SIZE[1]], size=(num_outliers, 2))
    outliers2 = rng.uniform([0.0, 0.0], [IMAGE_SIZE[0], IMAGE_SIZE[1]], size=(num_outliers, 2))

    points1 = np.concatenate([pixels1_noisy, outliers1], axis=0)
    points2 = np.concatenate([pixels2_noisy, outliers2], axis=0)
    order = rng.permutation(points1.shape[0])
    points1 = points1[order]
    points2 = points2[order]

    # --- RANSAC fundamental matrix ---
    fundamental, inlier_mask, ransac_stats = estimate_fundamental_matrix_ransac(
        points1, points2, threshold_px=1.5, seed=42, min_inliers=20
    )
    assert np.linalg.matrix_rank(fundamental, tol=1e-8) == 2
    assert ransac_stats["final_inlier_count"] >= 0.7 * num_inliers

    points1_inliers = points1[inlier_mask]
    points2_inliers = points2[inlier_mask]

    # --- essential matrix + pose decomposition + cheirality ---
    essential = essential_from_fundamental(fundamental, intrinsic1, intrinsic2)
    assert np.linalg.matrix_rank(essential, tol=1e-8) == 2
    candidates = decompose_essential_matrix(essential)
    assert len(candidates) == 4

    rotation_est, translation_est, _, positive_mask, positive_depth_counts = (
        select_pose_by_cheirality(candidates, points1_inliers, points2_inliers, intrinsic1, intrinsic2)
    )
    assert len(positive_depth_counts) == 4

    rotation_error_deg = _rotation_angle_error_degrees(rotation_est, rotation_gt)
    translation_error_deg = _direction_angle_error_degrees(translation_est, translation_gt_unit)
    assert rotation_error_deg < 1.0
    assert translation_error_deg < 2.0

    positive_depth_ratio = np.sum(positive_mask) / points1_inliers.shape[0]
    assert positive_depth_ratio >= 0.9

    # --- triangulation + reprojection evaluation ---
    projection1 = projection_matrix(intrinsic1, np.eye(3), np.zeros(3))
    projection2 = projection_matrix(intrinsic2, rotation_est, translation_est)
    points_3d = triangulate_points_dlt(points1_inliers, points2_inliers, projection1, projection2)
    _, _, combined_error = two_view_reprojection_errors(
        points_3d, points1_inliers, points2_inliers, projection1, projection2
    )
    finite_combined = combined_error[np.isfinite(combined_error)]
    assert finite_combined.size > 0
    assert np.median(finite_combined) < 2.0
