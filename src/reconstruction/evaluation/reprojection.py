"""Reprojection error metrics for evaluating triangulated 3D points."""

from __future__ import annotations

import numpy as np

__all__ = [
    "project_with_projection_matrix",
    "reprojection_errors",
    "two_view_reprojection_errors",
]


def project_with_projection_matrix(
    points_3d: np.ndarray,
    projection: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points with a 3x4 projection matrix.

    Args:
        points_3d: 3D points, shape (N, 3). Rows containing NaN/Inf are
            marked invalid rather than raising.
        projection: Projection matrix P, shape (3, 4).
        epsilon: The homogeneous denominator must exceed this magnitude for
            a projection to be valid.

    Returns:
        A tuple ``(pixels, valid)``:
            pixels: shape (N, 2), NaN for invalid points.
            valid: shape (N,), boolean.

    Raises:
        ValueError: If ``points_3d`` does not have shape (N, 3) or
            ``projection`` is not a finite (3, 4) matrix.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}.")
    if projection.shape != (3, 4):
        raise ValueError(f"projection must have shape (3, 4), got {projection.shape}.")
    if not np.all(np.isfinite(projection)):
        raise ValueError("projection must contain only finite values (no NaN/Inf).")

    num_points = points_3d.shape[0]
    finite_input = np.all(np.isfinite(points_3d), axis=1)

    points_h = np.concatenate([points_3d, np.ones((num_points, 1), dtype=np.float64)], axis=1)
    projected_h = points_h @ projection.T

    denominator = projected_h[:, 2]
    valid = finite_input & (np.abs(denominator) > epsilon)

    safe_denominator = np.where(valid, denominator, 1.0)
    pixels = projected_h[:, :2] / safe_denominator[:, None]
    pixels[~valid] = np.nan

    return pixels, valid


def reprojection_errors(
    points_3d: np.ndarray,
    observed_points: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean reprojection errors in pixels.

    Args:
        points_3d: 3D points, shape (N, 3).
        observed_points: Observed pixel coordinates, shape (N, 2).
        projection: Projection matrix, shape (3, 4).

    Returns:
        Reprojection errors in pixels, shape (N,). Inf for points whose
        projection is invalid (see :func:`project_with_projection_matrix`).

    Raises:
        ValueError: If shapes are invalid, lengths differ, or
            ``observed_points`` contains non-finite values.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    observed_points = np.asarray(observed_points, dtype=np.float64)

    if observed_points.ndim != 2 or observed_points.shape[1] != 2:
        raise ValueError(
            f"observed_points must have shape (N, 2), got {observed_points.shape}."
        )
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d must have shape (N, 3), got {points_3d.shape}.")
    if points_3d.shape[0] != observed_points.shape[0]:
        raise ValueError(
            f"points_3d and observed_points must have the same N, got "
            f"{points_3d.shape[0]} and {observed_points.shape[0]}."
        )
    if not np.all(np.isfinite(observed_points)):
        raise ValueError("observed_points must contain only finite values (no NaN/Inf).")

    pixels, valid = project_with_projection_matrix(points_3d, projection)

    errors = np.full(points_3d.shape[0], np.inf, dtype=np.float64)
    errors[valid] = np.linalg.norm(pixels[valid] - observed_points[valid], axis=1)
    return errors


def two_view_reprojection_errors(
    points_3d: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    projection1: np.ndarray,
    projection2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute image-1, image-2, and combined reprojection errors.

    Combined error is defined as
    ``sqrt((error1 ** 2 + error2 ** 2) / 2)``, the root-mean-square of the
    two per-image pixel errors.

    Args:
        points_3d: 3D points, shape (N, 3).
        points1: Observed pixel coordinates in image 1, shape (N, 2).
        points2: Observed pixel coordinates in image 2, shape (N, 2).
        projection1: Camera 1 projection matrix, shape (3, 4).
        projection2: Camera 2 projection matrix, shape (3, 4).

    Returns:
        A tuple ``(error1, error2, combined)``, each shape (N,), in pixels.

    Raises:
        ValueError: If shapes are invalid or lengths differ (see
            :func:`reprojection_errors`).
    """
    error1 = reprojection_errors(points_3d, points1, projection1)
    error2 = reprojection_errors(points_3d, points2, projection2)
    combined = np.sqrt((error1**2 + error2**2) / 2.0)
    return error1, error2, combined
