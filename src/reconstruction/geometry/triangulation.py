"""Linear (DLT) triangulation and camera-frame depth utilities.

Convention: given projection matrices ``P1 = K1 [I | 0]`` (camera 1 as the
reconstruction reference frame) and ``P2 = K2 [R_21 | t_21]``, homogeneous
DLT triangulation recovers 3D points expressed in the camera 1 frame. See
:mod:`reconstruction.geometry.transforms` and
:mod:`reconstruction.geometry.epipolar` for the repository-wide
``T_target_source`` coordinate convention this builds on.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "projection_matrix",
    "triangulate_point_dlt",
    "triangulate_points_dlt",
    "camera_depths",
]


def _check_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if ``array`` contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _check_projection_matrix(matrix: np.ndarray, name: str) -> None:
    """Raise ValueError if ``matrix`` is not a finite (3, 4) projection matrix."""
    if matrix.shape != (3, 4):
        raise ValueError(f"{name} must have shape (3, 4), got {matrix.shape}.")
    _check_finite(matrix, name)


def projection_matrix(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Construct the camera projection matrix P = K [R | t].

    Args:
        intrinsic: Camera intrinsic matrix K, shape (3, 3).
        rotation: Rotation from the reconstruction reference frame to this
            camera's frame, shape (3, 3). Use the identity for the
            reference camera (camera 1).
        translation: Translation from the reconstruction reference frame to
            this camera's frame, shape (3,) or (3, 1). Use zero for the
            reference camera.

    Returns:
        Projection matrix, shape (3, 4).

    Raises:
        ValueError: If shapes are invalid or values are non-finite.
    """
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape == (3, 1):
        translation = translation.reshape(3)

    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {intrinsic.shape}.")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}.")
    if translation.shape != (3,):
        raise ValueError(
            f"translation must have shape (3,) or (3, 1), got {translation.shape}."
        )
    _check_finite(intrinsic, "intrinsic")
    _check_finite(rotation, "rotation")
    _check_finite(translation, "translation")

    extrinsic = np.concatenate([rotation, translation.reshape(3, 1)], axis=1)
    return intrinsic @ extrinsic


def triangulate_point_dlt(
    point1: np.ndarray,
    point2: np.ndarray,
    projection1: np.ndarray,
    projection2: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Triangulate one correspondence using homogeneous DLT.

    Builds the 4x4 linear system with rows::

        u1 * P1[2] - P1[0]
        v1 * P1[2] - P1[1]
        u2 * P2[2] - P2[0]
        v2 * P2[2] - P2[1]

    and solves ``A X_h = 0`` via SVD (``X_h = Vt[-1]``), then dehomogenizes
    ``X = X_h[:3] / X_h[3]``.

    Args:
        point1: Pixel coordinates (u1, v1) in image 1, shape (2,).
        point2: Pixel coordinates (u2, v2) in image 2, shape (2,).
        projection1: Camera 1 projection matrix, shape (3, 4).
        projection2: Camera 2 projection matrix, shape (3, 4).
        epsilon: The homogeneous scale ``|X_h[3]|`` must exceed this value
            for the point to be considered valid.

    Returns:
        3D point in the frame shared by ``projection1``/``projection2``,
        shape (3,). ``[nan, nan, nan]`` if the homogeneous scale is
        degenerate (``abs(X_h[3]) <= epsilon``, e.g. a point at infinity or
        a near-parallel ray pair).

    Raises:
        ValueError: If shapes are invalid or values are non-finite.
    """
    point1 = np.asarray(point1, dtype=np.float64)
    point2 = np.asarray(point2, dtype=np.float64)
    projection1 = np.asarray(projection1, dtype=np.float64)
    projection2 = np.asarray(projection2, dtype=np.float64)

    if point1.shape != (2,):
        raise ValueError(f"point1 must have shape (2,), got {point1.shape}.")
    if point2.shape != (2,):
        raise ValueError(f"point2 must have shape (2,), got {point2.shape}.")
    _check_finite(point1, "point1")
    _check_finite(point2, "point2")
    _check_projection_matrix(projection1, "projection1")
    _check_projection_matrix(projection2, "projection2")

    u1, v1 = point1
    u2, v2 = point2
    design_matrix = np.stack(
        [
            u1 * projection1[2] - projection1[0],
            v1 * projection1[2] - projection1[1],
            u2 * projection2[2] - projection2[0],
            v2 * projection2[2] - projection2[1],
        ],
        axis=0,
    )

    _, _, vt = np.linalg.svd(design_matrix)
    point_h = vt[-1]

    if abs(point_h[3]) <= epsilon:
        return np.full(3, np.nan, dtype=np.float64)

    return point_h[:3] / point_h[3]


def triangulate_points_dlt(
    points1: np.ndarray,
    points2: np.ndarray,
    projection1: np.ndarray,
    projection2: np.ndarray,
) -> np.ndarray:
    """Triangulate N correspondences with per-point homogeneous DLT.

    Correspondences whose DLT solve is degenerate (near-parallel rays /
    homogeneous scale near zero) are returned as NaN rather than raising,
    so a batch with a few bad rays does not abort the whole reconstruction;
    filter NaN rows downstream (e.g. together with a cheirality/reprojection
    filter).

    Args:
        points1: Pixel coordinates in image 1, shape (N, 2). N may be 0.
        points2: Pixel coordinates in image 2, shape (N, 2), same N.
        projection1: Camera 1 projection matrix, shape (3, 4).
        projection2: Camera 2 projection matrix, shape (3, 4).

    Returns:
        Triangulated 3D points, shape (N, 3). Degenerate rows are NaN.

    Raises:
        ValueError: If shapes are invalid, lengths differ, or values are
            non-finite.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(f"points1 must have shape (N, 2), got {points1.shape}.")
    if points2.ndim != 2 or points2.shape[1] != 2:
        raise ValueError(f"points2 must have shape (N, 2), got {points2.shape}.")
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            f"points1 and points2 must have the same N, got {points1.shape[0]} "
            f"and {points2.shape[0]}."
        )
    _check_finite(points1, "points1")
    _check_finite(points2, "points2")
    projection1 = np.asarray(projection1, dtype=np.float64)
    projection2 = np.asarray(projection2, dtype=np.float64)
    _check_projection_matrix(projection1, "projection1")
    _check_projection_matrix(projection2, "projection2")

    num_points = points1.shape[0]
    points_3d = np.empty((num_points, 3), dtype=np.float64)
    for i in range(num_points):
        points_3d[i] = triangulate_point_dlt(points1[i], points2[i], projection1, projection2)
    return points_3d


def camera_depths(
    points_camera1: np.ndarray,
    rotation_camera2_camera1: np.ndarray,
    translation_camera2_camera1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return point depths (Z coordinate) in camera 1 and camera 2.

    Args:
        points_camera1: 3D points in the camera 1 frame, shape (N, 3). Rows
            may be NaN (propagated to the outputs).
        rotation_camera2_camera1: Rotation R_21, shape (3, 3).
        translation_camera2_camera1: Translation t_21, shape (3,) or (3, 1).

    Returns:
        A tuple ``(depth1, depth2)``, each shape (N,): ``depth1 =
        points_camera1[:, 2]``; ``depth2`` is the Z coordinate after
        transforming into camera 2's frame,
        ``X2 = points_camera1 @ R_21.T + t_21``.

    Raises:
        ValueError: If shapes are invalid.
    """
    points_camera1 = np.asarray(points_camera1, dtype=np.float64)
    rotation = np.asarray(rotation_camera2_camera1, dtype=np.float64)
    translation = np.asarray(translation_camera2_camera1, dtype=np.float64)
    if translation.shape == (3, 1):
        translation = translation.reshape(3)

    if points_camera1.ndim != 2 or points_camera1.shape[1] != 3:
        raise ValueError(
            f"points_camera1 must have shape (N, 3), got {points_camera1.shape}."
        )
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation_camera2_camera1 must have shape (3, 3), got {rotation.shape}."
        )
    if translation.shape != (3,):
        raise ValueError(
            "translation_camera2_camera1 must have shape (3,) or (3, 1), got "
            f"{translation.shape}."
        )

    depth1 = points_camera1[:, 2]
    points_camera2 = points_camera1 @ rotation.T + translation
    depth2 = points_camera2[:, 2]
    return depth1, depth2
