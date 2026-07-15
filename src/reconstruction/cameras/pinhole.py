"""Pinhole camera model: intrinsics, projection, and unprojection.

Convention
----------
- Camera-frame 3D points are ``(N, 3)`` arrays with columns ``(X, Y, Z)``,
  where ``Z`` is depth along the optical axis.
- Pixel coordinates ``(u, v)`` follow the standard image convention: ``u``
  increases rightward, ``v`` increases downward, and ``(0, 0)`` is the
  top-left corner of the image.
- ``image_size`` is always given as ``(width, height)``.
- Projection: ``u = fx * X / Z + cx``, ``v = fy * Y / Z + cy``.
- Unprojection: ``X = (u - cx) * Z / fx``, ``Y = (v - cy) * Z / fy``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "create_intrinsic_matrix",
    "project_points",
    "unproject_pixels",
]


def _check_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if ``array`` contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _check_intrinsic_matrix(intrinsic: np.ndarray) -> None:
    """Raise ValueError if ``intrinsic`` is not a valid (3, 3) intrinsic matrix."""
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {intrinsic.shape}.")
    _check_finite(intrinsic, "intrinsic")


def create_intrinsic_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Create a 3x3 pinhole camera intrinsic matrix.

    Args:
        fx: Focal length along the x-axis, in pixels. Must be > 0.
        fy: Focal length along the y-axis, in pixels. Must be > 0.
        cx: Principal point x-coordinate, in pixels.
        cy: Principal point y-coordinate, in pixels.

    Returns:
        Intrinsic matrix K of shape (3, 3), dtype float64::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Raises:
        ValueError: If fx or fy is not finite and positive, or if cx/cy is
            not finite.
    """
    values = np.array([fx, fy, cx, cy], dtype=np.float64)
    _check_finite(values, "fx/fy/cx/cy")

    fx, fy, cx, cy = values
    if fx <= 0:
        raise ValueError(f"fx must be > 0, got {fx}.")
    if fy <= 0:
        raise ValueError(f"fy must be > 0, got {fy}.")

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_points(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int] | None = None,
    min_depth: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame 3D points onto the image plane.

    Projection equations::

        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Args:
        points_camera: Camera-frame 3D points, shape (N, 3).
        intrinsic: Intrinsic matrix, shape (3, 3).
        image_size: Optional (width, height) in pixels. When given, points
            whose pixel coordinates fall outside ``[0, width)`` x
            ``[0, height)`` are also marked invalid.
        min_depth: Points with ``Z <= min_depth`` are marked invalid.

    Returns:
        A tuple ``(pixels, depth, valid)``:
            pixels: shape (N, 2), pixel coordinates (u, v). NaN for invalid
                points.
            depth: shape (N,), the original Z values.
            valid: shape (N,), boolean validity mask.

    Raises:
        ValueError: If ``points_camera`` does not have shape (N, 3) or
            ``intrinsic`` is not a valid (3, 3) matrix.
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            f"points_camera must have shape (N, 3), got {points_camera.shape}."
        )
    _check_finite(points_camera, "points_camera")
    _check_intrinsic_matrix(intrinsic)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]

    depth = z.copy()
    valid = z > min_depth

    # Avoid division-by-zero warnings by substituting a safe divisor for
    # invalid entries; their results are discarded via NaN below.
    safe_z = np.where(valid, z, 1.0)
    u = fx * x / safe_z + cx
    v = fy * y / safe_z + cy

    if image_size is not None:
        width, height = image_size
        in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid = valid & in_bounds

    pixels = np.stack([u, v], axis=1)
    pixels[~valid] = np.nan

    return pixels, depth, valid


def unproject_pixels(
    pixels: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Unproject pixels and depth into camera-frame 3D points.

    Unprojection equations::

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Args:
        pixels: Pixel coordinates (u, v), shape (N, 2).
        depth: Depth values Z, shape (N,). Must be strictly positive and
            finite.
        intrinsic: Intrinsic matrix, shape (3, 3).

    Returns:
        Camera-frame 3D points, shape (N, 3).

    Raises:
        ValueError: If shapes are inconsistent, ``intrinsic`` is invalid, or
            ``depth`` contains non-positive or non-finite values.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}.")
    if depth.ndim != 1:
        raise ValueError(f"depth must have shape (N,), got {depth.shape}.")
    if pixels.shape[0] != depth.shape[0]:
        raise ValueError(
            f"pixels and depth must have the same N, got {pixels.shape[0]} "
            f"and {depth.shape[0]}."
        )
    _check_finite(pixels, "pixels")
    _check_finite(depth, "depth")
    _check_intrinsic_matrix(intrinsic)

    if np.any(depth <= 0):
        raise ValueError("depth must be strictly positive for all points.")

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = pixels[:, 0]
    v = pixels[:, 1]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy

    return np.stack([x, y, depth], axis=1)
