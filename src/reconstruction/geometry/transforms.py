"""SE(3) rigid-body transform utilities.

Coordinate convention used throughout this repository
------------------------------------------------------
- 3D points are stored as ``(N, 3)`` numpy arrays (row-major, one point per row).
- A homogeneous rigid-body transform is a ``(4, 4)`` numpy array.
- Transform variables are named ``T_target_source``: applying ``T_target_source``
  to points expressed in ``source`` frame yields points expressed in ``target``
  frame.
- The mathematical definition is::

      p_target = R_target_source @ p_source + t_target_source

  Since points are stored as ``(N, 3)`` row-arrays, the equivalent numpy
  expression operating on a whole batch of points is::

      points_target = points_source @ R.T + t

- Transform composition follows::

      T_target_source = T_target_intermediate @ T_intermediate_source
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_transform",
    "transform_points",
    "invert_transform",
    "compose_transforms",
]

_ROTATION_ORTHOGONALITY_ATOL = 1e-6
_ROTATION_DETERMINANT_ATOL = 1e-6
_HOMOGENEOUS_ROW_ATOL = 1e-8


def _check_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if ``array`` contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _check_rotation_matrix(rotation: np.ndarray) -> None:
    """Raise ValueError if ``rotation`` is not an approximately valid SO(3) matrix."""
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}.")
    _check_finite(rotation, "rotation")

    should_be_identity = rotation.T @ rotation
    if not np.allclose(
        should_be_identity, np.eye(3), atol=_ROTATION_ORTHOGONALITY_ATOL
    ):
        raise ValueError(
            "rotation is not orthogonal: R.T @ R is not approximately the "
            "identity matrix."
        )

    determinant = np.linalg.det(rotation)
    if not np.isclose(determinant, 1.0, atol=_ROTATION_DETERMINANT_ATOL):
        raise ValueError(
            f"rotation must have determinant ~1.0 (proper rotation), got "
            f"{determinant}."
        )


def _check_transform_matrix(transform: np.ndarray, name: str = "transform") -> None:
    """Raise ValueError if ``transform`` is not a valid (4, 4) homogeneous transform."""
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}.")
    _check_finite(transform, name)

    last_row_expected = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(transform[3, :], last_row_expected, atol=_HOMOGENEOUS_ROW_ATOL):
        raise ValueError(
            f"{name} last row must be approximately [0, 0, 0, 1], got "
            f"{transform[3, :]}."
        )


def make_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Construct a 4x4 homogeneous transform T_target_source.

    Args:
        rotation: Rotation matrix R_target_source of shape (3, 3).
        translation: Translation vector t_target_source of shape (3,) or (3, 1).

    Returns:
        Homogeneous transform of shape (4, 4) mapping points from the source
        frame to the target frame: p_target = R @ p_source + t.

    Raises:
        ValueError: If shapes are invalid, values are non-finite, or
            ``rotation`` is not an approximately valid rotation matrix.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    if translation.shape == (3, 1):
        translation = translation.reshape(3)
    if translation.shape != (3,):
        raise ValueError(
            f"translation must have shape (3,) or (3, 1), got {translation.shape}."
        )
    _check_finite(translation, "translation")
    _check_rotation_matrix(rotation)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Transform Nx3 points using T_target_source.

    Args:
        points: Points in the source frame, shape (N, 3). N may be 0.
        transform: Homogeneous transform T_target_source, shape (4, 4).

    Returns:
        Points in the target frame, shape (N, 3).

    Raises:
        ValueError: If ``points`` does not have shape (N, 3), if ``transform``
            is not a valid (4, 4) homogeneous transform, or if inputs contain
            non-finite values.
    """
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    _check_finite(points, "points")
    _check_transform_matrix(transform, "transform")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert an SE(3) transform.

    Given T_target_source, returns T_source_target such that
    ``T_source_target @ T_target_source == I``.

    The inverse is computed using the SE(3) structure rather than a general
    matrix inverse: R_inv = R.T and t_inv = -R.T @ t.

    Args:
        transform: Homogeneous transform T_target_source, shape (4, 4).

    Returns:
        Homogeneous transform T_source_target, shape (4, 4).

    Raises:
        ValueError: If ``transform`` is not a valid (4, 4) homogeneous
            transform with a proper rotation block.
    """
    transform = np.asarray(transform, dtype=np.float64)
    _check_transform_matrix(transform, "transform")

    rotation = transform[:3, :3]
    _check_rotation_matrix(rotation)
    translation = transform[:3, 3]

    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation

    return make_transform(rotation_inv, translation_inv)


def compose_transforms(
    transform_target_intermediate: np.ndarray,
    transform_intermediate_source: np.ndarray,
) -> np.ndarray:
    """Compose two transforms to obtain T_target_source.

    Matrix multiplication order: the result is
    ``T_target_intermediate @ T_intermediate_source``, i.e. the first
    argument is applied *after* the second when transforming points
    (right-to-left composition, matching standard matrix-vector convention).

    Args:
        transform_target_intermediate: Transform T_target_intermediate,
            shape (4, 4).
        transform_intermediate_source: Transform T_intermediate_source,
            shape (4, 4).

    Returns:
        Composed transform T_target_source, shape (4, 4).

    Raises:
        ValueError: If either input is not a valid (4, 4) homogeneous
            transform.
    """
    transform_target_intermediate = np.asarray(
        transform_target_intermediate, dtype=np.float64
    )
    transform_intermediate_source = np.asarray(
        transform_intermediate_source, dtype=np.float64
    )

    _check_transform_matrix(
        transform_target_intermediate, "transform_target_intermediate"
    )
    _check_transform_matrix(
        transform_intermediate_source, "transform_intermediate_source"
    )

    transform_target_source = transform_target_intermediate @ transform_intermediate_source
    _check_transform_matrix(transform_target_source, "transform_target_source")
    return transform_target_source
