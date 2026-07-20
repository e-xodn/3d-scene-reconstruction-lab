"""Two-view epipolar geometry: normalized eight-point algorithm and metrics.

Coordinate and mathematical convention
---------------------------------------
This module follows the repository-wide ``T_target_source`` convention from
:mod:`reconstruction.geometry.transforms`, specialized to a calibrated
two-view setup where camera 1's frame is treated as the world frame:

    T_camera1_world = I
    P1 = K1 [I | 0]

    T_camera2_camera1 = [R_21 | t_21]
    X_camera2 = R_21 @ X_camera1 + t_21
    P2 = K2 [R_21 | t_21]

Homogeneous image coordinates are written in column-vector math notation as
``x = [u, v, 1]^T``. In this module, batches of ``N`` such points are stored
as ``(N, 3)`` row-arrays, so a single point's math-notation column vector
``x_i`` corresponds to row ``i`` of the array. Applying a 3x3 matrix ``M`` to
every point (``M @ x_i`` in math notation) is therefore implemented as
``points_h @ M.T`` on the whole batch (each output row equals
``(M @ x_i)^T == x_i^T @ M.T``).

Essential and fundamental matrices:

    E = [t_21]_x R_21
    F = K2^{-T} E K1^{-1}

Epipolar constraint and lines (math notation, column vectors):

    x2^T F x1 = 0
    l2 = F x1        (epipolar line in image 2 for a point in image 1)
    l1 = F^T x2       (epipolar line in image 1 for a point in image 2)

A line is represented as ``l = [a, b, c]``, with the point-on-line equation
``a*u + b*v + c = 0``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_homogeneous",
    "from_homogeneous",
    "skew_symmetric",
    "normalize_points_2d",
    "enforce_rank2",
    "canonicalize_fundamental_matrix",
    "essential_from_pose",
    "fundamental_from_pose",
    "estimate_fundamental_matrix",
    "epipolar_lines_in_image2",
    "epipolar_lines_in_image1",
    "algebraic_epipolar_residuals",
    "sampson_distances",
    "point_to_epipolar_line_distances",
]

_ROTATION_ORTHOGONALITY_ATOL = 1e-6
_ROTATION_DETERMINANT_ATOL = 1e-6
_MIN_TRANSLATION_NORM = 1e-12
_LINE_DEGENERACY_EPSILON = 1e-12


def _check_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if ``array`` contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _check_2d_points(points: np.ndarray, name: str) -> None:
    """Raise ValueError if ``points`` is not a finite (N, 2) array."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {points.shape}.")
    _check_finite(points, name)


def _check_3x3_matrix(matrix: np.ndarray, name: str) -> None:
    """Raise ValueError if ``matrix`` is not a finite (3, 3) array."""
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    _check_finite(matrix, name)


def _check_rotation_matrix(rotation: np.ndarray, name: str) -> None:
    """Raise ValueError if ``rotation`` is not an approximately valid SO(3) matrix.

    This is a minimal, module-local validation helper. It intentionally does
    not import the private rotation validator from
    :mod:`reconstruction.geometry.transforms`, since that helper is not part
    of that module's public API.
    """
    _check_3x3_matrix(rotation, name)

    should_be_identity = rotation.T @ rotation
    if not np.allclose(
        should_be_identity, np.eye(3), atol=_ROTATION_ORTHOGONALITY_ATOL
    ):
        raise ValueError(
            f"{name} is not orthogonal: R.T @ R is not approximately the "
            "identity matrix."
        )

    determinant = np.linalg.det(rotation)
    if not np.isclose(determinant, 1.0, atol=_ROTATION_DETERMINANT_ATOL):
        raise ValueError(
            f"{name} must have determinant ~1.0 (proper rotation), got "
            f"{determinant}."
        )


def _check_intrinsic_matrix(intrinsic: np.ndarray, name: str) -> None:
    """Raise ValueError if ``intrinsic`` is not a valid (3, 3) intrinsic matrix."""
    _check_3x3_matrix(intrinsic, name)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{name} must have fx > 0 and fy > 0, got fx={fx}, fy={fy}.")


def _check_not_collinear(points: np.ndarray, image_name: str) -> None:
    """Raise ValueError if ``points`` are numerically coincident or collinear."""
    centered = points - points.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)
    if rank == 0:
        raise ValueError(
            "Fundamental matrix estimation is degenerate because all points "
            f"in {image_name} are coincident."
        )
    if rank < 2:
        raise ValueError(
            "Fundamental matrix estimation is degenerate because points in "
            f"{image_name} are collinear."
        )


def _normalize_line_coefficients(lines: np.ndarray) -> np.ndarray:
    """Scale each line [a, b, c] so that sqrt(a^2 + b^2) == 1."""
    norm = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    if np.any(norm <= _LINE_DEGENERACY_EPSILON):
        raise ValueError(
            "Degenerate epipolar line: coefficients a and b are both ~0 for "
            "at least one correspondence."
        )
    return lines / norm[:, None]


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert Euclidean 2D points to homogeneous coordinates.

    Args:
        points: Euclidean 2D points, shape (N, 2). N may be 0.

    Returns:
        Homogeneous points, shape (N, 3), with a column of ones appended.

    Raises:
        ValueError: If ``points`` does not have shape (N, 2) or contains
            non-finite values.
    """
    points = np.asarray(points, dtype=np.float64)
    _check_2d_points(points, "points")

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.concatenate([points, ones], axis=1)


def from_homogeneous(points_h: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Convert homogeneous 2D points to Euclidean coordinates.

    Args:
        points_h: Homogeneous 2D points, shape (N, 3).
        epsilon: The homogeneous scale (last coordinate) must exceed this
            magnitude for the point to be well-defined.

    Returns:
        Euclidean points, shape (N, 2).

    Raises:
        ValueError: If ``points_h`` does not have shape (N, 3), contains
            non-finite values, or has a homogeneous scale with
            ``abs(w) <= epsilon`` for any point.
    """
    points_h = np.asarray(points_h, dtype=np.float64)
    if points_h.ndim != 2 or points_h.shape[1] != 3:
        raise ValueError(f"points_h must have shape (N, 3), got {points_h.shape}.")
    _check_finite(points_h, "points_h")

    scale = points_h[:, 2]
    if np.any(np.abs(scale) <= epsilon):
        raise ValueError(
            f"points_h has a homogeneous scale with abs(w) <= {epsilon} for "
            "at least one point."
        )
    return points_h[:, :2] / scale[:, None]


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]_x such that [v]_x @ w == cross(v, w).

    Args:
        vector: A 3-vector, shape (3,) or (3, 1).

    Returns:
        The skew-symmetric matrix, shape (3, 3)::

            [[ 0,  -vz,  vy],
             [ vz,   0, -vx],
             [-vy,  vx,   0]]

    Raises:
        ValueError: If ``vector`` does not have shape (3,) or (3, 1), or
            contains non-finite values.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape == (3, 1):
        vector = vector.reshape(3)
    if vector.shape != (3,):
        raise ValueError(f"vector must have shape (3,) or (3, 1), got {vector.shape}.")
    _check_finite(vector, "vector")

    vx, vy, vz = vector
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=np.float64,
    )


def normalize_points_2d(
    points: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize 2D points using isotropic Hartley normalization.

    Translates points so their centroid is at the origin, then scales them
    so the mean distance from the origin is ``sqrt(2)``.

    Args:
        points: Euclidean 2D points, shape (N, 2), with N >= 2.
        epsilon: The mean centroid distance must exceed this value; points
            that are all (numerically) coincident are rejected.

    Returns:
        A tuple ``(normalized_points, transform)``:
            normalized_points: shape (N, 2).
            transform: The 3x3 homogeneous transform T such that, in math
                notation, ``x_normalized = T @ x`` for homogeneous ``x``.

    Raises:
        ValueError: If ``points`` does not have shape (N, 2), has fewer than
            2 points, contains non-finite values, or the mean distance from
            the centroid is <= epsilon (points are coincident).
    """
    points = np.asarray(points, dtype=np.float64)
    _check_2d_points(points, "points")
    if points.shape[0] < 2:
        raise ValueError(
            f"normalize_points_2d requires at least 2 points, got "
            f"{points.shape[0]}."
        )

    centroid = points.mean(axis=0)
    centered = points - centroid
    distances = np.linalg.norm(centered, axis=1)
    mean_distance = distances.mean()

    if mean_distance <= epsilon:
        raise ValueError(
            "Cannot normalize points: mean distance from the centroid is "
            f"<= {epsilon} (points are coincident)."
        )

    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    points_h = to_homogeneous(points)
    normalized_h = points_h @ transform.T
    normalized_points = from_homogeneous(normalized_h, epsilon=epsilon)
    return normalized_points, transform


def enforce_rank2(matrix: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto the closest rank-2 matrix (Frobenius norm) via SVD.

    Args:
        matrix: A 3x3 matrix, typically an estimated fundamental matrix.

    Returns:
        The closest rank-2 matrix to ``matrix``, shape (3, 3).

    Raises:
        ValueError: If ``matrix`` does not have shape (3, 3) or contains
            non-finite values.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    _check_3x3_matrix(matrix, "matrix")

    u, singular_values, vt = np.linalg.svd(matrix)
    singular_values = singular_values.copy()
    singular_values[-1] = 0.0
    return u @ np.diag(singular_values) @ vt


def canonicalize_fundamental_matrix(
    fundamental: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Fix the arbitrary scale and sign of a fundamental matrix for comparison.

    Fundamental matrices are only defined up to a nonzero scale factor
    (including sign), since ``x2^T F x1 == 0`` is scale-invariant. This
    normalizes ``fundamental`` to unit Frobenius norm and a fixed sign
    convention (the largest-magnitude entry is positive) so two fundamental
    matrices that represent the same epipolar geometry compare equal.

    Args:
        fundamental: A 3x3 matrix, typically a fundamental matrix.
        epsilon: The Frobenius norm of ``fundamental`` must exceed this
            value.

    Returns:
        The canonicalized matrix, shape (3, 3), with Frobenius norm ~1.

    Raises:
        ValueError: If ``fundamental`` does not have shape (3, 3), contains
            non-finite values, or has Frobenius norm <= epsilon.
    """
    fundamental = np.asarray(fundamental, dtype=np.float64)
    _check_3x3_matrix(fundamental, "fundamental")

    frobenius_norm = np.linalg.norm(fundamental)
    if frobenius_norm <= epsilon:
        raise ValueError(
            f"Cannot canonicalize fundamental matrix: Frobenius norm <= "
            f"{epsilon}."
        )

    normalized = fundamental / frobenius_norm
    max_index = np.unravel_index(np.argmax(np.abs(normalized)), normalized.shape)
    if normalized[max_index] < 0:
        normalized = -normalized
    return normalized


def essential_from_pose(
    rotation_camera2_camera1: np.ndarray,
    translation_camera2_camera1: np.ndarray,
) -> np.ndarray:
    """Compute the essential matrix E = [t_21]_x R_21 from a relative pose.

    Args:
        rotation_camera2_camera1: Rotation R_21, shape (3, 3).
        translation_camera2_camera1: Translation t_21, shape (3,) or (3, 1).

    Returns:
        The essential matrix E, shape (3, 3). E has rank 2 whenever
        ``translation_camera2_camera1`` is nonzero, since [t]_x has rank 2
        and multiplying by the full-rank rotation preserves rank.

    Raises:
        ValueError: If shapes are invalid, values are non-finite,
            ``rotation_camera2_camera1`` is not a valid rotation matrix, or
            ``translation_camera2_camera1`` has norm ~0 (undefined for a
            zero-baseline / pure-rotation camera pair).
    """
    rotation = np.asarray(rotation_camera2_camera1, dtype=np.float64)
    translation = np.asarray(translation_camera2_camera1, dtype=np.float64)
    if translation.shape == (3, 1):
        translation = translation.reshape(3)
    if translation.shape != (3,):
        raise ValueError(
            "translation_camera2_camera1 must have shape (3,) or (3, 1), "
            f"got {translation.shape}."
        )
    _check_finite(translation, "translation_camera2_camera1")
    _check_rotation_matrix(rotation, "rotation_camera2_camera1")

    translation_norm = np.linalg.norm(translation)
    if translation_norm <= _MIN_TRANSLATION_NORM:
        raise ValueError(
            "translation_camera2_camera1 has norm ~0: the essential matrix "
            "is undefined for a zero-baseline (pure rotation) camera pair."
        )

    return skew_symmetric(translation) @ rotation


def fundamental_from_pose(
    intrinsic1: np.ndarray,
    intrinsic2: np.ndarray,
    rotation_camera2_camera1: np.ndarray,
    translation_camera2_camera1: np.ndarray,
) -> np.ndarray:
    """Compute F = K2^{-T} [t_21]_x R_21 K1^{-1} from intrinsics and relative pose.

    Args:
        intrinsic1: Camera 1 intrinsic matrix K1, shape (3, 3).
        intrinsic2: Camera 2 intrinsic matrix K2, shape (3, 3).
        rotation_camera2_camera1: Rotation R_21, shape (3, 3).
        translation_camera2_camera1: Translation t_21, shape (3,) or (3, 1).

    Returns:
        The ground-truth fundamental matrix, shape (3, 3), with the rank-2
        constraint enforced and canonicalized to unit Frobenius norm with a
        fixed sign.

    Raises:
        ValueError: If any intrinsic matrix is invalid (wrong shape,
            non-finite, non-positive fx/fy, or singular), or if the pose is
            invalid (see :func:`essential_from_pose`).
    """
    intrinsic1 = np.asarray(intrinsic1, dtype=np.float64)
    intrinsic2 = np.asarray(intrinsic2, dtype=np.float64)
    _check_intrinsic_matrix(intrinsic1, "intrinsic1")
    _check_intrinsic_matrix(intrinsic2, "intrinsic2")

    essential = essential_from_pose(
        rotation_camera2_camera1, translation_camera2_camera1
    )

    try:
        intrinsic1_inv = np.linalg.inv(intrinsic1)
        intrinsic2_inv = np.linalg.inv(intrinsic2)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "intrinsic1 or intrinsic2 is singular and cannot be inverted."
        ) from exc

    fundamental = intrinsic2_inv.T @ essential @ intrinsic1_inv
    fundamental = enforce_rank2(fundamental)
    return canonicalize_fundamental_matrix(fundamental)


def estimate_fundamental_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Estimate the fundamental matrix using the normalized eight-point algorithm.

    Convention: the estimated F satisfies ``x2^T F x1 == 0`` for each
    correspondence ``(x1, x2)``.

    Procedure: validate correspondences, independently Hartley-normalize
    ``points1`` and ``points2``, build the linear design matrix in the
    normalized coordinates, solve via SVD, enforce the rank-2 constraint,
    denormalize, re-enforce rank-2, and canonicalize the result.

    Args:
        points1: Points in image 1, shape (N, 2), with N >= 8.
        points2: Points in image 2, shape (N, 2), with the same N.

    Returns:
        The estimated fundamental matrix, shape (3, 3), canonicalized to
        unit Frobenius norm with a fixed sign.

    Raises:
        ValueError: If shapes are invalid, N < 8, points1/points2 lengths
            differ, values are non-finite, points in either image are
            coincident or collinear, or the eight-point design matrix has
            insufficient rank.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)

    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(f"points1 must have shape (N, 2), got {points1.shape}.")
    if points2.ndim != 2 or points2.shape[1] != 2:
        raise ValueError(f"points2 must have shape (N, 2), got {points2.shape}.")
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same number of "
            f"correspondences, got {points1.shape[0]} and {points2.shape[0]}."
        )
    num_points = points1.shape[0]
    if num_points < 8:
        raise ValueError(
            "estimate_fundamental_matrix requires at least 8 "
            f"correspondences, got {num_points}."
        )
    _check_finite(points1, "points1")
    _check_finite(points2, "points2")

    _check_not_collinear(points1, "image 1")
    _check_not_collinear(points2, "image 2")

    points1_normalized, transform1 = normalize_points_2d(points1)
    points2_normalized, transform2 = normalize_points_2d(points2)

    u1, v1 = points1_normalized[:, 0], points1_normalized[:, 1]
    u2, v2 = points2_normalized[:, 0], points2_normalized[:, 1]
    ones = np.ones(num_points, dtype=np.float64)
    design_matrix = np.stack(
        [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, ones], axis=1
    )

    design_rank = np.linalg.matrix_rank(design_matrix)
    if design_rank < 8:
        raise ValueError(
            "Fundamental matrix estimation is degenerate: the eight-point "
            f"design matrix has rank {design_rank} (< 8), so the solution "
            "is not unique."
        )

    _, _, vt = np.linalg.svd(design_matrix)
    fundamental_normalized = vt[-1].reshape(3, 3)
    fundamental_normalized = enforce_rank2(fundamental_normalized)

    fundamental = transform2.T @ fundamental_normalized @ transform1
    fundamental = enforce_rank2(fundamental)

    return canonicalize_fundamental_matrix(fundamental)


def epipolar_lines_in_image2(
    points1: np.ndarray,
    fundamental: np.ndarray,
) -> np.ndarray:
    """Compute epipolar lines l2 = F x1 in image 2 for points in image 1.

    Args:
        points1: Points in image 1, shape (N, 2).
        fundamental: Fundamental matrix, shape (3, 3).

    Returns:
        Lines [a, b, c] with ``a*u + b*v + c = 0``, normalized so
        ``sqrt(a^2 + b^2) == 1``, shape (N, 3).

    Raises:
        ValueError: If shapes are invalid, values are non-finite, or a
            resulting line has both a and b ~0.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    fundamental = np.asarray(fundamental, dtype=np.float64)
    _check_2d_points(points1, "points1")
    _check_3x3_matrix(fundamental, "fundamental")

    points1_h = to_homogeneous(points1)
    lines = points1_h @ fundamental.T
    return _normalize_line_coefficients(lines)


def epipolar_lines_in_image1(
    points2: np.ndarray,
    fundamental: np.ndarray,
) -> np.ndarray:
    """Compute epipolar lines l1 = F^T x2 in image 1 for points in image 2.

    Args:
        points2: Points in image 2, shape (N, 2).
        fundamental: Fundamental matrix, shape (3, 3).

    Returns:
        Lines [a, b, c] with ``a*u + b*v + c = 0``, normalized so
        ``sqrt(a^2 + b^2) == 1``, shape (N, 3).

    Raises:
        ValueError: If shapes are invalid, values are non-finite, or a
            resulting line has both a and b ~0.
    """
    points2 = np.asarray(points2, dtype=np.float64)
    fundamental = np.asarray(fundamental, dtype=np.float64)
    _check_2d_points(points2, "points2")
    _check_3x3_matrix(fundamental, "fundamental")

    points2_h = to_homogeneous(points2)
    lines = points2_h @ fundamental
    return _normalize_line_coefficients(lines)


def algebraic_epipolar_residuals(
    points1: np.ndarray,
    points2: np.ndarray,
    fundamental: np.ndarray,
) -> np.ndarray:
    """Compute the signed algebraic residual x2^T F x1 for each correspondence.

    This residual is scale-dependent (it changes if ``fundamental`` is
    rescaled) and does not directly correspond to a geometric pixel error;
    use :func:`sampson_distances` for a geometrically meaningful metric.

    Args:
        points1: Points in image 1, shape (N, 2).
        points2: Points in image 2, shape (N, 2), with the same N.
        fundamental: Fundamental matrix, shape (3, 3).

    Returns:
        Signed residuals, shape (N,).

    Raises:
        ValueError: If shapes are invalid, lengths differ, or values are
            non-finite.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    fundamental = np.asarray(fundamental, dtype=np.float64)
    _check_2d_points(points1, "points1")
    _check_2d_points(points2, "points2")
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same number of "
            f"correspondences, got {points1.shape[0]} and {points2.shape[0]}."
        )
    _check_3x3_matrix(fundamental, "fundamental")

    points1_h = to_homogeneous(points1)
    points2_h = to_homogeneous(points2)
    f_x1 = points1_h @ fundamental.T
    return np.sum(points2_h * f_x1, axis=1)


def sampson_distances(
    points1: np.ndarray,
    points2: np.ndarray,
    fundamental: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute the first-order Sampson distance for each correspondence.

    ``sampson = (x2^T F x1)^2 / (Fx1[0]^2 + Fx1[1]^2 + F^Tx2[0]^2 + F^Tx2[1]^2)``.

    When ``points1`` and ``points2`` are in pixel coordinates, the result is
    approximately in squared pixels: a first-order approximation of the
    squared geometric (reprojection-like) distance implied by the epipolar
    constraint.

    Args:
        points1: Points in image 1, shape (N, 2).
        points2: Points in image 2, shape (N, 2), with the same N.
        fundamental: Fundamental matrix, shape (3, 3).
        epsilon: The denominator must exceed this value for every
            correspondence.

    Returns:
        Non-negative Sampson distances, shape (N,).

    Raises:
        ValueError: If shapes are invalid, lengths differ, values are
            non-finite, or the denominator is <= epsilon for any
            correspondence (degenerate epipolar geometry).
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    fundamental = np.asarray(fundamental, dtype=np.float64)
    _check_2d_points(points1, "points1")
    _check_2d_points(points2, "points2")
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same number of "
            f"correspondences, got {points1.shape[0]} and {points2.shape[0]}."
        )
    _check_3x3_matrix(fundamental, "fundamental")

    points1_h = to_homogeneous(points1)
    points2_h = to_homogeneous(points2)
    f_x1 = points1_h @ fundamental.T
    ft_x2 = points2_h @ fundamental

    numerator = np.sum(points2_h * f_x1, axis=1) ** 2
    denominator = f_x1[:, 0] ** 2 + f_x1[:, 1] ** 2 + ft_x2[:, 0] ** 2 + ft_x2[:, 1] ** 2

    if np.any(denominator <= epsilon):
        raise ValueError(
            f"Sampson distance denominator is <= {epsilon} for at least one "
            "correspondence (degenerate epipolar geometry)."
        )

    return numerator / denominator


def point_to_epipolar_line_distances(
    points: np.ndarray,
    lines: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute the Euclidean distance from each point to its epipolar line.

    ``distance = abs(a*u + b*v + c) / sqrt(a^2 + b^2)``.

    Args:
        points: Points, shape (N, 2).
        lines: Lines [a, b, c], shape (N, 3), with the same N.
        epsilon: ``sqrt(a^2 + b^2)`` must exceed this value for every line.

    Returns:
        Non-negative distances, shape (N,). In the same units as ``points``
        (pixels, if ``points`` are pixel coordinates).

    Raises:
        ValueError: If shapes are invalid, lengths differ, values are
            non-finite, or a line has ``sqrt(a^2 + b^2) <= epsilon``.
    """
    points = np.asarray(points, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.float64)
    _check_2d_points(points, "points")
    if lines.ndim != 2 or lines.shape[1] != 3:
        raise ValueError(f"lines must have shape (N, 3), got {lines.shape}.")
    _check_finite(lines, "lines")
    if points.shape[0] != lines.shape[0]:
        raise ValueError(
            f"points and lines must have the same N, got {points.shape[0]} "
            f"and {lines.shape[0]}."
        )

    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    denominator = np.sqrt(a**2 + b**2)
    if np.any(denominator <= epsilon):
        raise ValueError(
            f"Degenerate epipolar line: sqrt(a^2 + b^2) <= {epsilon} for at "
            "least one line."
        )

    numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c)
    return numerator / denominator
