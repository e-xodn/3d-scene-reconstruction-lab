"""Essential matrix computation, decomposition, and cheirality-based pose selection.

Convention (matches :mod:`reconstruction.geometry.transforms` and
:mod:`reconstruction.geometry.epipolar`): camera 1's frame is the
reconstruction reference frame. ``T_camera2_camera1 = [R_21 | t_21]`` maps
points from camera 1 to camera 2: ``X_camera2 = R_21 @ X_camera1 + t_21``.

Essential matrix relation: ``E = K2^T F K1``, or from a known pose,
``E = [t_21]_x R_21``.

Scale ambiguity: the essential matrix determines the translation only up
to an unknown positive scale (and, before cheirality resolves it, sign).
:func:`decompose_essential_matrix` always returns a *unit-norm* translation
direction. Consequently, the triangulated 3D points and the recovered
translation have no absolute metric scale — the reconstruction is only
correct up to a single unknown global scale factor, standard for
two-view (uncalibrated-baseline) reconstruction.
"""

from __future__ import annotations

import numpy as np

from reconstruction.evaluation.reprojection import two_view_reprojection_errors
from reconstruction.geometry.triangulation import (
    camera_depths,
    projection_matrix,
    triangulate_points_dlt,
)

__all__ = [
    "essential_from_fundamental",
    "enforce_essential_constraints",
    "decompose_essential_matrix",
    "select_pose_by_cheirality",
]

_W_MATRIX = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_MIN_NORM_EPSILON = 1e-12


def _check_finite(array: np.ndarray, name: str) -> None:
    """Raise ValueError if ``array`` contains NaN or Inf values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _validate_intrinsic_matrix(intrinsic: np.ndarray, name: str) -> None:
    """Raise ValueError unless ``intrinsic`` is a valid, invertible (3, 3) camera matrix.

    Checks: shape (3, 3), finite values, fx > 0, fy > 0, last row
    approximately [0, 0, 1], and invertibility.
    """
    if intrinsic.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {intrinsic.shape}.")
    _check_finite(intrinsic, name)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{name} must have fx > 0 and fy > 0, got fx={fx}, fy={fy}.")
    if not np.allclose(intrinsic[2, :], np.array([0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(
            f"{name} last row must be approximately [0, 0, 1], got {intrinsic[2, :]}."
        )
    if abs(np.linalg.det(intrinsic)) <= _MIN_NORM_EPSILON:
        raise ValueError(f"{name} is singular and cannot be inverted.")


def _fix_orthogonal_sign(matrix: np.ndarray, flip_column: bool) -> np.ndarray:
    """Return a copy of an orthogonal ``matrix`` with determinant forced to +1."""
    if np.linalg.det(matrix) >= 0:
        return matrix.copy()
    corrected = matrix.copy()
    if flip_column:
        corrected[:, -1] *= -1
    else:
        corrected[-1, :] *= -1
    return corrected


def enforce_essential_constraints(essential: np.ndarray) -> np.ndarray:
    """Project a matrix onto the essential-matrix manifold.

    Computes ``U, S, Vt = svd(essential)``, corrects the sign of ``U``/``Vt``
    if needed so later rotation candidates have determinant +1, averages
    the two largest singular values (``s = (S[0] + S[1]) / 2``), and
    reconstructs ``E = U @ diag([s, s, 0]) @ Vt``.

    Args:
        essential: A 3x3 matrix (typically ``K2^T F K1``) to project.

    Returns:
        The corrected essential matrix, shape (3, 3): first two singular
        values equal, third singular value 0 (rank 2).

    Raises:
        ValueError: If ``essential`` does not have shape (3, 3), contains
            non-finite values, or is ~zero (Frobenius norm <= 1e-12).
    """
    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape != (3, 3):
        raise ValueError(f"essential must have shape (3, 3), got {essential.shape}.")
    _check_finite(essential, "essential")
    if np.linalg.norm(essential) <= _MIN_NORM_EPSILON:
        raise ValueError(
            "essential matrix is ~zero; cannot project it onto the essential "
            "manifold."
        )

    u, singular_values, vt = np.linalg.svd(essential)
    u = _fix_orthogonal_sign(u, flip_column=True)
    vt = _fix_orthogonal_sign(vt, flip_column=False)

    corrected_scale = (singular_values[0] + singular_values[1]) / 2.0
    return u @ np.diag([corrected_scale, corrected_scale, 0.0]) @ vt


def essential_from_fundamental(
    fundamental: np.ndarray,
    intrinsic1: np.ndarray,
    intrinsic2: np.ndarray,
) -> np.ndarray:
    """Compute the essential matrix E = K2^T F K1 from a fundamental matrix.

    Args:
        fundamental: Fundamental matrix, shape (3, 3).
        intrinsic1: Camera 1 intrinsic matrix K1, shape (3, 3).
        intrinsic2: Camera 2 intrinsic matrix K2, shape (3, 3).

    Returns:
        The essential matrix, shape (3, 3): rank 2, with the essential
        singular-value constraint enforced (see
        :func:`enforce_essential_constraints`) and rescaled to Frobenius
        norm ``sqrt(2)`` (equivalently, singular values [1, 1, 0]) as a
        canonical, unit-translation-consistent scale.

    Raises:
        ValueError: If ``fundamental`` is not a finite (3, 3) matrix, if
            either intrinsic matrix is invalid (see
            :func:`_validate_intrinsic_matrix`), or the result is ~zero.
    """
    fundamental = np.asarray(fundamental, dtype=np.float64)
    intrinsic1 = np.asarray(intrinsic1, dtype=np.float64)
    intrinsic2 = np.asarray(intrinsic2, dtype=np.float64)

    if fundamental.shape != (3, 3):
        raise ValueError(f"fundamental must have shape (3, 3), got {fundamental.shape}.")
    _check_finite(fundamental, "fundamental")
    _validate_intrinsic_matrix(intrinsic1, "intrinsic1")
    _validate_intrinsic_matrix(intrinsic2, "intrinsic2")

    essential_raw = intrinsic2.T @ fundamental @ intrinsic1
    essential_constrained = enforce_essential_constraints(essential_raw)

    frobenius_norm = np.linalg.norm(essential_constrained)
    if frobenius_norm <= _MIN_NORM_EPSILON:
        raise ValueError("Cannot normalize essential matrix: Frobenius norm is ~0.")
    return essential_constrained * (np.sqrt(2.0) / frobenius_norm)


def decompose_essential_matrix(essential: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose an essential matrix into the four candidate relative poses.

    Given ``U, _, Vt = svd(essential)`` (with signs corrected so
    ``det(U) == det(Vt) == 1``) and
    ``W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]``, the two candidate rotations
    are ``R1 = U @ W @ Vt`` and ``R2 = U @ W.T @ Vt``, and the translation
    direction is ``t = U[:, 2]`` (normalized to unit norm). Since the true
    scale and sign of ``t`` cannot be recovered from ``E`` alone, all four
    combinations ``(R1, +t), (R1, -t), (R2, +t), (R2, -t)`` are returned;
    exactly one is physically valid for a given scene (see
    :func:`select_pose_by_cheirality`).

    Args:
        essential: Essential matrix, shape (3, 3).

    Returns:
        A list of exactly 4 ``(rotation, translation)`` tuples:
        ``rotation`` has shape (3, 3) with ``det(rotation) ~= 1`` and
        ``rotation.T @ rotation ~= I``; ``translation`` has shape (3,) with
        unit norm. Each candidate owns independent arrays (no shared
        references), so mutating one candidate cannot affect another.

    Raises:
        ValueError: If ``essential`` does not have shape (3, 3), contains
            non-finite values, or the recovered translation direction has
            norm ~0.
    """
    essential = np.asarray(essential, dtype=np.float64)
    if essential.shape != (3, 3):
        raise ValueError(f"essential must have shape (3, 3), got {essential.shape}.")
    _check_finite(essential, "essential")

    u, _, vt = np.linalg.svd(essential)
    u = _fix_orthogonal_sign(u, flip_column=True)
    vt = _fix_orthogonal_sign(vt, flip_column=False)

    rotation1 = u @ _W_MATRIX @ vt
    rotation2 = u @ _W_MATRIX.T @ vt
    if np.linalg.det(rotation1) < 0:
        rotation1 = -rotation1
    if np.linalg.det(rotation2) < 0:
        rotation2 = -rotation2

    translation = u[:, 2].copy()
    translation_norm = np.linalg.norm(translation)
    if translation_norm <= _MIN_NORM_EPSILON:
        raise ValueError(
            "decompose_essential_matrix: recovered translation direction has "
            "norm ~0; this should not occur for a valid rank-2 essential "
            "matrix produced by enforce_essential_constraints."
        )
    translation = translation / translation_norm

    return [
        (rotation1.copy(), translation.copy()),
        (rotation1.copy(), -translation.copy()),
        (rotation2.copy(), translation.copy()),
        (rotation2.copy(), -translation.copy()),
    ]


def select_pose_by_cheirality(
    pose_candidates: list[tuple[np.ndarray, np.ndarray]],
    points1: np.ndarray,
    points2: np.ndarray,
    intrinsic1: np.ndarray,
    intrinsic2: np.ndarray,
    min_depth: float = 1e-8,
    min_positive_ratio: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Select the pose candidate with the most points in front of both cameras.

    Cheirality: only one of the four algebraically valid
    ``(R_21, t_21)`` decompositions of an essential matrix corresponds to a
    scene where the triangulated points are actually in front of (positive
    depth in) *both* cameras — the other three place most points behind one
    or both cameras, which is not physically realizable. Triangulating with
    each candidate and counting positive-depth points therefore
    disambiguates the correct pose.

    For each candidate ``(R, t)``: builds ``P1 = K1 [I | 0]``,
    ``P2 = K2 [R | t]``, triangulates all correspondences
    (:func:`~reconstruction.geometry.triangulation.triangulate_points_dlt`),
    and computes camera-1 depth ``z1`` and camera-2 depth ``z2`` (via
    :func:`~reconstruction.geometry.triangulation.camera_depths`). A point
    is "positive" if it triangulated to a finite value and
    ``z1 > min_depth`` and ``z2 > min_depth``. The candidate with the most
    positive points wins; ties are broken by the smaller median combined
    reprojection error.

    Args:
        pose_candidates: List of ``(rotation, translation)`` candidates,
            e.g. from :func:`decompose_essential_matrix`.
        points1: Matched pixel coordinates in image 1, shape (N, 2).
        points2: Matched pixel coordinates in image 2, shape (N, 2).
        intrinsic1: Camera 1 intrinsic matrix, shape (3, 3).
        intrinsic2: Camera 2 intrinsic matrix, shape (3, 3).
        min_depth: Minimum positive depth (see
            :func:`~reconstruction.cameras.pinhole.project_points` for the
            analogous M1 convention).
        min_positive_ratio: The best candidate's positive-depth ratio
            (positive_count / N) must reach at least this value.

    Returns:
        A tuple:
            rotation_camera2_camera1: shape (3, 3), the selected R_21.
            translation_camera2_camera1: shape (3,), the selected unit-norm
                t_21.
            triangulated_points_camera1: shape (N, 3), triangulated with
                the selected pose (NaN rows for degenerate rays).
            positive_depth_mask: shape (N,), boolean, True for points with
                positive depth in both cameras under the selected pose.
            positive_depth_counts: list of length ``len(pose_candidates)``,
                the positive-depth count for each candidate in order.

    Raises:
        ValueError: If ``pose_candidates`` is empty, ``points1``/``points2``
            shapes are invalid or mismatched, or either intrinsic matrix is
            invalid.
        RuntimeError: If every candidate's positive-depth ratio is below
            ``min_positive_ratio`` (e.g. the correspondences are dominated
            by outliers, or the motion is a near-pure rotation with no
            usable parallax).
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    intrinsic1 = np.asarray(intrinsic1, dtype=np.float64)
    intrinsic2 = np.asarray(intrinsic2, dtype=np.float64)

    if len(pose_candidates) == 0:
        raise ValueError("pose_candidates must be non-empty.")
    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(f"points1 must have shape (N, 2), got {points1.shape}.")
    if points2.shape != points1.shape:
        raise ValueError(
            f"points1 and points2 must have the same shape, got {points1.shape} "
            f"and {points2.shape}."
        )
    _validate_intrinsic_matrix(intrinsic1, "intrinsic1")
    _validate_intrinsic_matrix(intrinsic2, "intrinsic2")

    num_points = points1.shape[0]
    projection1 = projection_matrix(intrinsic1, np.eye(3), np.zeros(3))

    best_index = -1
    best_positive_count = -1
    best_median_error = np.inf
    best_points_3d: np.ndarray | None = None
    best_positive_mask: np.ndarray | None = None
    positive_depth_counts: list[int] = []

    for index, (rotation, translation) in enumerate(pose_candidates):
        projection2 = projection_matrix(intrinsic2, rotation, translation)
        points_3d = triangulate_points_dlt(points1, points2, projection1, projection2)
        depth1, depth2 = camera_depths(points_3d, rotation, translation)

        positive_mask = (
            np.all(np.isfinite(points_3d), axis=1)
            & (depth1 > min_depth)
            & (depth2 > min_depth)
        )
        positive_count = int(np.sum(positive_mask))
        positive_depth_counts.append(positive_count)

        _, _, combined_error = two_view_reprojection_errors(
            points_3d, points1, points2, projection1, projection2
        )
        finite_combined = combined_error[np.isfinite(combined_error)]
        median_error = float(np.median(finite_combined)) if finite_combined.size else np.inf

        is_better = positive_count > best_positive_count or (
            positive_count == best_positive_count and median_error < best_median_error
        )
        if is_better:
            best_index = index
            best_positive_count = positive_count
            best_median_error = median_error
            best_points_3d = points_3d
            best_positive_mask = positive_mask

    best_ratio = (best_positive_count / num_points) if num_points > 0 else 0.0
    if best_index < 0 or best_ratio < min_positive_ratio:
        raise RuntimeError(
            f"No pose candidate reached the minimum positive-depth ratio "
            f"({min_positive_ratio}): the best candidate had "
            f"{best_positive_count}/{num_points} = {best_ratio:.3f} "
            "positive-depth points. This usually indicates a near-pure "
            "rotation (little/no parallax), outlier-dominated "
            "correspondences, or an incorrect fundamental/essential matrix."
        )

    rotation_selected, translation_selected = pose_candidates[best_index]
    return (
        rotation_selected,
        translation_selected,
        best_points_3d,
        best_positive_mask,
        positive_depth_counts,
    )
