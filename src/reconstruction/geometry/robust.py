"""Robust fundamental matrix estimation via RANSAC over the normalized eight-point algorithm.

This module builds on :mod:`reconstruction.geometry.epipolar` (reused, not
reimplemented): :func:`~reconstruction.geometry.epipolar.estimate_fundamental_matrix`
provides the minimal-sample (and final refit) solver, and
:func:`~reconstruction.geometry.epipolar.sampson_distances` provides the
per-correspondence error used for inlier classification.
"""

from __future__ import annotations

import numpy as np

from reconstruction.geometry.epipolar import (
    estimate_fundamental_matrix,
    sampson_distances,
)

__all__ = ["estimate_fundamental_matrix_ransac"]

_SAMPLE_SIZE = 8
_MAX_LOCAL_REFINEMENTS = 3
_RATIO_CLIP = 1.0 - 1e-9


def _check_2d_points(points: np.ndarray, name: str) -> None:
    """Raise ValueError if ``points`` is not a finite (N, 2) array."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values (no NaN/Inf).")


def _required_iterations(inlier_ratio: float, sample_size: int, confidence: float) -> int:
    """Adaptive RANSAC iteration count for a given observed inlier ratio.

    ``required = log(1 - confidence) / log(1 - inlier_ratio ** sample_size)``,
    with ``inlier_ratio`` clipped away from 1.0 to keep the logarithm finite
    (avoiding a log(0) division-by-zero warning when the ratio is very high).
    """
    clipped_ratio = min(inlier_ratio, _RATIO_CLIP)
    base = 1.0 - clipped_ratio**sample_size
    needed = np.log(1.0 - confidence) / np.log(base)
    return max(1, int(np.ceil(needed)))


def estimate_fundamental_matrix_ransac(
    points1: np.ndarray,
    points2: np.ndarray,
    threshold_px: float = 1.5,
    max_iterations: int = 5000,
    confidence: float = 0.999,
    seed: int = 42,
    min_inliers: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Robustly estimate F with eight-point minimal sampling (RANSAC).

    Repeatedly draws minimal samples of 8 correspondences (without
    replacement), fits a candidate fundamental matrix with the normalized
    eight-point algorithm, and scores it by the number of correspondences
    whose Sampson distance falls at or below ``threshold_px ** 2``. The
    best-scoring model's inlier set is used to refit F on all inliers, with
    up to 3 local-refinement passes (refit -> reclassify inliers -> refit).

    Units: ``threshold_px`` is a pixel distance; it is compared against
    :func:`~reconstruction.geometry.epipolar.sampson_distances`, which for
    pixel-coordinate input is approximately in squared pixels. The inlier
    rule is therefore ``sampson_distance <= threshold_px ** 2``.

    Args:
        points1: Points in image 1, shape (N, 2), with N >= 8.
        points2: Points in image 2, shape (N, 2), with the same N.
        threshold_px: Inlier Sampson-distance threshold, in pixels (squared
            internally). Must be > 0.
        max_iterations: Maximum number of RANSAC sampling iterations. Must
            be > 0.
        confidence: Target probability of having sampled at least one
            all-inlier minimal set; used for adaptive early stopping. Must
            be in (0, 1).
        seed: Seed for the random sample generator, for reproducibility.
        min_inliers: Minimum number of inliers required for a successful
            estimate. Must be >= 8.

    Returns:
        A tuple ``(fundamental, inlier_mask, statistics)``:
            fundamental: The refined fundamental matrix, shape (3, 3),
                rank 2, canonicalized.
            inlier_mask: Boolean mask, shape (N,), True for inliers under
                the final refined model.
            statistics: Dict with at least ``attempted_iterations``,
                ``valid_models``, ``skipped_degenerate_samples``,
                ``initial_match_count``, ``final_inlier_count``,
                ``inlier_ratio``, ``median_inlier_sampson_distance``,
                ``mean_inlier_sampson_distance``, ``threshold_px``.

    Raises:
        ValueError: If shapes are invalid, N < 8, values are non-finite, or
            ``threshold_px``/``max_iterations``/``confidence``/
            ``min_inliers`` are out of their valid ranges.
        RuntimeError: If fewer than ``min_inliers`` inliers can be found
            (e.g. correspondences dominated by outliers, or degenerate
            geometry such as a near-planar scene or pure rotation).
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    _check_2d_points(points1, "points1")
    _check_2d_points(points2, "points2")
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same number of "
            f"correspondences, got {points1.shape[0]} and {points2.shape[0]}."
        )
    num_points = points1.shape[0]
    if num_points < _SAMPLE_SIZE:
        raise ValueError(
            "estimate_fundamental_matrix_ransac requires at least 8 "
            f"correspondences, got {num_points}."
        )
    if threshold_px <= 0:
        raise ValueError(f"threshold_px must be > 0, got {threshold_px}.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}.")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if min_inliers < _SAMPLE_SIZE:
        raise ValueError(f"min_inliers must be >= 8, got {min_inliers}.")

    rng = np.random.default_rng(seed)
    sampson_threshold_sq = threshold_px**2

    best_inlier_mask: np.ndarray | None = None
    best_inlier_count = -1
    best_median_sampson = np.inf

    attempted_iterations = 0
    valid_models = 0
    skipped_degenerate_samples = 0
    required_iterations = max_iterations

    iteration = 0
    while iteration < required_iterations and iteration < max_iterations:
        attempted_iterations += 1
        sample_indices = rng.choice(num_points, size=_SAMPLE_SIZE, replace=False)

        try:
            candidate_fundamental = estimate_fundamental_matrix(
                points1[sample_indices], points2[sample_indices]
            )
        except ValueError:
            skipped_degenerate_samples += 1
            iteration += 1
            continue

        valid_models += 1
        sampson = sampson_distances(points1, points2, candidate_fundamental)
        inlier_mask = sampson <= sampson_threshold_sq
        inlier_count = int(np.sum(inlier_mask))

        is_better = inlier_count > best_inlier_count
        if not is_better and inlier_count > 0 and inlier_count == best_inlier_count:
            median_sampson = float(np.median(sampson[inlier_mask]))
            is_better = median_sampson < best_median_sampson

        if is_better:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_median_sampson = (
                float(np.median(sampson[inlier_mask])) if inlier_count > 0 else np.inf
            )

        if best_inlier_count > 0:
            required_iterations = min(
                max_iterations,
                _required_iterations(best_inlier_count / num_points, _SAMPLE_SIZE, confidence),
            )

        iteration += 1

    if best_inlier_mask is None or best_inlier_count < min_inliers:
        found = 0 if best_inlier_mask is None else best_inlier_count
        raise RuntimeError(
            f"RANSAC found only {found} inliers (< min_inliers={min_inliers}) "
            f"out of {num_points} correspondences after {attempted_iterations} "
            "iterations. The correspondences may be dominated by outliers, or "
            "the scene/motion may be degenerate (e.g. near-planar scene or "
            "pure rotation with no parallax)."
        )

    inlier_mask = best_inlier_mask
    fundamental = estimate_fundamental_matrix(points1[inlier_mask], points2[inlier_mask])
    for _ in range(_MAX_LOCAL_REFINEMENTS):
        sampson = sampson_distances(points1, points2, fundamental)
        new_inlier_mask = sampson <= sampson_threshold_sq
        if int(np.sum(new_inlier_mask)) < _SAMPLE_SIZE:
            break
        if np.array_equal(new_inlier_mask, inlier_mask):
            break
        inlier_mask = new_inlier_mask
        fundamental = estimate_fundamental_matrix(points1[inlier_mask], points2[inlier_mask])

    final_sampson = sampson_distances(points1, points2, fundamental)
    inlier_mask = final_sampson <= sampson_threshold_sq
    final_inlier_count = int(np.sum(inlier_mask))

    if final_inlier_count < min_inliers:
        raise RuntimeError(
            f"After local refinement, only {final_inlier_count} inliers remain "
            f"(< min_inliers={min_inliers}) out of {num_points} correspondences. "
            "The correspondences may be dominated by outliers, or the "
            "scene/motion may be degenerate."
        )

    inlier_sampson = final_sampson[inlier_mask]
    statistics: dict[str, float | int] = {
        "attempted_iterations": attempted_iterations,
        "valid_models": valid_models,
        "skipped_degenerate_samples": skipped_degenerate_samples,
        "initial_match_count": num_points,
        "final_inlier_count": final_inlier_count,
        "inlier_ratio": final_inlier_count / num_points,
        "median_inlier_sampson_distance": float(np.median(inlier_sampson)),
        "mean_inlier_sampson_distance": float(np.mean(inlier_sampson)),
        "threshold_px": threshold_px,
    }

    return fundamental, inlier_mask, statistics
