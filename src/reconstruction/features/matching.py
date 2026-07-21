"""Grayscale image loading, SIFT/ORB feature detection, and descriptor matching.

OpenCV (``opencv-python-headless``) is used only for image I/O, feature
detection/description, and k-nearest-neighbor descriptor matching in this
module. All downstream geometry (RANSAC, essential-matrix decomposition,
triangulation) is implemented independently with NumPy in
:mod:`reconstruction.geometry.robust`, :mod:`reconstruction.geometry.pose`,
and :mod:`reconstruction.geometry.triangulation`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "load_grayscale_image",
    "detect_and_describe",
    "ratio_test_matches",
    "mutual_ratio_matches",
    "matched_keypoint_coordinates",
]

_SUPPORTED_METHODS = ("sift", "orb")


def _check_method(method: str) -> str:
    """Normalize and validate a feature ``method`` name."""
    normalized = method.lower()
    if normalized not in _SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {_SUPPORTED_METHODS}, got '{method}'.")
    return normalized


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load an image as an 8-bit grayscale array.

    Args:
        path: Path to an image file readable by OpenCV.

    Returns:
        Grayscale image, shape (H, W), dtype uint8.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file exists but cannot be decoded as an image,
            or decodes to an empty image.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(
            f"Failed to read image (unsupported format or corrupt file): {image_path}"
        )
    if image.ndim != 2 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image is empty or not a valid 2D grayscale array: {image_path}")

    return image.astype(np.uint8, copy=False)


def detect_and_describe(
    image: np.ndarray,
    method: str = "sift",
    max_features: int = 4000,
) -> tuple[list, np.ndarray]:
    """Detect keypoints and compute descriptors.

    Args:
        image: Grayscale image, shape (H, W), dtype uint8.
        method: Feature method, one of "sift" (``cv2.SIFT_create``) or
            "orb" (``cv2.ORB_create``).
        max_features: Maximum number of features to detect. Must be > 0.

    Returns:
        A tuple ``(keypoints, descriptors)``: ``keypoints`` is a list of N
        ``cv2.KeyPoint`` objects; ``descriptors`` has shape (N, D) (D=128,
        dtype float32, for SIFT; D=32, dtype uint8, for ORB).

    Raises:
        ValueError: If ``image`` is not a 2D array, ``method`` is
            unsupported, or ``max_features`` is not positive.
        RuntimeError: If no keypoints/descriptors are detected, or the
            descriptor count does not match the keypoint count.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be a grayscale (H, W) array, got shape {image.shape}.")
    method = _check_method(method)
    if max_features <= 0:
        raise ValueError(f"max_features must be > 0, got {max_features}.")

    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        detector = cv2.ORB_create(nfeatures=max_features)

    keypoints, descriptors = detector.detectAndCompute(image, None)

    if not keypoints or descriptors is None or len(keypoints) == 0:
        raise RuntimeError(
            f"No keypoints/descriptors were detected with method='{method}'. "
            "Try a different image, feature method, or a scene with more texture."
        )
    if descriptors.shape[0] != len(keypoints):
        raise RuntimeError(
            f"Descriptor count ({descriptors.shape[0]}) does not match keypoint "
            f"count ({len(keypoints)})."
        )

    return list(keypoints), descriptors


def ratio_test_matches(
    descriptors_query: np.ndarray,
    descriptors_train: np.ndarray,
    method: str,
    ratio_threshold: float = 0.75,
) -> list:
    """Perform two-nearest-neighbor descriptor matching with Lowe's ratio test.

    Args:
        descriptors_query: Query descriptors, shape (Nq, D).
        descriptors_train: Train descriptors, shape (Nt, D), with Nt >= 2
            (required for k=2 nearest-neighbor matching).
        method: "sift" (L2 distance) or "orb" (Hamming distance).
        ratio_threshold: A candidate nearest match ``m`` is kept only if
            ``m.distance < ratio_threshold * n.distance``, where ``n`` is
            the second-nearest match. Must be in (0, 1).

    Returns:
        A list of ``cv2.DMatch`` objects that passed the ratio test,
        sorted by ascending ``distance``.

    Raises:
        ValueError: If ``method`` is unsupported, ``ratio_threshold`` is
            not in (0, 1), ``descriptors_query`` is empty, or
            ``descriptors_train`` has fewer than 2 descriptors.
    """
    method = _check_method(method)
    if not (0.0 < ratio_threshold < 1.0):
        raise ValueError(f"ratio_threshold must be in (0, 1), got {ratio_threshold}.")
    if descriptors_query.shape[0] == 0:
        raise ValueError("descriptors_query is empty; there are no keypoints to match.")
    if descriptors_train.shape[0] < 2:
        raise ValueError(
            "descriptors_train must contain at least 2 descriptors for k=2 "
            f"ratio-test matching, got {descriptors_train.shape[0]}."
        )

    norm_type = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(descriptors_query, descriptors_train, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio_threshold * second.distance:
            good_matches.append(best)

    good_matches.sort(key=lambda match: match.distance)
    return good_matches


def mutual_ratio_matches(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    method: str,
    ratio_threshold: float = 0.75,
) -> list:
    """Retain ratio-test matches that agree in both matching directions.

    Runs :func:`ratio_test_matches` from image 1 to image 2 and from image
    2 to image 1, then keeps only correspondences ``(i, j)`` where the
    forward match ``i -> j`` and the backward match ``j -> i`` agree
    (``queryIdx``/``trainIdx`` are mutual inverses). This is a stricter
    filter than a one-directional ratio test alone; callers that want the
    cheaper one-directional result can call :func:`ratio_test_matches`
    directly instead (e.g. controlled by a pipeline-level
    ``mutual_check`` flag).

    Args:
        descriptors1: Image 1 descriptors, shape (N1, D).
        descriptors2: Image 2 descriptors, shape (N2, D).
        method: "sift" or "orb" (see :func:`ratio_test_matches`).
        ratio_threshold: Lowe ratio threshold, applied in both directions.

    Returns:
        A list of ``cv2.DMatch`` objects (in the image1 -> image2
        direction: ``queryIdx`` indexes ``descriptors1``, ``trainIdx``
        indexes ``descriptors2``), deduplicated and sorted by ascending
        ``distance``.

    Raises:
        ValueError: See :func:`ratio_test_matches`.
    """
    forward_matches = ratio_test_matches(descriptors1, descriptors2, method, ratio_threshold)
    backward_matches = ratio_test_matches(descriptors2, descriptors1, method, ratio_threshold)

    backward_pairs = {(match.queryIdx, match.trainIdx) for match in backward_matches}

    mutual_matches = []
    seen_pairs: set[tuple[int, int]] = set()
    for match in forward_matches:
        pair = (match.queryIdx, match.trainIdx)
        reverse_pair = (match.trainIdx, match.queryIdx)
        if reverse_pair in backward_pairs and pair not in seen_pairs:
            seen_pairs.add(pair)
            mutual_matches.append(match)

    mutual_matches.sort(key=lambda match: match.distance)
    return mutual_matches


def matched_keypoint_coordinates(
    keypoints1: list,
    keypoints2: list,
    matches: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV keypoint matches to two Nx2 float64 pixel-coordinate arrays.

    Args:
        keypoints1: Image 1 keypoints (``cv2.KeyPoint`` list), as returned
            by :func:`detect_and_describe`.
        keypoints2: Image 2 keypoints.
        matches: List of ``cv2.DMatch``, with ``queryIdx`` indexing
            ``keypoints1`` and ``trainIdx`` indexing ``keypoints2``.

    Returns:
        A tuple ``(points1, points2)``, each shape (N, 2), dtype float64,
        where N = len(matches). Returns shape (0, 2) arrays if ``matches``
        is empty.

    Raises:
        ValueError: If any ``match.queryIdx`` or ``match.trainIdx`` is out
            of bounds for ``keypoints1``/``keypoints2``.
    """
    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    num_keypoints1 = len(keypoints1)
    num_keypoints2 = len(keypoints2)
    points1 = np.empty((len(matches), 2), dtype=np.float64)
    points2 = np.empty((len(matches), 2), dtype=np.float64)

    for i, match in enumerate(matches):
        if not (0 <= match.queryIdx < num_keypoints1):
            raise ValueError(
                f"match[{i}].queryIdx={match.queryIdx} is out of bounds for "
                f"{num_keypoints1} keypoints in image 1."
            )
        if not (0 <= match.trainIdx < num_keypoints2):
            raise ValueError(
                f"match[{i}].trainIdx={match.trainIdx} is out of bounds for "
                f"{num_keypoints2} keypoints in image 2."
            )
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt

    return points1, points2
