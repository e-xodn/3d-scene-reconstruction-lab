"""Tests for reconstruction.features.matching.

Real user images are never used here; all tests use small deterministic
synthetic descriptor/keypoint arrays (or monkeypatching) so results are
exact and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reconstruction.features.matching import (
    detect_and_describe,
    matched_keypoint_coordinates,
    mutual_ratio_matches,
    ratio_test_matches,
)


# ---------------------------------------------------------------------------
# load_grayscale_image / detect_and_describe validation
# ---------------------------------------------------------------------------


def test_load_grayscale_image_missing_file_raises(tmp_path: Path) -> None:
    from reconstruction.features.matching import load_grayscale_image

    with pytest.raises(FileNotFoundError):
        load_grayscale_image(tmp_path / "does_not_exist.png")


def test_detect_and_describe_rejects_non_grayscale_image() -> None:
    color_image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_describe(color_image, method="sift")


def test_detect_and_describe_invalid_method_raises() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_describe(image, method="not_a_method")


def test_detect_and_describe_invalid_max_features_raises() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_describe(image, method="sift", max_features=0)


def test_detect_and_describe_does_not_mutate_image() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    image_copy = image.copy()

    try:
        detect_and_describe(image, method="orb", max_features=200)
    except RuntimeError:
        pass  # a random-noise image may yield no keypoints; that's fine here

    np.testing.assert_array_equal(image, image_copy)


# ---------------------------------------------------------------------------
# ratio_test_matches: validation and matcher-selection
# ---------------------------------------------------------------------------


def test_ratio_test_matches_invalid_ratio_threshold_raises() -> None:
    descriptors = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        ratio_test_matches(descriptors, descriptors, method="sift", ratio_threshold=1.5)


def test_ratio_test_matches_too_few_train_descriptors_raises() -> None:
    descriptors_query = np.zeros((3, 4), dtype=np.float32)
    descriptors_train = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        ratio_test_matches(descriptors_query, descriptors_train, method="sift")


def test_ratio_test_matches_empty_query_raises() -> None:
    descriptors_query = np.zeros((0, 4), dtype=np.float32)
    descriptors_train = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        ratio_test_matches(descriptors_query, descriptors_train, method="sift")


def test_ratio_test_matches_sift_uses_l2_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_norm_types = []
    real_bfmatcher = cv2.BFMatcher

    def spy_bfmatcher(norm_type, *args, **kwargs):
        seen_norm_types.append(norm_type)
        return real_bfmatcher(norm_type, *args, **kwargs)

    monkeypatch.setattr("reconstruction.features.matching.cv2.BFMatcher", spy_bfmatcher)

    descriptors = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    ratio_test_matches(descriptors, descriptors, method="sift")

    assert seen_norm_types == [cv2.NORM_L2]


def test_ratio_test_matches_orb_uses_hamming_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_norm_types = []
    real_bfmatcher = cv2.BFMatcher

    def spy_bfmatcher(norm_type, *args, **kwargs):
        seen_norm_types.append(norm_type)
        return real_bfmatcher(norm_type, *args, **kwargs)

    monkeypatch.setattr("reconstruction.features.matching.cv2.BFMatcher", spy_bfmatcher)

    descriptors = np.array([[0, 0, 0, 0]] * 3, dtype=np.uint8)
    ratio_test_matches(descriptors, descriptors, method="orb")

    assert seen_norm_types == [cv2.NORM_HAMMING]


def test_ratio_test_matches_lowe_ratio_behavior() -> None:
    # 1-D descriptors so Euclidean distance is exactly |a - b|.
    descriptors_query = np.array([[0.0], [100.0], [55.0]], dtype=np.float32)
    descriptors_train = np.array(
        [[0.5], [50.0], [100.1], [100.2], [9.0], [60.0]], dtype=np.float32
    )

    matches = ratio_test_matches(descriptors_query, descriptors_train, method="sift", ratio_threshold=0.75)

    matched_query_indices = {m.queryIdx for m in matches}
    assert matched_query_indices == {0, 1}  # query 2 is ambiguous (tied nearest neighbors)

    by_query = {m.queryIdx: m.trainIdx for m in matches}
    assert by_query[0] == 0  # nearest train descriptor to query 0 is train[0] = 0.5
    assert by_query[1] == 2  # nearest train descriptor to query 1 is train[2] = 100.1

    # sorted by ascending distance
    assert all(matches[i].distance <= matches[i + 1].distance for i in range(len(matches) - 1))


def test_ratio_test_matches_does_not_mutate_descriptors() -> None:
    descriptors_query = np.array([[0.0], [100.0]], dtype=np.float32)
    descriptors_train = np.array([[0.5], [50.0], [100.1]], dtype=np.float32)
    query_copy = descriptors_query.copy()
    train_copy = descriptors_train.copy()

    ratio_test_matches(descriptors_query, descriptors_train, method="sift")

    np.testing.assert_array_equal(descriptors_query, query_copy)
    np.testing.assert_array_equal(descriptors_train, train_copy)


# ---------------------------------------------------------------------------
# mutual_ratio_matches
# ---------------------------------------------------------------------------


def test_mutual_ratio_matches_drops_asymmetric_matches() -> None:
    # image 1: X=0.0, Y=1.0, W=100.0
    # image 2: Z=0.4, V=50.0
    # Forward: X->Z, Y->Z, W->V all pass the ratio test individually.
    # Backward: Z->X passes; V has no unambiguous backward match.
    # So only X<->Z is truly mutual; Y->Z and W->V must be dropped.
    descriptors1 = np.array([[0.0], [1.0], [100.0]], dtype=np.float32)
    descriptors2 = np.array([[0.4], [50.0]], dtype=np.float32)

    mutual_matches = mutual_ratio_matches(descriptors1, descriptors2, method="sift", ratio_threshold=0.75)

    assert len(mutual_matches) == 1
    assert mutual_matches[0].queryIdx == 0
    assert mutual_matches[0].trainIdx == 0


def test_mutual_ratio_matches_does_not_mutate_descriptors() -> None:
    descriptors1 = np.array([[0.0], [1.0], [100.0]], dtype=np.float32)
    descriptors2 = np.array([[0.4], [50.0]], dtype=np.float32)
    copy1 = descriptors1.copy()
    copy2 = descriptors2.copy()

    mutual_ratio_matches(descriptors1, descriptors2, method="sift")

    np.testing.assert_array_equal(descriptors1, copy1)
    np.testing.assert_array_equal(descriptors2, copy2)


# ---------------------------------------------------------------------------
# matched_keypoint_coordinates
# ---------------------------------------------------------------------------


def test_matched_keypoint_coordinates_shape_and_values() -> None:
    keypoints1 = [cv2.KeyPoint(1.0, 2.0, 1.0), cv2.KeyPoint(3.0, 4.0, 1.0)]
    keypoints2 = [cv2.KeyPoint(10.0, 20.0, 1.0), cv2.KeyPoint(30.0, 40.0, 1.0)]
    matches = [cv2.DMatch(0, 1, 0.1), cv2.DMatch(1, 0, 0.2)]

    points1, points2 = matched_keypoint_coordinates(keypoints1, keypoints2, matches)

    assert points1.shape == (2, 2)
    assert points2.shape == (2, 2)
    assert points1.dtype == np.float64
    assert points2.dtype == np.float64
    np.testing.assert_allclose(points1, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(points2, [[30.0, 40.0], [10.0, 20.0]])


def test_matched_keypoint_coordinates_empty_matches() -> None:
    points1, points2 = matched_keypoint_coordinates([], [], [])
    assert points1.shape == (0, 2)
    assert points2.shape == (0, 2)


def test_matched_keypoint_coordinates_invalid_query_index_raises() -> None:
    keypoints1 = [cv2.KeyPoint(1.0, 2.0, 1.0)]
    keypoints2 = [cv2.KeyPoint(10.0, 20.0, 1.0)]
    matches = [cv2.DMatch(5, 0, 0.1)]  # queryIdx out of bounds

    with pytest.raises(ValueError):
        matched_keypoint_coordinates(keypoints1, keypoints2, matches)


def test_matched_keypoint_coordinates_invalid_train_index_raises() -> None:
    keypoints1 = [cv2.KeyPoint(1.0, 2.0, 1.0)]
    keypoints2 = [cv2.KeyPoint(10.0, 20.0, 1.0)]
    matches = [cv2.DMatch(0, 7, 0.1)]  # trainIdx out of bounds

    with pytest.raises(ValueError):
        matched_keypoint_coordinates(keypoints1, keypoints2, matches)
