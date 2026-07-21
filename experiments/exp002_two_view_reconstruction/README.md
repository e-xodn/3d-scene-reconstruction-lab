# Experiment: Two-View Sparse Reconstruction

## Objective

Validate the full M2B pipeline (SIFT matching -> RANSAC fundamental matrix
-> essential matrix -> pose decomposition -> cheirality -> DLT
triangulation -> reprojection filtering) on a real, calibrated image pair,
and quantify reconstruction quality against the dataset's ground-truth
camera parameters.

## Image Pair

- Dataset: Middlebury `templeRing` (Seitz, Diebel, Scharstein, Curless,
  Szeliski), 47 calibrated views.
- `data/sample/two_view/templeRing/templeR0001.png` and
  `templeR0003.png` (640x480), a skip-2 pair ~15 degrees apart on the
  capture ring.
- Baseline/camera-distance ratio (from ground-truth extrinsics): ~0.29.

## Camera Intrinsics

Both views share the dataset's published (ground-truth) calibration:

```
K = [[1520.4,    0.0, 302.32],
     [   0.0, 1525.9, 246.87],
     [   0.0,    0.0,    1.0]]
```

`intrinsics_source: ground_truth` for both cameras (not an approximation).

## Feature Configuration

- Method: SIFT, `max_features=5000`
- Lowe ratio threshold: 0.75
- Mutual (bidirectional) consistency check: enabled

## RANSAC Configuration

- `threshold_px=1.5` (Sampson-distance inlier rule: `sampson <= 1.5**2`)
- `max_iterations=5000`, `confidence=0.999`, `seed=42`, `min_inliers=30`

## Reconstruction Configuration

- `min_positive_depth=1e-8`
- `min_positive_depth_ratio=0.7`
- `max_reprojection_error_px=3.0`
- `max_points_for_pose_selection=1000`

## Command

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_two_view_reconstruction.py --config configs/two_view/real_pair.yaml
```

(see `command.sh` for the reconstruction run itself)

## Results

Full metrics in `metrics.json` (this directory). Summary:

| Metric | Value |
|---|---|
| Image 1 / image 2 keypoints | 801 / 779 |
| Ratio-test matches | 238 |
| Mutual matches | 204 |
| RANSAC inliers | 197 / 204 (ratio 0.966) |
| Median inlier Sampson distance | 0.0099 px^2 |
| Selected pose candidate | 1 (positive-depth counts: [0, 197, 0, 0]) |
| Positive-depth ratio | 1.000 (197/197) |
| Triangulated points | 197 |
| Filtered points (<= 3.0 px combined error) | 39 |
| Median reprojection error (image 1 / image 2) | 5.49 / 5.37 px |
| Median combined reprojection error | 5.44 px |

Recommended initial success criteria (RANSAC inliers >= 30, inlier ratio
>= 0.30, positive-depth ratio >= 0.70, filtered points >= 20): **4 of 5
met**. The median combined reprojection error (5.44 px) exceeds the < 3 px
guideline — see Failure Analysis.

## Failure Analysis

The 2D fit (Sampson distance, 0.0099 px^2 median) is excellent, and the
positive-depth ratio is perfect (1.0), so RANSAC's fundamental-matrix fit
and the cheirality-based pose selection are both working correctly. The
reprojection error is nonetheless elevated. To isolate the cause, the
same 197 inlier correspondences were re-triangulated and re-scored two
ways (see `reports/milestone-reports/M2B-two-view-reconstruction.md`,
Observations, for the full analysis):

1. With this run's **estimated** pose: median combined reprojection error
   5.44 px; rotation error vs. the dataset's ground-truth relative pose is
   ~1.80 degrees, translation-direction error ~6.98 degrees.
2. With the dataset's **ground-truth** relative pose (same K, same
   inliers): median combined reprojection error 0.059 px, and all 197
   points pass the 3 px filter.
3. Decomposing the **exact ground-truth fundamental matrix** (rather than
   the RANSAC estimate) through the same `essential_from_fundamental` ->
   `decompose_essential_matrix` -> `select_pose_by_cheirality` code
   recovers the ground-truth pose to ~0.005 degrees rotation error and
   ~0.0 degrees translation-direction error.

This isolates the error to F estimation noise (not a decomposition bug):
a Sampson-distance fit of ~0.01 px^2 is small in the 2D epipolar-constraint
sense, but essential-matrix decomposition amplifies small F errors into a
several-degree pose error here, most likely because (a) the object's
depth range is fairly small relative to camera distance (bounding-box
diagonal / distance ~ 0.39), and (b) the very long focal length (~1520 px)
means a given angular pose error translates into more reprojected pixels
than it would at a typical webcam focal length (~500-800 px). This is a
direct, real-data illustration of why a good 2D/Sampson fit does not
guarantee an accurate 3D reconstruction (see the milestone report's
Observations for the general discussion).

## Conclusion

The custom RANSAC / essential-decomposition / triangulation pipeline is
verified correct end-to-end (via the ground-truth-F control test above,
and via the synthetic integration test in `tests/test_two_view_integration.py`).
On this specific real pair, feature matching and RANSAC produce an
excellent 2D fit and a fully self-consistent geometry (positive-depth
ratio 1.0), but the recovered pose has enough error, amplified by the
scene's modest depth range and the dataset's long focal length, to push
median reprojection error above the 3 px guideline; 39 points still pass
the filter and form a plausible sparse point cloud.

## Next Experiment

Candidates to reduce reprojection error on this dataset: a wider-baseline
pair (e.g. `templeR0001` vs `templeR0004`, baseline ratio ~0.43) for
better-conditioned F estimation, or tightening the RANSAC Sampson
threshold; both are exactly the kind of image-pair/threshold trade-off
discussed in the milestone report rather than a pipeline bug fix. The
next milestone (M3, point-cloud registration / ICP, or M3A multi-view
tracks) is a separate scope question tracked in the milestone report's
"Next Step" section.
