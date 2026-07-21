# M2B: Two-View Sparse Reconstruction

## Objective

Reconstruct a sparse 3D point cloud from a real, calibrated two-image pair:
detect and match real image features (SIFT), robustly estimate the
fundamental matrix with a from-scratch RANSAC + normalized eight-point
algorithm, recover the essential matrix and decompose it into a relative
camera pose via cheirality, triangulate matched points with linear DLT,
and filter/validate the result with reprojection error. This extends M2A
(synthetic epipolar geometry) to real images and real correspondence
noise/outliers, and extends M1 (single-camera projection/unprojection) to
a calibrated two-camera system.

## Pipeline Overview

```
Image 1, Image 2
  -> SIFT feature detection (reconstruction.features.matching)
  -> Descriptor matching + Lowe ratio test + mutual consistency check
  -> Fundamental matrix RANSAC (reconstruction.geometry.robust, reuses M2A's
     normalized eight-point + Sampson distance)
  -> Essential matrix (reconstruction.geometry.pose)
  -> Essential matrix decomposition -> 4 pose candidates
  -> Cheirality test -> selected (R_21, t_21)
  -> Linear DLT triangulation (reconstruction.geometry.triangulation)
  -> Reprojection-error filtering (reconstruction.evaluation.reprojection)
  -> Sparse 3D point cloud (+ visualizations, PLY, metrics.json)
```

## Input Image Pair

- **Dataset**: Middlebury `templeRing` (Seitz, Diebel, Scharstein,
  Curless, Szeliski) — 47 calibrated views of a small plaster object,
  captured on the Stanford spherical light-field gantry.
- **Images**: `data/sample/two_view/templeRing/templeR0001.png` and
  `templeR0003.png`.
- **Resolution**: 640x480 for both.
- **Capture conditions**: a skip-2 pair on the ring, ~15 degrees apart
  (baseline/camera-distance ratio ~0.29 from the dataset's ground-truth
  extrinsics); static object, fixed calibrated camera/gantry, textured
  object against a mostly dark/plain background. See
  `data/sample/two_view/README.md` for the general good/bad capture-condition
  guidance this repository follows.
- **Intrinsics source**: `ground_truth` — the dataset's own published
  calibration (`templeR_par.txt`), not an approximation:
  `fx=1520.4, fy=1525.9, cx=302.32, cy=246.87`, identical for both views.

## Feature Detection and Matching

- Method: **SIFT** (`cv2.SIFT_create`), `max_features=5000`.
- Descriptor distance: L2 (SIFT convention).
- Lowe ratio test: keep a nearest match `m` only if
  `m.distance < 0.75 * n.distance` against the second-nearest `n`.
- Mutual (bidirectional) consistency check: enabled — a match is kept only
  if the reverse-direction ratio test independently agrees.

## Robust Fundamental Matrix

- Minimal-sample RANSAC: 8 correspondences per iteration (no replacement),
  normalized eight-point solve (`reconstruction.geometry.epipolar.estimate_fundamental_matrix`,
  reused unchanged from M2A).
- Inlier rule: Sampson distance <= `threshold_px**2` = 1.5^2 = 2.25 px^2.
  Best model = most inliers, ties broken by smaller median inlier Sampson
  distance.
- Adaptive early stopping from the observed inlier ratio
  (`log(1-confidence)/log(1-ratio**8)`).
- Local refinement: up to 3 refit-and-reclassify passes on the current
  inlier set after the best minimal-sample model is found.

## Essential Matrix

- `E = K2^T F K1`, followed by the essential singular-value constraint
  (`enforce_essential_constraints`: average the two largest singular
  values, zero the third) and a final rescale to Frobenius norm `sqrt(2)`
  (singular values `[1, 1, 0]`) as a canonical, unit-translation-consistent
  scale.
- **Translation scale ambiguity**: the essential matrix determines the
  translation direction only, never its magnitude — `decompose_essential_matrix`
  always returns a unit-norm `t_21`. The triangulated point cloud is
  therefore correct only up to one unknown global scale factor.

## Relative Pose Recovery

- `decompose_essential_matrix` returns all four algebraically valid
  `(R_21, t_21)` candidates: `(R1, +t), (R1, -t), (R2, +t), (R2, -t)`.
- `select_pose_by_cheirality` triangulates all inlier correspondences with
  each candidate and picks the one with the most points at positive depth
  in *both* cameras (ties broken by median reprojection error).
- Camera 2's center in the camera 1 (reference) frame:
  `C2 = -R_21^T @ t_21`.

## Triangulation

- Linear homogeneous DLT (`triangulate_points_dlt`), per-point SVD of the
  4x4 system built from `P1 = K1[I|0]` and `P2 = K2[R_21|t_21]`.
- Positive-depth check in both cameras (`camera_depths`,
  `min_positive_depth = 1e-8`).
- Final filter: finite 3D point, positive depth in both cameras, and
  combined reprojection error `<= max_reprojection_error_px = 3.0`.

## Coordinate Convention

Consistent with M1/M2A (`reconstruction.geometry.transforms`,
`reconstruction.geometry.epipolar`):

```
T_camera1_world = I                     (camera 1 is the reference frame)
T_camera2_camera1 = [R_21 | t_21]       X_camera2 = R_21 @ X_camera1 + t_21
P1 = K1 [I | 0]
P2 = K2 [R_21 | t_21]
```

All triangulated points and both camera centers are expressed in the
camera 1 frame. As above, the reconstruction's scale is **arbitrary** —
`t_21` is unit-norm by construction, so no measurement in this pipeline
can recover real-world (metric) units from two views alone.

## Quantitative Results

`pytest -q`: **132 passed**, 0 failed (68 from M1/M2A + 64 new M2B tests:
19 matching, 12 RANSAC, 15 pose, 11 triangulation, 9 reprojection, 1
end-to-end synthetic integration).

Real-image run (`configs/two_view/real_pair.yaml`, seed 42), from
`experiments/exp002_two_view_reconstruction/metrics.json`:

| Metric | Value |
|---|---|
| Image 1 keypoints | 801 |
| Image 2 keypoints | 779 |
| Ratio-test matches | 238 |
| Mutual matches | 204 |
| RANSAC inliers | 197 / 204 |
| RANSAC inlier ratio | 0.9657 |
| RANSAC attempted iterations / valid models / skipped degenerate samples | 6 / 6 / 0 |
| Median / mean inlier Sampson distance | 0.00987 / 0.08400 px^2 |
| Pose candidate positive-depth counts | [0, 197, 0, 0] |
| Selected pose candidate | 1 |
| Positive-depth count / ratio | 197 / 197 = 1.000 |
| Triangulated points | 197 |
| Filtered points (<= 3.0 px combined error) | 39 |
| Mean / median / max reprojection error (image 1) | 5.120 / 5.490 / 8.989 px |
| Mean / median / max reprojection error (image 2) | 4.966 / 5.374 / 8.603 px |
| Mean / median / max combined reprojection error | 5.044 / 5.439 / 8.798 px |
| `rotation_camera2_camera1` | `[[0.9992,-0.0398,-0.0075],[0.0405,0.9720,0.2313],[-0.0019,-0.2314,0.9729]]` |
| `translation_direction_camera2_camera1` | `[0.00671, -0.97067, 0.24034]` |
| `camera2_center_in_camera1_frame` | `[0.03304, 0.99941, -0.00927]` |
| `intrinsics_source` (camera1 / camera2) | ground_truth / ground_truth |

Against the recommended initial success criteria (RANSAC inliers >= 30,
inlier ratio >= 0.30, positive-depth ratio >= 0.70, median combined
reprojection error < 3 px, filtered points >= 20): **4 of 5 met** — median
combined reprojection error (5.44 px) exceeds the 3 px guideline. Root
cause analyzed below (Observations, "Reprojection error가 작더라도...").

**Supplementary ground-truth validation** (not part of the production
pipeline; a one-off analysis using the dataset's published per-image
extrinsics, computed for this report only): the ground-truth relative
pose between these two frames is `R_21_gt` (rotation) and a translation
direction `[0.0153, -0.9925, 0.1210]`. Comparing:

| Test | Rotation error | Translation direction error | Median combined reproj. error (same 197 inliers) |
|---|---|---|---|
| This run's estimated pose vs. ground truth | 1.797 deg | 6.975 deg | — |
| Ground-truth pose (correct scale) re-triangulating the same inliers | — | — | 0.059 px (197/197 pass <=3px) |
| Decomposing the **exact** ground-truth F (not RANSAC-estimated) through this repo's `essential_from_fundamental` -> `decompose_essential_matrix` -> `select_pose_by_cheirality` | 0.0055 deg | 0.0 deg | — |

This confirms the essential-decomposition/cheirality code itself is
correct (near-exact recovery from an exact F); the ~5 px reprojection
error in the main run traces to the RANSAC-estimated F carrying enough
residual noise to shift the decomposed pose by a few degrees. See
Observations for why a ~0.01 px^2 Sampson fit can still produce this.

## Qualitative Results

- `assets/figures/m02b_raw_matches.png`: all 204 mutual ratio-test matches
  between the two images (side by side), colored lines connecting each
  correspondence. Most matches correctly follow the object's geometry;
  a few visibly crossing lines show a small number of pre-RANSAC outliers.
- `assets/figures/m02b_ransac_inliers.png`: the 197 RANSAC-inlier matches
  only — visibly cleaner/more consistent than the raw match set.
- `assets/figures/m02b_epipolar_inliers.png`: 15 of the 197 inliers'
  epipolar lines (`l2 = F x1`) drawn over image 2, each passing very close
  to its corresponding point — a direct visual confirmation of the small
  Sampson-distance fit.
- `assets/figures/m02b_sparse_reconstruction.png`: the 39 filtered 3D
  points plus both camera centers and short forward-direction indicators,
  in the camera 1 (reference) frame; title reports filtered count (39),
  median reprojection error among the filtered points (1.995 px — lower
  than the pre-filter median, since filtering keeps the best-fit points),
  and positive-depth ratio (1.000), plus the "arbitrary scale" note.

## Observations

1. **Why does descriptor matching produce outliers?** SIFT descriptors
   only approximate true correspondence; repetitive local texture,
   partial occlusion, viewpoint/illumination change, and descriptors that
   are simply close in descriptor space without being the true match all
   produce a nearest-neighbor match that is geometrically wrong. Lowe
   ratio and mutual checks reduce, but do not eliminate, these.
2. **Why does the Lowe ratio test reduce ambiguous matches?** If a
   descriptor's nearest and second-nearest neighbor are similarly close
   (ratio near 1), the match is ambiguous — likely a repetitive or weak
   feature — and unreliable. Requiring the nearest to be substantially
   closer than the second-nearest (ratio < 0.75) keeps only matches with
   a confident, distinctive nearest neighbor.
3. **Why does mutual matching increase precision?** A one-directional
   ratio-test match can pass even if the reverse direction would pick a
   different correspondence (see `test_mutual_ratio_matches_drops_asymmetric_matches`
   for an exact worked example). Requiring both directions to agree
   discards these asymmetric, usually-wrong matches, at the cost of
   recall (204 mutual vs. 238 one-directional here).
4. **Why is RANSAC still necessary given the normalized eight-point
   algorithm?** The eight-point algorithm (even normalized) is a
   least-squares fit with no robustness to outliers — a handful of wrong
   correspondences can dominate the SVD solution and produce a badly
   wrong F. RANSAC repeatedly fits on minimal (8-point) samples and keeps
   the model consistent with the largest inlier set, so a wrong minimal
   sample only wastes one iteration rather than corrupting the estimate.
5. **What roles do F and E play?** F operates directly on raw pixel
   coordinates (`x2^T F x1 = 0`) and has no interpretable internal
   structure beyond rank 2 — it absorbs both intrinsics and pose. E
   operates on normalized camera-ray coordinates (`K^{-1} x`) and has the
   structural form `[t]_x R`, which is what makes the rotation/translation
   decomposition in `decompose_essential_matrix` possible; F alone cannot
   be decomposed into a pose without first removing the intrinsics via
   `E = K2^T F K1`.
6. **Why does E give four pose candidates?** SVD determines `U`, `V` only
   up to sign/column ambiguities consistent with `E`'s rank-2 structure,
   and both `R1 = UWV^T` and `R2 = UW^TV^T` are valid rotations satisfying
   the same `E` (via `[t]_x`'s own sign ambiguity), so all four
   sign/rotation combinations `(R1,+t), (R1,-t), (R2,+t), (R2,-t)`
   algebraically produce the same E.
7. **How does cheirality pick the right one?** Only one of the four
   candidates places the triangulated 3D points in front of *both*
   cameras simultaneously — the physical requirement for a point to have
   actually been photographed by both. In this run, exactly one candidate
   (index 1) achieved a positive-depth count of 197/197; the other three
   scored 0/197, an unusually clean separation confirming the geometry is
   well-conditioned for disambiguation (this is not always true near
   pure-rotation or degenerate baselines).
8. **Why is there no absolute scale?** The essential matrix constrains
   only the translation *direction* (`E = [t]_x R` is invariant to scaling
   `t`); nothing in a single image pair ties any measurement to a known
   physical length. `decompose_essential_matrix` always returns a
   unit-norm `t_21`, so every triangulated point and the camera-2 center
   are correct only up to one shared, unknown global scale factor —
   consistent with M2A's synthetic-geometry note on the same ambiguity.
9. **Can small reprojection/Sampson error still mean a wrong
   reconstruction?** Yes — this run is a direct real-data example. The
   RANSAC fit has a median Sampson distance of only 0.00987 px^2 (an
   excellent 2D epipolar-constraint fit), yet the *decomposed pose* has a
   rotation error of ~1.8 degrees and translation-direction error of
   ~6.98 degrees against the dataset's ground truth (see Quantitative
   Results), which is enough to push median reprojection error to ~5.4 px.
   Sampson distance only measures how well points satisfy the *epipolar*
   constraint (a 2D relationship); it does not directly measure 3D pose
   accuracy, and essential-matrix decomposition can amplify small F
   errors into a measurably larger pose error, especially when combined
   with a long focal length (here, ~1520 px — a given angular pose error
   sweeps more pixels than it would at a typical ~500-800 px webcam focal
   length).
10. **Why does a small baseline destabilize triangulation depth?** DLT
    triangulation finds the intersection of two rays; with a small
    baseline, the two rays are nearly parallel, so a tiny pixel
    perturbation (noise, quantization) swings the intersection point's
    depth by a large amount — the triangulation is poorly conditioned.
    This dataset's baseline/camera-distance ratio (~0.29 for the chosen
    pair) is moderate but not large, which is part of why decomposition
    noise here has an outsized effect on reprojection error.
11. **Why does pure rotation break triangulation?** With zero translation,
    `E = [0]_x R = 0` — the essential matrix is identically the zero
    matrix (which `essential_from_pose`/this module's zero-translation
    check explicitly rejects), and every ray pair for a pure rotation is
    related only by a 2D homography, never converging to a unique 3D
    point regardless of depth: triangulation is undefined, not just
    unstable.
12. **Why can near-planar scenes destabilize F-based reconstruction?**
    When all scene points lie close to a single plane, the point
    correspondences satisfy a homography (`x2 = H x1`) as well as (or
    better than) the general epipolar constraint, so the eight-point
    system becomes ill-conditioned / near rank-deficient in a way that
    admits many nearly-equally-good F solutions — the true F becomes
    only weakly distinguishable from degenerate alternatives. This
    dataset's object has a bounding-box depth range that is a moderate
    fraction of the camera distance (not fully planar), which likely
    limits — but, per Observation 9, does not eliminate — this effect.

## Failure Cases

Not all observed here, but relevant to this pipeline in general:

- **Low overlap** — too few correspondences survive matching/RANSAC.
- **Repetitive texture** — ambiguous matches even after the ratio test.
- **Low texture** — too few keypoints detected at all.
- **Motion blur** — degraded descriptors, fewer/worse matches.
- **Dynamic objects** — correspondences that violate the static-scene
  rigid-motion assumption, contaminating RANSAC's inlier set.
- **Pure rotation** — zero-translation essential matrix, undefined
  triangulation (see Observation 11).
- **Small baseline** — poorly conditioned triangulation depth (Observation
  10); this run's moderate baseline plus decomposition sensitivity is a
  contributing factor to its elevated reprojection error.
- **Inaccurate intrinsics** — directly degrades E, pose, and triangulation
  accuracy (this run used ground-truth intrinsics, so this was not a
  factor here).
- **Planar degeneracy** — ill-conditioned eight-point system (Observation
  12).
- **Incorrect matches surviving to RANSAC** — a small residual fraction
  can still bias the final refit even after the ratio/mutual/RANSAC
  filters.
- **Points behind a camera** — excluded by the cheirality/positive-depth
  filters, not triangulated into the final cloud.
- **Large reprojection error** — this run's central finding: filtered out
  by the `max_reprojection_error_px` threshold rather than silently kept,
  at the cost of retaining only 39 of 197 triangulated points.

## Limitations

- Two views only — no multi-view consistency or bundle adjustment.
- No bundle adjustment (pose and points are not jointly refined).
- Arbitrary reconstruction scale (structural, not fixable with better
  intrinsics or more RANSAC iterations).
- Intrinsics are either dataset-provided ground truth or a user-supplied
  approximation — this repository does not implement camera calibration.
- No lens distortion model (assumes an ideal pinhole camera).
- Sparse reconstruction only (one 3D point per surviving correspondence).
- No multi-view track management (no notion of a point observed across
  more than two images).

## Reproduction

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_two_view_reconstruction.py \
    --config configs/two_view/real_pair.yaml
```

## Next Step

This repository's roadmap continues M2B with **point-cloud registration
(ICP)** next, before moving on to a COLMAP multi-view SfM baseline:

- Near-term: **M3 — Point-cloud registration with ICP** (aligning sparse
  or dense point clouds from different views/scans).
- Following that: **M4 — COLMAP SfM baseline** (multi-view structure from
  motion, as a comparison point for this repository's from-scratch
  two-view pipeline).
- Deferred to that stage: multi-view tracks and incremental pose
  estimation (an M3A-style extension), bundle adjustment, and metric
  scale recovery (e.g. from known object size or an IMU/GPS baseline).
