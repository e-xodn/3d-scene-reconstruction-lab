# M2A: Synthetic Epipolar Geometry

## Objective

- Understand two-view correspondence and the epipolar constraint between a
  calibrated camera pair with known relative pose.
- Implement the normalized eight-point algorithm from scratch (Hartley
  normalization, linear design matrix, SVD, rank-2 enforcement,
  denormalization).
- Compare the estimated fundamental matrix against the ground-truth matrix
  derived analytically from the known intrinsics and relative pose, on both
  clean and pixel-noise-corrupted correspondences.

## Coordinate Convention

- Camera 1's frame is treated as the world frame: `T_camera1_world = I`,
  `P1 = K1 [I | 0]`.
- Camera 2's pose relative to camera 1: `T_camera2_camera1 = [R_21 | t_21]`,
  so `X_camera2 = R_21 @ X_camera1 + t_21` and `P2 = K2 [R_21 | t_21]`.
- Epipolar constraint (column-vector math notation):
  `x2^T F x1 = 0`, for homogeneous `x1 = [u1, v1, 1]^T`,
  `x2 = [u2, v2, 1]^T`.
- Epipolar lines: `l2 = F x1` (line in image 2 for a point in image 1),
  `l1 = F^T x2` (line in image 1 for a point in image 2). A line
  `[a, b, c]` satisfies `a*u + b*v + c = 0`.
- Batches of `N` points are stored as `(N, 2)` / `(N, 3)` row-arrays.
  Applying a 3x3 matrix `M` to every point (`M @ x_i` in math notation) is
  implemented as `points_h @ M.T` on the whole batch, since each output row
  equals `(M @ x_i)^T == x_i^T @ M.T`.

## Essential and Fundamental Matrices

- `E = [t_21]_x R_21`.
- `F = K2^{-T} E K1^{-1}`.
- Both `E` and `F` are defined only up to an arbitrary nonzero scale
  (including sign), since the epipolar constraint `x2^T F x1 = 0` is
  scale-invariant; `canonicalize_fundamental_matrix` fixes this by
  normalizing to unit Frobenius norm and a fixed-sign convention so two
  matrices representing the same geometry compare equal.
- Both matrices have rank 2: `[t]_x` has rank 2 for nonzero `t`, and
  multiplying by full-rank rotation/intrinsic matrices preserves rank.

## Normalized Eight-Point Algorithm

Implemented in `estimate_fundamental_matrix`:

1. **Correspondence validation** — shapes, matching length, `N >= 8`,
   finite values, and explicit rejection of coincident or collinear points
   in either image (checked via the rank of the centered point matrix).
2. **Hartley normalization** — `points1` and `points2` are independently
   translated to zero centroid and isotropically scaled so the mean
   distance from the origin is `sqrt(2)`.
3. **Design matrix construction** — each correspondence contributes a row
   `[u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]` in normalized
   coordinates, so that `row @ vec(F) == x2^T F x1` for `F` flattened in
   row-major order.
4. **SVD solution** — the design matrix's right singular vector for the
   smallest singular value gives the normalized `F`, reshaped `(3, 3)`.
5. **Rank-2 constraint** — the smallest singular value of the normalized
   `F` is zeroed via SVD (`enforce_rank2`).
6. **Denormalization** — `F = T2^T @ F_normalized @ T1`, followed by a
   second `enforce_rank2` pass (denormalization can perturb the rank-2
   property numerically).
7. **Canonical scale normalization** — `canonicalize_fundamental_matrix`
   fixes scale and sign for comparison against ground truth.

Degenerate configurations (fewer than 8 points, coincident points, or
collinear points in either image, or a design matrix with insufficient
rank) are rejected with specific error messages rather than a generic SVD
failure.

## Error Metrics

- **Algebraic epipolar residual** (`x2^T F x1`): the direct SVD/least-squares
  objective. It is scale-dependent (rescaling `F` rescales the residual)
  and has no direct geometric (pixel) interpretation, so it is only useful
  as a relative/consistency check, not an absolute error.
- **Point-to-epipolar-line distance**: the Euclidean pixel distance from an
  observed point to its epipolar line in the *other* image. This is
  geometrically meaningful but asymmetric (it only measures error in one
  image at a time).
- **Sampson distance**: a first-order approximation to the sum of squared
  geometric distances in both images, derived from the algebraic residual
  and the gradient of the epipolar constraint. It is symmetric, scale
  independent (up to the same convention for `F`), and approximately in
  squared-pixel units when points are in pixel coordinates — a much better
  proxy for reprojection-style error than the raw algebraic residual.
- **Scale-invariant fundamental matrix error**: `min(||Fc_est - Fc_gt||_F,
  ||Fc_est + Fc_gt||_F)` on canonicalized matrices, used to compare an
  estimated `F` against the ground truth despite the inherent scale/sign
  ambiguity.

## Implementation

- `src/reconstruction/geometry/epipolar.py` — `to_homogeneous`,
  `from_homogeneous`, `skew_symmetric`, `normalize_points_2d`,
  `enforce_rank2`, `canonicalize_fundamental_matrix`, `essential_from_pose`,
  `fundamental_from_pose`, `estimate_fundamental_matrix`,
  `epipolar_lines_in_image2`, `epipolar_lines_in_image1`,
  `algebraic_epipolar_residuals`, `sampson_distances`,
  `point_to_epipolar_line_distances`, plus private validation helpers
  (`_check_finite`, `_check_2d_points`, `_check_3x3_matrix`,
  `_check_rotation_matrix`, `_check_intrinsic_matrix`,
  `_check_not_collinear`, `_normalize_line_coefficients`). Reuses no
  private helpers from `transforms.py`; a minimal local rotation validator
  is defined instead, per the module's own public/private boundary.
- `src/reconstruction/geometry/__init__.py` — extended to export the new
  epipolar functions alongside the existing M1 transform functions.
- `scripts/demo_epipolar_geometry.py` — loads the YAML config, builds
  `K1`/`K2` (`create_intrinsic_matrix`, reused from M1) and `R_21`/`t_21`
  (Euler-angle rotation with a fixed `Rz @ Ry @ Rx` order), rejection-samples
  synthetic 3D points visible in both cameras using M1's `project_points`
  and `transform_points`/`make_transform`, adds Gaussian pixel noise,
  computes ground-truth `E`/`F`, estimates `F` from clean and noisy
  correspondences, computes all error metrics, and saves a single-plot
  epipolar-line visualization (with a private line-clipping helper,
  `_clip_line_to_image`, handling vertical/horizontal/no-intersection
  cases).
- `configs/two_view/synthetic_epipolar.yaml` — camera intrinsics, relative
  pose, scene sampling ranges, and visualization settings.
- `tests/test_epipolar.py` — 41 unit tests covering homogeneous
  coordinates, skew-symmetric matrices, Hartley normalization, rank-2
  enforcement, ground-truth `E`/`F` construction, the eight-point
  algorithm's design-matrix convention and noiseless recovery, epipolar
  line/distance formulas, Sampson distance behavior, and degenerate-input
  validation.

## Validation Results

`pytest -q`: **68 passed** (27 from M1 + 41 new M2A tests), 0 failed.

Demo run (`configs/two_view/synthetic_epipolar.yaml`, seed 42):

- Number of correspondences: **120**
- Clean correspondences:
  - Mean absolute algebraic residual: **3.098e-12**
  - Median Sampson distance: **1.502e-21**
  - Maximum Sampson distance: **5.803e-21**
  - Scale-invariant F error: **6.341e-16**
- Noisy correspondences (`noise_std_px = 0.5`):
  - Mean absolute algebraic residual: **2.920e-02**
  - Median Sampson distance: **9.910e-02**
  - Maximum Sampson distance: **1.985e+00**
  - Median point-to-line distance in image 2: **4.485e-01** pixels
  - Scale-invariant F error: **3.291e-03**
- Fundamental matrix rank: **2** (both ground truth and estimated,
  verified via `np.linalg.matrix_rank`)
- Figure saved to `assets/figures/m02a_epipolar_geometry.png`

The clean-correspondence results are well within the recommended
thresholds (mean absolute algebraic residual `< 1e-8`, scale-invariant F
error `< 1e-6`); remaining error is float64 rounding. The demo was run
twice with the same config/seed and produced byte-identical stdout,
confirming reproducibility, and produced no runtime warnings (checked with
`python -W error::RuntimeWarning`).

## Visualization

`assets/figures/m02a_epipolar_geometry.png` shows image 2's epipolar
geometry: all 120 noisy correspondence points (small blue dots), 12
highlighted "selected correspondences" (orange, outlined), each with its
corresponding epipolar line `l2 = F x1` (computed from the noisy-estimated
`F`) clipped to the image boundary, the principal point of camera 2 (green
X), and the image boundary rectangle. The v-axis increases downward
(image convention). Each highlighted point sits close to its own line,
visually matching the reported median Sampson distance; the small
systematic offset between points and lines is the expected effect of the
0.5-pixel Gaussian noise added to both images.

## Observations

1. **Why does the fundamental matrix have an arbitrary scale?** The
   epipolar constraint `x2^T F x1 = 0` is homogeneous in `F` — multiplying
   `F` by any nonzero scalar leaves every solution unchanged. The
   eight-point algorithm's SVD only ever recovers `F` up to this scale (and
   sign), so comparing two fundamental matrices requires first fixing a
   common scale/sign convention (`canonicalize_fundamental_matrix`).
2. **Why is point normalization needed before the eight-point algorithm?**
   Raw pixel coordinates (e.g. `u ~ 300`, homogeneous `1`) produce design
   matrix columns spanning many orders of magnitude (`u2*u1 ~ 1e5` vs the
   constant column `1`), which badly conditions the SVD and amplifies
   noise sensitivity. Hartley normalization rescales all coordinates to
   the same order of magnitude (`~sqrt(2)` from the origin), dramatically
   improving numerical conditioning without changing the underlying
   geometry (the transform is undone by denormalization).
3. **Why enforce the rank-2 constraint?** A true fundamental matrix is
   always singular (rank <= 2) because it is built from a rank-2 skew
   matrix. The unconstrained eight-point SVD solution is only exactly
   rank-2 in the noiseless case; with noise it is generically full rank,
   which does not correspond to any valid epipolar geometry (a full-rank
   `F` has no well-defined epipole). Projecting onto the nearest rank-2
   matrix restores this structural constraint.
4. **Why is the algebraic residual hard to interpret geometrically?** `x2^T
   F x1` mixes the (arbitrary) scale of `F` with the local scale of the
   epipolar line's gradient (`||[Fx1_0, Fx1_1, Ftx2_0, Ftx2_1]||`, which
   varies per point). The same physical pixel error produces different
   algebraic residuals depending on where the point lies relative to the
   epipole and how `F` happens to be scaled, so it cannot be compared
   directly to a pixel-error threshold.
5. **Why is Sampson distance more useful than the algebraic residual?**
   Sampson distance divides the squared algebraic residual by the squared
   gradient magnitude of the constraint, which is a first-order correction
   that approximates the true sum-of-squared reprojection-style distances
   in both images. It is in approximately squared-pixel units, symmetric
   between the two images, and much less sensitive to the arbitrary scale
   of `F`, making it directly comparable to a pixel-noise budget (as seen
   in this demo, where noisy-Sampson values are on the order of `noise_std_px^2`).
6. **Why is a pure-rotation (zero-baseline) pair hard for fundamental
   matrix estimation?** With `t_21 = 0`, the essential matrix `[t]_x R`
   is identically zero, so there is no well-defined epipolar constraint —
   every point pair related by the homography `H = K2 R K1^{-1}` satisfies
   `x2 = H x1` regardless of depth, and the eight-point algorithm has no
   depth-dependent signal to recover. `essential_from_pose` and
   `fundamental_from_pose` explicitly reject a near-zero translation norm
   for this reason (see `test_essential_from_pose_zero_translation_raises`).
7. **How does pixel noise affect F estimation and epipolar lines?** In this
   demo, adding `noise_std_px = 0.5` pixel Gaussian noise increased the
   mean absolute algebraic residual from `3.1e-12` to `2.9e-2`, the median
   Sampson distance from `~1.5e-21` to `9.9e-2` (roughly `(0.5)^2 * constant`,
   consistent with Sampson distance's squared-pixel units), and the
   scale-invariant F error from `6.3e-16` to `3.3e-3`. Visually, epipolar
   lines no longer pass exactly through their corresponding points; the
   median point-to-line distance (`0.45` px) reflects this offset.
8. **How does this differ from real-image reconstruction?** This
   experiment uses perfectly known intrinsics and relative pose to
   generate noiseless, outlier-free, one-to-one correspondences by
   construction — there is no feature detection, descriptor matching, or
   mismatches to contend with. Real image pairs require detecting and
   matching features (SIFT/ORB), filtering false matches (ratio test,
   RANSAC), and estimating pose from noisy, outlier-contaminated data with
   unknown scale — all deferred to M2B.

## Limitations

- No outliers: all correspondences are inliers by construction.
- No feature matching: correspondences are generated directly from known
  3D points and known camera geometry, not detected/matched from images.
- No RANSAC or other robust estimation.
- Both cameras' intrinsics and the relative pose are known inputs used to
  generate the synthetic scene, not estimated from data.
- Essential matrix decomposition, relative pose recovery, the cheirality
  test, and triangulation are out of scope for this milestone.

## Reproduction

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo_epipolar_geometry.py \
    --config configs/two_view/synthetic_epipolar.yaml
```

## Next Step

M2B will move from synthetic to real two-view data:

- Real image pairs (instead of synthetic correspondences).
- SIFT or ORB feature detection.
- Descriptor matching with a ratio test.
- RANSAC-based robust fundamental/essential matrix estimation to reject
  outlier matches.
- Essential matrix decomposition and relative pose recovery.
- The cheirality test to disambiguate the four possible pose solutions.
- Triangulation of matched points into a 3D point cloud.
- Reprojection error as the end-to-end accuracy metric.
