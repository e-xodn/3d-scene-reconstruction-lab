# M1: Camera Geometry

## Objective

Implement and validate the coordinate-transformation and pinhole-camera-model
primitives that later milestones (two-view reconstruction, SfM, dense MVS)
will build on: SE(3) rigid-body transforms between coordinate frames, and
projection/unprojection between camera-frame 3D points and image pixels.

## Coordinate Convention

- Transform variables are named `T_target_source`. Applying `T_target_source`
  to points expressed in the `source` frame yields points expressed in the
  `target` frame.
- Mathematical definition:

  ```text
  p_target = R_target_source @ p_source + t_target_source
  ```

- 3D points are stored as `(N, 3)` row-arrays, so the batched numpy
  implementation is:

  ```text
  points_target = points_source @ R.T + t
  ```

- Transform composition order:

  ```text
  T_target_source = T_target_intermediate @ T_intermediate_source
  ```

  i.e. `compose_transforms(T_target_intermediate, T_intermediate_source)`
  returns `T_target_intermediate @ T_intermediate_source`.

## Pinhole Camera Model

- Intrinsic matrix:

  ```text
  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
  ```

- Projection equation (camera frame -> pixel):

  ```text
  u = fx * X / Z + cx
  v = fy * Y / Z + cy
  ```

- Unprojection equation (pixel + depth -> camera frame):

  ```text
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = depth
  ```

- Positive-depth condition: a point is only projectable when `Z > min_depth`
  (default `1e-8`). Points at or behind the camera (`Z <= min_depth`) are
  marked invalid and given `NaN` pixel coordinates; their `Z` value is still
  reported in `depth`.

## Implementation

- [`src/reconstruction/geometry/transforms.py`](../../src/reconstruction/geometry/transforms.py)
  — `make_transform`, `transform_points`, `invert_transform`,
  `compose_transforms`, plus private validation helpers
  (`_check_finite`, `_check_rotation_matrix`, `_check_transform_matrix`).
- [`src/reconstruction/cameras/pinhole.py`](../../src/reconstruction/cameras/pinhole.py)
  — `create_intrinsic_matrix`, `project_points`, `unproject_pixels`.
- [`scripts/demo_camera_geometry.py`](../../scripts/demo_camera_geometry.py)
  — loads a YAML config, generates synthetic camera-frame points (in-bounds,
  out-of-bounds, and behind-camera), projects and unprojects them, and saves
  a visualization.
- [`configs/camera_geometry/example.yaml`](../../configs/camera_geometry/example.yaml)
  — example camera intrinsics and scene generation parameters.
- [`tests/test_transforms.py`](../../tests/test_transforms.py) and
  [`tests/test_pinhole.py`](../../tests/test_pinhole.py) — unit tests
  covering identity/translation/rotation transforms, inverse and composition
  correctness, projection/unprojection round trips, validity conditions, and
  input-validation error paths.

## Validation

- `pytest -q`: **27 passed**.
- Demo run (`configs/camera_geometry/example.yaml`, seed 42):
  - Total points: **120** (100 valid + 10 out-of-bounds + 10 behind-camera)
  - Positive-depth points: **110**
  - Points inside image: **100**
  - Invalid points: **20**
  - Mean round-trip error: **8.9e-18**
  - Max round-trip error: **4.4e-16**
  - Figure saved to `assets/figures/m01_camera_projection.png`

Both the mean and max round-trip errors are many orders of magnitude below
the `1e-6` completion threshold, consistent with projection/unprojection
being exact algebraic inverses of one another (float64 rounding is the only
error source).

## Observations

- **Small Z amplifies projection sensitivity.** Since `u = fx * X / Z + cx`,
  the partial derivative of `u` with respect to `X` scales as `fx / Z`. As
  `Z` shrinks toward zero, a fixed change in `X` or `Y` (or a fixed amount of
  numerical/measurement noise) produces an increasingly large change in pixel
  position — this is the geometric root of why near-camera points are
  numerically unstable and why `min_depth` guards against `Z` values close to
  zero.
- **Behind-camera points are excluded because the pinhole model is only
  defined for a forward-facing ray.** A point with `Z <= 0` does not
  correspond to a physically visible ray through the image plane; naively
  applying the projection formula would either divide by zero/negative
  values or produce a pixel that aliases with a point actually in front of
  the camera, silently corrupting downstream geometry.
- **Image-boundary validity is a separate check from depth validity**
  because they test different failure modes: `Z <= min_depth` means "no
  physical image of this point exists," while an out-of-bounds `(u, v)`
  means "the point exists and projects to a real pixel location, but that
  location falls outside the finite sensor/image array." A point can have
  perfectly valid positive depth and still be invisible simply because the
  camera's field of view is finite.
- **Projection followed by unprojection cannot recover unknown world
  scale** because the two operations are exact inverses only when depth
  (`Z`) is already known — `unproject_pixels` requires `depth` as an
  explicit input. Given only a pixel `(u, v)`, every point along the
  corresponding ray `(X, Y, Z) = ((u - cx) * Z / fx, (v - cy) * Z / fy, Z)`
  for `Z > 0` projects to the same pixel, so a single 2D observation is
  scale-ambiguous by construction; recovering scale requires additional
  information (known depth, stereo triangulation, or scene priors), which is
  exactly the problem later milestones (two-view reconstruction, SfM, dense
  MVS) address.

## Reproduction

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo_camera_geometry.py \
    --config configs/camera_geometry/example.yaml
```
