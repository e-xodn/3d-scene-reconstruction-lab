# 3D Scene Reconstruction Lab

A reproducible study repository for learning and evaluating 3D scene reconstruction.

## Objectives

- Understand camera geometry and multi-view geometry
- Implement two-view reconstruction and point-cloud registration
- Experiment with COLMAP SfM and MVS
- Compare NeRF and 3D Gaussian Splatting
- Build an outdoor reconstruction baseline on RELLIS-3D

## Roadmap

1. Repository foundation
2. Coordinate systems and camera geometry
3. Two-view reconstruction
4. Point-cloud registration
5. COLMAP SfM and dense MVS
6. RGB-D and LiDAR reconstruction
7. NeRF
8. 3D Gaussian Splatting
9. Learning-based reconstruction
10. RELLIS-3D reconstruction

See [ROADMAP.md](ROADMAP.md) for details.

## Current Milestone

**M2B: Two-View Sparse Reconstruction** — real-image SIFT matching,
custom RANSAC fundamental matrix estimation, essential matrix
decomposition with cheirality-based pose selection, and DLT triangulation
into a sparse, colored 3D point cloud. See
[reports/milestone-reports/M2B-two-view-reconstruction.md](reports/milestone-reports/M2B-two-view-reconstruction.md).

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/run_two_view_reconstruction.py --config configs/two_view/real_pair.yaml
```

Output:

```
assets/figures/m02b_raw_matches.png
assets/figures/m02b_ransac_inliers.png
assets/figures/m02b_epipolar_inliers.png
assets/figures/m02b_sparse_reconstruction.png
```

Note: the recovered translation and 3D point cloud have **arbitrary
scale** — two-view reconstruction alone cannot recover absolute metric
scale (see the milestone report's Coordinate Convention section).

Previous milestones: **M2A: Synthetic Epipolar Geometry** — normalized
eight-point fundamental matrix estimation on a synthetic calibrated
two-camera scene. See
[reports/milestone-reports/M2A-epipolar-geometry.md](reports/milestone-reports/M2A-epipolar-geometry.md)
(`python scripts/demo_epipolar_geometry.py --config configs/two_view/synthetic_epipolar.yaml`).
**M1: Camera Geometry** — coordinate transformations and pinhole camera
projection/unprojection. See
[reports/milestone-reports/M1-camera-geometry.md](reports/milestone-reports/M1-camera-geometry.md)
(`python scripts/demo_camera_geometry.py --config configs/camera_geometry/example.yaml`).

## Repository Structure

```text
configs/       Experiment configurations
data/          Dataset documentation and local links
environment/   Conda and container environments
experiments/   Reproducible experiment records
src/           Reusable Python modules
scripts/       Executable pipelines
reports/       Milestone and paper notes
tests/         Automated tests
