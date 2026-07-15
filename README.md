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

**M1: Camera Geometry** — coordinate transformations and pinhole camera
projection/unprojection. See
[reports/milestone-reports/M1-camera-geometry.md](reports/milestone-reports/M1-camera-geometry.md).

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo_camera_geometry.py --config configs/camera_geometry/example.yaml
```

The demo saves a projection visualization to
`assets/figures/m01_camera_projection.png`.

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
