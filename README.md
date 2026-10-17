# modelforge

Camera-pose refinement for DJI drone video, built on COLMAP (unmodified, via
pycolmap). It produces a **dense mesh** (COLMAP MVS) and a **Gaussian splat**
(nerfstudio splatfacto) that share one metric ENU frame.

Stock COLMAP rediscovers every pose from the images. modelforge instead decodes
DJI's on-board telemetry (the `djmd` track) and time-aligns it to the video. The
telemetry leads the images by ~0.7 s. Every frame then starts at its 6-DoF prior,
and refinement is bundle adjustment with self-calibrating absolute and relative
pose priors.

## Quick start

Everything runs in Docker. Build the image once: `env UID=$(id -u) GID=$(id -g) docker compose build dev`.

```bash
scripts/process data/clip.MP4 --num-images 300     # video -> poses -> (GPU) mesh + splat
scripts/dev python -m modelforge.gui.app data/clip_frames   # GUI on http://localhost:8080
```

Step by step:

```bash
scripts/dev python -m modelforge.dji.extract data/clip.MP4 -o data/clip_frames --num-images 300
scripts/dev python -m modelforge.pipeline data/clip_frames                       # SfM, CPU is fine
MODELFORGE_GPU=1 scripts/dev python -m modelforge.pipeline data/clip_frames --stages dense
scripts/splat data/clip_frames                                                   # nerfstudio, GPU
```

## Outputs (in `data/<clip>_frames/`)

| Path | What |
|---|---|
| `images/`, `poses.json` | sampled frames (by travelled distance + view rotation), time-aligned priors, EXIF GPS |
| `sparse/refined/` | refined COLMAP model (poses, intrinsics, points) |
| `transforms.json`, `sparse_pc.ply` | nerfstudio dataset (ENU world) |
| `dense/fused.ply` | dense point cloud |
| `dense/meshed-delaunay.ply`, `dense/textured-delaunay/` | dense mesh, textured mesh (`mesh.ply` + `texture.png`) |
| `splat/splat_enu.ply` | Gaussian splat in the same ENU frame as the mesh |
| `report.json` | statistics of every stage |

## Requirements

- Docker with Compose.
- For dense + splat: an NVIDIA GPU with the NVIDIA Container Toolkit
  (`compose.gpu.yaml`). The pose refinement itself runs on CPU.

More detail: `docs/architecture.md` (design), `docs/djmd_format.md` (DJI
metadata), `CLAUDE.md` (development notes, results, conventions).
