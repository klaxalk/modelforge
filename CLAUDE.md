# CLAUDE.md

This file guides Claude Code when working in this repository. The repo root is
`/home/klaxalk/git/modelforge`.

## Project goal

**modelforge** aims to be a better nerfstudio pipeline for DJI drone footage.
The current pipeline is video → frames → COLMAP sparse reconstruction →
nerfstudio. It throws away the flight metadata that DJI drones record, and
modelforge fixes that. There are three workstreams:

1. **Metadata-aware frame extraction.** Extract frames from DJI videos together
   with per-frame metadata: 3D position, 3D orientation of the aircraft and
   gimbal, camera settings and timestamps. Write it in a form that COLMAP and
   nerfstudio can consume.
2. **Pose refinement, a standalone tool.** Start every frame at its prior pose,
   then triangulate and bundle-adjust with 6-DoF absolute and relative prior
   terms. COLMAP stays **unmodified** and is used through pycolmap. Full
   design: `docs/architecture.md`.
3. **Web GUI (viser)** to watch and control each stage, including live
   refinement. Output is a nerfstudio dataset (`transforms.json` +
   `sparse_pc.ply`).

## Layout

```
modelforge/
├── data/            # Input footage and outputs. Contents are gitignored (only data/.gitignore is tracked)
├── docs/            # architecture.md (design + milestones), djmd_format.md
├── modelforge/      # Project Python code (deps: numpy, opencv-python, pillow; ffmpeg on PATH)
│   └── dji/
│       ├── djmd.py     # djmd track decoder → per-frame CSV (stdlib only)
│       ├── srt.py      # SRT sidecar parser
│       └── extract.py  # frame extraction with time-aligned 6-DoF pose priors
├── resources/
│   ├── colmap/      # git submodule: github.com/colmap/colmap (SfM / MVS, C++)
│   ├── nerfstudio/  # git submodule: github.com/nerfstudio-project/nerfstudio (Python)
│   └── pyceres/     # git submodule: github.com/cvg/pyceres (full Ceres bindings)
├── docker/Dockerfile  # dev image (COLMAP + pycolmap + pyceres)
├── compose.yaml       # `dev` service; compose.gpu.yaml = CUDA override
├── scripts/dev        # run a command in the dev container
├── tests/             # pytest: M0 prior-BA check, GUI smoke test
├── build/             # gitignored local build logs
├── .gitmodules
└── CLAUDE.md
```

- Run `git submodule update --init --recursive` after cloning.
- Treat `resources/*` as read-only upstream code. COLMAP is reused through
  pycolmap (built from the submodule). Custom residuals are added to its Ceres
  problem from Python (pyceres) instead of patching C++. `resources/colmap/AGENTS.md` has
  upstream's own notes for agents.

## DJI metadata sources

The sample clip is `data/berlin_3_1.MP4`, recorded on a DJI NEO2: HEVC
1920x1080 at 29.97 fps, 8:46 long, about 1.2 GB. Metadata lives in two places:

- **`.SRT` sidecar**, one entry per frame (about 33 ms apart):
  - `FrameCnt`, wall-clock timestamp
  - camera settings: `iso`, `shutter`, `fnum`, `ev`, `ct`
  - position: `latitude`, `longitude`, `rel_alt`, `abs_alt`
  - `pp_target` / `pp_current`: 4-vectors that look like quaternions, probably
    stabilization/EIS (meaning not yet verified)

  The SRT carries **no explicit aircraft or gimbal attitude**.
- **MP4 data tracks**:
  - stream `0:1`, `djmd` ("CAM meta"): one protobuf packet per frame, holding
    the camera, aircraft (10 Hz) and gimbal (10 Hz) records. **Decoded** by
    `python -m modelforge.dji.djmd X.MP4 -o X_djmd.csv`, which needs only
    ffmpeg/ffprobe. The full field map is in `docs/djmd_format.md`.
  - stream `0:2`, `dbgi` ("CAM dbgi"): debug info, high bitrate, still
    undecoded.

Key facts (details in `docs/djmd_format.md`):
- Attitude is in 0.1° units. Velocity is NED in m/s. Aircraft yaw is a true
  heading.
- The SRT `pp_current` quaternion (w, x, y, z) is the gimbal/camera attitude.
- ⚠️ **The djmd aircraft and gimbal data lead the images by about 0.7 s (~20
  frames)**. The SRT `pp_current` is in sync. Time-align before using anything
  as a pose prior.
- The NEO2 gimbal **cannot yaw** (pitch only, earth-stabilized). So the camera
  heading is the aircraft yaw. The reported gimbal yaw is aircraft yaw −36.5°
  in an offset reference.
- djmd shutter is the exact exposure. The SRT rounds it to nominal stops.
- Altitude: use `rel_alt` (above takeoff). A local vertical reference is fine
  for this project. `abs_alt` is not plain MSL.

## Frame extraction (workstream 1)

`python -m modelforge.dji.extract X.MP4 -o OUT [--num-images N | --spacing X] [--sampling distance|time] [--start s --end s]`
writes:
- `OUT/images/frame_NNNNN.jpg`: one image per sampling target, the sharpest
  frame (Laplacian variance) within ±`window`/2 of the spacing. NNNNN is the
  true video frame index. EXIF GPS holds the aligned position, with
  altitude = `rel_alt`.
- `OUT/poses.json` / `poses.csv`, one entry per image:
  - WGS84 position, and local ENU position (origin = first image)
  - camera attitude
  - raw aircraft and gimbal values
  - `q_world_from_cam` (ENU ← OpenCV camera)
  - COLMAP `qvec`/`tvec` (cam_from_world)
  - an `alignment` report

How the extractor works:
- It estimates the telemetry lead per clip. The SRT method (`pp_current` vs.
  gimbal yaw, 0.1-frame grid) is preferred. Optical flow is the fallback, then
  a default of 20.7 frames. `--lead` overrides all of them.
- Each 10 Hz stream is rebuilt from its sample-and-hold runs, interpolated, and
  read at `frame − lead`.
- **Sampling** (default `distance`) places images at equal steps of a
  cumulative curve: ds = |v|·dt + w·Δ(optical-axis angle).
  - The default is N = 300 images, so spacing = total / N. `--spacing` gives
    the step in m directly.
  - Translation comes from integrated aligned velocity, which is exactly 0
    when hovering, so GPS jitter never accumulates.
  - w (m/°) credits pure rotation. The default is the median look-at depth ×
    π/180, assuming flat ground at takeoff height. `--rot-weight 0` gives pure
    distance.
  - `--sampling time` uses the same N/spacing interface on video time.
- Camera attitude = yaw from the aircraft + pitch/roll from the gimbal.
  `--yaw-source gimbal` exists only for models whose gimbal can yaw.

Results on the sample clip:
- Lead 21.6 frames (0.72 s), median yaw residual 0.05°.
- Pitch/roll vs. `pp_current`: median 0.17°.
- Frame indices verified exact against ffmpeg `select`.
- Full clip, default settings: 300 images in about 1.5 min (2869 m path,
  3754° of view rotation, w = 0.60 m/°).
- Comparison at N = 300:
  - Pure distance leaves gaps of up to 130° of view rotation in turns.
  - With the rotation credit, the worst gap between consecutive images is 30°
    and 17 m.
  - Time sampling also misses turns (105°), and 7% of its images land while
    hovering.

## Pipeline (M1, headless)

`scripts/dev python -m modelforge.pipeline PROJECT [--stages features,pairs,match,refine,export]`
works on an ingest output folder. There is deliberately **no stock-COLMAP
comparison**. Stages and artifacts:

| Stage | Artifact | Notes |
|---|---|---|
| `features` | `features.db` | Upright SIFT (gimbal keeps roll ≈ 0), one shared OPENCV camera. Matching works on a copy. |
| `pairs` | `pairs.txt` | Time neighbours (±5) plus look-at-point overlap (optical axis ∩ takeoff ground plane, axes within 60°). Catches revisits. |
| `match` | `database.db` | `match_image_pairs` on the pair list. |
| `refine` | `sparse/round_N[_tri]`, `sparse/refined` | Every frame registered at its prior, then 3 rounds of (triangulate with poses fixed → BA with free poses + `AbsolutePosePriorCost`). Thresholds 40 → 12 → 4 px. No gauge fixing. |
| `export` | `transforms.json`, `sparse_pc.ply` | nerfstudio, world = ENU, no `applied_transform`. |

Stats for every stage go to `report.json`. Use `point_error_px`
(median/p90) rather than the mean: right after BA, before filtering, a few
degenerate points make the mean meaningless.

Results on `berlin_3_1` (300 images, CPU):
- 300/300 registered, 43k points, track length 4.35, point error median 0.51
  px / p90 1.23 px.
- Focal length converges from the default 2304 px to **1085 px** (HFOV ≈ 83°).
  The 2161.78 djmd value is not fx at 1920 px.
- Deviation from priors: position median 0.76 m (p90 1.28 m), rotation median
  1.05° (p90 3.0°).
- Systematic residuals are small. Mean camera-frame offset 0.74° (heading bias
  −0.73°). The time offset from position-vs-velocity regression is 9 ms
  (R² 0.01), which **independently confirms the 0.72 s lead**.
- Rotation scatter (~1°) doesn't correlate with the EIS correction (r = 0.05)
  and only weakly with yaw rate (r = 0.11).
- Largest position deviations (2–5 m) are in the last ~10 s (landing).
- Runtime: matching 698 s (CPU, 2398 pairs) dominates. Refinement takes 83 s.

## GUI (M2, viser)

`scripts/dev python -m modelforge.gui.app PROJECT [--port 8080]`, then open
http://localhost:8080. compose publishes 8080. viser 1.1.1 and plotly 7.1.0 are
pinned in the image.

Code layout:
- `modelforge/gui/data.py`: a `Project` turns priors, snapshots (`priors`,
  `sparse/round_N[_tri]`, `refined`) and thumbnails into numpy. Poses are
  world_from_cam, the same convention as viser frustums (OpenCV, wxyz =
  scalar-first quaternion of R_wc).
- `modelforge/gui/app.py`, with four tabs:
  - **View**: snapshot picker and stats. Layers: points, cameras with
    thumbnails, prior cameras, prior→refined segments, trajectories. Cameras
    can be coloured by position or rotation deviation, reprojection error, or
    time.
  - **Inspect**: click a camera to see its image, with observations in green,
    residuals × gain in red, and unmatched keypoints in grey.
  - **Plots**: Plotly charts of deviation vs. video time and a point-error
    histogram.
  - **Run**: pairs, match, refine (live), export, each run in a background
    thread. Refine uses `pipeline.stage_refine(progress=...)`. BA iteration
    events move the existing frustums (throttled to 4 Hz), and each round
    reloads the snapshot.

Things to know:
- Points far outside the flight area are hidden, and the cloud uses float32.
  Unfiltered BA output contains points 100+ km away, which overflow viser's
  default float16.
- Live refine takes about 2× the headless time (171 s vs 83 s). The scene
  updates hold the GIL during Ceres' Python callbacks.
- Triangulation is RANSAC-based, so runs are not bit-identical. Medians are
  stable, extremes vary.
- Tests: `tests/test_gui_smoke.py` runs the app logic headlessly on
  `data/berlin_3_1_frames` (skipped if absent).

## Reusing COLMAP

The full table is in `docs/architecture.md`. The short version:
- C++ `PosePrior` / `PosePriorBundleAdjuster` handle position only, but
  pycolmap binds `AbsolutePosePriorCost` and `RelativePosePriorCost` (6-DoF,
  optionally covariance-weighted).
- `create_default_ceres_bundle_adjuster(...).problem` exposes the Ceres problem.
  Prior terms and callbacks go in through the full `pyceres` package. The
  minimal pyceres bundled in pycolmap can't add residuals.

## Environment (Docker)

**All development runs in the `dev` container.** Use `scripts/dev <cmd>`,
which starts the container if needed, or `scripts/dev` alone for a shell.
- The repo is mounted at `/workspace`, and `PYTHONPATH=/workspace`.
- Port 8080 is exposed for viser.
- The container runs as the host UID, so files written to `data/` are not
  root-owned.

The image is defined in `docker/Dockerfile` and `compose.yaml`. Rebuild with
`env UID=$(id -u) GID=$(id -g) docker compose build dev`.
- Ubuntu 24.04, system Ceres 2.2, Python 3.12 venv at `/opt/venv`.
- COLMAP is built from `resources/colmap` into `/opt/colmap` (CPU, no Qt GUI,
  no ONNX).
- pycolmap and pyceres are built from the submodules, both with
  **pybind11==3.1.0** and `--no-build-isolation`. That's what lets pycolmap's
  `ceres::Problem` and cost functions be used through pyceres. The build fails
  if both can't be imported.
- `resources/pyceres` is pinned at upstream `main` (`2eeeca7`, v2.7). The v2.6
  tag does not compile with pybind11 3.1.
- Changing a submodule means rebuilding the image (ccache cache mounts keep it
  fast).

**GPU:**
- This machine currently has **no NVIDIA device on PCI**. `lspci` shows only an
  AMD Radeon 680M, and the nvidia modules exist only for older kernels. So
  everything is CPU-only.
- `compose.gpu.yaml` gives a CUDA build: `nvidia/cuda:12.8.1-devel`,
  `CUDA_ENABLED=ON`, `gpus: all`. Use it with `MODELFORGE_GPU=1 scripts/dev ...`.
- Nerfstudio training will need a GPU host and is not in this image yet.

Tests: `scripts/dev pytest -q tests`.

## Conventions

- Default branch is `main`, but work is currently on `master`.
- Never commit large files. Frames, databases, sparse models and checkpoints go
  under `data/`.
