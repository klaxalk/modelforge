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
│   ├── dji/            # djmd decoder, SRT parser, frame extraction with time-aligned priors
│   ├── sfm/            # features, pairs, priors, refine (prior BA), robust (self-calibrating priors)
│   ├── export/         # nerfstudio transforms.json, splat_to_enu
│   ├── gui/            # viser app + data layer
│   ├── dense.py        # COLMAP MVS stages
│   └── pipeline.py     # CLI driving all stages
├── resources/
│   ├── colmap/      # git submodule: github.com/colmap/colmap (SfM / MVS, C++)
│   ├── nerfstudio/  # git submodule: github.com/nerfstudio-project/nerfstudio (Python)
│   └── pyceres/     # git submodule: github.com/cvg/pyceres (full Ceres bindings)
├── docker/Dockerfile  # dev image (COLMAP + pycolmap + pyceres)
├── compose.yaml       # `dev` service; compose.gpu.yaml = CUDA dev image + `nerfstudio` service
├── scripts/dev        # run a command in the dev container (MODELFORGE_GPU=1: CUDA image)
├── scripts/splat      # nerfstudio splatfacto train + export + ENU conversion
├── scripts/process    # video -> extract -> sfm -> (GPU) dense + splat, end to end
├── README.md          # human quick start
├── tests/             # pytest (19 tests): prior BA, robust priors, dense chain, splat, GUI
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
- Largest position deviations (2–5 m) are in the last ~10 s. That is not a
  landing: the drone hovers at about 11 m and the gimbal tilts up to −4°, so
  the views are near-horizontal and see mostly distant content. Camera
  position is weakly constrained there, and the priors hold it.
- Runtime: matching 698 s (CPU, 2398 pairs) dominates. Refinement takes 83 s.

## Robust priors (M3), `modelforge/sfm/robust.py`

A `PriorModel` is self-calibrated after every round, starting from round 1.
Round 0 still has the default focal length (2× off) and 40 px thresholds, so it
runs with absolute priors only. Each update does, in order:
1. **systematic**: boresight B = median camera-frame rotation residual;
   time offset δt = trimmed least squares of position residual on velocity.
   Both are estimated against the raw priors over *all* frames, then folded
   into the priors. This must come first: a big bias otherwise gets every
   prior flagged as an outlier before it can be estimated.
2. **calibrate**: rescale each block (abs rot, abs pos, rel rot, rel pos) so
   the median whitened d² = χ²₃ median. Clamped to ×2 per round, with floors.
3. **flag**: priors beyond χ²₆ at 99.9% lose their term in the next round.

Relative priors (`RelativePosePriorCost`) link time-consecutive images.
- COLMAP's cost is rigid in the camera frame: the prior displacement gets
  rotated by camera j's *absolute* attitude error. So
  σ_pos = a + b·baseline with b ≈ 0.008 (velocity) + 0.0175 (1° heading).
- σ_rot = a + c·turn, because 10 Hz attitude interpolation error grows with
  the rotation between images (~2% of the turn).
- Residuals are re-implemented in numpy and verified equal to the Ceres
  functors (`robust.residuals`).

Results on `berlin_3_1` (CLI flags `--no-relative`, `--no-calibration`,
`--no-systematic` for ablations):

| | points / obs | err median / p90 | prior dev pos median / max |
|---|---|---|---|
| M1, fixed σ | 42,487 / 185.3k | 0.505 / 1.214 px | 0.76 / 2.17 m |
| M3, no relative | 42,498 / 185.3k | 0.505 / 1.215 px | 1.17 / 2.72 m |
| **M3 full** | **42,987 / 187.6k** | **0.486 / 1.138 px** | **0.77 / 1.78 m** |

- Calibrated absolute σ: 0.63 m horizontal, 0.42 m vertical, 0.33° roll/pitch,
  0.98° yaw. Relative σ: 0.12° + 0.8% of the turn, 0.017 m + 1.4% of the
  baseline.
- B = 0.86° (mostly heading), δt = 11 ms.
- 3 absolute priors rejected, one of them in the final near-horizontal
  hover segment.
- 69 of 299 relative priors rejected. The tail is heavy and non-Gaussian, even
  for small turns.

Known limitations:
- Calibration is biased low in directions the images constrain weakly
  (synthetic test: about 0.7× the true σ). Fixing this needs redundancy-aware
  variance component estimation.
- Rounds 1–2 converge slowly with relative priors (~500 iterations). After
  100 iterations the median camera has settled to 4.6 cm, but the weakly
  constrained near-horizontal frames at t ≥ 515 s still move up to 3.6 m /
  3.8°, so the iteration cap must stay. Cost reaches 0.25% of final by
  iteration 100.
- Pair gating was **not** implemented, by measurement. Geometries are
  uncalibrated (F only), and the ~2.5% false pairs are already removed by
  triangulation filtering.

Tests: `tests/test_m3_robust.py` (synthetic ground truth). Calibration recovers
injected σ, a 3 m / 10° outlier is rejected and the image-only pose recovered
to 0.03 units, a 2° boresight comes back within 0.11°, and δt = 300 ms comes
back as 304 ms.

## Dense mesh (COLMAP MVS) and Gaussian splat (nerfstudio)

The end products are a **dense mesh** and a **Gaussian splat**, both in the
project's ENU frame, so they line up with each other and with the refined
cameras.

**Dense** (`modelforge/dense.py`, pipeline stage group `dense`, Run tab in the
GUI), run in the CUDA container:
`MODELFORGE_GPU=1 scripts/dev python -m modelforge.pipeline PROJECT --stages dense [--mesh delaunay|poisson|both] [--dense-max-image-size N]`

| Stage | Output in `PROJECT/dense/` | Device |
|---|---|---|
| `undistort` | `images/`, `sparse/` (PINHOLE) | CPU (8 s for 300 images, 464 MB) |
| `stereo` | `stereo/{depth,normal}_maps` (PatchMatch, 20 sources/view from the refined tracks) | **CUDA only**, clear error otherwise |
| `fusion` | `fused.ply` + `fused.ply.vis` | CPU |
| `mesh` | `meshed-<m>.ply`, `preview-<m>.ply` (≤300k faces, COLMAP `mesh_simplifier`, for the GUI) | CPU |
| `texture` | `textured-<m>/mesh.ply` + `texture.png` (COLMAP `mesh_texturer`) | CPU |

- **Delaunay** is the default. It is visibility-based, keeps open surfaces, and
  runs through the `colmap` CLI because it has no pycolmap binding. Poisson
  closes surfaces, and its trim step (10) deletes everything in sparse clouds.
- Fusion picks overlapping images from the **sparse tracks**: with no shared
  sparse points, nothing fuses. Fusion output type must be `"ply"` to get the
  `.vis` file Delaunay needs.
- A perfectly planar scene crashes Delaunay's graph cut (degenerate CGAL
  triangulation). Real scenes are fine, and the synthetic test uses relief.
- Untested here: PatchMatch on the real clip (no GPU). Expect a long run: 300 ×
  1913×1074 images, 4 maps each, about 10 GB of depth/normal maps.
  `--dense-max-image-size 1200` shrinks both time and size.

**Splat**: `scripts/splat PROJECT [ns-train args]`, on a host with an NVIDIA GPU.
1. Trains `ns-train splatfacto` in the official
   `ghcr.io/nerfstudio-project/nerfstudio:1.1.5` image (the `nerfstudio`
   service in `compose.gpu.yaml`, run as the host UID, viewer on :7007).
   splatfacto loads our `sparse_pc.ply` (`load_3D_points=True` in its method
   config).
2. Exports with `ns-export gaussian-splat` → `PROJECT/splat/splat.ply`, in
   nerfstudio's normalized frame.
3. Converts with `python -m modelforge.export.splat_to_enu` →
   `PROJECT/splat/splat_enu.ply`, using `dataparser_transforms.json`
   (x_ns = s·(R x + t)). Positions, log-scales and wxyz quaternions are mapped
   back, and the SH bands are rotated (degree ≤ 3, D_l by least squares over
   sampled directions).

Tests:
- `tests/test_splat_to_enu.py` checks that positions, covariances and
  view-dependent colour render identically.
- `tests/test_dense_chain.py` runs exact depth/normal maps of a bumpy ground
  through fusion (≤1.6 cm error), Delaunay (0.6 mm median), Poisson
  (1.9 mm), texturing and preview.
- `tests/test_gui_dense.py` covers the GUI layers and runs a dense stage from
  the GUI.

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
  - **View**: snapshot picker and stats, including the calibrated prior model.
    A "Dense / splat" folder shows `fused.ply` (≤1.5M points), the preview
    mesh and `splat_enu.ply` (viser gaussian splats), each enabled once its
    file exists. Layers: points, cameras with
    thumbnails, prior cameras, prior→refined segments, trajectories. Cameras
    can be coloured by position or rotation deviation, reprojection error, or
    time.
  - **Inspect**: click a camera to see its image, with observations in green,
    residuals × gain in red, and unmatched keypoints in grey.
  - **Plots**: Plotly charts of deviation vs. video time and a point-error
    histogram.
  - **Run**: pairs, match, refine (live), export, and the dense stages (a
    "Run dense (all)" button plus one per stage, with a mesh-method picker),
    each run in a background thread. Refine uses `pipeline.stage_refine(progress=...)`. BA iteration
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
