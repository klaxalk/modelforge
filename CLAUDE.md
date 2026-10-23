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

## Status (2026-10-17) and handoff

**Read this first if you are a new Claude instance on another machine.** The work
was developed on a laptop without an NVIDIA GPU and is being continued on a GPU
machine. Everything is committed and pushed to `origin/master`. Nothing lives
outside the repo except `data/` (videos and project folders; copy separately)
and the per-machine Claude memory, which holds nothing project-specific.

Done:
- M0–M3: prior-initialized refinement with self-calibrating robust priors.
- M5: dense mesh (COLMAP MVS) and Gaussian splat (nerfstudio splatfacto).
- Multi-video ingest: sources = video + time range, per-source offsets.
- Web GUI with two docked panels:
  - left (`gui.add_panel()`): Projects / Videos / Ingest / View / Inspect /
    Export / Settings (dark mode);
  - right (main): run state, **Run full pipeline**, **Stop**, Sparse /
    Splat / Splat mesh / Dense, history;
  - two windows floating at the bottom of the 3D view: the log (left) and the Hardware monitor (right).
  - Projects: list, open, create, rename, delete; final vs. intermediate
    results; clean.
  - Videos: a file manager over the shared video library: folders (create,
    delete), upload into any folder, clip info and used-by, add to project,
    download, rename, delete.
  - Sparse / Dense / Splat: run each part; live state above the tabs,
    history and log below, hover hints.
  - Export: mesh (glb / obj / ply) and splat (ply / .splat), saved under
    `export/` or downloaded in the browser.
- **One image for everything** (`compose.gpu.yaml`): COLMAP (CUDA) +
  modelforge in `/opt/venv`, and nerfstudio + torch cu128 + gsplat in
  `/opt/nerfstudio`. The separate nerfstudio container and the Docker socket
  are gone.

**Verified on the GPU machine** (RTX 3090, sm_86) on `data/berlin_test`, a
60 s piece of `berlin_3_1` (`--start 105 --end 165 --num-images 80`):
- The CUDA image with nerfstudio builds. The self-check prints `torch
  2.7.1+cu128 cuda 12.8 gsplat 1.4.0 prebuilt kernels True`. Three fixes were
  needed:
  - gsplat installs from its git tag: the PyPI sdist lacks the `glm`
    submodule (`glm/glm.hpp: No such file`).
  - `pillow<12` in the nerfstudio venv: its `pil_to_numpy` uses a Pillow
    internal that changed in 12.
  - `splat.py` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for nerfstudio:
    torch >= 2.6 refuses its checkpoints (numpy scalars) by default.
- SfM: 80/80 registered, error median 0.52 px, focal 1089 px. SIFT
  extraction and matching run on the GPU (matching 1.7 s for 417 pairs).
- Dense at `--dense-max-image-size 640`: 221k fused points, Delaunay mesh and
  texture. 864 s, of which stereo is 845 s.
- Splat, 2000 iterations: 328k Gaussians, 53 s for train + export + ENU.
  ICP of `splat_enu.ply` onto `fused.ply` leaves < 10 cm horizontally,
  0.1°, scale 0.999, so `splat_to_enu` is right. A 5–13 cm vertical offset
  remains that shrinks away from the centre: MVS at 640 px sits on the
  vegetation tops.
- **The whole pipeline from the GUI works on the GPU** (run by the user):
  ingest of 305–332 s (20 images) → SfM (20/20, error median 0.39 px) →
  dense at 640 px (stereo 170 s, 43.5k fused points, mesh, texture) → splat
  2000 iterations (204k Gaussians, 40 s) → View overlay. The GUI tests also
  pass on `berlin_test` (`MODELFORGE_TEST_PROJECT`).
- That image was built with a local override (`CUDA_ARCHITECTURES=86`,
  `GSPLAT_ARCH_LIST=8.6`, `JOBS=8`). The committed full arch list has not
  been built.

Still open on the GPU:
- PatchMatch speed is explained and the default is now the fast `low` preset
  (see "Quality presets"). Full-resolution dense on the full clip is still
  untested.
- The full test suite, and the GUI tests that need `berlin_3_1_frames` (not
  copied to this machine; rebuild it with extract + SfM).

Tested only partially:
- **Multi-video ingest.** Tested on synthetic data and on two pieces of the
  one sample clip (true offset 0, estimate 3 cm). **Not yet on two different
  flights.**

Open items by priority (high / medium / low; the user sets them; the TODO docs follow this list):

**High**
- (none open) Done 2026-10-23: **the loop-closure pose graph** (~37 min per revisit pass on kladno_dira 5681): the
  reference BLAS (now OpenBLAS) + CGNR instead of sparse Cholesky, dira2k loop graphs 368 s → 1–3 s with the same
  revisit split: `docs/todo_posegraph_speed.md`.

**Medium**
1. **TODO: cheap splat quality fixes** (exposure from djmd shutter / ISO / f-number, bilateral grid,
   anti-aliased rasterization, the depth prior outliers): `docs/todo_splat_quality.md`.
2. ~~Calibration biased low (~0.7×)~~ **closed 2026-10-23, by measurement**: the 0.7× was one synthetic draw
   (seed 1: its prior noise itself came out 0.81× nominal). A redundancy-aware estimate (Helmert VCE, tr(H) of the
   prior blocks by one perturbed re-solve, Hutchinson; `build/vce_proto.py`) found tr(H) 1–13 of 120 (images pin
   the cameras, the priors keep ~all their redundancy), also with 60 points per scene; median rule vs VCE on seeds
   1–4: 0.67–1.07 vs 0.79–1.00 of the true σ, each within the draw's own scatter. Not implemented (an extra BA
   solve per round for nothing measurable).
3. **TODO: robustness of long jobs** (heavy jobs out of the GUI process, memory estimates before heavy stages;
   **resume refine done** 2026-10-23: Sparse tab "Resume refine", CLI `--resume-refine`, a checkpoint after every
   round / chunk / pass in `sparse/refine_resume.pkl`): `docs/todo_robustness.md`.
4. **TODO: job queue with completion notices** (queue stages of any project with their settings, browser
   notification when done): `docs/todo_job_queue.md`.

**Low**
1. Run the full clip (all 300 images) through dense and splat (quality
   `low`, then higher if wanted).
2. Test multi-video ingest on real separate flights of one site.
3. The final ~10 s of `berlin_3_1` (t ≥ 515 s, a near-horizontal hover) is
   weakly constrained. Trim it with the source range.
4. M4 (accuracy): prior-guided matching, the EIS warp, rolling shutter.
5. **TODO: multi-camera DJI projects** (several drones / lenses in one project; zooming clips refused; one
   common training resolution). Plan and decisions: `docs/todo_multicamera.md`.
6. **TODO: level of detail for large splats** (merged coarse levels per tile / octree cell, chosen per camera
   under a Gaussian budget): `docs/todo_splat_lod.md`.
7. **TODO: normalize the lighting** (sunny vs overcast; sun visibility from the SfM mesh + per-photo gains in the
   splat plugin; waiting for the user's mixed-lighting project): `docs/todo_lighting.md`.
8. **Idea, not now: splat editing** (sky floaters, fog, selection, non-destructive delete):
   `docs/todo_splat_editing.md`.
9. **TODO: covisibility tiles too large** (kladno_dira 5187 photos: 479–2434 per tile, 2.5x slots; cap the extra
   photos per tile, balance the split): `docs/todo_tile_size.md`.

The user's working preferences, as seen so far:
- They want results committed and pushed when asked ("commit and push").
- They value a GUI that clearly shows state and progress, with hover hints.
- Don't run the full test suite unprompted while they use the machine. They
  stopped long test runs twice. Run the targeted tests for what changed, or
  ask.
- Mounting the host Docker socket into the container was refused, as a
  containment escape. Keep everything inside one container.

## Migrating to a GPU machine

The code is on GitHub: `origin` = `git@github.com:klaxalk/modelforge.git`,
branch `master`. `data/` and `build/` are gitignored and must be copied
separately.

1. Get the code:
   - `git clone --recurse-submodules -b master git@github.com:klaxalk/modelforge.git`
     (or `git submodule update --init --recursive` after a plain clone).
   - Pinned submodules: colmap `48af6d30`, nerfstudio `50e0e3c7` (v1.1.5+),
     pyceres `2eeeca7`. pyceres must stay on that commit, because v2.6 doesn't
     build with pybind11 3.1.
2. Copy the data (under `data/`). Minimum is `berlin_3_1.MP4` (1.2 GB) and
   `berlin_3_1.SRT`. Optional:
   - `berlin_3_1_frames/` (2.1 GB): the validated single-video project. It
     skips the ~17 min CPU run of extract + SfM.
   - `berlin_pieces/`: the two-source null test.
   - `site/`: the user's GUI trial project.
   - Databases and `dense/` can be regenerated, so you can leave them behind.
   - Keep file ownership as the host user (the container runs as host UID/GID).
3. Prepare the host:
   - Install an NVIDIA driver, Docker with Compose v2, and the **NVIDIA
     Container Toolkit**. Then run `sudo nvidia-ctk runtime configure --runtime=docker`
     and restart Docker.
   - Check with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
4. Build the image:
   `env UID=$(id -u) GID=$(id -g) docker compose -f compose.yaml -f compose.gpu.yaml build dev`
   - Result: `modelforge-dev:cuda`, ~20 GB. Takes roughly 45–90 min: COLMAP
     with CUDA, then torch + gsplat kernels + nerfstudio.
   - Architectures: COLMAP's `CUDA_ARCHITECTURES` is set in `compose.gpu.yaml`
     to `75;80;86;89;90;120` (Turing to Blackwell). gsplat's
     `GSPLAT_ARCH_LIST` (Dockerfile) is the same list in torch notation.
     Trim both to your GPU for a faster build.
   - Lower `JOBS` (default 6) if the build runs out of RAM.
   - The CPU image (`docker compose build dev`) is optional. It has no
     nerfstudio. `scripts/dev` without `MODELFORGE_GPU=1` uses it.
5. Use `MODELFORGE_GPU=1` for every `scripts/dev` call on that machine. The CPU
   and CUDA variants share the `dev` service name, so stop one before starting
   the other: `docker compose down`.

### GPU test checklist

```bash
MODELFORGE_GPU=1 scripts/dev python -c "import pycolmap; print(pycolmap.has_cuda)"   # True
MODELFORGE_GPU=1 scripts/dev /opt/nerfstudio/bin/python -c \
  "import torch, gsplat; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
MODELFORGE_GPU=1 scripts/dev pytest -q tests          # all pass except the no-CUDA error test (skipped)
# dense on the existing refined project (or build a project in the GUI)
MODELFORGE_GPU=1 scripts/dev python -m modelforge.pipeline data/berlin_3_1_frames --stages dense --mesh delaunay
#   default quality is superfast (320 px, fast PatchMatch); --dense-quality low|medium|high|max for more
scripts/splat data/berlin_3_1_frames --iterations 2000   # quick splat check; viewer on :7007
MODELFORGE_GPU=1 scripts/dev python -m modelforge.gui.app   # :8080; Projects, Dense, Splat, Export tabs; View → Dense / splat
```

What to check:
- `report.json` gets `dense.*` and `splat` entries.
- `dense/fused.ply` has millions of points.
- `dense/meshed-delaunay.ply` and `textured-delaunay/` exist.
- `splat/splat_enu.ply` overlays the mesh and the cameras in the GUI. A
  misalignment means a bug in `splat_to_enu` or in how the dataparser
  transform is read.
- Expected size: about 10 GB of depth/normal maps at full resolution.
- In the CUDA image, SIFT extraction and matching should also run on the GPU
  (pycolmap `Device.auto`). Matching took 698 s on CPU. Re-measure it.

If splat training fails in the nerfstudio venv:
- Likely suspects: gsplat kernels (they rebuild by JIT into `~/.cache` if
  the prebuilt ones don't load), torch/torchvision versions, or an
  `ns-train` CLI flag. `modelforge/splat.py:train_command` builds the
  command.
- Pins to adjust: `TORCH_VERSION`, `TORCHVISION_VERSION`, `TORCH_INDEX_URL`
  and `GSPLAT_ARCH_LIST` in `docker/Dockerfile`.

## Layout

```
modelforge/
├── data/            # Input footage and outputs. Contents are gitignored (only data/.gitignore is tracked)
├── docs/            # architecture.md (design + milestones), djmd_format.md
├── modelforge/      # Project Python code (deps: numpy, opencv-python, pillow; ffmpeg on PATH)
│   ├── dji/            # djmd decoder, SRT parser, per-video extraction, multi-video ingest
│   ├── video/          # ingest for videos without metadata (project kind "video"): motion sampling
│   ├── rosbag/         # ROS 2 bags (kind "rosbag"): bag reading, TF buffer, ingest (poses, camera_info, LiDAR)
│   ├── sfm/            # features, pairs, priors, refine (prior BA), robust (self-calibrating priors), chunked (large projects: chunks + pose graph), reconstruct (incremental SfM)
│   ├── export/         # nerfstudio transforms.json, splat_to_enu, products (Export tab), ckpt_dump (live splat)
│   ├── gui/            # viser app; tabs: projects.py, ingest.py; data.py (numpy layer)
│   ├── dense.py        # COLMAP MVS stages
│   ├── splat.py        # splatfacto train + export + ENU (runs /opt/nerfstudio CLIs)
│   ├── splat_plan.py   # splat training plan from the cameras (area, GSD, image count) and GPU memory
│   ├── splat_tiles.py  # large flights: splat per tile (cells by what the photos see), cropped + merged
│   ├── evaluate.py     # splat quality on held-out images (PnP pose + export/splat_eval.py)
│   ├── semantic.py     # semantic search: C-RADIOv3 features per Gaussian (sidecar) + text queries
│   ├── subset_sweep.py # calibration sweep of image subsets (every N, keyframes) against held-out quality
│   ├── splat_mesh.py   # mesh from the splat: TSDF fusion of rendered depth (export/splat_tsdf.py in /opt/nerfstudio)
│   ├── nsplugin/       # nerfstudio plugin run in /opt/nerfstudio: splatfacto-lidar (LiDAR depth loss), mf_train (checkpoint on Stop, resume fix)
│   ├── projects.py     # project folders: list / create / status / clean
│   ├── jobs.py         # stopping jobs: Token (check / cancel / run killable subprocesses)
│   ├── videos.py       # shared video library: list / probe / usage / upload / rename / delete (Videos tab)
│   └── pipeline.py     # CLI driving all stages
├── resources/
│   ├── colmap/      # git submodule: github.com/colmap/colmap (SfM / MVS, C++)
│   ├── nerfstudio/  # git submodule: github.com/nerfstudio-project/nerfstudio (Python)
│   └── pyceres/     # git submodule: github.com/cvg/pyceres (full Ceres bindings)
├── docker/Dockerfile  # dev image (COLMAP + pycolmap + pyceres [+ nerfstudio venv when NERFSTUDIO=ON])
├── compose.yaml       # `dev` service (CPU); compose.gpu.yaml = CUDA + nerfstudio, ports 8080 + 7007
├── scripts/dev        # run a command in the dev container (MODELFORGE_GPU=1: CUDA image)
├── scripts/splat      # wrapper: modelforge.splat in the CUDA container
├── scripts/process    # video -> extract -> sfm -> (GPU) dense + splat, end to end
├── README.md          # human quick start
├── tests/             # pytest: prior BA, robust priors, multi-source, projects, dense, splat, export, GUI
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

The sample clip is `data/videos/berlin/berlin_3_1.MP4` (tests find it anywhere in
`data/` via `tests/sample_clip.py`, or `$MODELFORGE_TEST_VIDEO`), recorded on a DJI NEO2: HEVC
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

**The SRT is optional.** Everything the pipeline uses comes from the MP4's
djmd track: position, altitude, velocity, attitude and camera settings. The
SRT only helps two things:
- **Lead.** The SRT method takes 21.6 frames, instantly. Without an SRT,
  optical flow gives 21.5 (corr 0.546, just above the 0.5 threshold) in
  ~50 s, else the default 20.7.
  - `ingest._cached_flow_lead` stores the flow result in the video's folder
    (`.modelforge_leads.json`, keyed by size + mtime, so it survives
    renames), so a clip pays it once.
- **Start time** (orders sources, places each video on the project
  timeline). `ingest._wall_start` returns UTC for every clip.
  - The SRT timestamp is local time (ms); the MP4 `creation_time` is UTC
    (1 s). On the sample clip they differ by exactly 2 h.
  - With both, the SRT's zone offset (rounded to 15 min) is removed.
  - Without an SRT, the MP4 tag is used. This fixed a 2 h skew when videos
    with and without SRT were mixed.

**Other models (2026-10-18): DJI Mavic 4 Pro.** The user's `dira.MP4` failed with "object of type
'float' has no len()": the decoder took the gimbal record for the aircraft record.
- Records are now found by the type in their header (1 camera, 2 aircraft, 3 gimbal; NEO2 3.2/3.4/3.5,
  Mavic 4 3.2/3.3/3.4). The camera and gimbal fields are chosen per layout. NEO2 output is
  unchanged.
- `djmd.model_name` reads the model ("DJI NEO2", "DJI Mavic4 Pro L3A"); `Clip.model`.
- **Yaw source `auto`** (new default in Settings, CLI and GUI; existing ingest.json keep their value),
  `Clip.yaw_source`: by model whether the gimbal can yaw. NEO → aircraft yaw; Mavic 4 (360° gimbal,
  true-heading gimbal yaw) → gimbal yaw; unknown models → gimbal unless its yaw is over 20° from
  the aircraft's. poses.json records `yaw_source_used` per source.
- **Lead** (no SRT, optical flow), `extract.estimate_lead_flow`:
  - Long clips are sampled in 4 windows of 45 s (`flow_windows`), each estimated on its own. The
    lead is the median over windows with |corr| ≥ 0.5 and a best lead inside the search range
    (−15…45 frames; a negative lead means the telemetry trails the images).
  - Why: decoding a 33 min 4K clip in full took over 10 min, and one correlation over a whole
    clip is weak. On the NEO2 sample it was 0.55, while single minutes reach −0.99 and others break
    down in fast turns.
  - NEO2 sample: 22.0 frames (SRT: 21.6) from 20 s of decoding. Mavic 4 `dira.MP4`: 0.5 frames in
    125 s, cached per clip.
  - `Clip.telemetry` and `sampling_curve` clamp at both ends now (negative leads read ahead).
- Tests: `tests/test_djmd_models.py` (both layouts from hand-built packets, the auto yaw rules,
  the windows).

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

## Project kinds

`ingest.json` holds `"kind"` (`dji.ingest.KINDS`; missing = `dji`). The kind is
chosen when the project is created (Projects tab "Kind",
`projects.create(kind=)`) and picks the front end and the sparse method. Dense,
splat, export and the View tab are shared: they only need images + poses +
intrinsics in one frame.

| | `dji` | `video` (phone, action cam, …) |
| *(kind `rosbag`: see "Kind rosbag" below)* | | |
|---|---|---|
| Ingest back end | `modelforge/dji/ingest.py` (telemetry, lead) | `modelforge/video/ingest.py` (same interface: `load_clip`, `make_plan`, `preview`, `run`) |
| Frame spacing | distance + rotation credit, or time | **motion** (cumulative dense optical flow on a 160 px copy, in image widths; cached in the video's folder as `.modelforge_motion.json`), or time |
| poses.json | priors per image | no poses (`priors.load` gives NaN priors; `has_poses` False) |
| Features | upright SIFT, OPENCV camera | oriented SIFT (handheld footage rolls), **RADIAL** camera (one focal length) |
| Pairs | look-at overlap from priors | `pairs.sequential`: exhaustive ≤150 images, else ±10 neighbours + quadratic jumps + a coarse grid between sources |
| Sparse | `refine` (prior BA) | `reconstruct` (`sfm/reconstruct.py`: incremental / global / hierarchical mapper, largest model) |
| World | ENU, metric | up-oriented, centred, median scene depth = 10 units (no metric scale) |

Video-kind notes:
- **Up.** `reconstruct.up_alignment` tries, in order: the plane of the camera
  path; the dominant RANSAC plane of the points (the ground, if within 60° of
  the cameras' up, so a facade is not taken for it); the cameras' mean up (−y).
  - On berlin 105–117 s treated as a phone video (a straight path), the ground
    cue gave camera pitch −19.9° against the gimbal's −20.1°.
  - The mean camera up alone would have been ~20° off, since a drone looks
    down.
- **RADIAL camera.** OPENCV's separate fx / fy drifted to 1938 / 1262 px on 20
  frames. RADIAL gives f = 1085.2 px, matching the DJI refinement's 1085.
  `nerfstudio.opencv_params` converts the simpler models for the export.
- **Overlap.** SfM without priors needs it.
  - 20 images over 60 s (spacing 0.46 image widths) registered only 3. Over
    12 s, all 20 registered.
  - The Ingest tab warns when the motion spacing exceeds 0.25 image widths.
  - 120 images over 60 s: 120/120 registered, 0.45 px, 156 s.
- **Portrait clips.** `extract.video_size` returns the size as decoded:
  ffmpeg applies the display-matrix rotation, so ±90° swaps w / h. Before,
  portrait phone video would have been read with the wrong frame size.
- **Mapper** (Sparse tab "Mapper", CLI `--mapper`; `reconstruct.MAPPERS`):
  - **incremental** (default) runs in pycolmap: Stop through its
    `CancellationToken`, and progress shows each registered image.
  - **global** (GLOMAP in COLMAP) and **hierarchical** run the `colmap`
    CLI (`global_mapper` / `hierarchical_mapper`) through `jobs.Token.run`:
    pycolmap offers them without a Stop hook. Their log lines are the
    progress detail.
  - On 20 images (berlin 105–117 s):

    | Mapper | Registered | Error median | f | Time |
    |---|---|---|---|---|
    | incremental | 20/20 | 0.419 px | 1085.4 | 17.6 s |
    | global | 20/20 | 0.419 px | 1085.0 | 6.4 s |
    | hierarchical | 17/20 (split into clusters) | 0.388 px | 1087.4 | 19.7 s |

  - Hierarchical is for thousands of images.
- **Features and Matching options** (Sparse tab folders, for both kinds;
  CLI `--features --max-features --upright/--no-upright --pairs --neighbors
  --ratio --guided --min-inliers`):
  - Extractor: `sift` (GPU) or `sift-robust` (DSP-SIFT + affine shape, CPU).
    On 20 images it gave +32% inlier matches and +33% points at 24×
    the extraction time (1.0 → 23.9 s).
  - Max features (8192). Upright (default: on for DJI, off for video).
  - Pairs: `auto` (DJI priors / video exhaustive ≤150 else sequential),
    `sequential` (with Neighbours) or `exhaustive`.
  - Ratio test (0.8), Guided matching, Min inliers (15).
  - Matching now gets Stop through pycolmap's `CancellationToken`.
    pycolmap reports a cancel as `InterruptedError`, which matching and
    incremental mapping turn into `jobs.Cancelled`; before, Stop would have
    shown as a failure.
  - Learned features / matching (ALIKED, LoMa, LightGlue) exist in this
    COLMAP but need `ONNX_ENABLED=ON`. Selecting ALIKED in the current image
    **aborts the process**, so they are not offered.
  - COLMAP also has `pose_prior_mapper` (incremental with GPS priors).
    Relevant for phones with per-frame GPS or the ROS 2 kind, not for
    videos without metadata.
- **One camera.** As for DJI, a project is one camera (sources must share a
  resolution). Per-video intrinsics for different phones are future work.
- **GUI.**
  - The Ingest tab hides lead / rotation credit / yaw source / the 3D flight
    plan.
  - The Sparse tab shows reconstruct instead of refine and the BA controls.
  - The View tab skips prior frustums, lines and deviations.
- Tests: `tests/test_video_kind.py`.
### Kind `rosbag` (ROS 2 bags; ROS 1 bags too)

**ROS 1 (Noetic) `.bag` files** (the user asked, 2026-10-22: the same requirements as ROS 2) go through the same kind
and code: rosbags' `AnyReader` reads both; ROS 1 bags carry their message definitions and get the ROS 2 type names
(`sensor_msgs/msg/Image`) and times (sec / nanosec). Differences handled: `bag.is_bag` / `find_bags` take `.bag` files
(`BAG_FILE_EXTS`), CameraInfo spells `K` / `D` in ROS 1 (`camera_info` reads either), frame ids with a leading "/"
are stripped (the TF buffer already did). The library lists / uploads / renames `.bag` like `.mcap`; GUI labels say
"ROS bag". Tests: `test_small_bag[ros1]` (a Noetic bag written by rosbags' ROS 1 writer: Header seq, K / D / R / P,
"/"-prefixed frames; poses to 1e-6, camera_info, extraction, LiDAR), `test_rosbag_project_through_the_gui[ros1]`
(the berlin-derived bag as ROS 1: extract 20, SfM), `test_bag_uploads` (a .bag upload). rosbags' Noetic type store
lacks `tf2_msgs/TFMessage`: the tests register it for writing (real bags carry it).

Built while the user was away (2026-10-18), from their spec: image topic
(usually compressed), a world frame (TF world → camera), or an odometry topic
for the pose, plus optional point cloud topics (several) used for sparse, dense
and the splat.

- **Reading.** The `rosbags` library (`rosbags==0.11.8`, a late Dockerfile
  layer) reads sqlite3 and MCAP bags, no ROS install.
  - `modelforge/rosbag/bag.py`: inventory, TF, image stamps, decoding
    (CompressedImage, raw rgb8 / bgr8 / mono8 / mono16), camera_info,
    odometry, PointCloud2 (x / y / z / intensity fields of any type).
  - `modelforge/rosbag/tf.py`: a TF buffer. Static + dynamic edges, lookups
    along the tree, linear translation + slerp interpolation, None outside a
    dynamic edge's range (±0.1 s).
- **Settings.** `Settings.ros` (new optional field, older ingest.json still
  load). `ros_defaults` guesses:
  - the first CompressedImage topic and its CameraInfo;
  - the TF root above the camera as world;
  - TF poses when a dynamic edge reaches the camera, else odometry;
  - all PointCloud2 topics.
- **Poses** (`rosbag/ingest.py` `Poser`):
  - `tf`: `T_world_cam = lookup(world, camera_frame, stamp)`;
  - `odometry`: `T_world_odomframe(TF) · T_odomframe_child(interpolated)
    · T_child_cam(TF)`.
  - The camera frame is the optical frame (OpenCV), the image header
    frame_id by default.
- **Images and priors.** Spaced by distance + rotation credit (0.15 m/°
  default) or time, the sharpest per window.
  - poses.json carries priors in the DJI format (`enu_*` = world xyz,
    finite-difference velocity, `ground_z` from the LiDAR's 2nd percentile
    z), so refine runs unchanged (`sfm_stages("rosbag")` = DJI stages).
  - camera_info → `camera.json`: plumb_bob → OPENCV (k3 dropped),
    equidistant → OPENCV_FISHEYE; it is scaled if the image size differs.
  - `features.extract(camera_params=)` uses it, and refine passes
    `refine_intrinsics=False` (`pipeline.known_camera`). Upright defaults off.
- **LiDAR.**
  - **Ingest:** each cloud topic's scans in the source's range (≤400 scans
    per topic) are transformed into world at their stamps, range-cropped,
    voxel-downsampled (5 cm) and merged into `lidar/cloud.ply`.
  - **Sparse:** `refine` reports `lidar_check`, the SfM points' distance to
    the LiDAR over the covered points (nearest within 2 m; OpenCV FLANN).
  - **Dense:** `dense.add_lidar` after fusion (Dense tab "Use LiDAR points").
    - Every LiDAR point gets the cameras that see it (in front, inside the
      frame, occlusion-checked against the geometric depth maps) as `.vis`
      entries, up to 5 and at least 2.
    - Index = position in the dense model's `reg_image_ids()`, as COLMAP's
      `mvs::Model` numbers images.
    - Colour from the nearest camera, normal towards them. The image-only
      cloud stays as `fused_mvs.ply`.
  - **Splat:** `export.lidar.merged_cloud` puts LiDAR points, coloured from
    the nearest seeing camera (no occlusion), into `sparse_pc.ply`, which
    splatfacto / splatfacto-big start from (Splat tab "Start from LiDAR
    points"; re-exports before training).
- **Test bag.** `tests/rosbag_fixture.py` builds a realistic one from
  `data/fullrun_test` + the clip:
  - images 960×540 @ 5 Hz, camera_info (refined intrinsics, plumb_bob);
  - `/tf` world → base_link 30 Hz (DJI telemetry poses), static base_link →
    camera_optical / lidar / lidar2, world → odom (offset, rotated);
  - `/odom`, and two LiDARs from the fused cloud.
  - The user's `data/videos/synthetic_bag` (306–326 s) and the project
    `data/rosbag_exp` were made with it.
- **Measured on the test bag** (40 images, superfast dense):
  - Poses from TF and from odometry match the truth to 1e-15 m.
  - SfM: 40/40, 0.26 px median, intrinsics kept fixed; priors moved 0.24 m
    (telemetry error).
  - SfM vs LiDAR (70% covered): median 8.5 cm, about the test cloud's
    spacing.
  - Dense: +260,911 LiDAR points (98% passed the visibility test); the
    Delaunay mesh has 76k faces vs 38k image-only.
  - sparse_pc.ply: 10k SfM + 266k LiDAR points.
- GUI:
  - Projects "Kind: ROS 2 bag".
  - The Ingest tab lists bags; a "ROS 2 bag" folder holds topic / frame
    pickers, a checkbox per cloud topic, voxel and range. 3D shows the track
    and a frustum preview.
  - Upload is hidden (bags are folders).
- Tests: `tests/test_rosbag_kind.py` (TF, a tiny in-test bag, the GUI flow on
  the berlin-derived bag).
- **Splat depth supervision** (the user asked; 2026-10-18): `splatfacto-lidar` /
  `splatfacto-big-lidar`, a nerfstudio plugin in `modelforge/nsplugin/`.
  - **Nothing installed.** `splat._env` puts `modelforge/nsplugin` on
    PYTHONPATH and registers the methods and the `lidar-data` dataparser
    through `NERFSTUDIO_METHOD_CONFIGS` / `NERFSTUDIO_DATAPARSER_CONFIGS`. So
    plugin edits need no image rebuild.
    - The dataparser is its own module (`mf_lidar_dataparser.py`):
      nerfstudio discovers dataparsers while `base_datamanager` is half
      imported, so importing the model there is a circular import.
  - **Dataparser:** `nerfstudio-data` + transforms.json
    `lidar_ply_file_path` (written by the export when `lidar/cloud.ply`
    exists), mapped like `ply_file_path` into `metadata["lidar_xyz"]`.
  - **Model:** after splatfacto's render (depth on during training), the
    LiDAR points are projected into the training camera: the undistorted one
    the datamanager hands over, at the current training resolution, with the
    optimized pose.
    - Per pixel the nearest return is kept, then dropped if it lies more than
      5% behind the nearest return in a 5 px window. This removes returns
      seen through the gaps of a sparse cloud.
    - Loss: weight × mean(min(|D − z| / z, 1)), with D = expected depth.
      Relative, so near and far count alike; the clamp bounds wrong returns.
    - Why projecting on the fly instead of nerfstudio's `depth_file_path`:
      nerfstudio only *crops* depth images when it undistorts, so they would
      misalign; on-the-fly projection is exact, follows the resolution
      schedule and the camera optimizer, and writes no files.
    - The cloud is a non-persistent buffer, so checkpoints stay splatfacto's
      (live preview, ns-export).
  - **Report:** every 500 steps the plugin prints `[lidar] step N: depth error
    median … p90 … within 5% …`. `splat.run` stores it in
    `report.json["splat"]["lidar_depth"]`; the History line shows it.
  - **Use:** Splat tab "LiDAR depth weight" (ROS 2 bag projects, default
    0.2, 0 = plain splatfacto), or `scripts/splat PROJECT --lidar-depth 0.2`.
    Weight 0 through the CLI still trains the -lidar method, which then only
    reports the error.
  - **Measured** on `data/rosbag_exp` (40 images, 3000 steps, both starting
    from the LiDAR-initialized cloud):

    | | depth error median / p90 | within 5% | eval PSNR / SSIM / LPIPS | train |
    |---|---|---|---|---|
    | weight 0 (report only) | 4.4% / 82% | 53% | 19.88 / 0.440 / 0.380 | 87 s |
    | **weight 0.2** | **0.58% / 2.4%** | **96%** | **20.26 / 0.452 / 0.372** | 115 s |

    The test bag's LiDAR comes from the MVS fused cloud, so it agrees with
    the images by construction. Real LiDAR (noise, motion, other time) is
    untested.
  - Tests: `test_splat_lidar_command`,
    `test_lidar_depth_image_in_nerfstudio_venv` (occlusion filter, method
    registration).
- **Uncalibrated camera_info / camera not in TF** (the user's `franta` bag, 2026-10-19: a downward
  camera on a UAV; images stamped `uav64/rpi_camera_front`, a frame missing from TF; camera_info all
  zeros, size 0 × 0):
  - `bag.camera_info` skips messages without a calibration (size or focal ≤ 0); none left = no
    camera.json, so SfM estimates the intrinsics. Extraction used to die with a division by zero.
  - **Extrinsic RPY** (Ingest → ROS 2 bag: "Use extrinsic RPY" + vector; `Settings.ros["camera_rpy"]`,
    None = the camera frame is the optical frame): the Camera frame is then a mount frame in ROS body
    convention (x forward, y left, z up) and `rosbag.ingest.mount_rotation` gives R_frame_optical =
    Rz(yaw) Ry(pitch) Rx(roll) · FORWARD_OPTICAL (extrinsic, about the frame's fixed axes; 0, 0, 0 =
    looking forward, image top up; 0, 90, 0 = straight down, image top to the front).
  - On franta with Camera frame `uav64/fcu`: without RPY the optical axis pointed straight up
    (median z +1.00), with 0, 90, 0 straight down (−1.00), 2216/2216 poses. But 0, 90, 0 gave SfM
    **0 points** (focal drifted to 8390 px, k2 −3.7) despite 4.5M inlier matches: the image's top
    points to the back on that drone. **0, 90, 180** (found by the user) works. A wrong mount by a
    multiple of 90° shows exactly like that: great matches, no triangulated points, a runaway camera.
    The refine History line now says so: < 100 points with ≥ 200 inlier matches per camera → "the pose
    priors do not fit the images" (ROS 2 bag: check Camera frame / Extrinsic RPY, e.g. yaw 180; DJI:
    yaw source / lead); fewer matches → "too little overlap" as before.
  - The Ingest tab warns when the images' frame is not in TF.
  - Extraction streams: `_sharpest` yields a window's sharpest frame as soon as decoding passes the
    window's end. Before, it decoded every window first and held all best frames (franta: 200 × 12 MP
    ≈ 7 GB in the GUI process), so the progress sat at "decoding up to the first selected frame" and
    then jumped to done. Test: `test_sharpest_streams_per_window`.
  - Tests: `test_extrinsic_rpy` (the test bag's forward mount reproduced from `base` + 0, 0, 0),
    `test_uncalibrated_camera_info`.
- **Images-only bags** (pose source **none**; the user asked, 2026-10-19): no TF to the camera and no
  odometry. `ros_defaults` picks it when neither reaches the camera (empty topics don't count).
  - Sampling: **motion** (`image_motion`: Farneback flow on 160 px grey frames from
    `bag.iter_gray_small`, OpenCV reduced JPEG decode; cached in `.modelforge_motion.json` next to
    the bag, keyed by size + mtime + topic) or time; distance falls back to motion.
  - poses.json records carry no pose (as for videos), LiDAR is skipped (it needs poses), no 3D preview.
  - `pipeline.sparse_kind(project)`: a rosbag project without priors reconstructs like a video
    (features oriented / RADIAL unless camera_info is calibrated, sequential pairs, incremental
    mapper, up-oriented frame); the GUI's Sparse tab, the full pipeline and the CLI use it.
  - Ingest tab: Pose source none hides the frame / odometry / mount fields; the preview shows frame +
    time.
  - Checked on `synthetic_bag` forced to none: 20 images extracted in 0.9 s, reconstruct 18/20,
    0.26 px. Test: `test_images_only_bag`.
- Not done yet:
  - per-point LiDAR timestamps (motion distortion within a scan);
  - LiDAR as a BA constraint (point-to-plane residuals on SfM points);
  - fisheye images through dense / splat (export supports
    OPENCV_FISHEYE, untested);
  - multiple cameras per bag.

## Frame extraction (workstream 1)

A project can be built from **one or more videos**. Each **source** is a video
plus a time range. The same video can appear several times, as several pieces.
Use the GUI Ingest tab, or the CLI:
`python -m modelforge.dji.extract A.MP4 [B.MP4 ...] -o PROJECT [--num-images N | --spacing X] [--sampling distance|time] [--start s --end s] [--overwrite]`.
`--plan ingest.json` takes the sources and settings from a plan file instead.
The code is split in two:
- `modelforge/dji/extract.py`: per-video machinery (streams, lead, sampling
  curve, sharpest-frame selection) and the CLI.
- `modelforge/dji/ingest.py`: `Source`, `Settings`, `make_plan`, `preview`,
  `run`. It combines sources into one project.

Output:
- `PROJECT/images/<video stem>_NNNNN.jpg`: one image per sampling target, the
  sharpest frame (Laplacian variance) within ±`window`/2 of the spacing.
  NNNNN is the true video frame index. EXIF GPS holds the aligned position,
  with altitude = `alt_m`. Older single-video projects use `frame_NNNNN.jpg`,
  and the loaders don't care.
- `PROJECT/poses.json` / `poses.csv`, one entry per image:
  - `source` (index into `sources[]`), `frame`/`pts` (per video), `time_s`
    (project timeline)
  - WGS84 position, and local ENU position
  - camera attitude
  - raw aircraft and gimbal values
  - `q_world_from_cam` (ENU ← OpenCV camera)
  - COLMAP `qvec`/`tvec` (cam_from_world)
  - top level: `sources[]` (alignment, alt offset, sampling per source), `plan`
    (copy of ingest.json at extraction), `ground_z`, `enu_origin_lla`
- `PROJECT/ingest.json`: the editable plan (sources + settings). The GUI saves
  it on every edit.
- Re-extracting needs `--overwrite` (or the GUI checkbox). Old images and
  everything derived from them (databases, sparse, dense, splat, report) are
  **moved** to `PROJECT/previous/<time>/`, never deleted.

Multiple sources:
- **Order.** Sources are ordered chronologically by their UTC start (MP4
  creation_time, refined to ms by the SRT; else list order). The first one is the reference: the ENU origin is its first image,
  and z comes from its `rel_alt`.
- **Altitude.** Other flights are lifted by the difference of their takeoff
  altitudes, median(`abs_alt` − `rel_alt`). On the sample clip that value is
  −31.011 m with zero spread, i.e. constant per flight. Pieces of the same
  flight get 0.
- **One spacing for all.** S_total over all sources / N, so images spread in
  proportion to each piece's distance + rotation. w is one value for the
  project.
- **One camera.** Sources must share one resolution (a single COLMAP camera);
  `run` refuses otherwise.
- **No cross-source chains.** Relative priors and time-neighbour pairs stay
  within a source. Look-at overlap pairs connect the sources.
- **Per-source offsets.** Refinement estimates a position offset per source
  for each flight's GNSS bias; see "Robust priors".

How the extractor works (per video):
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

Results on the sample clip (single source):
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

**Images used: a subset of the extraction** (Sparse tab folder, `gui/subset.py`;
`projects.apply_subsample`; the user asked, 2026-10-19: extracting a large video is expensive, so
extract once, then run the pipeline on subsets of it, in the same project):
- Keep every N (per source, capture order), Starting at (offset), Time range (project timeline).
- The first time, `images/`, `poses.json`, `poses.csv` move / copy to `PROJECT/extracted/` (the
  pool). `images/` then holds hard links to the selected frames and `poses.json` / `poses.csv` the
  selected entries, so every stage runs unchanged on the subset. poses.json records `subsample`.
- Results of the previous image set (features, matches, sparse, transforms, dense, splat, splatmesh,
  nerfstudio, export, images_N, report) move to `previous/<time>/` (`ingest.archive_outputs(names=)`);
  then run the pipeline again. Keep every 1 over the whole range uses all images again.
- `extracted/` is in `ingest.OUTPUTS` (re-extraction archives it with the rest) and kept by Clean.
  Disk sizes in the Projects status count the hard-linked frames in both folders.
- A first version made a new project per subset; the user wanted it inside the project, so it was
  replaced.
- **Keyframes from measured overlap** (the user asked for an automatic, non-uniform choice, DJI / ROS;
  `modelforge/sfm/overlap.py`):
  - **Analyze extracted images** (optional job, stage `analyze`): features of all extracted images +
    matches of each with the next 16 of its source, in `PROJECT/analysis/` (kept when a subset is
    applied, archived on re-extraction, deleted by Clean). `analysis/overlap.json` = inliers per pair,
    keypoints per image. Overlap(a, b) = inliers / min(keypoints): no scene geometry needed.
  - `select_keyframes`: per source, jump to the farthest of the next 16 images with overlap ≥
    min_overlap to the current keyframe and ≥ ½ of that to the keyframe before (≈ 3 views per point);
    none → the next image. Levels extra dense / dense / normal / sparse = **0.50 / 0.30 / 0.20 / 0.12**, default
    **dense** (2026-10-22, the user asked: shifted one step denser; before dense / normal / sparse = 0.20 / 0.12 /
    0.06, so the calibration table below uses the old names: its "dense" is today's normal, its "normal" today's
    sparse, and its "sparse" (0.06) is gone).
  - **Default: keyframes, density dense** (the user asked, 2026-10-22; `gui/subset.py DEFAULT_MODE`): a project that
    never chose a subset (none recorded in poses.json) starts on keyframes, so Run full pipeline runs the overlap
    analysis and the keyframe selection first. A chosen every N-th is kept, also every 1 over the whole range ("all
    images": `apply_subsample` now records it instead of nothing).
  - GUI: Selection "every N-th image" / "keyframes", Keyframe density; the preview shows the
    neighbours' overlap (weakest / 5% / median) for either selection once the analysis exists.
  - Progress: "features: image k / n · name", then "matching: pair k / n" (the match stage shows
    "pair k / n" too: `pipeline._watch_matches` polls database.db's matches table, as features.db is
    polled for features).
  - **Run full pipeline applies the panel's choice first** (stage `images`, `SubsetPanel.apply_selection`; 2026-10-22:
    switched to every N-th, the pipeline stayed on the keyframes, as only keyframes were applied): keyframes as below,
    or every N-th when it differs from the images in use (else nothing moves).
  - **Every N-th after keyframes** (moric, 2026-10-22: "now using 1 of 400 images"): in keyframes mode the panel never
    set Keep every / Starting at / Time range, so switching to every N-th used the slider's placeholder 0-1 s, and
    that range got stored. Now `refresh` always sets them, a stored range holding < 2 images opens on the whole
    flight, and fewer than 2 images cannot be applied. Test: `test_gui_every_nth_after_keyframes`.
  - **Run full pipeline with Selection = keyframes** starts with `analyze` (reused when
    `overlap.is_current`: overlap.json covers exactly the extracted images) and `keyframes`
    (`SubsetPanel.apply_keyframes`: applies the chosen density unless those images are in use; the
    report and priors are reloaded for the new set), then SfM → splat → splat mesh.
- **Calibration sweep** (`modelforge/subset_sweep.py`, `modelforge/evaluate.py`,
  `export/splat_eval.py`): a copy of a project with every 10th image held out; per configuration
  SfM → splat (fixed 30000 steps, ×2, no camera optimizer) → held-out evaluation. The held-out
  images are located in each subset's model by PnP (SIFT to the 6 nearest model images in time,
  2D-3D from their tracks, `pycolmap.estimate_and_refine_absolute_pose`), so they do not help the
  SfM; splat_eval renders them (pinhole, photo undistorted) and scores PSNR / SSIM / LPIPS, plus
  `*_view<i>.jpg` side by sides. Smoke run on fullrun_test (1500 steps, ×4): 8/8 located, poses
  visibly aligned; keyframes-normal (58 images) 21.07 dB vs every 2nd (36) 20.56 dB.
  - **Calibration on kladno_dopravnik** (2026-10-19; 80 extracted 4K images over 20 s, 8 held out,
    72 in the pool, 6000 steps at full resolution; the user rated 80 images good, 40 OK, 20 not
    converging):

    | run | images | PSNR | PSNR p10 | SSIM | LPIPS | neighbour overlap p5 | SfM |
    |---|---|---|---|---|---|---|---|
    | every 1st | 72 | 25.45 | 24.44 | 0.772 | 0.250 | 0.33 | 76 s |
    | keyframes dense | 57 | 25.50 | 24.47 | 0.774 | 0.252 | 0.28 | 49 s |
    | keyframes normal | 43 | 24.44 | 23.50 | 0.741 | 0.273 | 0.15 | 28 s |
    | every 2nd | 36 | 23.55 | 22.11 | 0.708 | 0.284 | 0.12 | 22 s |
    | keyframes sparse | 32 | 22.83 | 20.87 | 0.686 | 0.294 | 0.07 | 15 s |
    | every 3rd | 24 | 20.51 | 19.13 | 0.572 | 0.339 | 0.06 | 10 s |
    | every 4th | 18 | 17.75 | 16.72 | 0.403 | 0.409 | 0.02 | 6 s |

    The scores reproduce the user's judgement. Keyframes beat every-N at equal image count (normal
    +0.5 dB, sparse +0.3 dB over the every-N curve), and dense equals all images with 21% fewer. The
    weakest 5% neighbour overlap predicts quality: ≥ ~0.25 full, ~0.12–0.15 "OK", < 0.06 broken.
    Training time was the same for all (fixed steps, 265–280 s); fewer images saved SfM time. With
    the plan's 20 steps per image (min 30000), image count saves training time only above ~1500
    images. One site and 8 held-out photos (±0.3 dB); a karlak sweep (800 images) was started and
    stopped by the user.
- Too sparse a subset breaks SfM: kladno_dopravnik at every 8th image (10 of 80) matched 1,074
  inliers over 37 pairs and refine ended with **0 points** (focal drifted to 4940 px vs ~3200), and
  splatfacto then died with an `IndexError` on the empty sparse_pc.ply. Now `splat.run` refuses an
  empty sparse_pc.ply with a clear message, and the refine History line warns below 100 points.
- Tests: `tests/test_projects.py` (`test_subsample`, `test_gui_subsample`).

## Pipeline (M1, headless)

`scripts/dev python -m modelforge.pipeline PROJECT [--stages features,pairs,match,refine,export]`
works on an ingest output folder. There is deliberately **no stock-COLMAP
comparison**. Stages and artifacts:

| Stage | Artifact | Notes |
|---|---|---|
| `features` | `features.db` | Upright SIFT (gimbal keeps roll ≈ 0), one shared OPENCV camera. Matching works on a copy. |
| `pairs` | `pairs.txt` | Time neighbours (±5) plus look-at-point overlap (optical axis ∩ takeoff ground plane, axes within 60°). Catches revisits. On a re-run, also the pairs from look-ats on the terrain of the earlier model (see below). |
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
- **Terrain look-ats** (the user asked, 2026-10-21): the flat plane at takeoff height was far off on kladno_dira (a
  quarry pit: surface median −7.6 m, 2–98 % −27…+6 m vs −1.8 m): 2501 of 2883 look-at depths changed > 25 %, and
  drones inside the pit (below the plane) had their look-at clamped 1 m in front of the lens. `pairs.Terrain` =
  median z per grid cell (extent / 400, ≥ 0.5 m) of 3D points, holes from the nearest cell; `look_at_points(terrain=)`
  marches each optical axis to the first point at or below the surface. `stage_pairs` uses it when the project has
  a `sparse_pc.ply` with ≥ 1000 points (a re-run of SfM on these images) and adds those pairs to the flat ones
  (`report.json pairs.flat_pairs / terrain_points`). A fresh project has no points and keeps the flat plane.
  kladno_dira: +10,441 pairs (8,375 between passes > 60 s apart); 5,589 of them verify (≥ 15 inliers), 4,245 between
  passes with 1.09M inliers, against 7,084 cross-pass pairs / 1.57M inliers in the flat set (+60 % / +69 %);
  matching them took 48 s. Test: `test_multi_source.py::test_terrain_look_ats_pair_views_into_a_pit`.

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
   Cameras with fewer than 100 observations (`min_obs_to_reject`) are never
   flagged: they can't overrule their prior, and losing it lets them drift.

**Loss on the absolute priors is Huber** (scale 3σ). Relative priors stay
Cauchy.
- Why: Cauchy's pull fades with distance. A camera seeing few points, each in
  a 2-view track that slides with it, escaped 35 m in round 0 and stayed
  there. This happened on two pieces of `berlin_3_1`, frame 2998, 15
  observations.
- Huber's pull stays constant, and gross prior outliers are rejected by step 3
  anyway.
- Single-video `berlin_3_1`, Cauchy → Huber: error median/p90 0.517/1.250 →
  0.490/1.139 px, prior deviation max 1.81 → 1.77 m. Refine takes 121 → 181 s.

Relative priors (`RelativePosePriorCost`) link time-consecutive images.
- COLMAP's cost is rigid in the camera frame: the prior displacement gets
  rotated by camera j's *absolute* attitude error. So
  σ_pos = a + b·baseline with b ≈ 0.008 (velocity) + 0.0175 (1° heading).
- σ_rot = a + c·turn, because 10 Hz attitude interpolation error grows with
  the rotation between images (~2% of the turn).
- Residuals are re-implemented in numpy and verified equal to the Ceres
  functors (`robust.residuals`).

**Per-source offsets** (multi-video projects; `update_source_offsets`):
- When: after every round's BA, **starting with round 0**.
- What: one ENU position offset per source, added to that source's priors.
  The reference source (the one with the first image) stays at 0. This covers
  each flight's GNSS bias and any takeoff-height error.
- Bootstrap: biased videos first triangulate separate copies of the shared
  scene. While split points outnumber merged ones on a source pair, the
  offsets are solved by least squares on the source graph from
  median(X_a − X_b) over verified cross-source matches.
  `split_point_edges` reads the inlier matches from `database.db`. The
  cameras of that source are moved at once, so the next triangulation
  merges the copies.
- Split points are the **only** estimator. Once the copies merge, the
  leftover split points measure ~0 and the offsets settle.
- A per-source median prior residual was tried and dropped. On two pieces of
  one flight (true offset 0) it reported ~1 m that wandered 0.5 m per round,
  i.e. GNSS drift between the pieces, while the split points measured 6 mm.
- Real null test, `berlin_3_1` cut at 263 s into 2 sources: final offset
  3 cm, max prior deviation 2.7 m, error p90 1.165 px.
- δt estimation subtracts the offsets first.
- Synthetic test: a 0.6/0.4/0.3 shift is recovered to 2 cm in round 0, and
  round 1 has already merged.
- No per-source heading or time offset: the lead is estimated per video, and
  rotations are image-constrained.

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
- ~~Calibration is biased low in weakly constrained directions~~: not so (2026-10-23, see the priority list,
  medium 2): the synthetic 0.7× was a single random draw.
- **BA stops when the cameras settle** (`refine.Settle`), not at Ceres'
  convergence:
  - The rule: no camera centre moved > 0.5 mm, rotated > 0.001°, and no
    focal length changed > 0.01 px, measured across the last 10 iterations.
  - Why: COLMAP's solver options have `function_tolerance = 0` and
    `parameter_tolerance = 0`, so Ceres only stops on the gradient test
    (1e-4). The user saw minutes of BA with a flat cost and no visible pose
    changes.
  - Measured on berlin_test (80 images): the cameras are within 1 mm of
    final by iteration 80 / 10 / 9 (rounds 0 / 1 / 2), but Ceres ran 215 /
    274 / 345 iterations nudging points (−0.05–0.12% cost).
  - With the rule: 93 / 22 / 20 iterations. Refine 169 s → 42 s. Points,
    errors, prior deviations and focal are unchanged (focal within
    0.01 px).
  - The window displacement keeps a slow creeper going: the full clip's
    near-horizontal frames at t ≥ 515 s moved up to 3.6 m after iteration
    100 under the old cap. The 500-iteration cap stays as a backstop.
    Untested on the full 300-image clip.
  - `prior_ba(settle=None)` runs to Ceres' convergence. The report's
    termination says `SETTLED`.
  - The GUI shows BA progress against the cap: "iteration 120 / 500" live, "BA 120 / 500 iterations
    (SETTLED)" when a round ends (the `round_done` event carries `ba` and `max_iterations`; also in the
    log line), and the refine History line lists every round ("BA iterations 93, 22, 20 / 500"). The
    user asked. Test: `test_local_ba.py::test_refine_shows_iterations_of_the_cap`.
- **BA iteration cap and Skip** (Sparse tab, `stage_refine(max_iterations=,
  skip=)`, CLI `--ba-iterations`):
  - The cap applies per round; the default is 500, a backstop only.
  - **0 = no BA**: one triangulation at round 0's threshold, poses at the
    priors.
    - The user wanted 0 as the default ("the priors are super strong"). On
      berlin_test, 0 gives a reprojection error median 14.9 px / p90
      24.5 px, focal stuck at 2304 px (2× off) and track length 2.99. BA
      gives 0.42 / 1.10 px, focal 1082 px, in 8 s.
    - Shown those numbers, the user chose the default of 500, stopped by
      the settle rule.
  - **Skip refinement** (enabled while refine runs) sets an event. The
    running BA ends successfully with its current state (termination
    `SKIPPED`), the remaining rounds are skipped, and the final
    triangulation uses the last round's threshold. The pipeline then
    continues (export / splat / dense).
  - Ceres reports iteration 0 before any step, so a skip that early keeps
    the priors exactly.
  - Unlike Stop, Skip is not a failure and writes `sparse/refined`.
  - **Skip this BA round** (the user asked: the cameras visibly stop moving before BA ends): ends only
    the running BA with its current state (termination `SKIPPED_ROUND`, `skip_round` event in
    `refine._Progress` / `prior_ba`, cleared after the solve); the next round, chunk round or polish
    window runs as usual. Test: `test_m3_robust.py::test_skip_round_ends_only_the_running_ba`.
- **Large projects: chunked refinement** (the user asked, 2026-10-18: the 4000-image Mavic 4
  project `kladno_dira` ran > 30 min in round 0's BA alone, and it must work for 10k+ images).
  Sparse tab "BA mode" (auto / global / chunked), "Chunk size", "Chunk overlap", "Polish sweeps";
  CLI `--ba-mode --chunk-size --chunk-overlap --polish-sweeps`. `modelforge/sfm/chunked.py`,
  `pipeline._refine_chunks`.
  - **Default BA mode: auto = global if it fits in RAM, else chunked** (the user asked, 2026-10-23, after kladno_dira
    at 5681 images was OOM-killed at 30 GB in round 1 of a global refine; `pipeline.auto_ba_mode`, GUI and CLI):
    estimated peak = 1 GB + 0.7 GB per million inlier matches (report.json `match`; measured 0.49–0.72 on dira2k,
    berlin, kladno_dira 2883; 5681 images = 82.7M → ~59 GB), global if ≤ 80 % of MemAvailable at refine start. Without
    match stats: chunked from 2500 images (`Chunked.min_images`). The Sparse tab shows the choice under BA mode
    ("auto → global: needs ~12 GB …", every 5 s); report `refine.ba.auto`. Before: global (2026-10-22, chunks left
    revisits apart, fixed since: see below), auto = chunked from 2500 images (was 1000 until 2026-10-20).
  - A chunked refine (`stage_refine`):
    1. calibrates intrinsics + prior model on a subset (`refine.calibration_subset`: 5 runs of 60
       consecutive images spread over the flight; rounds 0–1, global BA; snapshots `calib_N`);
    2. cuts the capture order into chunks (500 images, 100 shared with the next) and runs each
       through rounds 1–2 **on its own**: only its images registered, so COLMAP loads only its
       matches (`DatabaseCache` filters by the registered images), intrinsics fixed, its own copy of
       the prior model (rejections are merged);
    3. a **pose graph** over all cameras, no points (`chunked.pose_graph`): edges i → i+1/4/16/64
       inside each chunk from its solution (σ 5 mm + 0.2 % of the baseline, 0.02°, Cauchy) plus the
       absolute priors (Huber, as in the BA). Shared cameras tie the chunks; ~6 parameters per
       camera, so 10k+ cameras take seconds;
    4. triangulates all images at those poses and polishes with one sweep of 300-image windows
       (`refine.local_ba(global_pass=False)`), which also brings in the matches between chunks
       (revisits); snapshot `posegraph`;
    5. the usual final triangulation with refined points.
  - Memory and time per chunk do not grow with the project. Still global: steps 4–5 (one
    triangulation of all images; COLMAP reads every pair's inlier matches, ~0.5 GB at 4000 images,
    and keeps the correspondence graph) and the pose graph. `report.json["refine"]["peak_rss_gb"]`
    records the peak. Revisits between chunks enter only through the polish and the final
    triangulation, not the pose graph.
  - Why not windows alone: a window holds its neighbours fixed, and since images bind neighbours
    in mm and the priors only in 0.6 m, it snaps onto their (prior) poses; more sweeps barely help
    (synthetic chain: global 2.1 cm, windows 6.9 cm, priors 8.6 cm). Chunks hold nothing fixed; the
    pose graph spreads the corrections.
  - Tried first, on kladno_dira round 1: a global BA on a thinned copy of the points (~300 per
    image, random, reprojection weighted ×9, stopping at 90 % of cameras settled) + windows gave the
    full global BA's image fit (0.586 vs 0.585 px; thinned pass alone 0.594) in 10 instead of 37
    min, but it is still a BA over all cameras, and the process peaked at ~20 GB of 31. Kept as
    `refine.local_ba` (the polish uses its windows) and `refine.thin_points`.
  - As the user noted, the GPS is not ground truth: absolute distance to it, or to another
    solution (~17 cm, smooth), is not the target; the image fit (final reprojection error) is.
  - **Also for all projects:**
    - Triangulation without COLMAP's point BA. `pycolmap.triangulate_points` ends with up to 5
      points-only global BAs (poses fixed): 10.5 of 12.5 min per triangulation on 4000 images,
      redone by the prior BA right after. Now `ba_global_max_refinements = 0` plus COLMAP's point
      filter called directly: 3.3 min. The final triangulation refines its points
      (`refine.refine_points`, poses + intrinsics constant). fullrun_test (80 images): final
      error 0.475 / 1.229 → 0.457 / 1.066 px, poses unchanged, refine 47 → 38 s.
    - `refine_points` solves in batches of 100k points (`POINT_BATCH`; a config with only
      variable points gives constant-pose residuals). One problem over all points needs ~0.9 KB
      per observation: on kladno_dira at 8000 images (~10M points) the GUI was OOM-killed at
      30 GB in the final refine_points, after 4 h of refine. Batched results are identical
      (fullrun_test: points within 1e-12); a batch peaks at ~0.5 GB.
    - Direct sparse Schur at any size: COLMAP's iterative Schur above 1000 images took 27–150 s
      per iteration on 4000 images, direct 11 s.
    - Relative priors skip pairs more than 10 images apart in capture order
      (`robust.MAX_ORDER_GAP`), so the calibration subset's blocks are not linked.
  - **Measured:** fullrun_test (80 images, chunks of 30, GUI test): final 0.465 / 1.072 px vs
    global 0.457 / 1.066. karlak (500 images, chunks of 150 / overlap 40 → 4 chunks): final
    0.640 / 1.296 px, 268k points, 262 s, peak RSS 2.7 GB (chunks 11–31 s each, pose graph 0.4 s,
    polish 30 s); its earlier global refine (before the triangulation changes, so not like for
    like) 0.693 / 1.605 px, 249k points, 358 s. Not yet run on 4000+ images end to end.
  - Tests: `tests/test_local_ba.py` (synthetic chain with local visibility: chunks + pose graph ≈
    global BA; windows alone stay near the priors; thinning; chunk / window cover; the GUI run).
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

**Dense** (`modelforge/dense.py`, pipeline stage group `dense`, Dense tab in the
GUI), run in the CUDA container:
`MODELFORGE_GPU=1 scripts/dev python -m modelforge.pipeline PROJECT --stages dense [--mesh delaunay|poisson|both] [--dense-quality low|medium|high|max] [--dense-max-image-size N]`

**Quality presets** (`dense.QUALITY`; the user wants the quickest dense by
default, so the default is `superfast`):

| Preset | Image size | PatchMatch | berlin_test, 20 views, stereo |
|---|---|---|---|
| **superfast** (default) | 320 px | `window_step 2`, `num_samples 7` | not measured (the user tests it) |
| low | 640 px | `window_step 2`, `num_samples 7` | 39 s, 38.2k fused points |
| medium | 1200 px | `num_samples 7` | |
| high | full | `num_samples 7` | |
| max | full | COLMAP defaults | 170 s at 640 px, 43.5k points |

Why PatchMatch was slow:
- Per pixel and sweep, it draws `num_samples` (15) source images by Monte
  Carlo sampling and evaluates 4 NCC costs over an 11×11 window. So the cost
  scales with `num_samples × window² / window_step²`, not with the number of
  source images (20 → 6 sources saved only 31%).
- During a run the GPU is at 100% and full boost clock (RTX 3090), so it is
  compute time, not a setup problem.
- Accuracy against the SfM points was the same for all of these settings
  (median depth error 0.56–0.63%). `window_step 2` costs ~12% of the fused
  points; `num_samples 7` alone costs nothing measurable and is 1.9× faster.
- Benchmark scripts: `build/pm_bench.py`, `build/pm_e2e.py` (gitignored).

| Stage | Output in `PROJECT/dense/` | Device |
|---|---|---|
| `undistort` | `images/`, `sparse/` (PINHOLE) | CPU (8 s for 300 images, 464 MB) |
| `stereo` | `stereo/{depth,normal}_maps` (PatchMatch, 20 sources/view from the refined tracks). Runs the `colmap patch_match_stereo` CLI (killable by Stop; "Processing view i / n" drives the progress) | **CUDA only**, clear error otherwise |
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
- Full resolution (`high` / `max`) on 300 × 1913×1074 images means about
  10 GB of depth/normal maps and a long run; untested on the full clip.

**Splat**: there are two ways to run it. Both call `modelforge.splat.run`
inside the CUDA dev container, where nerfstudio sits in its own venv at
`/opt/nerfstudio`:
- GUI: Splat tab → method, iterations (0 = auto), **Cache images on disk** (default on),
  **Downsample factor** (auto / 1 / 2 / 4 / 8, default auto), **Densification** (auto / default / mcmc),
  **Max gaussians** (0 = auto), **Camera pose refinement** (auto / off / SO3xR3 / SE3), then
  **Train + export splat**. The progress bar shows the training step, there is a Stop button,
  and the result is reloaded into the View tab.
- Shell: `scripts/splat PROJECT [--method splatfacto-big] [--iterations N] [--cache disk|cpu|gpu] [--downscale N] [--strategy default|mcmc] [--max-gaussians N] [--camera-opt MODE] [--no-plan] [ns-train args]`.
- **Training plan** (`modelforge/splat_plan.py`; the user asked, 2026-10-19, after a bad splat of a
  500 × 500 m world whose small parts trained fine). Auto values come from the cameras and the GPU:
  - **Why:** splatfacto pauses refinement for (training images + 100) steps after each opacity
    reset and resets every 3000 steps; gsplat refines only while `step % 3000 >= pause`. From ~2900
    training images on, **nothing is ever densified or pruned**. The 8000-image kladno_dira run
    showed it: 8,026,112 Gaussians at step 17,500 = its sparse cloud. Also, the means / camera
    learning rates decay over a fixed 30000 steps in nerfstudio's method config, and every eval
    image was rendered every 1000 steps.
  - `survey`: cameras from transforms.json (or, before SfM, the poses.json priors: image count
    only). **No ground plane** (the user: flights start in the air, terrain drops; arbitrary 3D
    space): the scene is a random 10k subset of `sparse_pc.ply`. Each point counts the cameras it
    projects into (in front, in frame, nearer than 4× that camera's median depth; no occlusion test)
    → overlap = mean count; GSD = median over cameras of median point depth / focal; surface = Σ
    footprints / overlap. Without sparse points the scene size is unknown.
  - **Iterations default 50000** (the user's choice, 2026-10-23; `splat_plan.DEFAULT_ITERATIONS`): the Splat tab's
    Iterations starts at it and the CLI uses it (per tile in a tiled run, as every explicit value); 0 = the auto rule
    below; `--no-plan` keeps splatfacto's 30000.
  - `suggest`: iterations = 20 × training images in [30k, 300k]; Gaussians needed = all image pixels
    / (overlap × (2 px × factor)²), rough (karlak ×1: 14M estimated, 2.0M grown by default densification);
    GPU fits ~0.35M per GB above 4 GB. **Since 2026-10-20 (the karlak investigation, below; the user asked):**
    - downsample auto = the smallest factor that trains at ≤ 2700 px width (`MAX_TRAIN_WIDTH`,
      `auto_downscale`): ×2 for 4K, ×1 up to 2.7K. Before, full resolution was the default (the user had
      preferred it); on karlak 4K was no better than ×2 even scored at 4K, in 2.4× the time. Pick 1 for it.
    - densification auto = **default** always, with the reset period max(3000, 2 × (training images + 100));
      **mcmc** only when chosen (budget = need, within the GPU estimate, ≥ the sparse points): it caused the
      near-camera fog. Before, auto switched to mcmc when the need exceeded the GPU (and splatfacto-big at half
      the need, `GROWTH`, removed).
    - **scene unit 10 m** (`WORLD_UNIT_M`, plan `scale_factor` 0.1 → `--auto-scale-poses False --scale-factor 0.1`;
      2026-10-20): nerfstudio's own unit is 1 / max |camera position|, and splatfacto's position learning rate and
      size thresholds are in it, so large flights trained differently (see "Splat on dira2k").
    - splitting stops at 1/2 (mcmc 5/6); camera optimizer SO3xR3. Explicit values win and the dependent ones
      follow. Untested: default densification without a cap on thousands of images (kladno_dira); if it runs
      out of GPU memory, pick mcmc.
  - **Per method:** the plan's flags apply to splatfacto, splatfacto-big and their -lidar variants
    (checked with nerfstudio's parser). splatfacto-big only differs by a lower split gradient (0.0005)
    and cull opacity (0.005), so it grows more Gaussians. Under mcmc both behave alike (no gradient
    threshold; cull 0.005 set).
  - Now on existing projects: fullrun_test (1080p) ×1, karlak / kladno_dopravnik / kladno_dira (4K) ×2,
    all default densification; kladno_dira 144,000 steps with the reset every 14,600.
  - `train_command(schedule=)` adds `--pipeline.model.strategy / max-gs-num / stop-split-at /
    reset-alpha-every / camera-optimizer.mode`, `--optimizers.{means,camera-opt,bilateral-grid}.scheduler.max-steps`;
    `--steps-per-eval-all-images 0` always. All flags checked against nerfstudio's own parser.
  - The Splat tab shows the plan and why (images, surface ha, overlap, GSD, Gaussians needed vs. GPU,
    steps); stats get `report.json["splat"]["schedule"]`. Validated on karlak (471 images) by the
    held-out experiments below; not on thousands of images.
  - Tests: `tests/test_splat_plan.py`.
- Disk cache = nerfstudio `--pipeline.datamanager.cache-images disk`: images
  stay on disk and are streamed by worker processes, for large datasets. Off
  means `gpu` (nerfstudio's default, which falls back to CPU above 500 images).
- Downsample N trains on `PROJECT/images_N/`, written by
  `splat.prepare_downscaled` (floor(w/N), as nerfstudio rescales the
  cameras). The factor is always passed explicitly (`nerfstudio-data
  --downscale-factor N`, last on the command line), because nerfstudio
  otherwise auto-picks a leftover `images_2/` for images wider than 1600 px.
  `images_N/` is an intermediate (Clean removes it; re-extraction archives it).
  - It decodes JPEGs at reduced size (PIL draft, DCT scaling by 1/2, 1/4, 1/8)
    in 8 threads, shows "downscaling images ×N: k / n" in the state line and
    honours Stop. Before, kladno_dira (8000 4K images) spent ~16 min here
    silently, at 117 ms per image, and Stop waited for it; now 5 ms per image.
  - Images are written as `.part.jpg`, then renamed, so a stopped run leaves
    no truncated image that the up-to-date check would keep.

**Stop, Pause and Resume** (the user asked, 2026-10-19: a stopped splat should be usable, and
training should pause and resume):
- Training runs as `python -m mf_train` (`modelforge/nsplugin/mf_train.py`, ns-train + two patches).
  - **Checkpoint on request:** when `PROJECT/splat/control` appears (checked every 10 steps), the
    trainer saves a checkpoint at the current step, writes `control.done` {step, dir} and exits 0.
    nerfstudio saves only every `steps_per_save` and nothing on Ctrl-C.
  - **Resume fix:** nerfstudio's `--load-dir` is broken for splatfacto. `load_state_dict` replaces
    the Gaussian parameters with new tensors of the checkpoint's size, but the optimizers keep the
    initial tensors and the loaded Adam state, so the first step would fail on shapes or train tensors
    nobody renders. The patch rebinds the optimizers to the new parameters, keeping the state.
- `jobs.Token.run(soft_stop=, soft_timeout=300)`: on cancel such a process is asked (the control
  file) and waited for instead of terminated; terminated if it does not exit in time.
- **Stop** (top button): checkpoint, `splat/resume.json` (the run's settings + checkpoint dir +
  step + transforms.json mtime), then the partial splat is exported and converted with a fresh token
  (the job's is cancelled) → `splat.StoppedPartial`; the GUI shows it and it is exportable;
  `report.json["splat"]["stopped_at_step"]`.
- **Pause** (Splat tab): the same without the export → `splat.Paused`; the live preview stays.
- **Resume training** (Splat tab, stage `splat-resume`; CLI `--resume`): `splat.resume` reruns with
  the stored settings (not the tab's) and `--load-dir`; step, optimizer state and learning-rate
  schedules continue. `resume_info` refuses when the checkpoint is gone or transforms.json changed
  (other cameras / scene frame). A new training run deletes resume.json; a finished one too.
- Stopped before the first step reaches the check (e.g. while images load): no checkpoint, a plain
  stop. ROS 2 bag projects skip the pre-training re-export on resume (it would change transforms.json).
- Tests: `tests/test_splat_pause.py` (a fake trainer + ns-export: pause → resume continues at
  step + 1 with --load-dir → stop exports the partial splat; soft stop; mf_train patches load in
  nerfstudio's venv). Real run on a fullrun_test copy (2000 steps): paused at 710, resumed at 711, densified on (78k → 157k Gaussians), stopped at 1130 with the partial splat exported.

**Live preview** (Splat tab, on by default):
- The user wanted to watch the splat form in the GUI's 3D view.
- With **Preview every N steps** (default 500, as the user asked), `ns-train` gets
  `--steps-per-save N`.
- `splat._watch_checkpoints` polls the run folder. On each new checkpoint it
  runs `export/ckpt_dump.py` in nerfstudio's venv (torch + numpy only),
  writing `PROJECT/splat/live.npz` (means, log-scales, wxyz quats, opacity
  logits, SH band 0, step).
- `gui.data.load_live_splat` maps the snapshot to ENU with
  `splat_to_enu.transform_vertices` (band 0, so no SH rotation) and
  subsamples to ≤500k for the browser. The GUI draws it as
  `/dense/splat_live`, replacing the old splat during training; the
  finished splat replaces it at the end.
- **All gaussians** (View → Dense / splat, **on by default** since 2026-10-19; the user asked):
  unticked, the browser gets a random subset, ≤500k Gaussians live and ≤1M for
  the finished splat (`gdata.LIVE_SPLAT_SHOWN` / `SPLAT_SHOWN`). Ticked, it sends
  all of them; the last live snapshot is reloaded at once. The status lines
  say "N gaussians (k shown …)" when a subset is drawn.
- `ns-train` gets `--method-name <method>`. splatfacto-big (and -mcmc) keep
  nerfstudio's `method_name="splatfacto"`, so their runs landed in
  `nerfstudio/*/splatfacto/`, where neither the watcher nor `latest_config`
  (the export) looked: the preview never appeared with splatfacto-big. Each
  method now has its own run folder.
- A half-written checkpoint fails the dump and is retried on the next poll. A
  preview error never stops training.
- The watcher reports to the GUI log and the server console ("waiting for the
  first checkpoint", "watching <run>", repeated read failures with the error,
  "live preview at step N").
  - Why: the first real GUI attempt showed nothing and left no trace: the
    dump never succeeded and failures were skipped silently. The cause was
    not found; with the diagnostics it worked on the next try (steps 1000 to
    9000).
  - Dumps time out after 5 min.
- The Splat tab shows a live status line. During training, the View tab's
  Splat checkbox is enabled and ticked and toggles the live layer (it used
  to be disabled until a finished splat existed).
- Measured on berlin_test: snapshots at steps 500/1000/1500 (47k → 146k →
  225k Gaussians), each redraw ≤0.3 s. A real 1.07M-Gaussian checkpoint
  dumps and converts in 0.6 s.
- Each checkpoint is ~0.75 GB per million Gaussians (it includes optimizer
  state), so very small intervals slow training.

**Splat quality on large scenes: the karlak investigation** (2026-10-20; the user: splats of larger scenes
look smudgy / low-res while the sparse model is sharp, and seem to get worse with more training; camera pose
refinement off did not help):
- **Setup** (`build/splat_exp.py`, gitignored; workspace `build/karlak_exp/`): a copy of karlak (471 4K
  keyframes of 2000 extracted, "normal" density; SfM 0.52 px). **24 held-out photos** = extracted frames
  between the keyframes (never trained on), located by PnP (`evaluate.pose_holdout`, cached in
  `located.json`); a snapshot every 1000–2000 steps (live.npz → ENU ply, band 0 only) scored with
  `evaluate.run(splat_path=, located=, report=False)`; the final splat with all SH bands too.
- **Results** (held-out, final splat with SH, rendered at 1920 px; camera optimizer off; MCMC unless noted):

  | run | steps | Gaussians | PSNR | SSIM | LPIPS ↓ | train |
  |---|---|---|---|---|---|---|
  | base6k (the user's settings, GPU cache) | 6000 | 2.09M | 22.01 | 0.643 | 0.481 | 381 s |
  | fullres6k (num_downscales 0) | 6000 | 1.99M | 21.40 | 0.617 | 0.539 | 1280 s |
  | depth6k (SfM points as depth targets, splatfacto-lidar 0.2, disk cache) | 6000 | 2.20M | 21.62 | 0.638 | 0.486 | 405 s |
  | base18k4m (budget 4M) | 18000 | 3.54M | 22.78 | 0.683 | 0.395 | 4815 s |
  | **ds2def18k** (×2 = 1920 px, **default** densification, splitting to 9000) | 18000 | 1.90M | **22.99** | **0.700** | **0.333** | **672 s** |
  | def18k (full 4K, default densification, splitting to 9000) | 18000 | 1.99M | 22.79 | 0.690 | 0.358 | 1614 s |

  Band-0 snapshot curves (PSNR / LPIPS) at steps 8k / 12k / 18k: base18k4m 20.06 / 0.559, 20.73 / 0.481,
  21.30 / 0.415; ds2_18k4m (×2, MCMC 4M, stopped at 12k) 20.37 / 0.522, 21.25 / 0.429, –; ds2def18k 20.93 /
  0.475, 21.41 / 0.400, 21.54 / 0.363. Per held-out photo, ds2def18k has the best LPIPS of all 5 runs on 21 of 24 (vs base18k4m: 20 of 24).

- **Findings:**
  - **More training is better on average**: base18k4m's held-out curve rises steadily (LPIPS 0.63 at 4k → 0.42
    at 18k); a dip around step 6000 (the switch to full resolution + the budget filling) looks like "worse".
  - **But floaters ("fog") build up in front of some viewpoints**: held-out frame 3612 (8 m, pitch −48°) went
    PSNR 17.8 → **9.5** from 6k to 18k: semi-transparent Gaussians hanging near the camera path, explaining
    other training views, hide this one. This is the user's "worse with more training".
  - **Quality depends on the flight part**: low passes looking down (~8 m, pitch −90…−48°; millimetre GSD)
    are poor (LPIPS 0.65–0.83: smudged, needle-like Gaussians), near-horizontal views ~9 m good (0.16–0.3).
    One Gaussian budget for the whole scene starves the high-detail parts.
  - **splatfacto's coarse-to-fine schedule helps**: training at full resolution from step 0 was worse and 3.4×
    slower. A 6000-step run reaches full resolution only at its end (num_downscales 2, every 3000 steps).
  - **Short runs are under-densified**: MCMC grows at most 5 % per 100 steps until stop_split_at, so 6000
    steps end at ~2.1–2.6M, far below the 7M budget.
  - **GPU memory at 4K**: ~0.2M Gaussians per GB (rendering buffers grow with resolution), not the 0.35M the
    plan assumed: the 7M budget ran out of memory at step ~6000 (4.41M, 21.9 GB, 5.8 GB of it fragmentation);
    4M fits. splat.py now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - **Image cache**: the GPU cache took 11.7 GB for 471 4K photos (and made full-res training run out of
    memory); the disk cache was only ~6 % slower (405 vs 381 s) — the long runs are slow because of 4K
    rendering with 4M Gaussians, not decoding. New Splat tab **Image cache: auto / gpu / cpu / disk**
    (`splat.auto_cache`: gpu if the photos take < 1/4 of the GPU memory, else cpu if **twice** their size is
    < 1/2 the free RAM, else disk; karlak ×1 → disk, ×2 → gpu). CLI `--cache auto` (default). cpu holds the
    photos twice (trainer + the data workers' shared memory): with the first rule (1×) 2000 photos at 1920 px
    (12.4 GB decoded) took 11.7 + 12.1 GB and the kernel OOM-killed the training (dira2k, 2026-10-20).
  - The semantic text worker held 3.8 GB of GPU memory (moved to the GPU before halving); fixed (half, then
    `.cuda()`, then `empty_cache`).
- **The user's practical result** (2026-10-20): karlak at **downsample ×2, 30000 steps** looked **good**. That
  run: 1920 px, the plan's **default** densification (at ×2 the estimated need fits the GPU, so no MCMC; reset
  period scaled, splitting to 15000), camera optimizer SO3xR3, disk cache, 1.81M Gaussians, 1451 s (24 min).
  It differs from the smudgy runs in resolution, steps **and** strategy (they were MCMC), so MCMC itself is a
  suspect (it relocates Gaussians into low-opacity spots: a plausible source of the near-camera fog).
- **MCMC is the fog** (2026-10-20, `ds2def18k`): ×2 with default densification beats every MCMC run, including
  full-res 4K MCMC over the same 18000 steps (LPIPS 0.333 vs 0.395), with half the Gaussians in 1/7 of the time.
  The fog frame 3612 is fine with it: PSNR 19.5 / LPIPS 0.448 (MCMC 18k: 9.5 / 0.830; MCMC 6k: 17.8). At equal
  ×2 resolution MCMC was behind at every snapshot (12k: LPIPS 0.429 vs 0.400). ×2 alone already helps (ds2 MCMC
  at 12k ≈ full-res MCMC at 18k), but the evaluation renders at 1920 px, which favours ×2 runs.
- **Full 4K with default densification** (`def18k`): no fog either (frame 3612: 19.5 dB), GPU memory peaked at
  only 8.4 GB, but it grew to just 2.0M Gaussians (the plan's ~14M "need" is far off for default densification:
  its gradient threshold is resolution-normalized) and is **no better than ×2 even scored at 4K**. Band-0 final
  snapshots, PSNR / SSIM / LPIPS at 1920 px → 3840 px: ds2def18k 21.54 / 0.668 / 0.363 → 21.29 / 0.705 / 0.438;
  def18k 21.41 / 0.662 / 0.385 → 21.32 / 0.712 / 0.452; base18k4m (MCMC) 21.30 / 0.660 / 0.415 → 21.18 / 0.711
  / 0.471. So 4K training costs 2.4× the time for nothing here, unless more Gaussians are allowed to grow.
- **TODO:**
  1. DONE 2026-10-20: the plan's defaults are now default densification and ×2 for 4K (see "Training plan").
     Possible next experiment: full 4K default with a lower
     split gradient threshold (splatfacto-big's 0.0005) so 4K detail gets more Gaussians (8.4 GB peak leaves
     room).
  2. `depth18k4m`: depth supervision from the SfM points over 18000 steps — does it stop the fog (frame 3612)?
     Less pressing now that default densification has no fog.
  3. Other defaults: splat plan budget at 4K (~0.2M / GB), iterations vs image count, floater
     control (depth supervision, opacity / scale regularization, near-camera pruning), possibly ×2 for large
     projects; and for very large projects (8000 photos) per-tile splats (CityGS / VastGaussian style).
  - Run them with `python build/splat_exp.py VARIANT ...` (`--eval-snaps VARIANT` scores a stopped run's
    snapshots) in the container (results append to
    `build/karlak_exp/results.json`); variants are defined in VARIANTS there.

**Large flights: SfM splits revisits into offset copies** (found 2026-10-20 on `data/dira2k`: every 4th of
kladno_dira's 8000 frames, 2000 4K photos, one 30-min Mavic 4 flight over ~380 × 380 m; SfM 26 min, chunked refine,
2000/2000, 0.535 px):
- **Symptom:** every splat failed: held-out PSNR ~13 dB / LPIPS ~1.0 (fog), also on *training* views and in
  nerfstudio's own `ns-eval` (15.1 dB at 4000 steps), the Gaussian count shrinking 2.6M → 0.69M. Not the evaluation,
  the camera optimizer, the image cache or nerfstudio's scene scale (each ruled out by a 4000-step run). A patch of
  150 consecutive photos trains fine alone (19.4 dB at 4000 steps), but the same area is bad (11–14 dB) inside the
  full run. Probably the user's earlier "bad splat of a 500 × 500 m world whose small parts trained fine".
- **Cause:** 1.29M of 1.33M sparse points are seen only within 60 frames of capture order: when the drone returns
  over the same ground it is triangulated again as a new copy, and the copies sit **1–3.4 m apart in height**
  (2 m cells seen by photos of different 250-frame groups, median dz up to 3.4 m, p10 1.8 m). Each time segment
  follows its own drifted prior (altitude drift over the flight); 3 m ≈ 200 px at 50 m, far beyond COLMAP's track
  merging, and the chunked refine's pose graph carries no revisits. The reprojection error cannot show it.
- The matches exist: 5,912 verified pairs > 300 frames apart, 1.33M inliers (database.db).
- **Fix: position drift over time** (`robust.update_drift`, `PriorModel.drift`; CLI `--no-drift`): per source an ENU
  offset linear in time between knots every 30 s (`drift_knot_s`), added to the priors like the per-source offsets
  (`corrected_priors`, also subtracted in the δt estimate). Measured from the split points between images of one
  source (`_split_pairs`, shared with `split_point_edges`): one equation per image pair with ≥ 20 split matches,
  o(t_j) − o(t_i) = median(X_i − X_j), weight √n (≤ 20), Huber 0.5 m (IRLS); first differences between knots
  (a stretch without revisits stays flat), a small ridge, and the mean over the source's images = 0 (GPS keeps the
  absolute position). The cameras are shifted by the increments, so the next triangulation merges the copies.
  - Where: global refine after each round's BA from round 1 (round 0's focal is ~2× off); not in chunks (throwaway
    model copies) and **not on the calibration subset**: its separate blocks leave the gaps unconstrained, and the
    first version (second differences) extrapolated noise there into 4.7 m of fake drift that the chunks then
    followed (revisit split 4.9 m). Chunked refine: after the pose graph, up to 3 passes of triangulate all →
    update_drift → shift (`pipeline.DRIFT_PASSES`, stop below 5 cm), then the polish with the corrected priors.
  - GUI: state line "drift pass k", refine History "position drift up to X m removed", View stats per source.
  - Tests: `test_multi_source.py::test_drift_is_recovered` (two passes over a synthetic scene, 0.8 units of drift:
    split 0.68 measured in the first update, merged after it, cameras within 3 cm up to the gauge),
    `test_drift_without_revisits_stays_zero`.
  - **dira2k:** drift passes split median 2.42 → 0.61 → 0.41 m (increments still ≤ 1 m at the 3-pass limit),
    drift ≤ 3.1 m, height range 5 m over 33 min (a 4 m dip around minutes 18–20, agreed on by many pairs). Points
    linking images > 250 frames apart 1.5 % → 6.1 %; height offset between 250-image groups over the same 2 m
    cells: median of group pairs 1.69 → **0.14 m** (max 3.40 → 1.10); per cell |dz| median 1.77 → 0.42 m. Final
    0.574 px (was 0.535), 1.32M points. Splat at 4000 steps: 15.3 → 16.5 dB held-out (ns-eval 15.1 → 16.5), and
    the render now lines up with the photo (before: structure in the wrong place); the rest is under-training.
  - Not converged in 3 passes (0.4 m left): faster drift than 30 s knots, or false matches on repetitive gravel.
  - **Chunked vs global on dira2k (with drift):** final revisit split (image pairs > 60 s apart, ≥ 20 split
    matches; `build/split_check.py`) median / p90: no drift 2.56 / 3.95 m (22 % of their matches merged), chunked
    0.24 / 0.88 m (64 %), **global 0.05 / 0.15 m (80 %)**, at the measurement's floor (pairs < 60 s: 0.06 m).
    Finer knots would only reach ~0.13 m: the windowed polish holds everything else fixed and cannot pull two
    passes together, a global BA can (its drift stayed ≤ 0.5 m). Global: 67 min, 7.2 GB; chunked 24 min, 3.9 GB.
    → BA mode auto now chunks from 2500 images (was 1000). Larger projects still need a cheap final global step
    (idea: `refine.local_ba(global_pass=True)` on thinned points after the drift passes; untested).
  - **kladno_dira (2883 images, chunked) had the same split** (2026-10-21): pairs > 60 s apart 0.39 m median, p90
    1.57 m, 51 % merged; the drift passes stopped at 0.47 m. The tiled splats could not even fit their training
    photos (17 dB). The user suspected the priors; they are loose (0.75 / 0.5 m; the solution deviates 1.4 m median),
    but in chunked mode they are the only link between passes.
  - **Loop closures in the pose graph** (`chunked.revisit_edges`, `Chunked.revisit_passes` = 3, before the drift
    passes; the user asked): triangulate all → every image pair beyond the chunk edges' reach (order gap > 64)
    with ≥ 20 inlier matches on different 3D points gives an edge = current relative pose with the second camera
    moved by median(X_a − X_b) (σ max(3 cm, the offset's own), 0.1°; within and across sources) → pose graph
    again (chunk edges + priors + these) → repeat until the median split < 5 cm. Log / state line "revisit pass k";
    History "revisits split a → b m". Synthetic two passes as separate chunks, drifting priors (`test_local_ba.py::
    test_revisit_edges_join_two_passes`): split 0.67 → 0 in one pass, camera spread 0.34 → 0.048 (a global BA with
    those tight priors stays at 0.34). fullrun_test (no real revisits): final error unchanged.
  - **Loop closures measured on kladno_dira** (2026-10-21, chunked, old pairs): revisit split 1.53 → 0.73 → 0.76 m
    over 3 passes (p90 3.1 → 2.5 m), drift passes after it 0.74 → 0.51 m; no better than without. Likely cause: the
    revisit edges go through the same Cauchy (3σ) as the chunk edges, so a 1.5 m split at σ 3 cm (50σ) weighs ~0.4 %
    (untested), and the chunk edges (5 mm, 0.02°) keep each chunk's drift rigid. Tile 3 retrained on it: no better
    (held-out 16.3 dB, training views 16.6).
  - **Fixed (2026-10-23): Huber loop edges + global passes on thinned points.**
    1. Loop-closure edges are Huber (3, `Chunked.revisit_loss`; `pose_graph(loops=)`), chunk edges stay Cauchy
       (Huber there brought nothing). Synthetic 3x drift, sigma 0.3: Cauchy split 0.31 → 1.98 (diverges), spread 1.03;
       Huber split → 0, spread 0.11 (`test_revisit_edges_join_two_passes[...]`).
    2. After the drift passes, `Chunked.global_passes` (2) x (triangulate all → `chunked.global_ba`: one BA over all
       cameras on ~100 points per image (`global_points_per_image`, thin_points in place), weighted, intrinsics fixed,
       stops at 90 % settled) → the polish. The revisit split is logged before each pass and after (report
       `chunked.global`, `global_split`; History "global BA x2: revisits split a → b m"). On the 40-image synthetic
       scene it hurts (its 400 points cannot hold the passes against the drifting priors); on real photos it works.
    - **dira2k** (2000 images, `build/dira2k_chunk2`, `build/split_check.py`, pairs > 60 s): revisit passes 2.51 →
      0.39 → 0.34 m, drift passes → 0.29, global passes → 0.12 → 0.078 m (own metric); like-for-like **0.05 / 0.20 m,
      80 % merged** (chunked before 0.24 / 0.88, 64 %; global 0.05 / 0.15, 80 %). 50 min (old chunked 24, global 67),
      peak 3.9 GB, final 0.59 px (global 0.574). The global BAs took 45 + 101 s.
    - **kladno_dira 5681** (2026-10-23, GUI, auto → chunked): revisit passes 1.91 → 0.35 → 0.30 m (pose graphs
      35–52 min each: `docs/todo_posegraph_speed.md`), drift passes no gain (0.27–0.30 m), global passes 0.30 → 0.20 →
      **0.14 m** (p90 1.22 → 0.83 m; BAs 210 s + ~3 min on ~140k points), polish 29 min; final 0.58 / 1.29 px, 6.2M
      points, 5.1 h, peak 14.6 GB. A third global pass would likely gain more.
  - **Global refine on kladno_dira** (2026-10-21, with terrain pairs, stopped after round 1 to diagnose): round 1 BA
    79 min (158 iterations, 15 CHOLMOD "not positive definite" step failures), point error 1.085 → 0.706 px
    median, revisits merged (far split 5 cm). Its log said 3.5 px: the after-BA stats were taken after the drift shift
    had moved the cameras (fixed: `_rounds` evaluates before offsets / drift).
- **Passes no other pass ties in** (`modelforge/sfm/isolated.py`; the user asked, 2026-10-21): on kladno_dira a
  crossing at 24-42 m (1586-1613 s, 12-17 m/s) over ground the other passes saw from ~6 m shares no 3D point with
  any other pass, and only 9-45 points with its neighbours in time; its camera path fits the GPS with scale 0.965
  (other 20 s segments 0.99-1.02, no trend with speed or altitude), sits 4 m off its priors after rounds 0 and 1
  (no prior was rejected then: `start_round` 1), and its ground copy lies ~1 m (up to 3.8 m) off the others'.
  - **SIFT does not bridge it** (`build/hl_match.py`, 24 high/low pairs of the same ground): 8k features 0/24
    verified, 16k 1-2/24, 32k at 3840 px 2/24; ratio 0.9 gives 24/24 with a uniform ~19 inliers (chance level).
    Pair selection also misses them: of 1,485 such pairs 35 were in pairs.txt.
  - **Export leaves them out** (Sparse tab "Export: leave out isolated photos", default on; CLI `--keep-isolated`):
    a photo with < 2 % of its points also seen > 60 s apart, whose own points lie > 1 m off the other passes'
    ground in over 20 % of its 1 m cells (per cell and 60 s pass the median height; reference = median over the
    passes not next to the photo's), is left out of transforms.json, and the points only such photos see out of
    sparse_pc.ply. Photos over ground no other pass saw stay, and flights without revisits are untouched.
    report.json `export.isolated`; History "N isolated photos left out".
    - Per cell, not per photo: a first version used the photo's median height offset (> 0.5 m) and missed the
      user's ghost in tile 3: those copies are tilted or noisy (dira_48332: median +0.16 m, 63 % of its cells
      > 1 m off). Per-photo shares of cells > 1 m off: tied photos median 7 % (p90 26 %: steep quarry walls,
      machines), weakly tied (< 0.5 % linking) 31 %. Linking points alone are too sparse as the reference.
    - kladno_dira (global refine, round 1 skipped at 75 iterations): 689 weakly tied, 309 dropped (10.7 %),
      268k of 2.4M points; tile 3's 2 m cells with a pass > 1 m off the others 175 → 52 (the crossing's +1.8 m
      copy, 77 cells, gone; 30 %: 243 photos, 102 cells). 45 s on 2883 photos. Test: `tests/test_isolated.py`.
    - **Flat ground decides** (berlin, 2026-10-22; the user saw a big chunk of cameras missing near a tile seam):
      the cell rule left out **1095 of 3142** berlin photos, mostly low passes under trees and a flight inside the
      halls: their cells differ because they see other surfaces (ground under canopy, a floor under a roof). Renders
      of the splat trained without them lined up with the photos. On open flat cells (the other passes' ground level
      within 0.3 m over the 3 x 3 cells around) berlin_3_3 / 3_4 agreed to 0.09 m, kladno_dira's drops were 0.9 m off
      (all flat cells), berlin_3_2's "offsets" ~10 m (floor vs roof). Now a candidate (the cell rule) is a copy when
      >= 5 of its flat cells lie within 4 m of the others and >= half of those > 0.5 m off (`isolated.MIN_FLAT`,
      `FLAT_*`); candidates < 5 s apart are decided together by those with flat cells, a stretch without any stays.
      berlin 1095 -> 117 (88 of berlin_3_2 beside / inside the halls, e.g. 3_2_00425: 34 flat cells at -1.2 m; 29 of
      the end of berlin_3_1, t >= 504 s); kladno_dira 309 -> 192, all 48 of the known crossing. report
      `export.isolated.candidates`. Test: `test_different_surfaces_are_not_a_copy`.
    - **Progress** (the user asked): `stage_export(progress=)` shows reading / finding isolated photos (point k / n,
      photo k / n) / "triangulating: image k / n" (COLMAP reports it only in its log: `jobs.stderr_lines` taps file
      descriptor 2 during `pycolmap.triangulate_points`, `refine.triangulate(progress=)`) / "refining points: batch
      k / n" / colours / writing. `refine_points` now uses all cores (Ceres defaults to one; on kladno_dira the export's
      refinement ran > 4 min on one core): fullrun_test 5.1 → 2.7 s. kladno_dira export: points 2.41M → 2.15M.
    - **The points are triangulated again without them** (the user asked): their frames are deregistered and
      `refine.triangulate` (poses fixed, refine's final 4 px threshold, points refined) writes `sparse/export`
      (a View snapshot "export"), whose points make sparse_pc.ply. Without a database.db only the points the left-out
      photos alone see are removed. report `export.isolated.points_before / points / triangulated`.
    - Tile 3 trained on it (global refine + this export, 30k steps ×2, before the re-triangulation): held-out
      16.89 / 0.269 / 0.653 vs chunked 16.59 / 0.221 / 0.687 (same 15 photos), own training views 18.5 dB (was
      16.6-17.3).
- **SfM points as depth targets** (the user asked, 2026-10-21): the Splat tab's depth weight ("Depth weight (LiDAR /
  SfM points)") is shown for every kind; without LiDAR the lidar-data dataparser takes `ply_file_path`
  (sparse_pc.ply, per tile its cropped cloud) as the depth points (`splat.depth_source`). Default 0 except ROS 2
  bags (0.2). CLI `scripts/splat --lidar-depth W`. On karlak (6k steps, MCMC) it was 21.62 vs 22.01 dB without.
- **Splat on dira2k, still open** (held-out PSNR / LPIPS, ×2, default densification): 4000 steps (camera optimizer
  off): no drift 15.3 / 0.97, chunked + drift 16.5 / 0.92, global + drift 17.0 / 0.86; renders now line up with
  the photos. But longer plan runs get **worse**: 12000 steps 14.9 (chunked and global alike), 36000 steps 12.9;
  already behind at equal steps (12k run at step 3000: 13.9 vs the 4k run at 2000: 15.8), the Gaussian count
  falling at every opacity reset (2.3M → 0.63M over 36k). Ruled out: the opacity reset (none: 13.8, 2.7M), the
  camera optimizer (−0.7 dB at 4k). Suspects for the next session: what depends on the step count — the position
  learning-rate decay (in nerfstudio units of 1 / max |camera position| ≈ 200 m here, 7× karlak; the unit-28-m
  test ran on the drift-broken model, so redo it: `schedule["scale_factor"]`) and the split stop. A 150-photo
  patch trains to 19.4 dB at 4000 steps. The site has no moving objects, steady light and fixed exposure (the user).
- **Solved: the scene unit** (2026-10-20, global + drift model, held-out PSNR / SSIM / LPIPS, `build/splat_exp.py`):

  | run | unit | final | Gaussians |
  |---|---|---|---|
  | g12k (12000 steps, the plan) | auto ≈ 198 m | 14.9 / 0.197 / 0.942 | 1.24M |
  | g12k_lr10 (position LR ÷ 10 only) | auto | 15.5 / 0.201 / 0.914 | 1.17M |
  | g12k_unit28 | 28 m | 17.0 / 0.278 / 0.782 | 1.21M |
  | **g12k_unit10** | **10 m** | **17.2 / 0.289 / 0.743** | 1.25M |
  | g12k_unit4 | 4 m | 17.0 / 0.281 / 0.743 | 1.23M |
  | drift36k (36000) | auto | 12.9 / 0.185 / 0.977 | 0.58M (falling) |
  | g36k_unit28 (36000) | 28 m | 16.9 / 0.262 / 0.626 | 2.2M (growing) |

  The learning rate alone is a small part; the size thresholds (clone below 0.01 units = 2 m at auto scale, cull
  above 0.5 = 100 m) matter more. With a fixed unit more steps help again (LPIPS 0.87 → 0.645 from 6k to 36k);
  PSNR flattens near 16.5–17 dB: not moving objects or light (none, per the user); renders are blurry with needle-like Gaussians, ~1.2M for ~14 ha. karlak at 10 m (ds2def18k_unit10): 22.99 / 0.700 / 0.333
  → **23.24 / 0.714 / 0.301** in the same time; its auto unit was ~31 m (kladno_dopravnik 26, fullrun_test 27). →
  The plan's default is now one unit = 10 m. Median scene depth as the unit would give dira 72 m: not supported.
- Splat side, found on the way: nerfstudio scales poses by 1 / max |camera position| and splatfacto's thresholds
  (`densify_size_thresh`, `cull_scale_thresh`) and position learning rate are in those units (`scene_scale=1.0`);
  `schedule["scale_factor"]` fixes the scale (`--auto-scale-poses False --scale-factor`), now the default (10 m).

**Splat tiles** (`modelforge/splat_tiles.py`; the user asked, 2026-10-20: worlds of up to ~800 photos train
well, dira2k's 2000 in one splat stayed blurry at ~17 dB with needle-like Gaussians, more Gaussians or steps
helped little):
- **When:** plan `tiles` (Splat tab "Tiles" auto / on / off, CLI `--tiles`): auto above 800 training images
  (`TILE_ABOVE`). The plan note says so.
- **Partition:** each camera's look-at point = median x/y of the sampled sparse points it sees (as
  `splat_plan.survey`); ⌈cameras / `PHOTOS_PER_TILE`⌉ cells (500, "Photos per tile"), cut recursively along the
  longer axis (k parts → ⌊k/2⌋ + ⌈k/2⌉). Outer cells open to infinity.
  - **Where to cut** (`_cut`; 2026-10-20, the user saw an excavator the drone circled split between tiles 1 and 2 on
    kladno_dira, its far-side views missing from one): a plain median of the look-ats landed on the most
    photographed spots, since an orbit puts many photos there. Now a cut may move up to `SIZE_SLACK` (50 %; the user:
    tiles of up to ~2× the photos still train well) of the smaller side's photos off balance, and goes where it splits
    the least: cost = sparse points within `GAP_WIDTH` (2 % of the extent) of the line, each weighted by the photos
    seeing it, in the worst 4 × `GAP_WIDTH` stretch along the line (summed over the whole line an object drowns in the
    ground); score = cost / median cost + `BALANCE` (0.3) × off-balance as a share of the slack. Look-at density alone
    did not work: an orbit's look-ats spread around the object, beyond it. kladno_dira (2883 photos): the most viewed
    spot (~500 photos per point, tall: the excavator) was 12 m from a cut, now ≥ 35 m; of the 20 most viewed 10 m
    spots 1 lies within 8 m of a cut (`BALANCE` 0.5 / 1: 3); 6 tiles of 168–813 core photos, ≤ 1147 with neighbours
    (the plain median gave 8 of ~360: 2883 → 1441 → 720 → 360).
  - A tile's photos: those looking into its cell + every photo with ≥ 25 % (`MIN_VISIBLE`) of its seen points
    inside it, or ≥ 10 % (`MIN_IN_CELL`) with its look-at within half the margin of the cell (far-side views of edge
    objects). kladno_dira: 4,763 photo slots (+12 % for the second rule; a "≥ 25 % in cell + margin" rule gave +37 %).
    dira2k (old rule): 4 tiles of 740 / 784 / 822 / 662 photos, 2.7 s.
- **Training:** `PROJECT/tiles/tile_KK/` = transforms.json with its frames, sparse_pc.ply cropped to the cell +
  margin (25 % of the cell's longer side), `images/` and `images_N/` as relative links to the project's; then
  `splat.run` with its own plan (`suggest(survey(tile), tiles="off")`: steps from its photos, the tab's explicit
  values apply per tile), one tile after the other, image cache auto per tile.
- **Rounds** (`ROUND_STEPS` = 10000 (the user's default since 2026-10-23; 4000 before, 6000 at first), Splat tab "Steps per round", CLI `--round-steps`, 0 = one tile after the other;
  the user asked, to see the whole scene early and have it improve evenly): every tile advances that many steps per
  round (`splat.run(stop_at=)`: the trainer checkpoints and ends as on Stop, resume.json written, the splat so far
  exported, no exception) and continues from its checkpoint next round, so step, optimizer state and learning-rate /
  densification schedules run as in one go. The cleaned tiles are merged into `splat/splat_enu.ply` after each round
  (seconds). tiles.json: per tile `step`, status `pending` / `training` / `round` / `paused` / `partial` / `done`, the
  run's `round`. Every tile's dataset and plan are written up front, so progress counts the steps of all tiles. Cost:
  ~1.5–2 min per tile and round (restart, photo loading, export; ~50 s of it the export), so 30000 steps in rounds of
  4000 (8 rounds) on 6 tiles add ~70–100 min to ~2.5 h of training. The live preview replaces a tile's last-round part with its
  training one.
- **View → Tile boundaries** (`splat_tiles.overlay`, `App._draw_tiles`): a red box per cell (clipped to the scene
  box, from the 2 % to the 98 % height of its sparse points), the training one orange, labels "tile k · N photos ·
  status"; quick bar "Tiles"; without a tiled run the planned cells with the tab's Photos per tile (computed in a thread).
- **Clean** (ENU), only at the sides shared with another tile (the user asked, 2026-10-20: outer tiles should keep
  the view away from the scene): keep Gaussians with the centre in the cell (outer sides at infinity) whose 3σ reach
  crosses no shared side by more than the margin (reach ≤ distance to the nearest shared side + margin: a large
  background blob far out stays, none spreads over a neighbour); `FAR` (2 × the scene box's size beyond it) only
  drops absurd outliers; no height cut (`clean`). Before: also the scene box and 3σ ≤ margin everywhere.
  kladno_dira at round 2: +2–11 % kept per tile (2.03M → 2.08M) → `tile_KK/splat/clean_enu.ply`. Smoke run (300 steps): ~45 % kept.
- **Merge:** concatenation → `PROJECT/splat/splat_enu.ply` (export, splat mesh, semantic, View use it as before).
- **State:** `tiles/tiles.json` (cells, frames, status per tile). Pause / Stop act on the running tile; Stop merges
  the done + partial tiles; `splat/resume.json` `{"tiles": true, "tile": k, ...}` → `splat.resume_info / resume`
  continue that tile, then the rest. The live preview shows the finished tiles (cleaned) + the training one
  (cropped to its cell) through `splat/live_tiles.npz` and an identity transform. Progress "tile k / n: training
  step …"; report.json `splat.tiles`, `per_tile`.
- `tiles/` is an intermediate (Clean), archived on re-extraction.
- **Set up tiles turns View → Tile boundaries on** (the user asked, 2026-10-22), so the new cells show at once.
- **Set up tiles / Train tiles** (the user asked, 2026-10-21; Splat tab, CLI `--setup-tiles`, `--tile K`):
  `splat_tiles.setup` = partition + every tile's dataset, all pending, nothing trained (replaces tiles/; the GUI asks
  when trained tiles would go). A run reuses the set-up while Photos per tile and transforms.json are the same
  (`current`: tiles.json `transforms_mtime`), else sets up again. "Train tiles" all / tile k: `run(only=k)` trains
  that tile from scratch in one go with the tab's current settings (its own plan); the others keep theirs; its last
  clean_enu.ply stays in the merge until replaced (merge = every tile with a clean_enu.ply). tiles.json `only`
  (Resume trains just that one), `merged` (tiles, counts, size + mtime of the merged file; `merged_tiles`).
- **Semantic features per tile** (the user asked): when splat/splat_enu.ply is the tiles' merge (`merged_tiles`;
  for runs before the record: the cleans add up to it and are not newer), `semantic.run` computes one PCA basis
  over the project's photos (`splat/semantic/pca.npz`, the worker's `pca_out`), then per tile its whole splat
  rendered from the tile's photos for the occlusion test, features kept for its cleaned part (`keep`, `pca_in`) →
  `tiles/tile_KK/semantic/` with a key (clean_enu size + mtime, PCA), concatenated in merge order. A retrained tile
  alone is recomputed. The full-splat run on kladno_dira (3.7M Gaussians, 2883 photos) itself worked: 17 min.
- **Quality of the 30k-step tiles on kladno_dira** (2026-10-21, `build/tile_exp.py`: tile 3, 828 photos, 24 held-out
  extracted frames between its keyframes looking into its cell, PnP-located): the merged tiles looked soft. Tile 3
  as trained (30k steps in rounds, ×2): PSNR 16.5 / LPIPS 0.66 at 1920 px (karlak: 23 / 0.30). Renders line up
  with the photos but surfaces are needle-like Gaussians. Not the photo count (keyframe overlap worst 5 % / median
  0.21 / 0.33, more than karlak's 0.12 / 0.26), but the site was flown ~3× higher (GSD 1.7 vs 0.6 cm). 53k steps
  (64 per photo) ×2: 16.3 / 0.61 (2.2M Gaussians): steps alone barely help; split gradient 0.0005 + scale
  regularization ×2 at 30k: 4.3M Gaussians, 15.9 / 0.647 (band 0; variant 1 at 30k: 16.0 / 0.684). **Cause: the
  SfM**, not the splat: tile 3 scores 17.3 dB on its own *training* views, and revisits are offset (pairs > 60 s
  apart: split median 0.39 m, p90 1.57 m, 51 % merged; < 60 s: 0.09 m) because 2883 photos refined chunked and the
  drift passes stopped at 0.47 m (`build/split_check2.py MODEL PROJECT`). Fix to try: refine with BA mode global
  (dira2k: 0.05 / 0.15 m). The experiments were stopped (workspace `build/dira_t3_exp`).
- **Tile layout "covisibility"** (the user asked, 2026-10-22: straight cuts cannot bend around an orbited object or a
  building; `modelforge/tile_regions.py`, Splat tab "Tile layout", CLI `--tile-layout`). **The Splat tab's and the
  CLI's default since 2026-10-23** (the user's choice;
  `splat_tiles.DEFAULT_LAYOUT`); the library functions keep "rectangles" (old tiles.json / resume.json without a
  layout). It needs the SfM model (synthetic test projects without one name rectangles). A project's existing set-up
  keeps its layout (the GUI adopts it) until Set up tiles runs again. On kladno_dira (5187 photos, 11 tiles, 2026-10-23) 2 regions vanished (narrow, ragged: no cell BAND_M from a
  seam, so no watershed seed; their tiles got 0 points and the splat failed): a region narrower than 2 x BAND_M now
  seeds from its deepest half (`test_narrow_region_keeps_its_seeds`). Its tiles then hold 479–2434 photos (13k slots,
  2.5x; mostly the MIN_VISIBLE rule). Dense graph:
  photos linked by shared 3D points (+ 0.3 x shared 2 m ground cells, so passes at different heights stay together, +
  a tiny link along capture order), recursive spectral bisection (normalized cut, size slack 0.25); regions = a 1 m
  label map: points vote 1 / d^2 for the parts whose photos see them, majority filter, then a geodesic watershed
  within 25 m of the seams (steps through features above the local ground are cheap: objects go wholly to one side,
  seams settle on flat ground). `tiles/regions.npz`; tiles carry `region`; crop, clean (reach vs the distance to
  the nearest seam), live crop, overlay (outlines) and the photo-joining rule use it (`in_tile`, `clean_tile`;
  margin = 0.25 x sqrt(region area)). berlin (1734 photos, 4 tiles; prototype `build/covis_tiles.py`): points > 3 m
  above ground within 3 / 10 m of a seam 1.2 / 3.9 % vs rectangles 2.4 / 9.5 %, seams 1.9 vs 1.7 km; the right-hand
  orbits stay in one tile (the rectangles cut them). Plain vote maps were ragged (seams 3x longer) and an MRF with
  iterated local updates could not move seams. Not yet for 10k photos (dense n x n graph, dense eigh).
  - A training comparison against rectangles (`build/tiles_cmp.py`) was cancelled by the user on 2026-10-23
    (covisibility was already the default); only rectangles tiles 2 + 3 had trained. Do not resume it.
  - The mesh depth cache key uses the mesh's real path (a workspace linking sfmmesh/ cleared the project's maps).
- **Photos join the tiles they show, by the SfM mesh** (berlin, 2026-10-22; the user's idea): a building shot from
  a low oblique pass (berlin_3_1_12380-12585) rendered at 11.7 dB in the merged splat, 20.1 dB in tile 1's own
  splat. It lay 90 % in tile 3's region, but the pass trained only tiles 0 and 1: such a photo sees half the site in
  the distance (12447: 43k of the 50k sampled sparse points), so by count only 1 % of its points were in region 3
  (rule MIN_VISIBLE 25 %), while 67 % of its pixels were. Tile 1's good copy was cut by the clean (outside its
  region), tile 3 had only junk. Over all 2047 training views, merged vs the best own tile: 311 lost > 2 dB, 83 > 4 dB;
  pixel share in tiles lacking the photo correlated 0.68 with the loss (> 4 dB: median 0.60; < 1 dB: 0.00).
  - Now `setup` raycasts the SfM mesh from every camera (`export/mesh_share.py`, nerfstudio's venv, Open3D, 240 px
    wide; berlin 17 s for 2047 cameras), labels each hit by region / cell (`_cell_labels` for rectangles), and a photo
    also joins every tile with >= `MIN_PIXELS` (5 %) of its pixels (`partition(mesh=)`, tile `by_pixels`). berlin:
    1008 / 1106 / 715 / 355 / 713 -> 1169 / 1235 / 752 / 441 / 824 photos (+13 %), covering all 83 broken views (1 %
    "any pixel": +26 %, tile 3 355 -> 635, mostly horizon slivers). Without a mesh or nerfstudio: the old rule only.
    The View's planned overlay (no tiles.json) does not raycast. Needs Set up tiles again (and a retrain).
  - Diagnostics in `build/tile_diag/` (gitignored): `render.py` (photo vs splats at a camera), `scan.py` (every view,
    merged vs own tiles), `mesh_share.py`, `rules.py`.
- **Unrecorded checkpoints** (berlin, 2026-10-22: a tile saved its round checkpoint, the run then ended before
  recording it, and Resume started the tile from scratch): a tile marked training at step 0 adopts the newest
  checkpoint in its own folder (`adopt_checkpoint` writes its splat/resume.json; a new run or set-up clears the
  folder, so only this run's are there). Test: `test_unrecorded_checkpoint_is_adopted`.
- **Leaving tiles out** (the user asked: edge tiles may not be worth training): View → Tile boundaries draws a ball
  under each tile's label, green = included, grey = left out; a click toggles it (`App._toggle_tile`; a set-up is
  needed). `tiles/skip.json` (`splat_tiles.skipped / set_skipped`; not tiles.json, which a running job rewrites):
  `run` trains only the others (re-read before every tile, so a click during a run counts from the next tile; an
  explicit `only` tile always trains) and `merge` leaves them out (without a job the GUI merges again at once). The
  area stays empty; neighbours are still cut at the shared seam. CLI `--skip-tiles 1,4`. Test: `test_skip_tiles`.
- Not done: seam fine-tune, visibility pruning of floaters, per-tile splat mesh (it runs on the merged splat), level
  of detail for very large merged splats.
- Tests: `tests/test_splat_tiles.py` (partition on a synthetic grid, clean, plan switch, a tiled run with the fake
  trainer: Stop in tile 2 → merged partial → Resume → every Gaussian of the scene box in exactly one tile; set-up +
  one tile; semantic per tile with a stub worker), `test_gui_dense.py::test_set_up_tiles`.

**Mesh from the SfM points** (`modelforge/sfm_mesh.py`, Sparse tab **Run SfM mesh**, stage `sfmmesh`,
`python -m modelforge.sfm_mesh PROJECT [--depth N] [--trim K]`; the user asked, 2026-10-21, as a later prior for the
splat): the points of sparse/export (else refined) without strays (outside the 0.5-99.5 % box + 10 %, or 16th
neighbour > 10x its median distance: on fullrun_test a few points stretched Poisson's box to 1.45 km and its cells
to 1.4 m), normals by PCA over 16 neighbours turned towards a camera seeing the point, COLMAP `poisson_mesher`
(killable; its density trim off, depth from extent / (2 x point spacing), 8-12), then faces with a vertex farther
than 6 point spacings from every point cut. `PROJECT/sfmmesh/` mesh.ply (vertex colours), points.ply, preview.ply
(≤ 300k faces), stats.json; report.json `sfm_mesh`. View tab Mesh **sfm**. fullrun_test: 43.7k points, depth 11
(0.17 m cells), 175k of 225k faces kept, 82 % of the points within 0.3 m of a vertex, 5 s. Not yet used as a prior
(the plan: rendered depth / normals per training view for the depth loss, Gaussians seeded on it).
- **Patch holes** (Sparse tab **Patch holes in SfM mesh** + "Max hole size (m)", stage `sfmmesh-fill`, run by hand;
  `--fill [MAX]`; the user asked): Poisson's surface is closed (fullrun_test: 12 open edges of 338k), so the faces
  the trim cut form pieces (`components`: faces sharing an edge, union-find by hooking + pointer jumping); anything
  open to the outside joins one large piece around the scene, and the enclosed pieces are Poisson's smooth surface
  across a hole. Those up to the max size across (bounding box; 0 = 40 point spacings) are put back: mesh.ply
  becomes the patched mesh, `patched.npy` marks its patched faces, mesh_trimmed.ply keeps the trimmed one; SfM mesh
  again undoes it. fullrun_test: 186 holes ≤ 5.8 m, +3,930 faces, 1 s. Interpolated, not measured.
- **Colour from the photos** (Sparse tab **Colour SfM mesh from the photos**, stage `sfmmesh-colour`, by hand; the
  user wanted it pretty in the GUI): `export/mesh_color.py` (nerfstudio's venv): every transforms.json photo (decoded
  at ≤ 1920 px by JPEG DCT scaling) projects the vertices with its OPENCV distortion; a vertex counts when in frame,
  facing it (cos > 0.1) and not hidden (ray from the camera hits the mesh no earlier than 5 cm / 0.2 % before it);
  colour = mean weighted (cos / distance)^4. → `mesh_coloured.ply` + `preview_coloured.ply` (≤ 1M faces), which the
  View tab prefers; mesh.ply (the depth prior and its cached maps) is untouched; SfM mesh / Patch holes drop the
  colours. fullrun_test: 95 % of 93k vertices coloured from 80 photos in 5 s (0.07 s per photo); roof tile rows and
  edges show where the SfM-point colours were mottled.
  - **Live and kept on Stop** (the user asked): `mesh_color.py --live sfmmesh/colour_live.npy --every 100` saves the
    colours so far (uint8 RGB per vertex of mesh.ply); `sfm_mesh.colour(on_live=)` passes each save on, and the GUI
    (`App._on_colour_live`, ≤ every 15 s, View Mesh switched to **sfm**) maps the shown mesh's vertices once to the
    nearest mesh.ply vertex and re-sends it recoloured. Stopped, the last saved colours become mesh_coloured.ply +
    preview (`.json` "partial", "photos_done"). Progress "colouring: photo k / n" on the state line and the bar.
  - **Previews keep the colours**: COLMAP's `mesh_simplifier` writes positions only (and names the face list
    `vertex_index`), so every preview vertex takes the colour of the nearest mesh vertex (`transfer_colours`); before,
    large SfM meshes (kladno_dira) showed grey in the View tab, coloured or not.
- **Which model the mesh uses**: `sparse/export` only while it is not older than `sparse/refined` (else refined), and
  every export first deletes the previous `sparse/export` (it writes one only when it leaves photos out), so a mesh
  never comes from an earlier refine's or export's points. Run full pipeline runs export → SfM mesh → patch holes
  (Run SfM (all) stops at export); the splat then trains on that export (transforms.json, sparse_pc.ply; tiles set up again when
  transforms.json changed) with the mesh prior ticked when a mesh first appears (a user's untick holds).
- **View opens on the export model** (the user asked): snapshot **export** whenever `sfm_mesh.model_dir` picks it
  (not older than refined), else **refined** (`App._main_snapshot`), on start, on opening a project and after
  refine / export / Run SfM (all) / the full pipeline.
- **GUI: its own tab "Sparse Mesh"** (right panel, after Sparse; the user asked): **Run all (mesh → patch holes →
  colour)** (stage `sfmmesh-all`; colouring skipped without nerfstudio), **Build mesh**, **Max hole size (m)**,
  **Patch holes**, **Colour from the photos**. With pose priors the hole size starts at the mean distance between
  consecutive prior cameras (per source; kladno_dira 4.4 m, fullrun_test / karlak 2.0 m; `App._camera_spacing`), else 0.
- **In the pipelines** (the user asked): Run full pipeline ends the sparse part with `sfmmesh` and `sfmmesh-fill`
  (GUI sequence; the CLI pipeline does not) and colours it (`sfmmesh-colour`) as its last step (before COLMAP dense),
  so the splat does not wait for it (~14 min on kladno_dira). **Run SfM (all) does not** (2026-10-22, the user asked:
  the Sparse Mesh is its own thing); it ends at export.
- **SfM mesh prior for the splat** (Splat tab "Depth prior: SfM mesh"; ticked by itself when a project has a mesh or
  one appears, `App._refresh_dense`; the user's untick holds while the mesh stays): the depth loss uses the mesh's
  vertices (`sfmmesh/mesh.ply`, patched if done; ~0.11 m spacing on kladno_dira) instead of the SfM points: the
  lidar-data dataparser's `--depth-ply` (`splat.train_command(depth_cloud=)`, `splat.run / resume`,
  `splat_tiles.run`: the whole mesh for every tile; stored in resume.json / tiles.json). Ticking the prior (by itself
  or by hand) sets Depth weight to 0.2 when it is 0 (`App._mesh_prior_weight`; the user asked for 0.2 as the prior's
  default); without a mesh the weight stays 0 (no depth loss) except for ROS 2 bags.
- **Starting Gaussians from the SfM mesh** (the user asked, 2026-10-22: single-pass flat ground stayed empty; Splat
  tab "Start from the SfM mesh too", default on when the mesh exists; CLI `--seed-mesh`; `splat.run(seed_mesh=)`,
  also tiles / resume): `sfm_mesh.seed_points` spreads points evenly over the mesh (holes patched; spacing = its SfM
  point spacing, coarser above 4M candidates), keeps those with no SfM point of the dataset's sparse_pc.ply within
  2.5 spacings (exact FLANN search: an approximate one overstated distances and seeded duplicates), with face
  normals and colours (mesh_coloured.ply if coloured) -> `splat/mesh_seeds.ply`; the lidar-data dataparser's
  `--seed-ply` appends them to the SfM points (metadata "mf_seed_normals", rotated into nerfstudio's frame); the
  plugin starts them flat: rotation taking z to the normal (`flat_quats`, checked against gsplat), width = the size
  splatfacto gives (3 nearest points), thickness 0.1 of it. fullrun_test (3000 steps, 8 eval frames, PSNR / SSIM /
  LPIPS): flatten 0.02 21.26 / 0.521 / 0.454; + seeds (124k on 47.5k SfM points) 21.27 / 0.525 / 0.449, depth error
  0.69 -> 0.62 %; baseline without either 21.15 / 0.515 / 0.467. Test: `test_seed_points_fill_where_sfm_points_are_thin`.
- **Flattening** (the user asked, 2026-10-22: flat ground seen from a single straight pass at ~45° stays soft;
  Splat tab "Flatten Gaussians (weight)", CLI `--flatten W`, default 0.02 (`splat.FLATTEN`), 0 = off;
  `splat.run(regularize={"flatten": W})`, also tiles / resume): the plugin adds W x mean(smallest / middle scale),
  the middle scale detached (only the smallest axis shrinks: flat discs, no needles). On without a depth weight it
  trains the -lidar method at weight 0. **Bug fixed:** with the middle scale free the optimizer widened Gaussians
  into huge sheets: berlin tile 1 ran out of GPU memory at step ~1500 (21.7 GB vs 1.8 GB; `build/oom_check.py`).
  - A **far-field mask** (pixels deeper than K x the photo's median depth out of the losses, default 1.5) was tried
    alongside and **removed**: it did not work well for the user. Old run records may still name `far_mask`;
    it is ignored.
- **Dense mesh prior** (Splat tab "Mesh prior: dense (rendered depth)", default on; the user asked: the vertices
  supervise only ~1 pixel in 10 at ×2 on kladno_dira): `export/mesh_depth.py` (nerfstudio's venv, Open3D CPU
  raycasting) renders the mesh's z-depth at every training camera **as nerfstudio trains it**: intrinsics × 1/N,
  size floor(w / N), then undistorted and cropped like `nerfstudio.data.utils.dataloaders._undistort_image` (K − 0.5,
  `getOptimalNewCameraMatrix` alpha 0, roi crop, + 0.5), one ray per pixel centre → `sfmmesh/depth_x<N>/<image>.npy`
  (float16 m, 0 = no hit; manifest keyed by mesh size + mtime, N, camera; shared by the tiles). `splat.mesh_depth_maps`
  runs it before training (`splat.run(depth_dense=True)`, also resume / tiles); the dataparser's `--depth-maps`
  attaches the files to the train split (`metadata["depth_files"]`, `depth_scale` = dataparser scale) and the plugin
  takes the map of `camera.metadata["cam_idx"]`, scaled, resized (nearest) to the current resolution, as the
  target: the same robust relative loss over every covered pixel. The camera optimizer's pose changes are ignored.
  - Alignment on fullrun_test against each photo's own SfM points projected with nerfstudio's undistorted camera:
    |dz| / z median 0.18 % (×1) / 0.19 % (×2), p90 1.3 %. 0.17 s per 1920 × 1080 map. 1000 steps, weight 1.0:
    depth error 1.68 % → 0.68 % median (step 500 → 1000), gpu and disk caches alike.
Test: `tests/test_sfm_mesh.py`.

**Mesh from the splat** (`modelforge/splat_mesh.py`, right panel "Splat mesh" tab, or
`python -m modelforge.splat_mesh PROJECT [--quality coarse|medium|fine] [--width 960] [--voxel M] [--max-depth M]`):
an alternative to COLMAP's dense mesh, asked for by the user (who suggested an ESDF over the splat;
a TSDF fused from rendered depth is the standard form of that idea, as in 2DGS and nerfstudio's TSDF
export).
- **How.** `export/splat_tsdf.py` runs in nerfstudio's venv (gsplat + Open3D, GPU).
  - Inputs: `splat/splat_enu.ply` and the transforms.json cameras, at a virtual pinhole of each
    camera's intrinsics (render width 960 by default).
  - Per view it renders RGB + expected depth + alpha and keeps pixels with alpha ≥ 0.7 whose depth
    differs from each 4-neighbour by < 5%. This drops the blended depths at silhouettes, which would
    fuse into "curtains".
  - Depth beyond 3 × the median depth is not fused (background, sky floaters).
  - Fusion: Open3D `VoxelBlockGrid` on CUDA, truncation 4 voxels. Marching cubes over voxels seen
    by ≥ 2 views. Connected pieces under 0.2% of the faces are dropped.
  - Output `PROJECT/splatmesh/`: `mesh.ply` (vertex colours, ENU), `preview.ply` (≤ 300k faces,
    quadric decimation), `stats.json`; stats also in `report.json["splat_mesh"]`.
  - Quality = the voxel as pixels at the median depth: coarse 8, medium 4 (default), fine 2.
- **GPU memory at marching cubes** (karlak, 471 views, 2.1M Gaussians, 9.5 cm voxel: 389k voxel blocks;
  Open3D's CUDA marching cubes ran out of the ~11 GB free after fusing): torch's cached render memory is
  released first, and blocks none of whose voxels reached `min_weight` views are erased (`prune_blocks`;
  136k of 389k there), after which it fit (2.25M faces). Retrying after a CUDA out-of-memory does not work
  (Open3D's CUDA state is broken: "illegal memory access"), and moving a large grid to the CPU crashed
  Open3D 0.20 (segfault), so an out-of-memory now ends with a clear message (coarser quality).
- **Large grids** (kladno_dira, 2026-10-22: 4.0M Gaussians, 2574 views, 433k voxel blocks after pruning):
  - Open3D 0.20's marching cubes crashes from 2^18 = 262,144 blocks on (CUDA: illegal memory access, CPU:
    segfault; 262,000 ran in 3 s with 18 GB free), so the karlak "out of memory" was probably this too. It now runs
    in pieces of ≤ 200k blocks (`marching_cubes`, `MC_MAX_BLOCKS`, param `mc_max_blocks`): each piece a grid of its
    blocks + their 26 neighbours, keeping the triangles whose centroid lies in its own blocks; shared vertices
    merged. Same triangles as one pass (checked on 250k blocks of kladno_dira and the test scene).
  - A hash map filled to exactly its capacity drops random keys on the GPU (stdgpu "bucket and excess list full"):
    the piece grids get 2× the room, and a missing block is an error.
  - Views with nothing to fuse (1–2 stray pixels in range; 14 of 2574) are skipped: Open3D aborted the run on
    them ("No block is touched in TSDF volume").
  - kladno_dira medium (11.6 cm voxel): 23.4M faces, 9 min.
  - **Preview** (berlin, 2026-10-23: 24.3M faces, the process OOM-killed at 27.6 GB after writing mesh.ply): Open3D's
    quadric decimation of the whole mesh needs ~1 KB per face, while the splat, the voxel blocks and the
    marching-cubes arrays were still held. Now they are freed first, and `preview_mesh` clusters vertices first
    (cell = voxel x sqrt(faces / 4 x max preview faces), colours averaged), then quadric-decimates to 300k: on that
    mesh 10 s, peak 3.0 GB (the mesh itself).
  - **Tiles** (berlin fine, 6.6 cm voxel, 2026-10-23: one grid ran out of GPU memory on 24 GB): a cheap guess from the
    Gaussian centres (blocks holding centres x 2.5) decides; above half the budget (0.6 x free GPU memory / 10.5 KB
    per block) one pass counts the blocks exactly (Open3D's per-view block coordinates into a GPU hash set,
    `count_blocks`) and cuts x / y at medians into tiles that fit (`plan_tiles`, margin 2 blocks). A tile fuses only
    the pixels whose surface lands in it + margin (`in_blocks`) from the views that reach it, its grid allocated at
    1.3 x its count (no doubling); its own triangles kept, seam vertices merged (`merge_parts`). Same mesh as one grid
    (synthetic, 81 tiles: identical faces and vertices; `test_mesh_in_tiles`). torch's cache must be emptied after
    `active_blocks` (its GPU copies, ~5 GB, made the next tile not fit). berlin fine: 3.0M blocks, 4 tiles of ~0.76M,
    99M faces / 52.9M vertices, 469 s.
  - **CPU side in numpy** (the same run was OOM-killed at 27 GB in Open3D's `cluster_connected_triangles`, ~420 B per
    face): merge, small pieces (vertex-connected union-find, int32), normals (bincount), binary PLY in chunks
    (float32 positions / normals, uint8 colours), preview by numpy vertex clustering + one quadric pass (it stops near
    434k on that mesh). On berlin fine's mesh these steps peak at 9.1 GB (5.2 GB of it reading the file). `jobs.Token.run`'s error line skips lines that are
    only colour codes (Open3D ends its red errors with one).
- **Patch holes** (Splat mesh tab "Max hole size (m)" + **Patch holes**, stage `splatmesh-fill`; `python -m modelforge.splat_mesh PROJECT --fill [MAX]`; the user asked for the SfM
  mesh's option here, 2026-10-22). Marching cubes leaves real gaps (no Poisson surface underneath as in the SfM mesh),
  so `splat_mesh.fill` closes border loops: every loop that is not its piece's widest (the outer border; for the
  ground the scene's edge, for a floater its rim) and is at most the size across (bounding box; 0 = 40 voxels; the GUI
  starts at the camera spacing like the Sparse Mesh tab) gets a fan of triangles from its mean point, colour and
  normal the rim's mean. Where two holes touch at a vertex (2 border edges in, 2 out; frequent in marching cubes),
  turning around it through its faces finds the other hole's outgoing edge, so each hole gets its own fan; more
  holes at one vertex stay open. `mesh_unpatched.ply` keeps the marching-cubes mesh (patching again starts from it;
  Build mesh deletes it), `patched.npy` marks the new faces, stats / report `splat_mesh.fill`. kladno_dira at 4.4 m:
  94,511 of 94,607 loops closed with 1.03M faces, 4 min (mostly reading, the piece labels and the preview).
  - `sfm_mesh.read_mesh` reads triangle PLYs as one block (plyfile `known_list_len`: 23M faces 0.1 s instead of a
    Python object per face) and `write_mesh` writes binary directly (23M faces 12 s); `sfm_mesh.union_labels` is the
    shared union-find.
- **Open3D gotcha:** the legacy `ScalableTSDFVolume` (Open3D 0.20 in the nerfstudio venv)
  returns no surface at all, even for a synthetic wall integrated 5 times. The tensor
  `VoxelBlockGrid` works; its `extract_triangle_mesh` defaults to weight ≥ 3 (views).
- **Depth bias:** gsplat's expected depth blends Gaussian *centre* depths, so oblique views pull
  depth toward the camera by about the Gaussian size (synthetic 9 cm discs at ~50°: 6.8 cm,
  1.1%). Ray–Gaussian intersection depth (2DGS, RaDe-GS) would remove it; gsplat 1.4 lacks it.
- **Measured on `data/rosbag_exp`** (40 views, the LiDAR-supervised splat), against the LiDAR cloud:

  | | faces | mesh → LiDAR median / p90 | LiDAR covered (< 30 cm) | time |
  |---|---|---|---|---|
  | splat mesh medium (13.8 cm voxel) | 610k | 14.5 / 51 cm (100% within 2 m) | 77% | 9 s |
  | splat mesh fine (6.9 cm) | 2.04M | 13.8 / 38 cm | 73% | 24 s |
  | COLMAP Delaunay, superfast 320 px | 76k | 26.8 / 121 cm (10% within 2 m) | 37% | |

  The COLMAP row is the 320 px preset, so it is not a like-for-like comparison. The test LiDAR is
  the MVS fused cloud (~8.5 cm spacing), which bounds the medians.
- GUI: quality, render width, max depth (0 = auto), **Build mesh from splat**. Not part of
  Run full pipeline since 2026-10-22 (the user asked). The View tab's
  Mesh dropdown gets `splat` (vertex colours) and switches to it; the Export tab exports it
  (glb / obj / ply).
- Mesh export now rotates with `trimesh.apply_transform`, which keeps the stored vertex normals.
  Assigning vertices dropped them, and trimesh recomputes normals in a slow Python loop without
  scipy: 105 s for a 173k-face mesh, now < 1 s.
- Tests: `tests/test_splat_mesh.py` (a synthetic splat of ground + box: geometry within 1.5
  voxels, colours, View / Export wiring).

**Semantic search** (its own **Search** tab on the left, after View; `modelforge/semantic.py`, `gui/semantic.py`,
`export/semantic_features.py`, `export/semantic_text.py`; the user asked, 2026-10-19: C-RADIOv3, option A,
fine detail, a sidecar so the normal splat export stays unchanged):
- **Features** (job stage `semantic`, "photo k / n"): C-RADIOv3-L in nerfstudio's venv on every photo,
  undistorted, at 2048 px (128 × 72 patches for 16:9). The SigLIP2 adaptor's patch features are **not**
  text-aligned as they are (they mimic SigLIP2's patch tokens: "a machine" was low on the machine); applying
  SigLIP2's attention-pooling head per patch without its attention (v_proj → out_proj → + MLP(LayerNorm), the
  MaskCLIP trick; head weights from google/siglip2-so400m-patch16-naflex) makes "a machine", "gravel" and
  "dirt road" light up where they belong. PCA (64 of 1152 dims, ~91 % of the variance) on patches of 24
  photos. Per photo: the splat's depth rendered (gsplat, input / 4), every Gaussian centre projected and kept
  when in frame and within 3 % of the rendered depth (occlusion), the reduced feature sampled bilinearly,
  accumulated weighted by opacity. Output `splat/semantic/features.npz` (feats float16, weight, PCA mean +
  basis) + `meta.json` with the splat's fingerprint (size + mtime): a new / resumed splat → "stale".
- **Query**: `semantic.TextEncoder` keeps a worker with the SigLIP2 text model (the first search loads it,
  ~17 s once the weights are cached; later ones ~0.3 s). `Features.relevance`: cosine in the full space
  (reduced features mapped back through the basis, norms precomputed), LERF-style relevance against
  "object / things / stuff / texture": min over negatives of sigmoid(10 (cos_q − cos_n)).
- **GUI**: Compute semantic features, Query + Search, Threshold (0.5–0.95, default 0.6), Show (highlight
  in red / matches only), Clear. Only the colours / selection sent to the browser change
  (`App.redraw_splat_decorated` + `SemanticPanel.decorate`; scores follow `gdata.splat_subset`, so a shown
  subset maps correctly).
- **Measured on kladno_dopravnik** (56 photos 4K, 1.02M Gaussians): 38 s, 900k Gaussians seen. "a car" →
  exactly the car (366 Gaussians); "gravel pile" → the piles; "dirt road" → the tracks; "a machine" → the
  crusher clearly, but at 0.6 also much of the ground around it (raise the threshold).
- **Packages**: RADIO needs einops, timm ≥ 1.0, transformers, which clash with nerfstudio's pins; they live in
  `/opt/radio-pkgs` (Dockerfile layer, CUDA image; not yet rebuilt) or `build/radio_pkgs` (installed by hand
  with the same `pip --no-deps --target` list), first on PYTHONPATH only for these subprocesses. Weights cache
  in `build/hf_cache` + `build/torch_hub` (`MODELFORGE_MODEL_CACHE`): C-RADIOv3-L 1 GB, SigLIP2-so400m
  4.3 GB. License: NVIDIA Open Model License.
- **Objects** (the user asked for bright turning 3D markers): `semantic.objects` groups the Gaussians above
  the threshold by 26-connected voxels (1/300 of the scene; `_components`: vectorized min-label propagation,
  no scipy), drops groups under "Min object size" (30), ranks by summed relevance above the threshold; each:
  centre, radius (95 % horizontal), z0 / z1. The panel lists them ("#1 · 5.5 m · 0.61 · 341 G"), draws a
  bright yellow cylinder (two rings + 8 struts, screen-width lines) with a label per object, turning at
  30°/s (one thread for the panel's lifetime: restarting it per redraw raced), and **◀ / Go / ▶** glide the
  camera (`App.glide_to`, smoothstep 1.2 s) to the registered camera that saw the object best (in frame
  with a margin, near, close to its axis; `best_view`), pulled back to ≥ 2.5 radii, looking at the object;
  else an elevated view fitting it.
  - kladno_dopravnik: "a car" → 1 object (5.5 m); "a machine" at 0.66 → the crusher (16 m, its
    conveyors) + 2 specks; "gravel pile" at 0.65 → 20 piles (neighbouring piles can merge: one 56 m).
- A new search **goes to the best object at once** (in the browser that pressed Search).
- **Enter searches**: the Query box is multi-line only so a newline (Enter) reaches the server (viser's text
  box has no submit event); the newline is removed and the search starts.
- **Auto threshold** (default on; `semantic.auto_threshold`; the user asked for an automatic threshold): the
  threshold is lowered from the best match in 0.01 steps, regrouping objects each step, and stops at the
  last value before (1) **specks**: ≥ 3 new groups under 3 × min size while no new substantial object
  (≥ 10 × min size) appears, or (2) a **merge**: the largest object grows ≥ 1.5 × by absorbing another
  substantial one of ≥ 30 % its size (fragments joining their own object are not merges); floor 0.55. A best
  relevance below 0.6 → "probably not in the scene" (threshold to the top, no markers, no camera jump).
  - Sweep on kladno_dopravnik (mesh filter on): "a machine" 0.66 (specks; crusher alone), "gravel pile" 0.68
    (specks; 18–19 piles, in the plateau before piles merge at 0.60), "a car" 0.55 (floor; one object all
    the way), "a person" / "a tree" absent (best 0.56 / 0.58), "dirt road" 0.65 (fragments: diffuse
    "stuff"). Earlier rule versions stopped gravel at 0.75 / 0.74 / 0.73 on fragments forming.
  - Statistics alone (median + k·MAD of the relevance) cannot do it: "a machine" needs k ≈ 2, "a car"
    k ≈ 12.
- **Near the mesh only** (default on, tolerance 3 mesh voxels; the user asked to prune with the mesh):
  matches whose centre is farther than the tolerance from the mesh count as 0 for the highlight, the counts
  and the objects. `semantic.search_mesh`: the splat mesh (its TSDF voxel) else COLMAP's Delaunay / Poisson
  (median edge length); `Surface`: nearest mesh vertex per Gaussian (OpenCV FLANN, approximate), **measured
  only for the matches** (relevance ≥ min(threshold, the auto floor 0.55)) and remembered while the mesh and the
  splat stay the same; the k-d tree is built in the background when a project opens (~15 s at 12M vertices).
  Before (2026-10-22) every Gaussian was measured once per splat + mesh: on kladno_dira (4.0M Gaussians, the
  patched 12.4M-vertex splat mesh) a search hung for minutes, also because plyfile read the mesh's 24M faces as
  Python objects (~20 GB); the user suggested checking only what the objects are made of. On kladno_dopravnik (splat mesh, 10.7 cm voxel) only 54 % of the
  Gaussians lie within 3 voxels of the mesh (74 % within 1 m: floaters, haze, terrain beyond the mesh's
  range); "a machine" at 0.66 → 1 object instead of 3 (944 floating matches ignored), "a car" loses 16 %
  of its matches (glossy parts), "gravel pile" 20 → 18 objects. It cannot fix matches bleeding onto real
  surfaces (the ground around the machine): that is the threshold's job.
- **In the full pipeline** (the user asked): after the splat, unless the Search tab's "Include in Run
  full pipeline" is off; the History line: "900,283 of 1,018,377 Gaussians from 56 photos (c-radio_v3-l,
  2048 px), 38 s".
- **Unloading for jobs** (the user asked): `App.run_stage` calls `SemanticPanel.unload(stage)` for every
  job (SfM, dense, splat, meshes, the semantic features, …): the text worker stops and waits to exit (~2.7 GB
  RAM, ~1 GB GPU freed). It loads again in the background on the first keystroke in Query, or when
  searching. viser does not report switching tabs (selection is client-side only), so typing is the
  first sign the Search tab is in use.
- **Preloading**: `main` calls `SemanticPanel.preload()`, which starts the text worker in the background when
  the GUI starts (~13 s; "text model: ready" in the tab); a search meanwhile waits for it. Opening a project
  with ready features also loads them and the splat positions in the background (`_warm`), so the first
  search takes ~0.4 s instead of ~6 s. Not done in `App()` itself: tests create apps.
- Tiled splats: features per tile, merged (see "Splat tiles").
- Not yet: tiling above 2048 px, distillation (option B).
- Tests: `tests/test_semantic.py` (scoring, staleness, the panel's decoration; synthetic features).

**Export** (`modelforge/export/products.py`, Export tab, or
`python -m modelforge.export.products PROJECT [--mesh-format glb|obj|ply] [--splat-format ply|splat]`):
- Writes to `PROJECT/export/` (a final result).
- Mesh:
  - `glb` is one textured file; `obj` writes `.obj` + `.mtl` + texture.
    Both are Y-up = (east, up, −north), as glTF and Blender expect.
  - `ply` copies COLMAP's files as they are, in ENU.
  - COLMAP's textured PLY has per-face UVs. They are unwelded to per-vertex
    UVs, then identical position+UV pairs are merged. COLMAP's UVs use the
    OpenGL convention (origin bottom-left), the same as trimesh; this was
    checked against image colours.
- Splat: `ply` (standard 3DGS, all SH bands) or `.splat` (32 bytes per
  Gaussian, colour only). Both are in ENU.
- `export/georef.json` records the ENU origin (lat, lon, alt) and the axes of
  each exported file.
- The GUI can also send the export to the browser (several files as a .zip).

Why a separate venv: nerfstudio pins viser 1.0.0 and opencv 4.10 and wants
numpy<2, which clash with `/opt/venv`. The two environments exchange only
files. `splat.py` strips `VIRTUAL_ENV`/`PYTHONPATH` and puts
`/opt/nerfstudio/bin` first on PATH for its subprocesses.

Steps:
1. Trains `ns-train splatfacto` with `--output-dir PROJECT/nerfstudio`; the
   viewer is on :7007. `compose.gpu.yaml` publishes 7007 and sets
   `shm_size: 8gb`. Progress comes from nerfstudio's table rows
   ("12000 (40.00%) ..."); ANSI cursor escapes are stripped. splatfacto
   loads our `sparse_pc.ply` (`load_3D_points=True` in its method config).
2. Exports with `ns-export gaussian-splat` → `PROJECT/splat/splat.ply`, in
   nerfstudio's normalized frame.
3. Converts with `python -m modelforge.export.splat_to_enu` →
   `PROJECT/splat/splat_enu.ply`, using `dataparser_transforms.json`
   (x_ns = s·(R x + t)). Positions, log-scales and wxyz quaternions are mapped
   back, and the SH bands are rotated (degree ≤ 3, D_l by least squares over
   sampled directions).
4. Stats go to `report.json["splat"]`: method, iterations, number of
   Gaussians, times.

Tests:
- `tests/test_splat_to_enu.py` checks that positions, covariances and
  view-dependent colour render identically.
- `tests/test_dense_chain.py` runs exact depth/normal maps of a bumpy ground
  through fusion (≤1.6 cm error), Delaunay (0.6 mm median), Poisson
  (1.9 mm), texturing and preview.
- `tests/test_gui_dense.py` covers the GUI layers and runs a dense stage from
  the GUI.

## GUI (M2, viser)

`scripts/dev python -m modelforge.gui.app [PROJECT] [--root data] [--port 8080] [--run STAGE,STAGE]`,
then open http://localhost:8080. Without PROJECT it opens the project opened
last (`last_project` in `ROOT/.modelforge_gui.json`, saved by `main` and on every
Projects → Open), else the newest project under `--root`. Before, it was always the
newest by file times, so a project only opened (not changed) was not reopened. `--run` starts those stages (e.g. `export,sfmmesh,sfmmesh-fill,splat,sfmmesh-colour`) at once as one job with the tabs' defaults, shown in the GUI like a button's (`run_stage` takes a tuple; test `test_run_a_stage_list`). compose publishes 8080. viser 1.1.1 and plotly 7.1.0 are
pinned in the image.

Code layout:
- `modelforge/gui/data.py`: a `Project` turns priors, snapshots (`priors`,
  `sparse/round_N[_tri]`, `refined`) and thumbnails into numpy. Poses are
  world_from_cam, the same convention as viser frustums (OpenCV, wxyz =
  scalar-first quaternion of R_wc).
  - Snapshots are read straight from COLMAP's binary files with numpy
    (`Project.read_snapshot`), not through `pycolmap.Reconstruction` and
    per-keypoint Python loops. kladno_dira (8000 images, 83M keypoints, 8M
    points): 271 s → 8.4 s. While it loaded under `App.lock`, the state line
    could not update, so a splat started meanwhile showed "idle".
    - `points3D.bin` is sorted by id; track lengths are counted from
      `images.bin`, which gives every record's offset without a loop.
    - Inspect reads the clicked image's keypoints from `images.bin` on demand
      (`Observations`), 0.08 s.
    - Results equal the pycolmap path (`test_read_snapshot_matches_reconstruction`);
      on an unexpected layout it falls back to pycolmap.
    - Live refine events read the snapshot the pipeline has just written.
- `modelforge/gui/projects.py`: the **Projects** tab (first tab), built on
  `modelforge/projects.py`.
  - Lists project folders under the root: any folder with `ingest.json` or
    `poses.json`.
  - **Open** switches the whole GUI via `App.open_project`: it clears the
    scene, reloads the Project, calls `IngestPanel.set_project`, and resets
    the Run state/history/log. Switching is refused while a job runs.
  - **Create** makes a new project, optionally copying the open project's
    `ingest.json`.
  - Status: sources, image count, stages done, and final vs. intermediate
    files with their sizes.
  - **Rename / delete** (`projects.rename` / `projects.delete`) acts on the
    selected project folder directly under the root.
    - Renaming the open project reopens it under the new path, and is
      refused while a job runs in it.
    - Delete needs a confirmation box and refuses the open project.
    - Paths inside a project are relative, so a rename breaks nothing; only
      old nerfstudio run configs keep the old absolute path, and they are
      not reused.
  - **Clean intermediates** needs a confirmation box. It deletes the
    databases, `sparse/round_*`, the dense workspace (depth maps),
    `nerfstudio/` runs, `splat/splat.ply` and `previous/`. It keeps the
    images, sources and final results.
- `modelforge/gui/videos.py`: the **Videos** tab, built on `modelforge/videos.py`.
  The user wanted a manager for the *videos*, not for the project's internal
  files; a project-file browser was built first and then removed.
  - The library is every clip under the data root outside project folders
    (`ingest.find_videos`). Browser uploads (video and .SRT) go to
    `ROOT/videos/`.
  - Per clip: size, length, resolution / fps and recording time (ffprobe,
    cached by size + mtime), SRT present, and the projects using it (their
    `ingest.json`).
  - Actions: add to the open project (`IngestPanel.add_source(video)`),
    download, rename and delete. Rename and delete move the .SRT along.
  - **The library is `data/videos/`** (`videos.LIBRARY_DIR`, `_lib`); projects
    live next to it in `data/`.
    - Why: the user could not create project "karlak" while a video folder
      `data/karlak/` existed, because both shared the top of `data/`.
    - `videos` is a reserved project name (`projects._check_name`).
    - Their `berlin/` and `karlak/` folders were moved into `data/videos/`
      with `update_references` (parking and fullrun_test now point at
      `data/videos/berlin/berlin_3_1.MP4`).
    - The Ingest tab still finds clips anywhere under `data/` outside
      projects.
  - **File manager** (the user asked for "a more traditional file manager"
    with context buttons, a + for upload and a new-folder button):
    - Toolbar button group (⬆ Up, 📁＋ New folder, ⟳ Refresh) plus the
      **＋ Upload** button (viser's upload widget cannot sit in a group).
    - One row per entry under a "Files" folder: a button group
      `[name][✏][🗑]` (stray .SRT: `[name][🗑]`). viser button groups share
      the width but never shrink below a label, so the name keeps its width.
    - Clicking a folder name opens it; clicking a clip shows it under
      "Selected clip" with Add to open project / Download.
    - ✏ / 🗑 / New folder open a viser modal (name input or "Delete …? There
      is no undo").
    - **➜ Move** a clip (with its .SRT) or a folder to another library folder
      (`videos.move`, destination picked from `all_folders` in a modal).
    - Renames and moves keep projects pointed at their clips:
      `update_references` rewrites the matching path strings in every
      project's `ingest.json` and `poses.json`, stored the way they were
      (relative or absolute).
      - So renaming a used clip is no longer refused. Delete still is.
      - The open project's (maybe unsaved) Ingest sources follow in memory
        (`_retarget_open_project`).
      - Both are refused while a job runs.
      - A clip's cached optical-flow lead moves with it.
      - The user's berlin_3_1.MP4 / .SRT now live in
        `data/videos/berlin/`.
    - Folders can be renamed too (`rename_folder`); a folder holding a
      project cannot be renamed, moved or deleted.
    - Rows are rebuilt on each refresh (`_refresh`, ≤300 rows).
    - **New folder** (`make_folder`); **Delete** a folder (`delete_folder`,
      refused for the root, for folders holding a project folder and for
      folders holding a clip a project uses).
    - Upload goes into the current folder (`save_upload(rel_dir=)`); the
      Ingest tab's upload still targets `videos/`.
    - Stray .SRT files (no video of that name) are listed and can be deleted
      (`delete_file`).
    - Project folders and hidden files are never listed, entered or deleted
      (`resolve_dir` refuses them).
  - Rename and delete are refused while a project uses the clip, including
    the open project's unsaved Ingest sources.
  - Delete needs a confirmation box.
- **ROS 2 bags in the library** (the user asked, 2026-10-18):
    - A bag is an `.mcap` file or a folder with `metadata.yaml`
      (`rosbag.bag.is_bag`). It is listed as one 🤖 row (size, duration, topic
      count, used-by), never opened as a folder. Details show the image /
      point-cloud / odometry topics and TF.
    - **Upload** (the same ＋ button) takes:
      - an `.mcap`;
      - a bag folder packed as `.zip` / `.tar` / `.tar.gz`, unpacked by
        `videos.extract_bags`. Loose bag files at the archive's top go into a
        folder named after the archive. Members leaving the folder are
        refused (zip checked by hand, tar with `filter="data"`). An archive
        without a bag keeps nothing;
      - the files of a bag folder (`metadata.yaml`, `.db3`) one by one into a
        folder. The browser picks one file at a time, so a zip is easier.
    - Rename / move / delete work as for clips. Projects follow because
      `_tree_moves` maps folders too (a bag folder is a source path) and
      `update_references` rewrites any path-like string that moved.
      Deleting a used bag, or a folder holding one, is refused.
    - Deleting a clip, bag or folder used by a project (including the open
      project's unsaved sources) opens an "… is in use" dialog naming the
      projects. Before, only the status line at the top of the tab said so,
      and the user took it for a broken delete.
      - **Remove from sources and delete** (the user asked) drops it from
        every project's sources (`videos.remove_from_projects`; the open
        project's Ingest list in memory too), then deletes it.
      - The projects keep their images and results; only re-extraction
        needs the source.
    - Download sends a bag folder as a `.zip` (stored, built in memory).
    - Add to open project only pairs bags with ROS 2 bag projects and clips
      with the others (also checked in `IngestPanel.add_source`).
    - Tests: `tests/test_videos.py` (`test_bag_uploads`,
      `test_bag_rename_move_delete_follow_projects`).
  - **Uploads.** viser buffers an upload in server memory and delivers it only
  when complete, so a 1.2 GB clip needs ~2.4 GB of RAM at peak.
  `App._watch_uploads` shows progress from viser's in-flight transfer table
  (`_current_file_upload_states`, private, read defensively). Files are
  written via a `.part` temp file, then renamed.
- `modelforge/gui/ingest.py`: the **Ingest** tab (first tab when the
  project has no images yet). `python -m modelforge.gui.app data/new_site`
  works on a new or empty folder.
  - **Videos**: lists the library's clips (data root incl. `data/videos/`;
    project folders are skipped), or takes a path. "Add as source" adds the
    whole clip. **Upload video / SRT** saves into the library, as the Videos
    tab does.
  - **Sources**: one folder per source, with a stats line, a range
    multi-slider, Enabled, a lead override, Preview, "Add piece" (a second
    range of the same video) and Remove. Clips decode in a background planner
    thread, a few seconds each, cached by `ingest.load_clip`.
  - **Sampling**: N over all sources, spacing, rotation credit, window, yaw
    source. The plan summary and warnings update live, and every edit saves
    `ingest.json`.
  - **Frame preview**: source + time slider. A project switch clears it, and an added source is
    selected and shows its first frame once decoded (a preview generation counter: an identical
    source label + time in the new project used to be skipped as already shown). It shows the video frame
    (ffmpeg accurate seek), the aligned telemetry and a white frustum in 3D.
    "Start here" / "End here" set the range.
  - **3D**: each source's track, bright in its colour inside the range and
    grey outside, plus the planned image positions. (Its plots of altitude /
    speed / motion were removed on 2026-10-19: the user found them not useful.)
  - **Extract**: runs `ingest.run` with progress. A project with images needs
    the Overwrite box ticked. Afterwards the app reloads the project
    (`App.load_project`) and the Sparse tab takes over.
- `modelforge/gui/app.py`. Layout, as the user asked:
  - The **left panel** (`self.left = gui.add_panel()`, `dock_left`, 420 px)
    holds Projects / Videos / Ingest / View / Inspect / Export. (A Plots tab
    with deviation and reprojection-error charts was dropped on 2026-10-19: unused.)
  - The **right (main) panel** (`main_panel.dock_right()`) only runs the
    methods: header, run state + progress, **Run full pipeline**, tabs
    Sparse / Splat / Splat mesh / Dense (COLMAP's MVS last: the full pipeline's
    dense result is the splat mesh), then History.
  - Browser tab: viser hardcodes `<title>Viser</title>` and its icon in its built `index.html`, so the Dockerfile
    runs `docker/patch_viser_page.py` (a late layer): title "ModelForge", our icon (`docker/favicon.svg`, an orange
    cube), and a script that follows the hidden `#mf-status` element (`App._status_html`, kept by `_watch_status`
    each second): the title reads "ModelForge · refine 42 % · berlin" / "ModelForge · idle · berlin" / "… failed …",
    the icon gets a green dot while a job runs, a red one after a failure (the user asked, 2026-10-22). Applied to the
    running container by hand too (`docker exec -u root … python docker/patch_viser_page.py INDEX docker/favicon.svg`;
    the original is kept as `index.html.orig_mf`). The panel label is `ViserServer(label=...)`.
  - **Hardware** window (bottom, right of the log, `modelforge/gui/hardware.py`, the user asked): live CPU, RAM, GPU
    utilization and GPU memory (bars + numbers + GPU temperature / power), and a 2-minute history plot.
    Sampled at 1 Hz from `/proc/stat`, `/proc/meminfo` (host-wide, also in the container) and
    `nvidia-smi` (no psutil / pynvml in the image). Tests: `tests/test_gui_hardware.py`.
  - **Settings** tab (left panel): **Dark mode**.
    - `App.set_dark` calls `configure_theme(dark_mode=...)` for every client
      (the default `control_layout` sends no dock command, so the panel
      placement survives).
    - The Plotly charts switch to the `plotly_dark` / `plotly_white`
      template.
    - The choice is remembered in `ROOT/.modelforge_gui.json`.
  - **Quick bar** (the user asked): a small panel floated at the top of the
    3D view (`self.quick`).
    - One button group in **two rows of 7** (the user found one row of 13, ~1250 px, too wide):
      ● / ○ / · Splat, Mesh, Dense (fused points), Points (sparse), Cams, Thumbs (camera
      thumbnails; available while Cams is on) / Resid (View "Prior → refined": the red prior →
      refined segments), Traj (View "Trajectories"), Plan, Tiles (View "Tile boundaries"), Orbit, ⌖ Center,
      Fly. Short labels (`QUICK_SHORT`): viser gives every button the longest label's width (~72 px) and
      wraps; the panel is sized for ⌈n / 2⌉ per row (13 buttons: 7 + 6, ~528 px).
    - **⌖ Center** (also View tab "Center the view"; `App.recenter`): looks at the cameras + sparse
      points (2–98 % range, so stray far points don't count) from the south, above them. The same
      framing runs when a browser connects and on a project switch. The user lost the scene of a
      video-kind project (arbitrary frame and scale) and asked for it.
    - **Default view** (the user asked, 2026-10-21): on start and on opening a project only **Points** and
      **Trajectories** are on (`_default_view`); splat, mesh, dense points, cameras, thumbnails, prior cameras,
      Prior → refined and tile boundaries are off (a large splat is no longer sent at start). The flight plan is
      on only for a project without images (nothing else to see). After an SfM run a project with nothing else
      to show still gets Cameras (≤ 2000) on (`_auto_show_sparse`); after a splat run the splat is shown.
    - **Plan** is the Ingest flight plan (each source's track plus dots at the
      planned image positions). The user asked to hide it from the 3D view:
      its checkbox existed only in the Ingest tab. The quick bar, View
      "Flight plan" and Ingest "Show flight plan in 3D" stay in step
      (`App._set_plan_visible`). These mirror the View tab controls
      (`_quick_click` flips them; `_sync_quickbar` keeps the labels in
      step).
    - viser button groups cannot relabel in place, so the row is rebuilt
      (same order) when a state changes.
    - The log window starts minimized and History collapsed, as the user
      asked.
  - The **log** is its own panel (`self.log_panel`), floated at the bottom of
    the 3D view (`float(x=8, y=-8, width=760, height=260)`; negative y is
    measured from the bottom). viser docks panels only to the left / right
    edges or above / below another panel, so "bottom", as the user asked, is
    a float. It shows the last 200 lines, newest first, because markdown
    cannot auto-scroll.
    - The Hardware monitor (`self.hw_panel`) floats right of it (x=776,
      480 px). The user wanted them split side by side; a floating panel
      cannot be split into columns (only docked ones can), so they are two
      windows. Both start minimized.
  - **Run full pipeline** (stage `all`) runs SfM → SfM mesh → splat → semantic features (Search tab
    "Include in Run full pipeline", default on; skipped without the RADIO packages) → colouring the SfM mesh, with
    the tabs' settings. **No splat mesh** (2026-10-22, the user asked; from 2026-10-18 until then it was the
    pipeline's dense result): Splat mesh tab by hand. COLMAP's MVS follows only with the Dense tab's "Include in Run full
    pipeline" (off by default). A project without images is extracted first from
    its Ingest plan (`IngestPanel.extract_now`). Splat / dense are skipped, with a
    History note, without nerfstudio / CUDA.
  - **Stop asks first** (2026-10-22, the user asked: a misclick cost a long run): a "Stop?" box naming the job, Stop /
    Keep running (`App._stop_clicked`; without a browser client, e.g. tests, it stops at once).
  - **Stop** (`App.stop`, `modelforge/jobs.py`) stops whatever runs: a
    stage, the full pipeline, splat training or an extraction. Every job
    gets a `jobs.Token`:
    - Subprocesses started through `Token.run` are terminated at once:
      stereo, Delaunay, texturing, mesh simplification, ns-train, ns-export.
    - Bundle adjustment aborts within an iteration (its Ceres callback
      returns `SOLVER_ABORT`). An aborted round writes no `round_N`, and a
      stopped refine leaves `sparse/refined` as it was.
    - Extraction checks the token at every frame.
    - In-process pycolmap calls (features, matching, triangulation, fusion)
      cannot be interrupted; the token is checked when they return, and the
      state line reads "⏹ stopping …" meanwhile.
    - A stopped run ends as "⏹ stopped during <stage>", not as a failure.
    - The splat-only Stop button was folded into this one.
  - Tabs:
  - **View**: snapshot picker and stats, including the calibrated prior model.
    A "Dense / splat" folder shows `fused.ply` (≤1.5M points), the mesh and
    `splat_enu.ply` (viser gaussian splats), each enabled once its file
    exists.
    - The mesh is the **textured** one (`textured-<m>/mesh.ply` +
      `texture.png`, loaded with `products.load_textured`) when the texture
      stage has run, it is ≤1.5M faces, and **Mesh texture** is on.
    - Otherwise it is the grey simplified preview: COLMAP's meshers write
      geometry only, so `meshed-*.ply` / `preview-*.ply` have no colour.
      The user asked why the mesh was grey.
    - Textured meshes in the user's projects: 94k–156k faces, an 11 MB GLB,
      ~2.6 s to load and convert.
    - The Splat checkbox is off on start / project open (see Default view), ticked after a splat run.
    - The layers refresh right after the splat stage and after fusion /
      mesh / texture, not only when the run ends. Before, the splat vanished
      during the full pipeline's dense stages: the live layer was gone and
      the final splat was not drawn yet.
    - `App._unchanged` remembers each layer's inputs (file + mtime +
      options), so these refreshes don't re-send unchanged layers (the
      splat is tens of MB).
    - Each layer has its own status line (`_dense_status`).
  - **Orbit** (View tab, the user asked): **Start / Stop orbit** + speed
    (°/s). **Since 2026-10-22 the browsers turn the view themselves** (the user could not zoom while orbiting: the
    server's 30 Hz poses overwrote each wheel step, a ~50 ms animation in the browser, before it was reported):
    `docker/patch_viser_camera.py` (was `patch_viser_keys.sh`) adds a per-frame `rotate` (as the arrow keys) at `data-orbit` °/s of the hidden
    `#mf-move-speed` element; the server only sets that (`App._set_orbit`), shared by all browsers; Fly, Center and
    glides turn it off. viser renders on demand with a 1 s tick, so it starts within a second; steps capped at 0.1 s.
    The notes below describe the earlier server-driven orbit (`_orbit`, removed).
    - `App._orbit` rotates the pressing client's camera about +z through its
      look-at point at 30 Hz (`client.atomic()` sets position + look_at
      together). One stop event per client.
    - Each tick starts from the browser's *current* camera, so mouse zoom /
      tilt / pan stick and the orbit continues from the new view, as the
      user asked.
    - A camera that matches none of the last 15 poses sent (echoes) counts
      as the user dragging; rotation then pauses until 0.6 s after the last
      such change, so it doesn't fight the mouse.
    - The browser reports the camera a few updates late. On an echo the
      orbit continues from the **latest** pose sent: restarting from the
      echoed older pose made it jump back in time, as the user saw. A camera
      it adopts (at start, after a drag) joins the sent list, so its late
      echoes don't count as a drag (a 0.6 s stutter).
    - Tested with a stand-in browser that lags 3 updates.
  - **Fly** (View tab folder + quick bar, the user asked): replays the camera path.
    - `gdata.FlightPath`: the shown snapshot's registered cameras in capture order, parametrized by
      **distance** (the user asked for 0.5 m/s by default, loop on): segment = max(distance, 0.02 m ×
      rotation in degrees), so turns on the spot take time (25°/s at 0.5 m/s); where the project
      timeline pauses over 3 s (between videos) the jump is shortened to at most 2 m. Scene units
      for videos without metadata.
    - Positions follow a cubic Hermite curve (finite-difference tangents); orientations are slerped.
      The curve passes exactly through every camera.
    - **The browsers fly it** (2026-10-23, the user asked; before, the server's `App._fly` sent the pose at 30 Hz):
      `docker/patch_viser_camera.py` adds `window.__mfFly` to the page, called from viser's per-frame camera hook.
      It reads the path from the hidden `#mf-fly-path` (`data-path`: path times, centres, wxyz quaternions,
      sent once per path) and the clock from `#mf-move-speed` `data-fly` (path id, epoch, t0, speed, loop, look-at
      distance, hold). It evaluates the same Hermite curve and slerp as `FlightPath` (checked in node: 0.1 mm, the
      sent rounding), maps them through viser's scene root transform `NL(e)` and calls camera-controls
      `updateCameraUp` + `setLookAt`. Look-at = ahead by the median depth of the sparse points (so the mouse pivot
      afterwards lies in the scene); up = −y of the OpenCV camera (roll reproduced).
    - **One flight for all browsers** (as Orbit). The server only runs the clock (`_fly_clock`: a new epoch on
      start, speed / Loop changes and slider jumps; `_fly_now`) and `_fly_watch` at 4 Hz (slider, status, photo
      window, the end with Loop off). A browser connecting during a flight joins it (no recenter); the last one
      closing ends it; opening a project ends it.
    - Controls: speed in m/s (default 0.5), Loop (default on), and "Camera's field of view" (on: the real
      camera's vertical FOV while flying; the browser's FOV and up come back afterwards).
    - The **Position** slider shows progress; dragging it jumps there, also when not flying.
    - A drag, wheel or held movement key on the 3D view **pauses** the flight and hands the camera to the user
      until 1 s after the last one (the user asked: not skip ahead). The page freezes its clock at once and reports
      "pause:k:t" / "resume:k:t" to the server through a hidden text input (`g_fly_report`; a `GuiUpdateMessage` sent
      with viser's own `sendMessage`, its uuid in `data-fly`). `App._fly_report` holds the clock at t (`paused`)
      while any browser holds it, for all browsers, the slider and the photo window, and continues from t when the
      last one lets go (or disconnects). If no answer comes within 3 s, the page follows the server's clock again.
    - **Photo window** (the user asked, 2026-10-22): a floating panel right of the quick bar (`photo_panel`,
      420 px wide, height from the images' aspect) shows the photo of the camera the flight is nearest to
      (`FlightPath.pose`'s index), also when the Position slider jumps; decoded at ~630 px with JPEG DCT
      scaling (`Project.photo`, ~20 ms) only when the camera changes.
    - **Navigation** (the user asked, 2026-10-22: zooming in made moving ever slower): viser orbits a pivot
      (`look_at`) and its wheel moves a fraction of the distance to it, never past it. **Ctrl + click** sets the pivot
      to the clicked spot (`App.ray_hit`: the nearest group of >= 3 sparse points within 0.6° of the ray, else the
      ground; a yellow ball for 2 s); **View → Carry the pivot forward** (default on, `_carry_pivot` on every camera
      update, not while Orbit / Fly drive it): within 1 % of the scene's size (>= 0.3 m) the pivot moves 8x that far
      ahead. viser's own keys: W A S D move, R / F up / down (viser's Q / E), arrows turn; speed View → **Move speed** (default 5 m/s, remembered; viser's fixed 2 m/s). Both by `docker/patch_viser_camera.py`, which makes viser's bundle loader edit the unpacked code (the speed is read from a hidden `#mf-move-speed` element at the top of the right panel while a key is held); a browser keeps the old code across GUI restarts: hard reload.
    - One camera driver: Fly stops Orbit and vice versa. Layers: points, cameras with
    thumbnails, prior cameras, prior→refined segments, trajectories. Cameras
    can be coloured by position or rotation deviation, reprojection error,
    time, or source; **time is the default** (2026-10-22, the user asked). Stats list the per-source offsets.
  - **Inspect**: click a camera to see its image, with observations in green,
    residuals × gain in red, and unmatched keypoints in grey.
  - **Sparse**: "Run SfM (all)", or features, pairs, match, refine (live),
    export one at a time.
    - A state line: ⏳ running stage k/n with stage/total time and a detail
      line (feature extraction: "image k / n · name", polled from features.db since
      `pycolmap.extract_features` has no callback (`features._watch`); BA round and iteration; in a chunked refine "calibration round", "chunk k/n",
      "pose graph" or "polish … window k/n"), then ✅ finished + next step, or ❌ failed + error.
    - BA mode / chunk size / overlap / polish sweeps (see "Large projects: chunked refinement").
    - A **History** of finished stages with key numbers, and a **Log** of
      the last 40 `pipeline.log` lines (`pipeline.LISTENERS`).
    - The ticker and the final message share `App.lock` plus a run id, so
      a finished job never shows "running".
    - Every widget has a hover `hint`.
    - Refine uses `pipeline.stage_refine(progress=...)`. BA iteration events
      move the existing frustums (throttled to 4 Hz), and each round reloads
      the snapshot.
  - **Dense**: "Run dense (all)" plus one button per stage, with a Quality
    picker (superfast default / low / medium / high / max) and a mesh-method picker. Each
    run is in a background thread.
  - **Splat**: method, iterations, disk cache, downsample factor, LiDAR start +
    LiDAR depth weight (rosbag), train, Stop.
  - **Splat mesh**: the mesh from the splat (see "Mesh from the splat").
  - **Export**: mesh and splat exports (see "Export" above), with
    `export/` status.

Things to know:
- Points far outside the flight area are hidden, and the cloud uses float32.
  Unfiltered BA output contains points 100+ km away, which overflow viser's
  default float16.
- Live refine takes about 2× the headless time (171 s vs 83 s). The scene
  updates hold the GIL during Ceres' Python callbacks.
- Triangulation is RANSAC-based, so runs are not bit-identical. Medians are
  stable, extremes vary.
- Lines use `thickness_units="screen"` (pixels). viser's default is world
  units.
- Tests:
  - `tests/test_gui_smoke.py` runs the app logic headlessly on
    `data/berlin_3_1_frames` (skipped if absent). It, `test_gui_dense.py` and
    `test_projects.py` take any refined project from `MODELFORGE_TEST_PROJECT`
    (e.g. `data/berlin_test`).
  - `tests/test_gui_ingest.py` goes empty project → two pieces → plan →
    preview → extract → features/pairs.

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
- The original development machine had **no NVIDIA GPU**, only an AMD Radeon
  680M. So everything up to now ran CPU-only. Setup on a GPU host is under
  "Migrating to a GPU machine".
- `compose.gpu.yaml`:
  - Overrides `dev` with a CUDA build: `nvidia/cuda:12.8.1-devel-ubuntu24.04`,
    `CUDA_ENABLED=ON`, `CUDA_ARCHITECTURES=75;80;86;89;90;120`,
    `NERFSTUDIO=ON`, `gpus: all`, ports 7007, `shm_size: 8gb`, image
    `modelforge-dev:cuda`. Use it with `MODELFORGE_GPU=1 scripts/dev ...`.
  - `NERFSTUDIO=ON` adds `/opt/nerfstudio`: torch 2.7.1+cu128, gsplat 1.4.0
    with kernels compiled at build time for `GSPLAT_ARCH_LIST`, and
    nerfstudio from `resources/nerfstudio`. tiny-cuda-nn is not installed,
    because splatfacto doesn't need it.
  - There is no separate nerfstudio service or image any more, and no Docker
    socket in the container. The user refused the socket as a containment
    escape.
  - The earlier CUDA image, without nerfstudio, was validated without a GPU:
    `has_cuda` is True and the tests pass.

Tests: `scripts/dev pytest -q tests`.

## Conventions

- Work is on `master`, which is the only branch on `origin` (GitHub).
- Never commit large files. Frames, databases, sparse models and checkpoints go
  under `data/`.
