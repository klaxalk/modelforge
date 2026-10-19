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
    Plots / Export / Settings (dark mode);
  - right (main): run state, **Run full pipeline**, **Stop**, Sparse /
    Dense / Splat, history;
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

Open items, in priority order:
1. Run the full clip (all 300 images) through dense and splat (quality
   `low`, then higher if wanted).
2. Test multi-video ingest on real separate flights of one site.
3. The final ~10 s of `berlin_3_1` (t ≥ 515 s, a near-horizontal hover) is
   weakly constrained. Trim it with the source range.
4. Calibration is biased low (~0.7×) in weakly constrained directions. The
   fix is redundancy-aware variance component estimation.
5. M4 (accuracy): prior-guided matching, the EIS warp, rolling shutter.

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
│   ├── splat_mesh.py   # mesh from the splat: TSDF fusion of rendered depth (export/splat_tsdf.py in /opt/nerfstudio)
│   ├── nsplugin/       # nerfstudio plugin run in /opt/nerfstudio: splatfacto-lidar (LiDAR depth loss)
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
    plan and plots the motion curve.
  - The Sparse tab shows reconstruct instead of refine and the BA controls.
  - The View tab skips prior frustums, lines and deviations.
- Tests: `tests/test_video_kind.py`.
### Kind `rosbag` (ROS 2 bags)

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
    pickers, a checkbox per cloud topic, voxel and range. The plot shows
    height + speed; 3D shows the track and a frustum preview.
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
- Calibration is biased low in directions the images constrain weakly
  (synthetic test: about 0.7× the true σ). Fixing this needs redundancy-aware
  variance component estimation.
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
- **Large projects: chunked refinement** (the user asked, 2026-10-18: the 4000-image Mavic 4
  project `kladno_dira` ran > 30 min in round 0's BA alone, and it must work for 10k+ images).
  Sparse tab "BA mode" (auto / global / chunked), "Chunk size", "Chunk overlap", "Polish sweeps";
  CLI `--ba-mode --chunk-size --chunk-overlap --polish-sweeps`. `modelforge/sfm/chunked.py`,
  `pipeline._refine_chunks`.
  - **auto = chunked from 1000 images on.** `stage_refine` then:
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
- GUI: Splat tab → method, iterations, **Cache images on disk** (default on),
  **Downsample factor** (1 / 2 / 4 / 8, default 1), then **Train + export splat**. The
  progress bar shows the training step, there is a Stop button, and the
  result is reloaded into the View tab.
- Shell: `scripts/splat PROJECT [--method splatfacto-big] [--iterations N] [--cache disk|cpu|gpu] [--downscale N] [ns-train args]`.
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
- **All gaussians** (View → Dense / splat, off by default; the user asked):
  the browser gets a random subset, ≤500k Gaussians live and ≤1M for the
  finished splat (`gdata.LIVE_SPLAT_SHOWN` / `SPLAT_SHOWN`). Ticked, it sends
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
- GUI: quality, render width, max depth (0 = auto), **Build mesh from splat**. The full
  pipeline always builds it after the splat (it is the pipeline's dense result). The View tab's
  Mesh dropdown gets `splat` (vertex colours) and switches to it; the Export tab exports it
  (glb / obj / ply).
- Mesh export now rotates with `trimesh.apply_transform`, which keeps the stored vertex normals.
  Assigning vertices dropped them, and trimesh recomputes normals in a slow Python loop without
  scipy: 105 s for a 173k-face mesh, now < 1 s.
- Tests: `tests/test_splat_mesh.py` (a synthetic splat of ground + box: geometry within 1.5
  voxels, colours, View / Export wiring).

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

`scripts/dev python -m modelforge.gui.app [PROJECT] [--root data] [--port 8080]`,
then open http://localhost:8080. Without PROJECT it opens the project opened
last (`last_project` in `ROOT/.modelforge_gui.json`, saved by `main` and on every
Projects → Open), else the newest project under `--root`. Before, it was always the
newest by file times, so a project only opened (not changed) was not reopened. compose publishes 8080. viser 1.1.1 and plotly 7.1.0 are
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
  - **Frame preview**: source + time slider. It shows the video frame
    (ffmpeg accurate seek), the aligned telemetry and a white frustum in 3D.
    "Start here" / "End here" set the range.
  - **3D**: each source's track, bright in its colour inside the range and
    grey outside, plus the planned image positions. **Plot**: altitude, speed
    and gimbal pitch on the project timeline, with planned images marked.
  - **Extract**: runs `ingest.run` with progress. A project with images needs
    the Overwrite box ticked. Afterwards the app reloads the project
    (`App.load_project`) and the Sparse tab takes over.
- `modelforge/gui/app.py`. Layout, as the user asked:
  - The **left panel** (`self.left = gui.add_panel()`, `dock_left`, 420 px)
    holds Projects / Videos / Ingest / View / Inspect / Plots / Export.
  - The **right (main) panel** (`main_panel.dock_right()`) only runs the
    methods: header, run state + progress, **Run full pipeline**, tabs
    Sparse / Dense / Splat, then History.
  - Browser tab title "ModelForge": viser hardcodes `<title>Viser</title>` in
    its built `index.html`, so the Dockerfile patches it (a late layer, so the
    rebuild takes seconds). The panel label is `ViserServer(label=...)`.
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
    - It is one button-group row: ● / ○ / · Splat, Mesh, Points (sparse),
      Cameras, Thumbs (camera thumbnails; available while Cameras is on),
      Residuals (View "Prior → refined": the red prior → refined segments),
      Dense (fused points), Plan, Orbit, Fly.
    - Points, Cameras, Thumbs and Prior cameras start off (the user asked):
      with thousands of cameras the frustums hide the splat / mesh.
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
  - **Run full pipeline** (stage `all`) runs SfM → splat → splat mesh with the
    tabs' settings: the dense result is the mesh from the splat, as the user asked
    (2026-10-18). COLMAP's MVS follows only with the Dense tab's "Include in Run full
    pipeline" (off by default). A project without images is extracted first from
    its Ingest plan (`IngestPanel.extract_now`). Splat / dense are skipped, with a
    History note, without nerfstudio / CUDA.
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
    - The Splat checkbox is ticked by default.
    - The layers refresh right after the splat stage and after fusion /
      mesh / texture, not only when the run ends. Before, the splat vanished
      during the full pipeline's dense stages: the live layer was gone and
      the final splat was not drawn yet.
    - `App._unchanged` remembers each layer's inputs (file + mtime +
      options), so these refreshes don't re-send unchanged layers (the
      splat is tens of MB).
    - Each layer has its own status line (`_dense_status`).
  - **Orbit** (View tab, the user asked): **Start / Stop orbit** + speed
    (°/s).
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
    - `gdata.FlightPath`: the shown snapshot's registered cameras in capture order, on the project
      timeline. Without usable times it uses one image per second. Pauses over 3 s (between videos)
      are shortened to 1 s.
    - Positions follow a cubic Hermite curve (finite-difference tangents); orientations are slerped.
      The curve passes exactly through every camera.
    - `App._fly` sets position, look-at (ahead by the median depth of the sparse points, so the
      mouse pivot afterwards lies in the scene) and up = −y of the OpenCV camera (roll reproduced)
      at 30 Hz.
    - Controls: speed × real time (default 4), Loop, and "Camera's field of view" (on: the real
      camera's vertical FOV while flying; the browser's FOV and up come back afterwards).
    - The **Position** slider shows progress; dragging it jumps there, also when not flying.
    - Mouse drags pause the flight for 1 s (detected as in Orbit), then it resumes on the path.
    - One camera driver per browser: Fly stops Orbit and vice versa. Layers: points, cameras with
    thumbnails, prior cameras, prior→refined segments, trajectories. Cameras
    can be coloured by position or rotation deviation, reprojection error,
    time, or source. Stats list the per-source offsets.
  - **Inspect**: click a camera to see its image, with observations in green,
    residuals × gain in red, and unmatched keypoints in grey.
  - **Plots**: Plotly charts of deviation vs. project time (colour = source)
    and a point-error histogram.
  - **Sparse**: "Run SfM (all)", or features, pairs, match, refine (live),
    export one at a time.
    - A state line: ⏳ running stage k/n with stage/total time and a detail
      line (BA round and iteration; in a chunked refine "calibration round", "chunk k/n",
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
