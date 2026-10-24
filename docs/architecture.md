# modelforge architecture

modelforge is a standalone pose-refinement tool for DJI drone video. It turns
a video into frames with time-aligned 6-DoF pose priors, refines those poses
against the images, and exports a nerfstudio dataset. Every stage can be
watched and controlled from a web GUI.

**COLMAP is an unmodified dependency** (`resources/colmap`, used through
pycolmap). We don't fork or patch it. Instead we reuse its features, matching,
two-view geometry, triangulation, and its Ceres bundle-adjustment problem.
Then we add our own residuals on top.

## Why not stock COLMAP

Stock COLMAP has to discover every pose from the images, adding them one at a
time during incremental mapping. It uses GPS only for nearest-neighbour pair
selection, and optionally as position priors in bundle adjustment. DJI
telemetry gives position, velocity and attitude for every frame. Once
time-aligned (see `djmd_format.md`), that's a full 6-DoF initial estimate. So
the work left is **refinement**: start every frame at its prior, triangulate,
and run bundle adjustment with prior terms.

## What we reuse from COLMAP (pinned `48af6d30`)

| Need | pycolmap API |
|---|---|
| Features, matching | `extract_features`, matching from an imported pair list (`ImportedPairingOptions`), LightGlue |
| Two-view geometry | `TwoViewGeometry` estimators |
| Tracks + triangulation | `IncrementalMapper` / triangulator (same path as `point_triangulator`) |
| Reprojection BA | `create_default_ceres_bundle_adjuster(...).problem` (a `ceres::Problem`) |
| 6-DoF absolute prior | `AbsolutePosePriorCost(cam_from_world_prior[, covariance 6×6])` |
| Relative prior | `RelativePosePriorCost(i_from_j_prior[, covariance])` |
| Model I/O | `Reconstruction`, cameras, binary/text models |

The full `pyceres` package (from the colmap org, not the minimal one bundled
in pycolmap) is needed to add residual blocks, custom Python costs and
iteration callbacks to that problem. It must be built against the same Ceres
as pycolmap.

## Pipeline

Each stage reads and writes artifacts in a **project directory**, so the CLI and
the GUI can show, re-run or resume any stage. A manifest records each stage's
input and parameter hashes.

```
ingest   video → frames/ + poses.json                     modelforge.dji.extract (done)
features → database.db (SIFT; GPU when available; ALIKED+LightGlue later)
pairs    → pairs.txt: view-frustum overlap + time neighbours + revisits of the same spot
verify   → two-view geometries; drop pairs that disagree with the priors
           (relative rotation, and translation direction when the baseline is large enough)
init     → Reconstruction: one shared camera, every frame at its prior pose
refine   → loop { triangulate → BA → filter → retriangulate/merge }, snapshot each round
export   → nerfstudio transforms.json + sparse_pc.ply
```

### Ingest (implemented)

- Decodes djmd and estimates the telemetry lead (about 0.72 s on NEO2), per
  video.
- Rebuilds the 10 Hz streams and reads them at `frame − lead`.
- Samples frames along ds = |v|·dt + w·Δθ, with N images as the user's input.
- **Several sources per project.** A source is a video plus a time range,
  saved in `ingest.json`.
  - All sources share one spacing and one ENU frame. The reference is the
    chronologically first source. Other flights are lifted by their
    takeoff-altitude difference.
  - Images carry a `source` index. Relative priors and time-neighbour pairs
    stay within a source.
  - Refinement estimates a position offset per source, bootstrapped from
    cross-source matches whose points were triangulated twice.
- Outputs JPEGs with EXIF GPS, plus `poses.json` (ENU, cam rotation, COLMAP
  qvec/tvec).
- GUI Ingest tab: pick videos and ranges, preview frames and telemetry, see
  the planned images, extract.
- See `CLAUDE.md` and `djmd_format.md`.

### Refinement cost

```
E = Σ reprojection (robust)                               COLMAP default BA terms
  + Σ_k  ρ( ‖ log( T_k^prior(δt, B)⁻¹ · T_k ) ‖²_Σabs )   absolute 6-DoF prior, loose
  + Σ_k  ρ( ‖ log( ΔT_k,k+1^telemetry⁻¹ · ΔT_k,k+1 ) ‖²_Σrel )  relative prior, tight
```

- **Loose absolute, tight relative.**
  - Consumer GNSS and compass errors drift slowly, so their absolute error is
    large: σ_pos about 2–3 m, σ_yaw a few degrees, σ_pitch/roll about 0.5°.
  - Frame-to-frame deltas are much more accurate. Attitude deltas are
    essentially gyro-integrated, and position deltas follow integrated
    velocity.
- **Shared nuisance parameters**, one set for the whole flight:
  - Boresight B (3 DoF): the rotation between the camera and the
    telemetry-attitude frame.
  - Residual time offset δt (1 DoF), applied as
    p(t + δt) ≈ p + v·δt, R(t + δt) ≈ R·exp(ω·δt).
- **Gauge.** The priors fix position, rotation and metric scale in ENU, so
  there's no gauge fixing and no Sim3 alignment.
- **Intrinsics.** One shared camera, refined jointly. The focal length is
  unknown, and it must not be held fixed, or it will bias the poses.
- **Fallback if pyceres can't share the problem.** Alternate a pycolmap-only
  solve with a numpy fit of B and δt from the residuals. They're low-dimensional
  and weakly coupled. Relative priors do need pyceres.

### Export: nerfstudio

The format follows `nerfstudio/process_data/colmap_utils.py:colmap_to_json`:

- `camera_model: OPENCV`, `fl_x, fl_y, cx, cy, w, h, k1, k2, p1, p2`.
- `frames[].transform_matrix` = c2w from COLMAP w2c, with the camera y and z
  axes flipped (`c2w[0:3,1:3] *= -1`, OpenCV → OpenGL).
- No `applied_transform`. Our world is ENU (z up, metric). Nerfstudio only
  permutes axes on its COLMAP path because COLMAP's world frame is arbitrary.
- `ply_file_path` → sparse points, used to initialize splatfacto.
- Optional per-frame timestamps, for appearance embeddings.

## GUI: viser

Nerfstudio's viewer is built on viser (pinned there at `viser==1.0.0`), a
Python-driven web 3D viewer with GUI widgets and Plotly panels. It runs in the
pipeline's own process, so there's no separate frontend to maintain.

- **3D view:**
  - camera frustums with thumbnails, and the sparse cloud
  - prior vs. refined trajectory, with per-frame correction arrows
  - cameras coloured by reprojection error or prior error
- **Stage panel:** run or re-run any stage with its parameters, see progress,
  and step through refinement snapshots.
- **Live refinement:** a Ceres iteration callback pushes camera poses to the
  scene every few iterations.
- **Inspector:** click a camera to see its image, keypoints, reprojections and
  matched pairs.
- **Plots:**
  - residual histograms, and prior correction vs. time
  - δt / B estimates
  - telemetry traces, and the sampling curve s(t)

If viser becomes limiting, the option is a custom FastAPI + three.js frontend.

## Code layout

```
modelforge/
  dji/            djmd.py srt.py extract.py           ingest (done)
  project.py      project dir, stage manifest, cache keys
  sfm/            features.py pairs.py verify.py init.py refine.py costs.py
  export/         nerfstudio.py
  gui/            app.py (viser); subscribes to stage progress events
  cli.py          modelforge run PROJECT [--until STAGE] | modelforge gui PROJECT
```

Stages are plain functions `(project, params, progress) → artifacts`. The CLI
and GUI both call them, and progress events feed the GUI.

## Milestones

- **M0, environment check.**
  - Build pycolmap from the submodule and pyceres against the same Ceres.
  - Confirm that `AbsolutePosePriorCost` can be added to the default BA
    problem and solved.
  - Confirm that a Python iteration callback fires.
- **M1, headless pipeline** on `berlin_3_1`, ingest through `transforms.json`.
  - No stock-COLMAP comparison (decided against). The checks are internal:
    reprojection error, registered frames, track length, deviation from the
    priors, systematic residuals (B, heading bias, δt), runtime.
  - Sanity check: `ns-train splatfacto`.
- **M2, GUI.** A read-only artifact viewer first, then live refinement and stage
  control.
- **M3, robustness.** B/δt, relative priors, prior-gated verification, and
  rejecting frames with bad priors.
- **M5, products.** Dense mesh via COLMAP MVS on the refined poses
  (undistort → PatchMatch [CUDA] → fusion → Delaunay/Poisson → texture), in the
  pipeline and the GUI. Gaussian splat via nerfstudio splatfacto
  (`scripts/splat`), converted back to ENU so the splat and mesh align.
- **M4, accuracy.** Prior-guided epipolar matching, EIS warp (`pp_target` vs.
  `pp_current`), rolling shutter. **Closed by measurement (2026-10-24), nothing implemented:**
  - *Rolling shutter* (`build/rs/rs_fit.py`): per observation of a refined model, the image motion J (px/s) from the
    camera's velocity (priors) and rotation rate (neighbouring refined poses), the row's read time (y / h - 1/2) T,
    and a least-squares fit of the readout T to the residuals. dira2k (Mavic 4, 4K, 8 m/s median) 0.3 ms, karlak
    (NEO2 4K) 0.1 ms, berlin's 4 flights (NEO2 1080p, 4-6 m/s, 7-16 deg/s) -0.0..0.2 ms, per-photo IQR within
    +-0.6 ms, residual variance explained <= 0.2 %. Injecting 20 ms into the residuals gives back 19.2 ms (IQR
    18.4-20.0, 67 % explained), so the fit would see it. Real sensors read out in 10-30 ms: DJI's video pipeline
    (EIS) evidently removes it, or BA absorbs a constant part (fy / fx = 1.003-1.004 on dira2k, karlak, berlin).
  - *EIS warp*: the refined rotation scatter does not correlate with the EIS correction (r = 0.05, berlin_3_1; see
    CLAUDE.md), and a pure rotation of the virtual camera is a pose change the BA solves anyway.
  - *Prior-guided matching*: the pairs already come from the priors (look-at overlap, terrain look-ats); where
    matching fails (kladno_dira's high / low passes, `build/hl_match.py`) SIFT itself does not correspond (ratio 0.9:
    a uniform ~19 inliers, chance level), so an epipolar band from the priors has nothing to keep.

## Risks

- **pycolmap ↔ pyceres binary compatibility.** Both must share one Ceres and a
  compatible pybind11. M0 checks this first.
- **EIS (`eis: normal`)** warps and crops every frame, so the intrinsics are
  not constant. Prefer recording with EIS off, or model the warp in M4.
- **Rolling shutter** at around 7 m/s is a second-order effect.
- **GPU.** CUDA is needed for SIFT-GPU and nerfstudio training, but not for
  refinement.
