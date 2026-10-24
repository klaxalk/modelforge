# TODO: multi-camera DJI projects

Priority: low. Status: planned 2026-10-23; **steps 1–3 done 2026-10-24** (SfM with several cameras), **4–5 done the same day** except the common training resolution, semantic features and the held-out evaluation; **6 (GUI) done the same day**. Scope: **DJI kind only** (every photo has a pose prior, so refine stays the
prior BA). Today a project is one camera: `dji/ingest.py` refuses sources of different resolutions, features use
`CameraMode.SINGLE`, transforms.json carries one global camera. Two drones that both record 4K pass the resolution
check today and wrongly share one set of intrinsics.

## Decisions taken

- A camera = drone model + resolution + lens (Mavic 4 Pro: wide and two tele cameras). Pieces of one drone's videos
  share a camera, as now.
- **Clips that zoom are refused** (the focal length changes per frame). Check first whether the djmd camera record
  has a zoom / focal-length field (a value that changes over the clip); otherwise detect zoom from the optical flow
  (radial scaling about the image centre).
- **Training resolution: one common target long side** (default **1920 px**), every camera scaled to the largest size
  within it, never upscaled; non-integer factors are fine (JPEG DCT decode to the nearest power of 2, then an exact
  resize). We write `images_N` ourselves with intrinsics scaled to match; nerfstudio gets factor 1. It replaces the
  integer `splat_plan.auto_downscale` rule (≤ 2700 px). Today's projects train identically (4K → 1920, 1080p as is);
  only a 2.7K camera would drop from 2688 to 1920. 3840 for 4K detail.

## Steps

1. **Ingest.** Drop the one-resolution refusal (`dji/ingest.py`); each source records its camera key (`Clip.model`,
   resolution, lens), poses.json a camera per image. Refuse zooming clips. The Ingest tab shows each source's camera.
   Lead, yaw source, sampling and the per-source GNSS offsets are per clip already.
2. **Features.** One `pycolmap.extract_features` call per camera group (`CameraMode.SINGLE`, that group's image
   names; images stay flat in `images/`). A starting focal length per model from measured values (NEO2 1085 px at
   1920, Mavic 4 ~3284 px at 3840), instead of the default that is 2× off.
3. **Refine.** Intrinsics per COLMAP camera (pycolmap's BA does this; `_refine_chunks` keeps `cams` per camera id
   already). The chunked calibration subset (`refine.calibration_subset`) must cover every camera. The prior model
   (`sfm/robust.py`) becomes per camera / drone: boresight B per camera, time offset δt per drone model, absolute
   prior sigmas per camera group. Drift, source offsets and isolated photos are per source / photo already.
4. **Export.** Per-frame intrinsics in transforms.json (`fl_x / fl_y / cx / cy / w / h / k1…` per frame;
   `export/nerfstudio.py`), images_N at the common target (above).
5. **Readers of the global camera** (`t["fl_x"]`, `t["w"]`) go per frame: `splat_plan` (GSD, downscale, cache),
   `splat.py`, `splat_tiles` (survey, mesh share), `export/mesh_depth.py` (undistortion per camera; camera in the
   cache key), `export/mesh_color.py`, `export/mesh_share.py`, `export/splat_tsdf.py`, `semantic.py`, `evaluate.py`.
   COLMAP dense needs nothing (its undistorter handles several cameras).
6. **GUI.** View stats per camera, frustums per camera, "Colour cameras by: camera", the refine History line with
   a focal length per camera.

## Tests

- Synthetic: two cameras of different focal length in one prior BA (both recovered), per-frame export, the splat
  stages with the fake trainer on mixed sizes.
- Real with known truth: two pieces of `berlin_3_1`, one resized to 1280 × 720 → refine must give f₂ ≈ f₁ × 2/3 at
  the same reprojection error.
- Then a real mixed-drone project (the user is building a mixed-lighting one, which may also mix drones).

Order: steps 1–3 + the berlin test (SfM with several cameras), then 4–5 (splat / meshes), then 6.

## Done 2026-10-24: steps 1–3

- **Camera key** (`ingest.Clip.camera`): drone model + resolution + zoom ratio, e.g. "DJI Mavic4 Pro L3A 3840x2160"
  (" zoom 3" when not 1). poses.json `cameras` (list of {key, model, width, height, zoom}), `camera` per image and
  per source. The plan warns "N cameras (...): each gets its own intrinsics"; the one-resolution refusal is gone.
- **Zoom**: djmd `FrameMeta.zoom` = the Mavic 4 camera record's `.23.1` (1.0 on every clip here: dira, spodek_1,
  spodek_2; the NEO2 has no zoom and gets 1.0). `.37.1` varies 23–30 at zoom 1, so it is no plain focal length.
  A source whose zoom changes > 1 % within its range is refused by `ingest.run` (warning in the plan). **Unverified:
  no zooming or tele clip exists to check that `.23.1` is the zoom / tells the lenses apart**; detecting zoom from the
  optical flow is the fallback if it is not.
- **Features**: one `extract_features` call per camera group (`features.camera_groups`), each `CameraMode.SINGLE`, so
  one COLMAP camera per camera. Projects without `camera` per image: one group, as before. The starting focal length
  per model was **not** added (COLMAP's default start, as before, converges in round 0; it keeps today's projects
  unchanged).
- **Refine**: pycolmap's BA already refines every camera; `calibration_subset(cameras=)` adds a run of its own for a
  camera none of the 5 runs covers; **boresight per COLMAP camera** (`PriorModel.boresight_by_camera`, image →
  camera in `image_camera`; with one camera nothing changes). The time offset δt and the prior sigmas stay one per
  project. `refine.evaluate` lists `cameras` (model, size, params) when there are several; the History line shows a
  focal length per camera.
- **Measured** (`build/multicam/two_cams.py`: fullrun_test, every other photo resized to 1280 x 720 as camera 1):
  global refine 80/80, f = 1084.59 / 723.10 px (ratio 0.66670, truth 2/3), distortion equal, 0.40 / 1.01 px;
  chunked (30 / 10) 1084.81 / 723.25 px, boresights per camera within 0.05° of each other (one drone).
  Test: `tests/test_multicamera.py` (camera key + zoom, calibration subset, the 40-photo two-camera refine).

## Done 2026-10-24: steps 4–5 (per-frame intrinsics)

- **Export** (`export/nerfstudio.py`): with several COLMAP cameras every frame carries its own `camera_model, fl_x,
  fl_y, cx, cy, k1, k2, p1, p2, w, h`; the top level keeps only `camera_model` (shared) and `mf_cameras` = the count:
  nerfstudio's dataparser takes a top-level key over the frames' own (a first version kept the first camera's at the
  top, and nerfstudio trained every photo as 1920 x 1080: "size of image (1280, 720) does not match"). One camera: unchanged.
- **Readers** go through `export/frame_camera.py` (`camera(t, frame)`, `cameras(t)`, `largest(t)`; standard library
  only, loaded by path from the nerfstudio-venv scripts): `splat_plan._cameras` / `survey` (per-camera fx / w / h;
  the widest camera's size for the resolution rules), `splat_tiles._seen`, the tiles' downscale and loss masks,
  `splat.auto_cache`, `splat_mesh.cameras` (one render size, each camera scaled to its width), `export/mesh_depth.py`
  (an undistorted camera each; manifest lists them), `export/mesh_color.py`, `export/mesh_share.py`.
- **Not yet, refused with a message:** semantic features (the worker undistorts every photo with one camera) and the
  held-out evaluation (`evaluate.run`).
- **Training resolution: still one integer downscale for all photos** (auto from the widest camera): a 4K + 1080p mix
  at x2 trains the 1080p photos at 960 px. The common target long side (decision above) changes how every project
  trains; left for a session where the splat runs can be compared.
- **Measured** on the two-camera fullrun_test copy: depth maps vs each photo's own SfM points |dz| / z median 0.27 %
  (1920 px camera) / 0.28 % (1280 px), so each camera's undistortion lines up; colouring 96 % of the vertices; survey
  overlap 49.2 / area 2700 m² vs the one-camera original's 49.9 / 2580 m². Test: `test_multicamera.py` (export per
  frame, survey, `frame_camera`).
- **Found on the way:** `mesh_depth.py`'s cache key had lost the mesh's mtime and the downscale to a comment, and it
  rewrote its manifest on every run (the tiles' loss masks, keyed by its time, were then always made again). Both
  fixed; existing depth maps are rendered once more.
- **Splat on per-frame intrinsics** (two-camera copy, 3000 steps, 192k Gaussians, 58 s): training views rendered with each
  camera's own K score PSNR 20.8 (1920 px) / 21.3 (1280 px) median (`build/multicam/render_check.py`).

## Done 2026-10-24: step 6 (GUI)

- View → Color cameras by **camera** (legend: the camera keys from poses.json `cameras`; `gui.data.Project.camera`,
  `camera_names`); the stats list every camera ("camera OPENCV 1280×720: f = …"); Inspect projects each photo with its
  own camera (`Snapshot.intrinsics["cameras"]` / `["by_image"]`, both snapshot readers); the refine History line has a
  focal length per camera. On the two-camera copy Inspect gives 0.33 / 0.27 px median for a photo of each camera.
- Not done: frustums sized per camera (they use each camera's pose only; the frustum shape is the first camera's).
