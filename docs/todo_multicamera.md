# TODO: multi-camera DJI projects

Priority: low. Status: planned 2026-10-23, not started. Scope: **DJI kind only** (every photo has a pose prior, so refine stays the
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
