# TODO: normalize the lighting (sunny vs overcast photos)

Priority: low. Status: discussed 2026-10-23; **steps 2-3 prototyped 2026-10-25 on berlin_lighting** (see the end). **Waiting for the user's test project** with mixed lighting (same site,
sunny and overcast flights, or flights at different times of day; keep the SRT / MP4s for UTC time; do not run the
full pipeline on it before the comparison). Goal: the splat looks like overcast, diffuse light, without hard
shadows, also when parts of the dataset were flown in direct sun.

## Why not a per-photo 2D relighting model

Single-image shadow removal / relighting (IC-Light, RGB↔X, Marigold-IID, Careaga & Aksoy intrinsics, NVIDIA
DiffusionRenderer) works per photo: each photo gets its own guess of what lies under a shadow, so views disagree
and the splat explains the disagreement with fog / view-dependent colour (as MCMC's near-camera fog). Also: they run
at 512–1024 px (tiling 4K, seams), lifting deep shadows amplifies noise and JPEG blocks, clipped areas are lost,
generative detail is invented. Worth only an A/B (step 4).

Not in the spherical harmonics either: SH are view-dependent colour fixed per Gaussian; sunlight depends on the sun
direction per photo. SH would average or bake the shadows.

## Plan, in order

1. **Bilateral grid** as a Splat tab option: splatfacto `--pipeline.model.use-bilateral-grid True` (off today;
   `splat.train_command` already sets its scheduler). Per-photo tone / exposure / white balance; no hard shadows.
   Together with the exact exposure from djmd: `docs/todo_splat_quality.md` §1.
2. **Sun per photo**: elevation / azimuth from GPS + UTC (solar ephemeris; `ingest._wall_start` gives UTC per clip,
   `time_s` per image) and a per-photo shadow-contrast score from the images, to see which projects mix lighting.
3. **3D shadow model in the splat plugin** (`modelforge/nsplugin`):
   - sun visibility V(pixel) per training view: from the pixel's surface point (SfM mesh, as `export/mesh_depth.py`
     renders depth at nerfstudio's training resolution and undistortion) a ray towards the sun in the mesh (Open3D
     raycasting); also cos(normal, sun); cached like the depth maps (`sfmmesh/sun_x<N>/`);
   - rendered colour × L, L = ambient_i + sun_i × V × cos, with learned per-photo RGB gains (6 numbers); the
     Gaussians hold the surface colour (band 0 + higher bands for real view dependence);
   - overcast photos learn sun_i ≈ 0 by themselves; render / export with ambient only (overcast look) or with sun
     added for any time of day.
   - Identifiability: a spot always shadowed (or always lit) in every photo cannot tell dark surface from shade; the
     sun / ambient ratio comes from overcast flights of the same ground, other times of day, or the sun moving
     ~7° over a 30-min flight. A single sunny flight needs a prior on the ratio (clear sky ~3–5× by sun elevation).
   - Weak spots: shadow edges where the mesh approximates the casters (trees, poles, Poisson closures); later the
     splat's own rendered depth as the caster after warm-up.
4. Optional A/B: a 2D relighting model into `images_relit_N/` for training only (never SfM: SIFT copes with
   shadows) against step 3.

## First run on berlin_lighting (2026-10-25)

The user's project: berlin_1_1 (4K, 13 July 17:11 UTC, sun 17° up, hard shadows) + berlin_3_1 (FHD, 15 July 18:20 UTC,
overcast), 2000 photos, two cameras. SfM 2000/2000, 0.65 px; 32 % of the cross-flight matches merged (44 % with two
manual tie points).
- **Step 2** (`modelforge/lighting.py`, `python -m modelforge.lighting PROJECT`): sun per photo (GPS + MP4 creation
  time + pts; `modelforge/sun.py`, shared with the sky box) and a shadow score (contrast, dark share): sunny 2.7 % dark
  vs overcast 1.1 %; the contrast measure does not separate them (2.56 vs 2.54 stops).
- **Step 3**: `export/sun_vis.py` (sunlight per pixel at the training camera: mesh hit, then a ray towards the photo's
  sun; S = visibility x cos with **smooth normals** (vertex normals interpolated at the hit: facet normals made a
  patchwork of every Poisson triangle), -1 unknown; started 0.3 m / 1 % of the depth towards the sun so the mesh's own
  bumps do not shade; 0.6 s per 1920 px map on the 2.7M-triangle mesh; `sfmmesh/sun_x<N>/`); the plugin's `sun_light`
  (per photo log RGB ambient + sun, the map blurred 0.4 % of the width; mean log ambient held at 0); Splat tab
  "Sunlight learned per photo", CLI `--sun-light`.
- **A/B, tile 2** (593 photos: 274 sunny, 319 overcast; 15000 steps, x2 / common resolution 1920, dense mesh prior,
  seeds, sky masks, exposure; `build/light_ab.py`), nerfstudio's held-out 10 %:

  | | sunny PSNR / cc / LPIPS | overcast PSNR / cc / LPIPS | train |
  |---|---|---|---|
  | base | 15.30 / 16.75 / 0.618 | 13.13 / 15.53 / 0.458 | 849 s |
  | sun_light | 13.76 / 17.68 / 0.600 | 11.78 / 15.51 / 0.476 | 1695 s |

  (low absolute values: black sky in the score, a tile's held-out photos see beyond it; raw PSNR lower because the
  splat's brightness sits between the flights' ambients.) Renders (`build/light_compare.jpg`): the base splat is full
  of smears where the flights disagree, the sun_light one visibly cleaner, but darker.
- Learned light (median): sunny ambient 2.79 / sun 0.16, overcast 0.49 / 0.015: the right direction, but the sun term
  adds only ~2 % on lit ground (sun 17°: S <= ~0.3); the flights' brightness difference went into the ambient.
  Likely: the mesh's casters (trees, bushes as blobs) and the bumpy ground make S too unreliable.
- Next: (1) the splat's look: tie the ambient to the overcast flight (or a chosen photo) instead of the mean; (2) speed
  (2x slower: the maps read and blurred per step: cache them); (3) the score without the sky; (4) crops of known
  shadows (removed or averaged?); (5) casters from the splat's own depth after a warm-up.
