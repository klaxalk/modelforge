# TODO: normalize the lighting (sunny vs overcast photos)

Priority: low. Status: discussed 2026-10-23, not started. **Waiting for the user's test project** with mixed lighting (same site,
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
