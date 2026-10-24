# TODO: cheap splat quality fixes

Priority: **medium**. Status: implemented 2026-10-23, **A/B on karlak 2026-10-24: exposure on by default, bilateral grid and anti-aliasing off** (see the end); the tile mask A/B is running. Both are flags or small changes plus an A/B run;
validate on a small project first (fullrun_test, karlak; held-out scoring as in `build/splat_exp.py`), not on
kladno_dira.

## 1. Exposure from the metadata

- djmd records the exact exposure per frame: shutter (exact, not the SRT's rounded stop), ISO, f-number
  (`docs/djmd_format.md`). Relative exposure per photo = shutter × ISO / f². Photos of one flight with auto exposure
  differ in brightness; today splatfacto has to absorb it (fog, view-dependent colour) or ignores it.
- Proposal: write the factor per frame into transforms.json (poses.json has the raw camera values per image) and, in
  the plugin (`modelforge/nsplugin`), scale the rendered image by it before the loss (or the photo by its inverse);
  exports render at a reference exposure (the median).
- On top: splatfacto's **bilateral grid** (`--pipeline.model.use-bilateral-grid True`; `splat.train_command` already
  sets its scheduler steps) for white balance / tone per photo, as a Splat tab option. It cannot remove hard shadows
  (spatially varying): see `docs/todo_lighting.md`.
- Check first: how much exposure actually varies in the user's flights (fixed exposure on kladno_dira per the user).
  Also useful for the mixed sunny / overcast project.

## 2. Anti-aliased rasterization

- gsplat's `antialiased` rasterize mode (Mip-Splatting's 2D filter; splatfacto `--pipeline.model.rasterize-mode
  antialiased`). Flights at different heights see the same ground at very different scales: plain 3DGS aliases
  there and grows needle-like Gaussians (seen on kladno_dira and dira2k).
- A Splat tab option (default after the A/B decides), stored in resume.json / tiles.json like the other settings.
- A/B: held-out PSNR / SSIM / LPIPS, plus renders from heights between the flight's passes.

## Also open (earlier sessions; the user's main complaint: flat ground seen from one straight pass at ~45° stays soft)

Done for it already (CLAUDE.md "Flattening", "Starting Gaussians from the SfM mesh"): flattening 0.02, mesh seeds
(default on); a far-field mask was tried and removed (it did not work well for the user). Not done, in the order
suggested:

1. Lower SH degree (1) for one-sided views: higher bands overfit view-dependent colour where every photo looks
   from one side. splatfacto `--pipeline.model.sh-degree 1`; A/B on a tile.
2. Full resolution ×1 for 4K with splatfacto-big's split gradient threshold (0.0005), so 4K detail gets Gaussians
   (karlak: 4K default densification peaked at only 8.4 GB and grew no more Gaussians than ×2; CLAUDE.md "karlak
   investigation").
3. Difix3D+ (NVIDIA, 2025): a single-step diffusion model cleans renders from off-path views, which are added as
   pseudo training views, then fine-tune. SuperGaussian was judged a poor fit.
- Judging it needs a berlin tile viewed **off the path**: held-out photos from the same pass cannot show the softness.
## 3. Depth prior outliers (check and fix as part of this TODO)

- The SfM-mesh dense depth prior (`export/mesh_depth.py` maps, loss in `nsplugin/mf_splatfacto_lidar.py`:
  weight × mean(min(|D − z| / z, 1)) over covered pixels) has wrong targets in places. berlin's last tiled run
  (2026-10-23, 7 covisibility tiles, 50000 steps, mesh 22:18 / transforms.json 22:14 of 2026-10-22; per tile
  `tiles/tile_KK/report.json` `splat.lidar_depth`), at step 50000:

  | tile | median | p90 | within 5 % |
  |---|---|---|---|
  | 0 | 1.3 % | 88 % (239 % at 40500) | 72 % |
  | 1 | 1.5 % | 54 % | 74 % |
  | 2–6 | 0.9–1.5 % | 7–13 % | 78–86 % |

  (Earlier berlin tile runs showed p90 ~400 %. The p90 pools only the last 500 training views, so it swings.)
- ≥ 100 % = the splat sees at least twice as far as the mesh: the mesh has surface in front of what the photos
  see. Suspects: Poisson closures (sky, scene edge, open gaps), patched holes (`sfmmesh/patched.npy`), surfaces the
  SfM points miss (berlin_3_2 beside / inside the halls: floor under roof, through openings). Each such pixel pulls
  with the clamp's full value towards something at mesh depth: a plausible source of fog / softness there.
- **Check first** (CPU, no training): per photo of tiles 0 and 1, the cached maps (`sfmmesh/depth_x<N>/`) vs the
  photo's own SfM points (as the alignment check in CLAUDE.md "Dense mesh prior"); render / overlay the worst
  views and see what the mesh puts there.
- **Fix candidates:** (a) no target on mesh faces without SfM support (patched faces, faces far from any point,
  Poisson closures), written as 0 in the maps; (b) after warm-up, drop a pixel's target when rendered and mesh
  depth disagree by > ~50 % (outlier gate in the plugin). A/B on tile 0: depth error, held-out PSNR / LPIPS, renders.

### Findings (2026-10-23, CPU + a few GPU renders; scripts in `build/dp_outliers/`)

- **Maps vs each photo's own SfM points** (`compare.py`, berlin tiles 0 + 1, 2014 photos): median |dz|/z 0.6 %; in the
  median photo 0.04–0.1 % of its points have the mesh at < half their depth (p90 photo 1.8–4.3 %). A few photos are
  wrong as a whole: berlin_3_2_05501–05574 inside a hall (60–100 %: a smooth Poisson bubble 0.5–4 m in front of the
  walls; berlin's patch-hole size was 50 m, its largest patch 38 m), berlin_3_4_006xx (40–60 %: fat blobs around
  bushes / crowns, distant closures above the horizon).
- **Support filtering does not pay** (faces kept only within k point spacings of an SfM point, 200 sample photos):
  k = 3 keeps 78 % of the faces, covered pixels 87 → 72 %, the worst 40 photos' front-50 % share only 0.32 → 0.22;
  k = 2: coverage 58 %. The unpatched (trimmed) mesh: 0.32 → 0.30, coverage 85 %.
- **The splat's own depth vs the maps** (`splat_vs_map.py`, tile 0's trained splat at the map's camera, 1049 views):
  pixel |D − z| / z median 1.5 %, p90 91 % (the training log's p90 reproduced), 12.6 % of covered pixels ≥ 50 % off,
  93 % of those with the splat *farther* than the mesh... but per view it is bimodal: median view 0.7 % bad pixels,
  174 views > 20 %. **They are the views that see mostly outside the tile** (`inregion.py`, share of the view's mesh
  pixels inside tile 0's region): < 20 % inside: 591 views, bad pixels mean 21 %; ≥ 20 % inside: 458 views, 0.3–0.4 %.
  Renders (`views/`): the splat paints fog close to the camera where the photo shows the neighbouring tiles; the
  mesh is right there. The tile has no starting points outside its region + margin, so the photometric loss of those
  pixels grows fog.
- **So the large "depth outliers" are mostly splat fog, not wrong targets.** Wrong targets exist (hall interiors,
  blobs at crowns), but are a small share.

### Fix to A/B (GPU)

1. **Tile mask**: per training view, drop the losses (photometric + depth) of pixels whose mesh surface lies outside
   the tile's region + margin (computed once from the depth maps and regions.npz, cached next to them; pixels without
   a mesh hit stay). Expected: no fog from views that mostly see the neighbours, faster tiles. A/B on berlin tile 0
   (10k steps each): held-out PSNR / LPIPS of in-region views, the depth error, and renders of the merged splat.
2. Then the remaining wrong targets: an outlier gate after warm-up (drop a pixel's depth target when |D − z| / z >
   50 %), and a smaller patch-hole size for the prior (berlin's 50 m patched whole hall interiors).

## A/B results (2026-10-24, `build/quality_ab.py`, karlak copy `build/karlak_ab`)

471 keyframes, 24 held-out extracted frames between them (PnP-located), 18000 steps x2, unit 10 m, camera optimizer
off, flatten 0.02, default densification; scores at 1920 px. `psnr_cc` = PSNR after a per-image 3 x 3 + bias colour
fit (leaves out exposure / white balance).

| arm | PSNR | PSNR cc | SSIM | LPIPS | PSNR p10 | Gaussians | train |
|---|---|---|---|---|---|---|---|
| base | 22.35 | 22.79 | 0.678 | 0.338 | 16.53 | 1.89M | 924 s |
| **exposure** (rendered x the photo's gain) | **22.59** | **23.02** | **0.681** | **0.338** | **16.91** | 1.90M | 842 s |
| exposure, rendered without the gain | 19.34 | 23.02 | 0.674 | 0.344 | 15.61 | | |
| bilateral grid | 20.99 | 22.32 | 0.677 | 0.336 | 16.00 | 1.82M | 1279 s |
| anti-aliased, rendered anti-aliased | 22.56 | 22.96 | 0.685 | 0.332 | 16.69 | 2.12M | 920 s |
| anti-aliased, rendered classic | 20.88 | 21.36 | 0.642 | 0.349 | 16.57 | | |

- **Exposure: default on** (`splat.EXPOSURE`; Splat tab "Exposure from the metadata", CLI `--no-exposure`; only where
  transforms.json carries `mf_exposure`, `splat.exposure_known`). A small but consistent gain, also in the colour-
  corrected PSNR (the splat's colours are cleaner, not just brighter). The splat holds the scene at the median
  exposure, so a render scored against a photo at its own exposure without its gain is ~3 dB lower: expected, and
  what a viewer shows (one normalized scene).
- **Bilateral grid: off.** Learned per training photo, so a new view gets none: -1.4 dB raw, -0.5 dB colour-corrected,
  38 % slower.
- **Anti-aliased: off** (Splat tab "Anti-aliased rasterization", CLI `--antialiased`). Better on every score when
  rendered anti-aliased (`evaluate.run(rasterize_mode=)`, splat_eval `rasterize_mode`), but its opacities assume the
  2D filter: in classic renderers (the GUI's viewer, most .ply viewers, the splat mesh's TSDF) it loses 1.5 dB.
  Worth it only with an anti-aliased viewer.
