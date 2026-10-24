# TODO: cheap splat quality fixes

Priority: **medium**. Status: implemented 2026-10-23, **A/B on karlak 2026-10-24: exposure on by default, bilateral grid and anti-aliasing off; tile masks on by default (berlin A/B); depth outlier gate on by default and the seams of masked tiles checked (2026-10-24)** (see the end). Still open: the "Also open" list below (SH degree 1, 4K with a lower split threshold, Difix3D+). Both are flags or small changes plus an A/B run;
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

## Tile mask A/B (2026-10-24, `build/tilemask_ab.py`, berlin tile 0 copied to `build/berlin_ab`)

10000 steps x1, mesh prior dense 0.2, seeds, flatten 0.02; scored on nerfstudio's eval photos of the tile (never
trained on), over the pixels whose mesh surface lies in the tile (`build/tilemask_eval.py`; 65 of 104 views have such
pixels):

| arm | in-tile PSNR | p10 | all pixels | depth error median / p90 | within 5 % | Gaussians | train |
|---|---|---|---|---|---|---|---|
| no mask | 22.22 | 18.21 | 19.02 | 7.4 % / 1287 % | 47.5 % | 879k | 802 s |
| **mask** | 22.17 | **19.02** | 15.99 | **0.6 % / 19.6 %** | **84.6 %** | 635k | **537 s** |

The mean is the same, the worst views gain 0.8 dB, the fog in front of mostly-other-tile views is gone (the depth
error's tail), and training is a third faster. "All pixels" drops because the other tiles' parts are no longer trained;
the merge cuts them anyway. → **On by default** for tiled runs (Splat tab "Tiles: train only the tile's pixels",
`splat_tiles.TILE_MASK`; CLI `--no-tile-mask`; needs the dense mesh prior, else ignored). Not yet checked: the seams
of a merged masked run (the masks keep each tile's margin).

## Depth outlier gate and the seams of masked tiles (2026-10-24)

**Gate** (`nsplugin/mf_splatfacto_lidar.py` `depth_loss`, config `lidar_depth_gate` 0.5, `lidar_depth_gate_after`
2000; dense mesh maps only, LiDAR untouched): after warm-up, a pixel whose rendered depth lies more than 50 % *behind*
the mesh loses its target: the photos see past the mesh there (Poisson bubbles in halls, closures above the horizon,
blobs at crowns). One-sided on purpose: a render in front of the mesh is fog, and its target stays. The loss stays
normalized by all target pixels; the report line adds "gated x %" (`splat.parse_lidar` `gated_pct`). Off:
`--pipeline.model.lidar-depth-gate 0` (extra ns-train args). The smaller patch-hole size for the prior was not done:
the gate drops those targets wherever they disagree, without a second mesh.

berlin tile 0, 10000 steps, masks on (`build/tilemask_ab.py maskgate`): ~10 % of the targets gated from step 2000 on
(more than the "small share" the CPU check suggested), depth error p90 19.6 → 17.4 %, in-tile PSNR 22.17 → **22.32**,
p10 19.02 → **19.71** dB, Gaussians 635k → 631k. → on by default.

**Seams** (`build/seam_ab.py`, `seam_eval.py`, `seam_view.py`): tile 2 (the longest seam with tile 0, 338 m) trained
10000 steps without and with masks, each arm's tiles 0 + 2 cleaned as a tiled run cleans them and merged; scored on the
101 photos neither tile trained on, per pixel by its mesh surface's distance to the 0|2 seam (pooled PSNR / mean of
views / views):

| merge | < 5 m | 5–15 m | 15–40 m | > 40 m |
|---|---|---|---|---|
| no masks | 26.35 / 25.67 / 6 | 24.23 / 24.17 / 15 | 21.68 / 23.33 / 38 | 22.39 / 21.66 / 82 |
| masks | 26.10 / 25.51 / 6 | 24.12 / 23.96 / 15 | 21.90 / 23.48 / 38 | 22.52 / 21.82 / 82 |
| masks + gate (tile 0) | **26.42 / 25.85** / 6 | **24.27 / 24.18** / 15 | **21.97 / 23.55** / 38 | **22.55** / 21.81 / 82 |

- Masks alone cost ~0.1–0.25 dB at the seam (few views) and gain ~0.15–0.2 dB away from it; with the gate the seam
  band is as good as without masks. Per view, 11 of the 15 seam views are equal or better masked. The worst one
  (berlin_3_4_02875, −2 to −3 dB in the band) is a low view past a tree crown. Its seam ground looks the same in
  all arms (`build/berlin_ab/seam_02875.jpg`); the visible difference is the far ground of tiles outside this
  two-tile merge, which masked tiles leave empty instead of fogging over. No seam line shows.
- Tile 2 alone: the mask removed its fog (depth p90 46.7 → 5.5 %, 1.27M → 0.97M Gaussians), but in-tile PSNR was
  22.32 → 22.16 (p10 19.34 → 18.99) and it was not faster (735 / 763 s). Tile 0 had gained 0.8 dB at p10, a third
  faster. The merged scores above favour the masks, so they stay on.
- Found on the way: a tile's loss masks only cover photos whose depth map exists. `splat.run` renders the missing
  maps only later, so the A/B script now renders them first. The real tiled run computes the masks after the maps
  (`splat_tiles.run`); not affected.
