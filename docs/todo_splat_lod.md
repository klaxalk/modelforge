# TODO: level of detail for large splats

Priority: low. Status: **step 1 done 2026-10-24** (the user asked): cells + merged levels + a cut per camera in the
View tab (`modelforge/splat_lod.py`, `modelforge/gui/lod.py`). Not done: Fly look-ahead, render fine-tuning, 3D Tiles.

## Done (2026-10-24)

- **Hierarchy** (`splat_lod.build`, `python -m modelforge.splat_lod PROJECT`): not per tile but a k-d split of the
  merged splat (median cuts along the box's longest axis, 1-99 % range) into cells of <= 100k Gaussians (`CELL_MAX`),
  which works the same for untiled splats. Per cell levels of ~1/4 each (`RATIO`), down to >= 1000 Gaussians, at most
  5: per level a cubic voxel size, found by bisection so the count comes out right; Gaussians larger than the voxel
  stay, the rest merge per voxel by moment matching with weight opacity x sigma_1 sigma_2 (the two largest axes);
  opacity = sum of the children's weights / the merged Gaussian's sigma_1 sigma_2, clipped to 1; colour the weighted
  mean (band 0: the GUI draws base colours). Every level is merged from level 0, so errors do not pile up.
  `splat/lod/`: index.json (keyed by splat_enu.ply's size + mtime) + per level npy files (float32 centres, float16
  covariance upper triangle, uint8 RGBA: viser's 32 bytes per Gaussian) memory-mapped by the GUI; `L<k>_parent.npy`
  maps level 0 to the original indices and each level-0 Gaussian to its merged one (the semantic search highlights a
  merged Gaussian by its best child). Clean removes it.
- **Measured:** karlak 2.27M Gaussians -> 32 cells, levels 2.27M / 567k / 142k / 35k, 18 s, peak 1.3 GB, 81 MB;
  kladno_dira 5.77M -> 64 cells, 5.77M / 1.44M / 360k / 90k, 48 s, peak 3.0 GB, 205 MB.
- **Quality** (`build/lod/render_cmp.py`, gsplat in nerfstudio's venv, 12 training cameras; the truth = level 0
  rendered at 4x and box-filtered; each level rendered at the resolution where its voxel is P px at the median depth):

  | P | | karlak L1 / L2 / L3 | kladno_dira L1 / L2 / L3 |
  |---|---|---|---|
  | 1.5 | merged level | 23.3 / 21.3 / 21.6 dB | |
  | 1.5 | level 0 itself | 20.6 / 18.3 / 18.2 | |
  | 1.5 | random subset of the same size | 20.8 / 17.8 / 16.7 | |
  | 2 | merged / level 0 / random | | 25.0 / 21.9 / 20.5 · 24.6 / 22.5 / 21.3 · 19.9 / 14.5 / 11.7 |
  | 3 | merged / level 0 | 24.2 / 21.6 / 20.1 · 23.5 / 20.3 / 18.1 | |
  | 6 | merged / level 0 | 24.4 / 20.9 / 19.0 · 28.1 / 22.8 / 20.1 | |

  Merged levels act as anti-aliasing: up to ~3 px per voxel they match or beat the full splat drawn at that size,
  and beat the random subset (what the GUI showed so far) by 2.5-9 dB. -> switch at `PIXELS` = 2 px. Covariance
  scale x0.6 / x1.6 instead of 1 gave no better worst cases (`COV_SCALE` = 1).
- **Cut** (`splat_lod.choose`, ~1 ms for 64 cells): every cell starts coarsest; a heap refines the cell whose voxel
  looks largest (voxel x focal / distance to its 2-98 % box; outside the view x0.25) one level at a time while the
  budget (View "LoD budget", default 2M) holds and the voxel is over 2 px. kladno_dira from 300 m up, from a photo's
  pose and from 10 m: ~2.0M each, 12-18 cells at full detail.
- **GUI** (View "Level of detail", default on; used when the splat has more Gaussians than the budget): every cell
  its own viser splat object (`/dense/splat_lod/cNNNN`; viser sorts all splat objects together, so cells blend
  correctly; checked in its source, `GaussianSplats.tsx`); the cut follows the browser that moved last, 0.4 s after
  its camera stops (every 2 s while it keeps moving), changed cells swapped in one `server.atomic()`. Without a
  hierarchy it is built in a job process of its own (if the RAM is free; ~0.6 GB per million Gaussians), meanwhile
  the random subset as before. The live training preview is unchanged.
- Viser notes: a hidden splat object is still sorted (moved off screen), so all levels cannot be preloaded and
  toggled; adding / replacing an object re-merges all buffers in the browser (local, not sent again).
- Test: `tests/test_splat_lod.py`.


## Why

The GUI sends a random subset of at most 1M Gaussians (`gdata.SPLAT_SHOWN`), or all with View → "All gaussians". A
random subset thins everything evenly (holes on near surfaces, detail spent on far parts nobody sees); all of them
works for 3–4M (kladno_dira) but tiled flights grow past what the browser's memory and its per-frame depth sort
handle. Goal: a fixed budget in the browser, fine near the camera, coarse far away, at any splat size.

## 1. Hierarchy, built once per splat (offline)

- Cells: the splat tiles are the top level (`tiles/tile_KK/splat/clean_enu.ply`, already cleaned and seamless); an
  octree inside a tile later. Untiled splats: one octree over the whole splat.
- Per cell coarse levels (e.g. 100 % / 25 % / 6 %) by merging neighbouring Gaussians into fewer, larger ones (moment
  matching, opacity-weighted: mean position, covariance = Σ w (Σ_child + (μ_child − μ)(μ_child − μ)ᵀ), opacity
  combined, colour = weighted SH band 0; higher bands dropped on coarse levels). As in hierarchical 3DGS (Kerbl et al.
  2024) and the LoD splat formats (Cesium 3D Tiles, Spark).
- Optional: fine-tune the coarse levels by rendering against the training photos (hierarchical 3DGS does); without
  it far areas look softer / fatter.
- Stored in `splat/lod/` (levels per cell + an index with each cell's box and Gaussian sizes), keyed by
  splat_enu.ply's size + mtime; built after training / the tile merge (a job, minutes), Clean removes it.

## 2. Choosing what to draw (per browser, server side)

- Per cell the projected size of its Gaussians at the camera picks the level: when the fine level's Gaussians would
  be under ~1–2 px, the coarse one looks the same. A total budget (e.g. 2–3M, a View setting) lets near cells win.
- The server already gets every browser's camera: when it stops or moves enough (debounced ~0.5 s), choose the cut
  and swap the changed cells, each cell its own viser splat object (show / hide / replace). A short fade against
  popping if viser allows it.
- Fly: the path is known ahead (`gdata.FlightPath`), so the cells ahead are loaded early.
- **Check first:** whether viser sorts several splat objects together (overlapping cells must blend in depth order).
  If not: one combined object per cut, rebuilt on change (simpler, more data per swap).
- A cell of 100k Gaussians is a few MB: fine on a local network. The live preview during training stays as today.

## 3. What the user sees

Opening a 10M-Gaussian project shows the whole scene at once from ~1M coarse Gaussians, without holes; zooming into
a corner sharpens the tiles around the camera about half a second after it stops, the far ones stay coarse; the
browser's load stays the same everywhere.

## Order

1. Tile-level LoD: every tile at 3 merged levels, the level by the camera's distance / projected size. Simple,
   since the tiles are cleaned and seamless.
2. An octree inside tiles (large tiles, untiled splats).
3. Render fine-tuning of the coarse levels.
4. A 3D Tiles export (KHR Gaussian splatting) for viewers outside the GUI (Cesium).
