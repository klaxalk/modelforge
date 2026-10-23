# TODO: level of detail for large splats

Priority: low. Status: proposed 2026-10-23, not started; the user will ask for it.

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
