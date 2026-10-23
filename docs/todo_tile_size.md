# TODO: keep covisibility tiles near their photo target

Priority: low. Status: proposed 2026-10-23, not started; the user asked to let the running kladno_dira splat finish
first.

## Symptom

kladno_dira, 5187 photos ("extra dense" keyframes of 8000), Photos per tile 500 → 11 covisibility tiles holding
479–2434 photos each, 13,021 photo slots (2.5x the photos). The earlier build of the same flight (2574 photos, 6
rectangles) had 528–805 per tile, 1.7x. Training took ~13 min per tile and round instead of ~8; large tiles may train
softer (~800 photos trained well before).

## Causes

1. **Unbalanced split**: `tile_regions.split_photos` bisects recursively, each cut up to `SLACK` (0.25) off
   balance; four levels compound, so own photos range 233–953.
2. **Joining rules scale with photo density, not area**: a photo joins a tile when >= `MIN_VISIBLE` (25 %) of its
   seen points lie in its region, or >= `MIN_IN_CELL` + look-at near, or >= `MIN_PIXELS` (5 %) of its pixels
   (`splat_tiles.partition`). The tile count follows the photo count, so twice the photos halve the regions
   (~8000 m², ~90 m across here) while an oblique view from 30–50 m still covers about one region: it then has
   25 % of its points in two or three. The largest tile: 953 own + 196 by pixels + ~1285 by points.

## Fix

1. **Cap per tile**: own photos + at most ~50 % extra (a setting), the extras ranked by how much of the tile they
   show (pixel share from the SfM mesh, else point share). Bounds every tile whatever the density.
2. **Balance**: smaller slack at the deeper cuts (or a final rebalancing of the parts), so no tile exceeds the target
   by ~2x.
3. Check on kladno_dira's set-up only (`splat_tiles.setup`, CPU, ~minutes): slots, per-tile counts, seams; then a
   short A/B on one tile (held-out PSNR / LPIPS, as `build/tile_exp.py`).
