# Covisibility tiles near their photo target (done 2026-10-24)

Priority: low. Status: **done 2026-10-24** (the set-up's photo counts; a training A/B on one tile is still to do).

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

## Fix (done)

1. **Balance**: the slack shrinks with the depth of the cut, `SLACK / (1 + depth)` (`tile_regions.split_photos`), so
   nested cuts do not compound.
2. **Cap per tile**: the photos joining a tile beyond its own (by any of the three rules) are ranked by the share of
   the tile they show (max of pixel share and point share) and at most `EXTRA_MAX` (0.5) x its own photos stay, plus
   every one showing >= `KEEP_SHARE` (0.5) of it (`splat_tiles._cap`, `partition(extra_max=)`; None = no cap).
   tiles.json records `capped` per tile; the set-up's log line says how many were left out.

## Measured: kladno_dira's set-up (`build/tile_size/exp.py`, read-only on the project; the mesh raycast included)

| variant | own photos per tile | photos per tile | slots |
|---|---|---|---|
| before (slack 0.25 at every depth, no cap) = the trained set-up | 233–953 | 479–2434 | 13,021 (2.5x) |
| slack / (1 + depth), no cap | 304–666 | 506–2105 | 12,996 |
| **+ cap 0.5 (the default)** | 304–666 | 479–1206 | **8,593 (1.66x)** |

The largest tile, 1206, keeps 666 photos beyond its 540 own because they show >= 50 % of it.

## Open

- A training A/B on one tile (held-out PSNR / LPIPS as `build/tile_exp.py`): does the cap lose anything at the seams?
  The berlin case the pixel rule fixed (a low oblique pass showing 67 % of a tile) is kept by `KEEP_SHARE`.
- kladno_dira keeps its current tiles until Set up tiles runs again (and a retrain).
