# TODO: manual tie points (human in the loop SfM)

Priority: **medium** (the user asked, 2026-10-25; option 1 first). Status: **option 1 done 2026-10-25** (see the end).

Where automatic matching fails (flights under different light, seasons, altitudes; berlin 3_2-3_4 still 3.39 m apart;
the kladno_dira crossing no other pass ties in), the user marks the same physical spot in two or more photos.

## Option 1 (first): tie points in the photos

- Workflow: SfM as usual -> the View shows where flights did not merge -> add tie points -> **Refine again** (refine
  only: features and matches kept). Before a first SfM it works too (photos found by the GPS priors).
- GUI: a "Tie points" panel: Ctrl + click in the 3D view lists the photos that see that spot (refined poses; other
  flights first); a photo opens large, a click marks the pixel (viser's image widget reports no clicks: a page script
  + hidden input, as the streaming upload's channel); with one marked photo the others show where the spot projects.
  List of tie points (name, photos, reprojection error after the refine), delete.
- Stored per project (`tie_points.json`: image name + pixel per observation), used by every later refine.
- Refine: each tie point = an extra track (triangulated from its rays) with a high weight in the BA; chunked refines
  also get it as a loop-closure edge in the pose graph. Report: their reprojection errors, the split before / after.

## Option 2 (maybe later): same point in the sparse cloud

Ctrl + click a spot on copy A, then on copy B: "the same" -> an equation for the per-flight offset / drift estimate
(as the split points). Coarser (the cloud's spacing), for gross offsets.

## Done 2026-10-25 (option 1)

- `modelforge/sfm/tie_points.py`: storage (`tie_points.json`: name + clicks {image, x, y} in COLMAP pixels),
  triangulation from the clicked rays, projection, errors. Per tie point the clicks of each pass (flight = source;
  passes = clicks > 60 s apart) are triangulated on their own; X_a - X_b between two passes feeds the refine where it
  moves whole flights / passes: `robust._split_pairs` yields it as 400 split matches (drift over time, the cross-flight
  drift, the chunked refine's loop closures) and `split_point_edges` lets it decide a flight pair's offset
  (`update_source_offsets`; the user's clicks override the SIFT split points there). `pipeline.stage_refine` runs
  with them (`tie_points.active`), reports `refine.tie_points` (px per point after the refine) and logs them; they
  are part of the refine's checkpoint key.
- **Not as BA residuals** (tried first): the BA satisfied them by turning the few clicked cameras (each dragging its
  own SIFT points), cheaper than moving a flight against its priors; the flight stayed 0.39 m off and the offset
  then measured ~0. As offset / drift measurements: synthetic two flights 0.6 / 0.4 / 0.3 apart with no cross-flight
  matches: second flight 0.78 m off without → 6 mm with 6 tie points (2 clicks per flight each), offset recovered to
  1 cm (`test_multi_source.py::test_tie_points_join_two_flights`).
- GUI: left tab **Tie points** (`gui/tiepoints.py`): tie point list (＋ New, 🗑 Delete), clicks per flight, Ctrl +
  click in 3D → "Photos seeing the spot" (registered photos it projects into, nearest first, 8 per flight), ◀ Open
  photo ▶, **Refine again (refine + export)**, the last refine's errors. The floating **Tie point** window (960 px)
  shows the photo: a click zooms in 5x around it, a second click records the pixel; markers: yellow the Ctrl + clicked
  spot, cyan the tie point so far, green this photo's click; Zoom out, Remove this click, Close. Clicks reach the
  server through `docker/patch_viser_page.py` (img[data-mf-click] → a hidden text input).
- Tests: `tests/test_tie_points.py` (storage; the tab on fullrun_test: pick → photos → zoom → record at the spot's
  projection to 0.01 px), the synthetic two-flight test.
- Not yet: tried on berlin_lighting by hand (its two flights: 32 % of the cross-flight matches merged).
