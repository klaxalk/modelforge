# TODO: manual tie points (human in the loop SfM)

Priority: **medium** (the user asked, 2026-10-25; option 1 first). Status: planned, after the lighting work.

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
