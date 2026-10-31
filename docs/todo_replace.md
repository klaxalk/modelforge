# Content-aware repair of a splat (Replace)

The user asked (2026-10-31): repairs of weak parts of large outdoor drone splats by copy/paste within the splat, "the
equivalent of Photoshop's content-aware fill". A broken low-res tree may be replaced by a *different*, sharp tree:
realism of the exact object does not matter. Generic vegetation (tree, bush, hedge). Two modes: **automatic**
(proposals to review, Apply checked) and **manual** (Ctrl + click an object, "what is it?", ranked candidates, pick).
A replacement = 1. remove the object, 2. fill the ground under it from similar ground elsewhere, 3. place the donor.

Why not a diffusion model: NVIDIA Fixer and Difix were tried on kn_odin (`build/fixer/`, 2026-10-31): Fixer erases fine
detail (cables, the view through windows) and the splat barely takes up even Difix's good views (+0.1 dB over a
photos-only fine-tune). No 3D-native model exists for drone splats (SplatFormer: objects; SceneSplat: indoor semantics).

## Code

- `modelforge/replace.py`: find objects, rate them, edits (remove / fill / donor), apply onto the base splat, undo.
- `modelforge/patchfill.py`: PatchMatch on a top-down cell grid (per-cell source offsets, coherent chunks).
- `modelforge/export/replace_rate.py` (nerfstudio venv): render vs photo per object.
- Tests: `tests/test_replace.py` (synthetic ground + two blob trees; stripes through a hole).
- Prototype and every experiment: `build/treeswap/` (berlin; `trees.py`, `swap.py`, `pmfill.py`, `rate.py`, `views.py`,
  `render.py`, `sheet.py` (labelled sheets), `lib_test.py`; workspace `build/replace_ws` links berlin).

## Findings on berlin (2026-10-31)

- **Query:** "a tree" catches only the clearest crowns (34k Gaussians at 0.6); "vegetation" (0.55) or "trees" (0.58),
  split by the height above the ground (> 1.5 m: object; below: ground cover), finds the trees and bushes.
- **Floaters** labelled vegetation (sky Gaussians, up to 285 m "tall trees") are dropped by size (leaves < 1 m).
- **Segmentation:** the plain watershed of the canopy split bushes into pieces (a swap then left the rest); merging
  basins unless the dip is >= max(1 m, 25 % of the lower peak) gives single trees (292 -> 219 objects on berlin).
- **Ground:** from the splat (rig.Ground) it ranged -351 ... +44 m on a site at about -13 m (floaters above and
  below the ground): 28 objects stood far underground. From the SfM points (10th percentile per cell, spikes > 2 m
  off the neighbourhood dropped, a 10 m grey opening against canopies): -26 ... +3 m, none.
- **Rating:** Gaussians per m2 is not "broken" (a big healthy tree drawn with fewer, larger Gaussians looked fine).
  Render vs the photos that see the object best (Laplacian energy after a 1 px blur; PSNR after a gain): the worst
  are fog / smear, the best match their photos.
- **Ground fill:** one cloned patch of the hole's shape came from 120-430 m away and did not match (brown in green).
  PatchMatch per cell blends; sources must have nothing above the copied layer (copy <= 1.5 m from cells with
  nothing above 1.5 m: copying <= 1 m from cells with up to 1.5 m cut low bushes into dark needle fragments; sources
  with nothing above 1 m were only bare paths: a sparse, pinkish fill). Sheet: `build/treeswap/ws_sheet3.jpg`.
- **Donors:** within berlin no tree has close-up detail (the site was flown at a distance): donors looked like coarse
  leaf clusters at close range, no better than the smear. Remove + fill alone looked better there.

- **Texture repair** (ground areas with no object): a detail map (render vs photo per 31 px window of 300 photos,
  placed on a 2 m grid by the rendered depth; 15 s on berlin) marks weak ground (< 0.3 of the photos' detail, >= 8 m2,
  no object on it): 187 areas, the largest dark scrub (g94 240 m2, g99 284 m2). The first fill made things worse
  (bright blobs, dark sticks): sources only needed to look similar, so equally blurry ground was copied, with large
  glare Gaussians. Sources now keep >= 0.6 of the photos' detail (GOOD_SOURCE) and copied Gaussians are < 1 m: no
  blobs, sharper grass blades (g99 off the path); from the photo positions the change is modest. Sheets:
  `build/treeswap/ground_sheet2.jpg`, the map `build/treeswap/detail_mid.jpg`.

## GUI

Left panel → **Replace** (`gui/replace.py`): Look for (the queries), Find and rate objects (job `replace`); Automatic
(Action remove + fill / replace by a donor, Suggest, the proposals with a checkbox and Go each, Apply checked); Manual
(Pick, Ctrl + click an object, donor candidates with Go, Remove + fill ground / Replace by the donor); Edits (Go /
Undo each, Undo all); **Ground texture** (Map weak ground: the detail map job; Show weak areas: magenta outlines on
the ground labelled g<id>; the areas weakest first with a checkbox (areas >= 32 m2 ticked) and Go each; Repair checked
areas; Pick a spot (Ctrl + click) + Radius (m) + Repair circle, a yellow ring previews it). Markers: orange selected,
red proposals, cyan donors. Test: `test_replace_panel_ground_texture`.

## Open

- A **donor library across projects** (the best close-up trees / bushes of karlak, kladno, ...; placed with the same
  colour match). The likely way to make donors worth it.
- False targets: graffiti inside a ruin (berlin object 139) read as vegetation; the review list lets the user untick.
- Dark patches remain under some removed bushes (berlin 136).
- Facades (a grid on the wall plane), ground repair of weak areas without an object (thin spots).
