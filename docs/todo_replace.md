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
- **Donors:** berlin has good trees (127 by the shed, 120 the birch, 40 the willow, 48, 220: sharp next to their
  photos; `build/treeswap/pool_sheet.jpg`), but a splat's tree is good **only from where it was photographed**: seen
  from the target's camera angles, the first donors were fog and needles even in their own place
  (`donor_self.jpg`). Most "sharp" donors were rated from 2-5 photos of one side. Fixes: donors any size from 0.5 to
  2 x (was 0.6-1.67, closely alike, denser than the median: it hid the good ones); a donor's own Gaussians only
  (haze and floaters around it came along as white clouds; dropping all > 0.5 m left hollow twigs), no sky-coloured
  specks; at least 10 photos, seen from all round ranks higher (`sides`: the photographed side per object); optional
  **turn** so the donor's photographed side faces the target's (sun side turns too). Only 5 of berlin's objects are
  sharp and seen from all round (77, 88, 120, 127, 165); with them + turning: 136 -> 165 and 107 -> 86 look like real
  shrubs, also 40 deg off the path (`donor_cov.jpg`); turning alone (one-sided donors) changed little
  (`donor_turn.jpg`).

- **Texture repair** (ground areas with no object): a detail map (render vs photo per 31 px window of 300 photos,
  placed on a 2 m grid by the rendered depth; 15 s on berlin) marks weak ground (< 0.3 of the photos' detail, >= 8 m2,
  no object on it): 187 areas, the largest dark scrub (g94 240 m2, g99 284 m2). The first fill made things worse
  (bright blobs, dark sticks): sources only needed to look similar, so equally blurry ground was copied, with large
  glare Gaussians. Sources now keep >= 0.6 of the photos' detail (GOOD_SOURCE) and copied Gaussians are < 1 m: no
  blobs, sharper grass blades (g99 off the path); from the photo positions the change is modest. Sheets:
  `build/treeswap/ground_sheet2.jpg`, the map `build/treeswap/detail_mid.jpg`.

- **Templates** (kladno_dira, 2026-10-31: a road was filled with a different ground's texture: the per-cell fill
  matched the area's rim, grass along the road). Ground repairs copy from a template disc: picked by the user (Ctrl +
  click + radius; sharpness not required), or automatic, **the most similar detailed patch** (`suggest_template`: a
  disc of the area's size, 3-8 m radius, open and detailed cells, nearest in mean colour + semantic feature to the
  weak area's *own* cells, which blur barely changes; 0.3 per 100 m). On kladno_dira the road's template is the same
  road 12 m on; the scrub-covered areas get the gravel with stones around them (`build/treeswap/kd_sheet.jpg`).
  Under removed objects the fill still matches the rim (the hole's content is the object).
- **Template search without distance, by what the photos show** (the user: similarity in colour, texture and
  semantics matters, not distance): the detail map also records per 2 m cell the photos' colour and texture (log
  detail energy after a 1 and a 3 px blur), blur-free; the comparison normalizes every number, then weights the
  groups (colour 0.5, texture 0.5, semantics 2: `TEMPLATE_W`). Equal weights sent kladno_dira's road (g35) to a dirt
  yard 185 m away with a car in it (photo colour + texture of a dusty road resemble a yard); with the semantics
  leading, the same road 12 m on (`build/treeswap/kd_sheet3.jpg`). The per-cell fill has no distance term either.
  (A first version weighted before normalizing: the weights had no effect.)
- **The transition** (the user asked, 2026-10-31; `build/treeswap/seam_eval.py` on a kladno_dira copy
  `build/kd_ws`, areas g35 road, g82, g84, g86): the biggest seams were (1) the blocky outlines of the 2 m map cells
  (now rounded: blurred at 1 m, thresholded) and (2) the fog over weak ground, which stayed above the new ground
  (`haze_over`: Gaussians > 0.5 m up to 6 m over a ground repair are removed). Then: **feathering** (the copy reaches
  1 m beyond the hole, crossfading by opacity with the original ground layer) and **gradient-domain colour blending**
  (the colour difference surroundings - copy along the edge, taken over 5 x 5 cell windows, smoothed along the edge,
  capped, harmonically interpolated over the patch and added at 0.7: the edge matches, the texture stays). A first
  blending from 3 x 3 windows painted dark blotches (g84) where one dark neighbour decided. Also fixed: the hole's
  large old ground Gaussians (the smear itself) were not removed (the copy's size limit was used for removal too).
  The seam score (low-pass edge gradient / surroundings) barely separates the variants (0.88-1.13); the renders do:
  `seam_sheet3.jpg` (from above), `seam_views.jpg` (from photo positions: the quarry floor's gravel now matches the
  photos, the road shows only a faint patch). Synthetic test: a 0.25 luminance step at the edge -> < 0.04, texture
  kept.
- **Relative thresholds:** kladno_dira's detail map has a median of 0.33 (berlin 0.62): fixed 0.3 / 0.6 made half
  of it weak (one 10 ha area) and few spots good. Now weak < min(0.3, 20th percentile), good >= min(0.6, 60th).
  Areas larger than 20 x 20 m are cut into such pieces.
- **Over the mesh only** (the user asked): weak areas and automatic templates only where the LiDAR mesh, else the SfM
  mesh's measured part (mesh_trimmed: no patched holes) has triangles (`mesh_cover`). kladno_dira: 587 -> 128 areas.

## GUI

Left panel → **Replace** (`gui/replace.py`): Look for (the queries), Find and rate objects (job `replace`); Automatic
(Action remove + fill / replace by a donor, Suggest, the proposals with a checkbox and Go each, Apply checked); Manual
(Pick, Ctrl + click an object, donor candidates with Go, Remove + fill ground / Replace by the donor); Edits (Go /
Undo each, Undo all); **Ground texture** (Map weak ground: the detail map job, finding the objects first if needed;
Template: Draw a template, a polygon of Ctrl + clicked corners (Undo corner / Finish / Clear), green; Copy from: the
most similar detailed patch (default) / similar sharp ground anywhere / my template; the templates applied repairs used
show as thin green rings / outlines; Draw the area to repair, a yellow polygon the same way, Repair the drawn area
(the user asked for polygons instead of circles, 2026-10-31; the library still takes circles); Show weak areas: magenta outlines on
the ground labelled g<id>; the areas weakest first with a checkbox (areas >= 32 m2 ticked) and Go each; Repair checked
areas; Pick a spot (Ctrl + click) + Radius (m) + Repair circle, a yellow ring previews it). Markers: orange selected,
red proposals, cyan donors. Test: `test_replace_panel_ground_texture`.

## Open

- More donors seen from all round: a **donor library across projects** (the best close-up trees / bushes of karlak, kladno, ...; placed with the same
  colour match). The likely way to make donors worth it.
- False targets: graffiti inside a ruin (berlin object 139) read as vegetation; the review list lets the user untick.
- Dark patches remain under some removed bushes (berlin 136).
- Facades (a grid on the wall plane), ground repair of weak areas without an object (thin spots).
