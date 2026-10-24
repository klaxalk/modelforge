# TODO: mitigate Gaussians that represent the sky

Priority: **high** (the user asked, 2026-10-24). Status: **sky masks done 2026-10-24, on by default for DJI projects** (see the end); open: a sky model for viewers, the spikes on crowns.

Splatfacto explains the sky in the photos with Gaussians: a dome or haze at some finite distance, often in front of
the horizon or hanging over the scene, which shows as floaters / fog from other viewpoints, bloats the splat and
breaks the splat mesh and the semantic search near the top of the scene.

Ideas to compare (A/B on a project with much sky, e.g. berlin's low oblique passes; held-out PSNR / LPIPS and renders
from off-path viewpoints, plus the count of Gaussians above / beyond the scene):
1. **Sky masks per photo**: segment the sky (a small segmentation model, or C-RADIO features we already run for the
   semantic search with a "sky" text query; or simply pixels with no SfM-mesh hit above the horizon) and drop those
   pixels from the photometric loss (the tile loss masks already do this per pixel: `loss_masks` /
   `batch["mask"]`), so nothing grows there.
2. **A background model** instead: a learned sky colour per direction (splatfacto's `background_color` "random" /
   an environment map, or an SH background), so the sky has a cheaper explanation than Gaussians.
3. **Pruning**: drop Gaussians beyond the scene (far from the SfM mesh / points, above the highest surface by some
   margin, or whose centre projects only onto sky pixels) during training and / or at export.
4. Depth prior: the mesh depth maps have no target for sky pixels; an explicit "far" target there.

Check first how much of the current splats is sky (karlak, berlin, kladno_dira): Gaussians above the mesh's top /
outside the scene box, and their share of the rendered pixels.

## Done: sky masks (2026-10-24; the user chose masks, an optional splat step, then on by default for DJI projects)

`modelforge/sky.py` + `export/sky_masks.py`: SegFormer-B5 (ADE20K, `nvidia/segformer-b5-finetuned-ade-640-640`, class
2 = sky; transformers from the RADIO packages, nerfstudio's venv) on every training photo at 1024 px, logits upsampled
to the training size (floor(w / N), before undistortion), the sky widened by 0.4 % of the width (`DILATE`, 8 px at
1920) -> `ROOT/sky/masks_x<N>/<image>.png` (0 = sky), ROOT = the project whose images/ the dataset uses (tiles share
it), manifest-keyed, made once per photo (~0.09 s per photo: berlin tile 0's 1049 in ~1.5 min). The lidar-data
dataparser's `--sky-masks` sets nerfstudio's `mask_filenames`: undistorted with the photo, out of the photometric loss
(combined with the tile masks); the plugin's `sky_alpha_lambda` (`sky.ALPHA` 0.3) adds the mean rendered opacity over
the sky's core (1 % of the width away from anything else: `sky_core_margin`) x its share. Splat tab "Sky masks (leave
the sky out)" (default from the kind: `sky.default_on`, DJI and only where the packages exist), CLI `--sky-masks` /
`--no-sky-masks`, `regularize["sky"]` (resume, tiles, queue carry it). `sky/` is an intermediate (Clean) and archived
on re-extraction. Tests: `tests/test_sky.py`, flags in `test_rosbag_kind.py`.

**berlin tile 0 A/B** (`build/tilemask_ab.py`, 10k steps, tile masks + depth gate, scored on nerfstudio's 104 eval
photos; ground = the tile's mesh pixels that are not sky; sky opacity = mean rendered alpha over the (widened) sky):

| arm | ground PSNR | p10 | sky opacity | sky pixels > 0.5 | Gaussians | train |
|---|---|---|---|---|---|---|
| no sky masks | 22.56 | 19.89 | 0.99 | 99.7 % | 631k | 700 s |
| sky, opacity 0.05 on all sky | 22.56 | 19.97 | 0.21 | 21 % | 584k | 716 s |
| sky widened, opacity 0.05 on the core | 22.55 | 19.95 | 0.21 | 20 % | 582k | 710 s |
| **sky widened, opacity 0.3 on the core** | **22.56** | **19.95** | **0.16** | **15 %** | 583k | **605 s** |

- The sky dome is gone; the ground is the same. A first score said -1 dB: the SfM mesh closes surfaces above the
  horizon, so "ground" pixels included sky (`build/sky_diag.py`: all the extra error in the top fifth of the image).
- What is left over the sky is mostly the widened edge band around crowns (the tree is there) and faint whitish
  spikes on the crowns (`build/berlin_ab/sky3_02178.jpg`): needle-like Gaussians in the edge band, which no loss
  supervises; the needle term (`--needle`) may take them.
- In a viewer the sky now shows the viewer's background (black in renders). Open: a sky model (an environment colour
  per direction, e.g. SH) for nicer renders without Gaussians.
