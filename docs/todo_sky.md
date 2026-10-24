# TODO: mitigate Gaussians that represent the sky

Priority: **high** (the user asked, 2026-10-24). Status: not started.

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
