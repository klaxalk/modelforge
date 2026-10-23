# TODO: cheap splat quality fixes

Status: proposed 2026-10-23, not started; the user will ask for it. Both are flags or small changes plus an A/B run;
validate on a small project first (fullrun_test, karlak; held-out scoring as in `build/splat_exp.py`), not on
kladno_dira.

## 1. Exposure from the metadata

- djmd records the exact exposure per frame: shutter (exact, not the SRT's rounded stop), ISO, f-number
  (`docs/djmd_format.md`). Relative exposure per photo = shutter × ISO / f². Photos of one flight with auto exposure
  differ in brightness; today splatfacto has to absorb it (fog, view-dependent colour) or ignores it.
- Proposal: write the factor per frame into transforms.json (poses.json has the raw camera values per image) and, in
  the plugin (`modelforge/nsplugin`), scale the rendered image by it before the loss (or the photo by its inverse);
  exports render at a reference exposure (the median).
- On top: splatfacto's **bilateral grid** (`--pipeline.model.use-bilateral-grid True`; `splat.train_command` already
  sets its scheduler steps) for white balance / tone per photo, as a Splat tab option. It cannot remove hard shadows
  (spatially varying): see `docs/todo_lighting.md`.
- Check first: how much exposure actually varies in the user's flights (fixed exposure on kladno_dira per the user).
  Also useful for the mixed sunny / overcast project.

## 2. Anti-aliased rasterization

- gsplat's `antialiased` rasterize mode (Mip-Splatting's 2D filter; splatfacto `--pipeline.model.rasterize-mode
  antialiased`). Flights at different heights see the same ground at very different scales: plain 3DGS aliases
  there and grows needle-like Gaussians (seen on kladno_dira and dira2k).
- A Splat tab option (default after the A/B decides), stored in resume.json / tiles.json like the other settings.
- A/B: held-out PSNR / SSIM / LPIPS, plus renders from heights between the flight's passes.

## Also open (earlier sessions; the user's main complaint: flat ground seen from one straight pass at ~45° stays soft)

Done for it already (CLAUDE.md "Flattening", "Starting Gaussians from the SfM mesh"): flattening 0.02, mesh seeds
(default on); a far-field mask was tried and removed (it did not work well for the user). Not done, in the order
suggested:

1. Lower SH degree (1) for one-sided views: higher bands overfit view-dependent colour where every photo looks
   from one side. splatfacto `--pipeline.model.sh-degree 1`; A/B on a tile.
2. Full resolution ×1 for 4K with splatfacto-big's split gradient threshold (0.0005), so 4K detail gets Gaussians
   (karlak: 4K default densification peaked at only 8.4 GB and grew no more Gaussians than ×2; CLAUDE.md "karlak
   investigation").
3. Difix3D+ (NVIDIA, 2025): a single-step diffusion model cleans renders from off-path views, which are added as
   pseudo training views, then fine-tune. SuperGaussian was judged a poor fit.
- Judging it needs a berlin tile viewed **off the path**: held-out photos from the same pass cannot show the softness.
- The SfM-mesh depth prior has p90 errors of ~400 % on berlin tiles (median 1.6 %): the mesh has surface where the
  photos see far things (Poisson closure, patched holes, scene edge). Not investigated; candidates: drop depth
  targets beyond the mesh's trusted range, or per pixel where rendered and mesh depth disagree by far.
