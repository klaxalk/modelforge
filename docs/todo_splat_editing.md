# TODO (idea, not now): splat editing, SuperSplat-like

Priority: low. Status: proposed 2026-10-23; the user said not now. Their use: removing sky floaters and fog, and opening holes that
should be open.

## Automatic selection

- Sky floaters: per-photo sky masks from the C-RADIO / SigLIP model the Search tab already runs
  (`export/semantic_features.py`); Gaussians that land on sky pixels where they are visible.
- Free-space fog: Gaussians mostly seen *through* in the rendered depth (the photos see surfaces behind them). Not by
  the SfM mesh: Poisson closes real openings.

## Manual selection

- viser `on_rect_select`: everything in the rectangle, or the front surface only.
- Click a blob → its connected group off the surface.
- A box with transform controls.
- The Search tab's semantic matches as a selection.

## Editing

Delete / undo as a non-destructive mask next to `splat/splat_enu.ply` (kept by export, splat mesh, semantic; a new
training run starts from it or ignores it, to decide). No lasso / brush without another viser bundle patch
(`docker/patch_viser_camera.py` is the pattern).
