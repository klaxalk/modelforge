# TODO: Unreal Engine 5 plugin for splats with level of detail

The user asked on 2026-10-26: show modelforge's Gaussian splats in Unreal Engine 5, with the same level of detail as
the GUI (View → Level of detail, `modelforge/splat_lod.py`, `gui/lod.py`).

## Status: waiting for the user

The user will provide these first. **Do not start before they are in place.**

1. **Unreal Engine 5 installed on this machine** (Linux, RTX 3090). The aim is that Claude can compile the plugin
   and render offscreen screenshots itself, to check its own work. Note down:
   - the engine version;
   - its install path;
   - whether it is a source build or the binary release;
   - how to reach it from the dev container, or whether it runs on the host.
2. **A sample splat visualization plugin** to start from. Note down:
   - where it lives;
   - its licence;
   - how it renders: Niagara particles, or its own shaders / scene view extension.

## Plan, as discussed

**Reused from modelforge:**
- **The LoD build.** `splat/lod/` holds `index.json` (the cells: box, and per level the offset, count and voxel
  size) and per level `L<k>_centers.npy` (float32), `L<k>_cov.npy` (float16 upper triangle) and `L<k>_rgba.npy`
  (uint8).
  - Possibly also an engine-friendly binary export: one file, little-endian, no numpy headers.
- **The cut.** `splat_lod.choose` is about 100 lines and gets ported to C++. In the engine it can run every frame or
  every few frames, rather than only after the camera stops as in the browser.
- **The frame.** ENU in metres → Unreal's left-handed Z-up frame in cm: flip one horizontal axis, ×100.
  `export/georef.json` holds the origin, and Cesium for Unreal could place the splat on the globe.

**Renderer (the open question; decide once the sample plugin is known):**
- **Niagara:** quick, but a poor fit for swapping cells in and out and sorting millions of particles.
- **Own scene view extension with compute shaders:** project, GPU depth sort, instanced quads, depth-tested
  against the opaque scene. The reference is Aras Pranckevičius' Unity Gaussian splatting (MIT licence).
- In either case, cells live in GPU buffers and only the changes are uploaded when the cut moves.

**Known limits:**
- The splats are unlit translucency: no Lumen lighting or shadows.
- TSR / TAA can ghost on them.
- Band 0 colour only, as the LoD levels store it.

**Check:** render the saved views (`PROJECT/views.json`) offscreen in Unreal and compare them with modelforge's
render of the same splat at the same cameras.
