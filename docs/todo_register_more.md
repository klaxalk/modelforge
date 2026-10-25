# TODO: register more images into an existing SfM model

Priority: **medium** (the user asked, 2026-10-25; after the tie points and the lighting work). Status: planned.

Idea (the user's): SfM on a subset (keyframes), then the other extracted frames placed into that model, so the splat
can use any denser subset without running SfM again.

- Interpolated poses alone are not enough: a splat needs ~1 px (0.04° at 1920 px, 83° HFOV); between keyframes
  ~1 s apart, gimbal corrections, wind and the 10 Hz telemetry leave a few tenths of a degree (5-10 px): the extra
  photos would blur the splat.
- So: start each extra frame at its interpolated pose (keyframe poses + the telemetry between them), then register it
  into the model with the model fixed: SIFT features, matches to the nearest keyframes (in time; same source), 2D-3D
  from their tracks, PnP + refinement (`pycolmap.estimate_and_refine_absolute_pose`), as `evaluate.pose_holdout`
  already does for held-out photos (COLMAP's image registrator). Optionally a local BA of the new cameras only.
  No global BA: cost = features + a few matches per frame, any number of frames.
- GUI: Sparse tab → Images used: "Register more images" (every N-th / all of the extracted pool between the
  keyframes) after SfM; transforms.json gets keyframes + registered frames (marked), the splat trains on them;
  frames that fail to register (too few inliers) are left out and listed.
- Expected value: SfM time scales with the keyframes only (large flights: hours saved); splat quality gains are
  likely small where keyframes already cover well (kladno_dopravnik: keyframes dense 25.50 dB = all images 25.45),
  larger where they are thin (fast turns, low passes) or for 4K training with more views. Measure on karlak.
