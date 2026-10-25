# TODO: register more images into an existing SfM model

Priority: **medium** (the user asked, 2026-10-25). Status: **done 2026-10-25** (see the end).

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

## Done 2026-10-25

- `modelforge/sfm/register.py`: candidates = extracted photos not in use (every N-th per flight in time);
  `evaluate.locate` (factored out of the held-out evaluation; now also the inliers' median reprojection error);
  the check: the pose must lie within 1 m / 2° of the flight's trajectory (`interpolated`: the metadata pose + the
  refine's corrections at the model photos interpolated in time; the user's point that the trajectory is smooth);
  `registered/` (hard links + registered.json keyed by sparse/refined); `add_frames` appends them to transforms.json
  (frames "mf_registered", intrinsics per frame for several cameras, exposure gains via `exposure.gains_for`), also
  after every export while the model is that one; `remove`.
- **Measured** on kladno_dopravnik (24 frames between 56 model photos, 4K; reprojection of the model's points each
  sees): raw metadata 54 px median, interpolated corrections 2.0 px (p90 7.7, max 33), registered 0.67 px (max 0.97);
  against a full SfM of all 80 both are at that comparison's floor (the 56 shared photos differ by 5.9 px between the
  two SfM solutions). Registered sit 7 mm / 0.015° from the interpolated trajectory (median). 3 s per 4K photo.
- GUI: Sparse tab → "Register more photos (for the splat)": every N of the unused, Register more photos (stage
  `register`, a job process), Remove registered photos, status (registered / stale after a new refine).
- Tests: `tests/test_register.py` (fullrun_test, half in use: registering photos the model also holds gives its own
  poses within 3 cm / 0.05°; transforms.json; export / Remove).
- Splat A/B (`build/reg_ab.py`, kladno_dopravnik copy, 6000 steps x2, 8 held-out registered photos): 56 model
  photos 22.28 dB / SSIM 0.595 / LPIPS 0.372 vs + 16 registered 22.34 / 0.590 / 0.386: no gain where the keyframes
  are already dense (as the keyframe calibration found); the value is SfM time (SfM on keyframes only) and thin
  stretches (fast turns, low passes), untested.
