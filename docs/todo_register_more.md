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
- **Choosing the photos like Images used** (2026-10-25, the user asked): the folder has the same Selection as Sparse
  tab → Images used (`gui/subset.py Selection`, shared): every N-th with Starting at and a Time range, or keyframes at
  a density (default **dense**: SfM on sparse keyframes, the denser ones registered; extra dense / sparse before 2026-10-26); the choice is made over the
  whole extracted pool and its photos in use are skipped (`register.candidates(selection=)`; the preview: "→ N photos
  to register (M chosen, K of them in use)"). Keyframes need the overlap analysis (Images used → Analyze). Before:
  only "every N of the unused photos" (still `register.run(every=)`). Tests: `test_candidates_from_a_selection`,
  `test_gui_register_selection`.
- **Next to isolated photos** (2026-10-26, the user asked): registration uses `sparse/refined`, which still holds the
  photos the export leaves out (sfm/isolated.py), so a photo next to them was located against their offset copy of
  the ground, passed the trajectory check (that trajectory runs through them too) and went into transforms.json,
  bringing the ghost back. Now `register.near_dropped`: a registered photo whose `NEIGHBORS` (6) nearest model photos
  of its flight are half or more left out stays registered but out of transforms.json (decided at every export;
  transforms.json `modelforge.registered_near_dropped`, the Sparse tab status and the run's log say how many). Test:
  `test_photos_next_to_isolated_ones_stay_out`.

**In the pipelines (2026-10-26, the user asked):** Register more photos is the last step of Run SfM (all) and of the
SfM part of Run full pipeline (after export, before the SfM mesh and the splat), with the panel's selection (default
keyframes, dense). Keyframes without a current overlap analysis run the analysis first; a selection with no photo
beyond those in use skips the step (History note). The folder sits below the stage buttons (after export) and
opens by default, as does Images used.
