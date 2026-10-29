# TODO: features and matching for hard pairs (berlin_all)

The user asked (2026-10-28): berlin_all (Mavic 3, 21 clips, 7753 photos in use) seems to fail at matching photos taken
from different viewpoints and scales. Which detectors / descriptors / matchers would do better? Licences are not a
concern (the user: non-commercial ones such as MASt3R and SuperPoint are fine).

## The benchmark (build/matchbench/, gitignored)

- `select_pairs.py` (in /opt/venv): covisibility from the refined model without matching. Each photo's observed 3D
  points go into 2 m voxels; overlap = shared voxels / the smaller photo's voxels. That gives 220k pairs with
  overlap >= 0.3. The test set is 72 pairs:
  - 56 "hard" pairs: in pairs.txt, more than 60 s apart, **0 verified SIFT inliers** in database.db. 14 each of:
    - scale: ground resolution (GSD) ratio >= 1.8;
    - viewpoint: viewing directions >= 30 deg apart;
    - cross-flight;
    - same-flight revisit.
  - 8 pairs never proposed in pairs.txt;
  - 8 controls (SIFT >= 100 inliers).
- `bench.py` (in build/matchbench/venv: nerfstudio's torch through a .pth file, plus LightGlue, RoMa, DeDoDe,
  MASt3R, GIM, transformers):
  - Each method runs on the photos as they are ("plain"), and with the finer photo downscaled to the other's GSD when
    the two differ by more than 1.3x ("eq").
  - Scoring: Sampson distance to the model's epipolar geometry (undistorted, in pixels), and a MAGSAC essential matrix.
  - A pair is **linked** when at least 15 RANSAC inliers also fit the model within 4 px.
- `report.py` prints the table below.
- `upright_check.py` and `colmap_feat_check.py` compare COLMAP's own SIFT under different settings.

## Results (RTX 3090, photos 1920 x 1080)

Linked pairs; "all" counts the 56 hard + 8 never-proposed pairs. The better of plain / eq is shown, and the variant
in parentheses when eq is the better one. Every method links all 8 controls.

| method | scale | viewpoint | cross-flight | revisit | never proposed | all | s / pair |
|---|---|---|---|---|---|---|---|
| COLMAP SIFT (database) | 0 | 0 | 0 | 0 | - | 0 | |
| OpenCV SIFT 8192, ratio + mutual | 0 | 8 | 9 | 8 | 0 | 25 | 1.0 |
| SIFT + LightGlue | 2 | 9 | 11 | 10 | 0 | 32 | 0.9 |
| SuperPoint + LightGlue | 1 | 8 | 13 | 8 | 0 | 30 | 0.3 |
| DISK + LightGlue (eq) | 7 | 6 | 11 | 9 | 3 | 36 | 0.4 |
| ALIKED + LightGlue (eq) | 2 | 9 | 13 | 10 | 3 | 37 | 0.2 |
| GIM-LightGlue (SuperPoint, eq) | 2 | 11 | 13 | 13 | 2 | 41 | 0.13 |
| DeDoDe v2 L + G, dual softmax | 1 | 6 | 12 | 10 | 0 | 29 | 0.5 |
| Efficient LoFTR (full resolution) | 0 | 7 | 13 | 9 | 0 | 29 | 0.5 (16 GB GPU) |
| MASt3R (512 px, eq) | 6 | 11 | 14 | 13 | 3 | 47 | 0.35 |
| **RoMa outdoor** | **11** | **11** | 13 | **14** | **4** | **53** | 0.55 |

## Findings

1. **RoMa is the strongest by far**, and the only method that handles scale changes (11 of 14). MASt3R comes second.
   Of the sparse methods, GIM-LightGlue is best, and ALIKED / DISK need the GSD equalization to reach it.
   Equalization helps the sparse methods on scale pairs; RoMa does not need it.
2. **COLMAP's SIFT is weaker than SIFT itself.** With the same 8192 features, OpenCV SIFT links 25 pairs where COLMAP
   links 0. COLMAP's own keypoints matched the same simple way give 0-18 correct matches.
   - Upright on or off makes no difference.
   - The cap does: COLMAP keeps the coarsest scales when it truncates. With 32768 features it links 14 of the 42
     viewpoint / flight / revisit pairs (8192: 2), and the controls get twice the inliers.
3. **Pair selection misses most revisits.** Of the 153k overlapping pairs more than 60 s apart, only 6 % were in
   pairs.txt. When proposed, 63 % of them were linked (SIFT).

## Built (2026-10-28; the user chose: a learned sparse method for all pairs, RoMa for the unlinked ones between passes)

- `scripts/install_matchers`: build/matchers/venv (nerfstudio's Python + torch through a .pth file; LightGlue, GIM,
  RoMa, kornia, omegaconf; weights in build/torch_hub). The image is unchanged.
- `modelforge/export/matchers.py` (the worker, in that venv): extract / match (LightGlue straight into database.db's
  matches table) / roma (blocks of 200 pairs, resumable) / query (photos outside the model).
- `modelforge/sfm/learned.py`: Sparse tab → Features → Extractor **aliked-lightglue** / **gim-lightglue** (CLI
  `--features`): descriptors in `features_learned/<method>/<photo>.npz`, keypoints in features.db (images / cameras /
  rigs / frames through `pycolmap.import_images`); `current.json` ties them to that features.db (a SIFT run forgets
  them). Matching: LightGlue + `pycolmap.verify_matches`. **Dense matcher (hard pairs): roma** (CLI `--dense roma`,
  `--dense-per-photo`, default 3): unlinked pairs > 60 s apart or of two flights, best covisibility first, each photo
  in at most N; RoMa's matches (2000 per pair, epipolar error 0.06 px median on fullrun_test) snap onto a keypoint
  within 1.5 px or become new ones (0.5 px grid) appended to the photo's keypoints in database.db, the pairs verified
  again. Works with SIFT features too.
- `pairs.covisible` (Sparse tab → Matching → "Pairs between passes (per photo)", default 10, CLI `--covis-pairs`): on
  a re-run, pairs > 60 s apart / of two flights that see the same surface in the earlier model (2 m voxels of each
  photo's observed points). berlin_all: 41,106 pairs, 36,360 of them new, 44 s.
- Register more photos and the held-out evaluation (`evaluate.locate`) use the project's learned features (the
  worker's query op): photos of the model located within 0.3 mm of their poses.
- **One extraction per photo** (the user asked): `features_learned/<method>/<photo>.npz` is a cache that holds the
  photo's size + mtime (`stamp`; the photos in use are hard links to the pool's, so they share it). SfM takes what
  is there and extracts the rest; Register more photos takes its photos from it and adds the ones it extracts. A new
  choice of Images used keeps the cache (`apply_subsample`), a new extraction archives it. features stats `cached`.
  The overlap analysis stays SIFT (its keyframe levels are calibrated on SIFT inliers).
- **Precision** (fullrun_test, 80 photos, global refine, error median / p90, points): SIFT 0.475 / 1.229 px, 47.5k;
  **ALIKED 0.585 / 1.534 px, 49.9k**; GIM 0.784 / 1.574 px, 21.4k (SuperPoint keypoints sit on the pixel grid). So
  ALIKED is the one to use for splats; GIM links a few more hard pairs (41 vs 37 of 64).
- Tests: `tests/test_learned.py` (snapping, the dense pair choice, covisible pairs, ALIKED + RoMa through refine and
  locate on fullrun_test).
- **Default auto** (the user chose it): `pipeline.auto_matching` picks ALIKED + RoMa when at least max(20, 1 % of
  the photos) look-at pairs from the priors are > 60 s apart on the project timeline (several flights or passes over
  the same ground; DJI's split files of one flight follow each other and do not count), SIFT otherwise or without
  the matchers. The features stage decides and records it (report `features.auto`); Dense matcher auto follows it.
  fullrun_test / kladno_dopravnik → SIFT; karlak 4,713, berlin 8,108, berlin_all 35,601, kladno_dira 12,222,
  dira2k 13,253 such pairs → ALIKED + RoMa.
- **First berlin_all run (2026-10-28)**: ALIKED on 7753 photos, LightGlue on 155,774 pairs + verification (30 min),
  RoMa on 6,912 of 44,360 unlinked candidates (67 min); then the job process was OOM-killed while merging RoMa's
  matches: each pair's slice of a block was a view of a freshly decompressed copy of the whole block (NpzFile reads
  the array on every access), ~44 GB. Fixed (arrays read once per block, slices copied: 0.25 GB, 0.8 s), and RoMa's
  finished pairs are now reused per pair by a later run (`_dense_done`; blocks per run under
  `features_learned/roma/run_*`, `meta.json` the match count), so a repeated Match skips the 67 min.
- **RoMa's matches need a check of their own** (2026-10-29): the second run linked all 6,917 RoMa pairs, and
  their inliers disagreed with the earlier model (median 32 px). RoMa on 60 pairs of photos > 400 m apart (no overlap
  possible; `build/roma_fp/`): **all 60 verified**, median 527 inliers (F, 4 px): over flat ground its warp always
  fits some fundamental matrix; its certainty does not separate them (half of those matches > 0.9). The test that
  does: lift both ends of each match into the earlier model (the photo's pose, the median depth of its 8 nearest
  observed points) and compare: the 60 pairs have **no** match within 10 m, berlin_all's RoMa pairs median 1,025 (81 %
  of the pairs >= 100). Now `learned.Ground` keeps a match within `GROUND_M` (10 m) and a pair with >= `GROUND_MIN`
  (100) of them; without an earlier model the dense pass is skipped (report `dense.skipped`). berlin_all, repaired in
  place (`build/repair_dense.py`: keypoints trimmed back to features.db's, LightGlue + verification of those pairs
  again, the dense pass with the check; backup `features_learned/database_before_ground_check.db`): 6,731 pairs
  checkable, 1,394 dropped, 5,337 linked, 6.7M matches. Pairs with < 10 % of their inliers within 4 px of the old
  model: covisible 37 → 26 %, look-at only 55 → 33 % (ALIKED's pairs between passes: 16 %); the rest agree on the
  ground within 10 m: mostly revisits the old model left apart, which is what they are for.

## Options considered

- Cheap: a higher SIFT cap (32k: features.db ~4x, slower matching), and more cross-pass pairs from the terrain
  look-ats or the model's covisibility after a first refine.
- Learned sparse features for all pairs: GIM-LightGlue or ALIKED + LightGlue (0.13-0.2 s per pair, keypoints fit
  COLMAP's database as-is). Through COLMAP's ONNX build (an image rebuild) or the Python packages (no rebuild).
- RoMa for the pairs between passes and flights only: berlin_all has 153k such overlapping pairs, ~23 h at 0.55 s, so
  a per-photo budget (e.g. the 5 best per photo: ~6 h) or RoMa only where the sparse method failed. Dense matches
  need quantizing to shared keypoints per photo (as hloc does) before they go into COLMAP.
