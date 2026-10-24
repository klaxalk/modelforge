# TODO: robustness of long jobs

Priority: **medium**. Status: §2 (resume refine) and §1 (job processes) **done 2026-10-23**, re-attaching after a
GUI restart **done 2026-10-24**; §3 deferred. Why: on 2026-10-23 the kernel OOM-killed the GUI
twice (07:26 next to a test run of mine, 09:53 alone at 30 GB in round 1 of kladno_dira's global refine, 5681
images). Each time the GUI and the job inside it died together and the refine was lost (round 0 alone took 52 min),
although `sparse/round_0` was on disk.

## 1. Heavy jobs in their own process

- Today refine, triangulation, export, SfM mesh and the sparse stages run **in the GUI process** (`App.run_stage`
  threads); splat, dense, splat mesh, semantic features and mesh colouring are already subprocesses
  (`jobs.Token.run`).
- Proposal: a job runner process per job (`python -m modelforge.pipeline ...` or a small worker), the GUI follows it
  through what the in-process events carry today (progress events, log lines, the live BA poses / snapshots for
  the 3D view), e.g. over a pipe or a JSON-lines file + the snapshots it already writes.
- Gains:
  - an OOM kills only the job, with a clear "out of memory" in the History, not the GUI;
  - the job can run under a memory limit (`resource.setrlimit` / cgroup `memory.max` if available) and fail
    cleanly before the kernel picks a victim;
  - the job survives a GUI restart; the GUI re-attaches on start (a `PROJECT/job.json` with pid, stage, log path);
  - Stop works the same way for every stage (terminate the process), also inside pycolmap calls that cannot be
    interrupted today (features, matching, triangulation, fusion).
- Careful: the live refine view (frustums moving per BA iteration) needs the poses streamed (cheap: 7 floats per
  camera at 4 Hz); Skip / Skip this BA round become messages to the process.

## 2. Resume refine

- Refine always restarts from round 0. Save the prior model (`robust.PriorModel`: sigmas, boresight, δt, rejected
  priors, source offsets, drift) next to each round snapshot (`sparse/round_N/prior_model.json`) and offer
  **Resume refine from round N** (Sparse tab, CLI `--resume-refine`), checked against the same database.db / pairs
  / images (sizes + mtimes, as `splat.resume_info` does).
- Chunked: save per finished chunk (its pose-graph edges, rejections, init poses: small) and resume at the next
  chunk; after the chunks, resume at the pose graph / revisit / drift / global / polish step from `posegraph_tri`.

## 3. Memory estimate before heavy stages (done 2026-10-24, see below)

- Like `pipeline.auto_ba_mode` (0.7 GB per million inlier matches for a global refine): an estimate per stage from
  the project's sizes, compared with MemAvailable, shown before the start and refused / warned when it does not fit
  next to what runs already (the GUI process' own RSS, a job in progress).
- Stages and what drives them: triangulation of all images (inlier matches, keypoints), export (points,
  observations), splat mesh (voxel blocks: `splat_tsdf.count_blocks` estimate, faces for the CPU stage), splat
  training with `cpu` image cache (2 × the decoded photos), semantic features, SfM mesh / colouring (vertices).
- Measured values to start from are in CLAUDE.md (refine peaks, splat mesh 9–13 GB on berlin fine, cpu cache 2×).

Order: 2 first (small, saves hours at once), then 1 (the real fix), then 3.

## Done: memory estimates before heavy stages (2026-10-24)

`modelforge/memory.py`. `estimate(project, stage, settings)` -> (GB, why), per stage key (refine split into
`refine:global` / `refine:chunked`, auto resolved as `pipeline.auto_ba_mode`; a group such as `all` / `sfm` / `dense`
takes its largest stage):
1. this project's last measured peak of that stage (report.json `memory.<key>`: peak + its driver) x 1.1, its part
   above the model's base scaled up when the driver grew (an older refine's `refine.peak_rss_gb` counts as a floor);
2. else the largest GB per driver unit any project measured on this machine (`ROOT/.modelforge_memory.json`, last 20
   per stage);
3. else the model `MODELS`: base + GB per driver unit. Drivers: inlier matches (refine, export), SfM points (SfM
   mesh), mesh faces / vertices (patching, colouring), decoded photos of a cpu image cache (splat, as
   `splat.auto_cache` picks it), splat-mesh faces (the last splat mesh's, x (voxel px ratio)^2 for another
   quality), Gaussians (semantic), depth + normal maps (fusion).

**Measured** (`build/memcal.py`: each stage in a child process on a copy of karlak, `memory.job_gb` every 0.5 s):
export 1.93 GB at 6.9M inlier matches (212 s), SfM mesh 1.50 GB at 0.65M points, patching 1.03 GB at 1.4M faces,
colouring 0.97 GB at 0.74M vertices, splat mesh coarse 2.94 GB at 1.4M faces, semantic 4.55 GB at 2.3M Gaussians.
The models' bases follow these; their slopes are kept above what one project can show (the records take over).

**GUI:** `App.run_stage` estimates before a job; more than MemAvailable -> every browser gets "Not enough memory?"
(Start anyway / Cancel; no browser: it starts, with a log line); the state line shows "memory: needs ~X GB (why), Y GB
available" while it runs. `memory.Sampler`: PSS of every process below the GUI (job processes, ns-train and its data
workers, COLMAP; not queue entries in their own process, which measure themselves: gui/headless.py is an App too)
plus the GUI's own growth since the stage began, once a second; each finished stage is recorded
(`App._mem_record`). **Queue:** `memory_need` = max(MIN_FREE_GB, the estimate); its entries skip the dialog
(`run_stage(check_memory=False)`: the queue waited already). Tests: `tests/test_memory.py`, `tests/test_jobqueue.py`.

Not covered: GPU memory (the splat plan and the splat mesh's block budget handle it), re-attached jobs after a GUI
restart are not sampled, and stages run from the CLI record nothing.

## Done: resume refine (2026-10-23)

`modelforge/sfm/checkpoint.py`: after every finished step (a global round; a calibration round, a chunk, the pose
graph, each revisit / drift / global pass and the polish of a chunked refine) `sparse/refine_resume.pkl` holds the
poses, intrinsics, prior model, stats and the chunked state (chunk edges, start poses, rejections); no points (each
step triangulates afresh). `stage_refine(resume=True)` (Sparse tab **Resume refine**, enabled while a checkpoint
exists; CLI `--resume-refine`) skips the finished steps; report `refine.resumed_after`. Key: database.db by content
(images, keypoint and verified-match totals: COLMAP touches its mtime on every open, even for reading), poses.json
size + mtime, the refine settings (rounds, BA mode, chunk settings, iteration cap, initial prior model). A finished
refine deletes it. Tests: `tests/test_refine_resume.py` (global stopped after round 1 → resumes at round 2; chunked
stopped after chunk 2 → calibration and chunks 1–2 skipped; other settings start afresh).

## Done: heavy stages in a job process (2026-10-23)

`modelforge/worker.py`: `worker.run(target, kwargs, callbacks, token, events, on_log)` starts the stage function in
a spawned process. Callbacks are proxied through a queue (a pycolmap.Reconstruction in them becomes `{"poses": {name:
params}}`, live BA events at most every 0.25 s), log lines come back through on_log, Stop cancels the child's token
(a BA aborts within an iteration) and terminates it after `STOP_GRACE` (60 s) if it is stuck in pycolmap, Skip / Skip
this BA round are forwarded. A child that dies by a signal raises `worker.Killed` ("the job process was killed
(SIGKILL): out of memory? The GUI kept running"). The GUI (`App._in_job`) runs features, pairs, match, refine,
reconstruct, export, the overlap analysis, SfM mesh build / patch holes and COLMAP dense that way and re-reads
report.json after each stage; `MODELFORGE_INPROCESS=1` runs them in the GUI process as before. Splat, splat mesh,
semantic features and mesh colouring were subprocesses already. Not done: a job that survives a GUI restart and is
re-attached (`PROJECT/job.json`), a memory limit per job. Tests: `tests/test_worker.py` (callbacks + log lines, Stop,
Skip, a stuck job terminated, SIGKILL reported, errors); the GUI tests run the stages through it.

## §3 deferred (2026-10-23)

With heavy stages in job processes an out-of-memory kill no longer takes the GUI, a killed refine resumes from its
last step, and BA mode auto already chooses chunked when a global refine would not fit. A per-stage estimate would
mostly add warnings of unproven accuracy; revisit when a stage other than refine runs out of memory again (measure
its peak first, as `auto_ba_mode` did).

## Done: re-attaching after a GUI restart (2026-10-24)

A job process already outlived its GUI: killed by a signal (a restart's SIGTERM), Python skips the exit handlers that
would terminate daemonic children, so the child finishes as an orphan (docker-init reaps it); only its pipes to the
GUI break, which the child now tolerates. What was missing was the GUI knowing about it:
- the child writes `PROJECT/job.json` (pid, target, label, start; at the end the result: done / cancelled / error +
  message), `PROJECT/job.log` (its log lines) and watches `PROJECT/job.cancel` (`worker.detached`, `alive` (not a
  zombie), `FileToken`, `clear`). The GUI that saw a job to its end removes job.json.
- `App._reattach` (on start and on opening a project): a running job shows as running "<stage> (re-attached)" with its
  log lines (no live progress or 3D updates), Stop through job.cancel, its end in the state line and History, then the
  snapshots / layers are reloaded; a job that finished (or died) while no GUI watched gets a History line.
- Only the stages run through `worker.run` (features … export, SfM mesh, COLMAP dense); splat training, splat mesh and
  colouring are subprocesses of the GUI's own thread (the trainer would go on, its tile rounds and export would not:
  Resume training covers that). The rest of a sequence (Run full pipeline) does not continue.
- Tests: `test_worker.py` (`test_job_outlives_the_gui`: a stand-in GUI killed with SIGTERM, the job finishes, job.json
  / job.log tell; Stop by file; a finished job leaves no job file; `test_gui_reattaches_after_a_restart`: a real App
  follows the job to its end and stops another).
