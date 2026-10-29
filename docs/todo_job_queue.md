# TODO: job queue with completion notices

Priority: **medium**. Status: **done**: first version 2026-10-23, memory check per entry and entries of other projects
without switching the view 2026-10-24 (see the end); open (optional): a command hook per notice.

## Why

Long jobs (refine of thousands of images, tiled splats, splat meshes) take hours, and today the GUI runs one job at a
time that has to be started by hand when the previous one ends: overnight work stops after the first stage the
user started (Run full pipeline covers one fixed sequence of one project).

## What

- **A queue** (left panel or under the run state): add entries "project + stage (+ the tab's settings as they are when
  added)", e.g. kladno_dira refine → export → Set up tiles → Train tiles, then berlin splat mesh fine. Reorder,
  remove, pause the queue; the running entry shows as today (state line, progress, Stop).
- **Settings are captured at queueing time** (a snapshot of the stage's arguments, as `splat/resume.json` stores a
  run's settings), so changing a tab later does not change queued work; an entry shows its settings on hover.
- **Policy on failure:** stop the queue (default) or skip to the next project's entries; a stopped entry (Stop) pauses
  the queue.
- **Across projects:** the queue runs entries of any project; the GUI shows the open project, and an entry of another
  project runs without switching the view (needs the job to not depend on the open project's `App` state: see
  `docs/todo_robustness.md` §1, jobs in their own process; until then, switch the project when its entry starts and
  refuse manual switching while the queue runs).
- **Persistent:** `ROOT/.modelforge_queue.json`, survives a GUI restart (an interrupted entry is marked, not rerun
  blindly; splat entries can resume through `splat.resume`).
- **Memory check per entry** before it starts (`docs/todo_robustness.md` §3; `pipeline.auto_ba_mode` today): wait or
  fail with a message instead of starting something that will not fit.

## Completion notices

- Browser notification (the Notification API from the page, through the viser bundle patch like `#mf-status`, which
  already drives the tab title / icon) when an entry or the whole queue ends, with the History line ("refine
  kladno_dira: 5681 cameras, 0.59 px, 3 h 10 min").
- Optional: a command hook per notice (e.g. `notify-send`, a webhook) set in Settings, run in the container.

## Depends on / order

Best after `docs/todo_robustness.md` §1 (jobs in their own process); a first version can queue within the current
single-job model (one entry at a time, the project switched when needed).

## Done: first version (2026-10-23)

`modelforge/gui/jobqueue.py`, right panel folder **Queue** (above History): Stage (Run full pipeline, Run SfM (all),
Refine, Resume refine, Export, SfM mesh (all), Set up tiles, Train splat, Resume splat, Splat mesh, Semantic features,
COLMAP dense (all)), **Add to queue** (the open project + the tabs' settings now), **Start queue** / **Pause queue**,
"On failure: go on" (skip the failed project's other entries; off: stop), Entry k: Up / Remove / Retry / Clear
finished.
- Settings: every input created in the right panel's tabs (Sparse, Sparse Mesh, Splat, Splat mesh, Dense; recorded
  while `_build_gui` builds them, `App._settings_names`, 44 on fullrun_test) plus the Search tab's "Include in Run full
  pipeline"; put back by `App._apply_settings` after the entry's project is opened (opening resets some of them).
  Not included: Images used (the subset panel keeps its choice per project).
- Runner: a thread starts the next pending entry when no job runs: opens its project if needed, applies its
  settings, `run_stage`; the end is read from the state line (✅ done, ⏹ / ⏸ stopped = the queue pauses, ❌ failed).
  `ROOT/.modelforge_queue.json`; after a GUI restart a running entry is "interrupted" and the queue waits.
- Notices: `App._notice(text)` (an entry or the queue ended) adds a History line and sets `#mf-status data-notice`;
  the page script (`docker/patch_viser_page.py`) shows a browser notification when it changes (permission asked
  after the first click; needs https or localhost).
- Tests: `tests/test_jobqueue.py` (file, restart, policies, editing; two projects queued with different Pairs settings
  run with their own).

## Done: memory check per entry (2026-10-24)

Before an entry starts, `jobqueue.memory_need` gives the RAM it needs available: at least `MIN_FREE_GB` (6), or 1.1 x
the peak its project's last such run recorded (report.json `refine.peak_rss_gb` for refine, Resume refine, Run SfM
(all), Run full pipeline; no other stage records one yet), or for a refine with BA mode **global** the estimate of
`pipeline.global_ba_need_gb` (BA mode auto picks chunked itself). With less available the entry waits ("waiting for
memory: 10.0 GB available, needs 16 GB (its last refine peaked at 15 GB)" on its line, one notice) and starts once the
memory is free. Why: what holds the memory is often outside the GUI (another job, a test run), and an OOM kill once
cost a 5 h refine. Test: `test_entry_waits_for_memory`.

## Done: entries of other projects without switching the view (2026-10-24)

The user asked. An entry of the open project runs in the GUI as before (live view, the top Stop). An entry of another
project runs in a process of its own: `modelforge/gui/headless.py` builds an `App` on that project without a browser
(viser on a free local port), puts the entry's settings into its tabs and calls `run_stage`, so every stage runs
exactly as from its button (no second implementation of the stages); it writes
`ROOT/.modelforge_queue.d/<id>.status.json` every second (state line, progress, at the end the result) and stops when
`<id>.cancel` appears.
- The queue (`QueuePanel._start_external` / `_follow_external`) shows its progress on the entry's line ("33 % · running
  refine · …"), the browser tab title says "queue: Refine (b)", **Stop entry** writes the cancel file (the entry ends
  "stopped" and the queue pauses, as with Stop).
- While it runs, the GUI refuses jobs in that project (the state line names the entry) and does not re-attach its job
  processes as its own; other projects can be opened and used. When it ends and its project is open, the view reloads
  its results.
- Started in a session of its own (`start_new_session`), so it outlives a GUI restart; the restarted queue keeps it
  "running" (not "interrupted") and follows it by pid. The GUI restart routine's `pkill` pattern does not match it.
- Still one entry at a time; the queue also waits while the GUI runs a job.
- Checked by hand (`build/queue_check/stop.py`): a refine of another project, stopped after 25 s → "stopped" within
  ~1 s, the queue paused. Tests: `test_queued_entries_run_with_their_settings` (the second project's entry runs in
  its own process, the GUI keeps its project), `test_external_entry_survives_a_gui_restart`.

**Stages and settings, 2026-10-26 (the user asked):** the Stage list follows the full pipeline's order and covers its steps:
- Run full pipeline;
- Run SfM (all);
- the overlap analysis; Images used (apply the selection);
- features, pairs, match, refine (the project's sparse method: videos get reconstruct), resume refine, export;
- register more photos;
- the SfM mesh, all or each step (build, patch holes, colour);
- set up tiles, train / resume splat;
- semantic features;
- Difix: suggest target views (job stage `difix-suggest`) and refine the splat;
- the splat mesh (build, patch holes);
- COLMAP dense.

An entry now also keeps the Sparse tab's Images used and Register more photos selections (`app.PANEL_SETTINGS`, by
dotted names: `subset.g_level`, `register.sel.g_mode`, ...). Before, they came from whatever the panels showed when
the entry ran. The full pipeline keeps the Images used choice across its extraction (the panel's refresh used to reset
it to the defaults).
