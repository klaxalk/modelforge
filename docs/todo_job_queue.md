# TODO: job queue with completion notices

Priority: **medium**. Status: **first version done 2026-10-23** (see the end); open: entries of other projects
without switching the view, a memory check per entry, a command hook per notice.

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
