"""Job queue (right panel, folder "Queue"; docs/todo_job_queue.md): stages of any project with the settings the tabs
had when they were queued, run one after the other; a browser notification when an entry or the queue ends.

An entry = project + stage (as App.run_stage takes it: "all", "sfm", "refine", "splat", ...) + the values of the right
panel's inputs at queueing time (App._settings: Sparse, Sparse Mesh, Splat, Splat mesh, Dense tabs; changing a tab
later does not change queued work). When it starts, its project is opened if another one is open, its settings are
put back into the tabs and the stage runs as if its button was pressed; the GUI shows it as any job (state line,
progress, Stop). Stop pauses the queue; a failure stops it, or (On failure: go on) skips that project's other entries.
An entry starts only when enough RAM is available (memory_need: MIN_FREE_GB, its project's last recorded peak of
that stage, a forced-global BA's estimate); it waits meanwhile, with a note. Kept in ROOT/.modelforge_queue.json: after a GUI restart a running entry is marked interrupted (not rerun) and the
queue waits to be started again.
"""

import json
import os
import threading
import time

FILE = ".modelforge_queue.json"
STAGES = {  # label -> App.run_stage stage
    "Run full pipeline": "all",
    "Run SfM (all)": "sfm",
    "Refine": "refine",
    "Resume refine": "refine-resume",
    "Export": "export",
    "SfM mesh (all: mesh, holes, colour)": "sfmmesh-all",
    "Set up tiles": "splat-tiles",
    "Train splat": "splat",
    "Resume splat": "splat-resume",
    "Splat mesh": "splatmesh",
    "Semantic features": "semantic",
    "COLMAP dense (all)": "dense",
}
MIN_FREE_GB = 6.0  # an entry starts only with this much RAM available (an OOM kill once took a 5 h refine) ...
PEAK_OF = {"refine": "refine", "refine-resume": "refine", "sfm": "refine", "all": "refine"}  # ... or the peak its
#        project's last run of that kind recorded (report.json <key>.peak_rss_gb), or a forced-global BA's estimate
ICONS = {"pending": "⚪", "running": "⏳", "done": "✅", "failed": "❌", "stopped": "⏹", "interrupted": "⚠️",
         "skipped": "⏭"}


class JobQueue:
    """The queue's state and file (no GUI): entries and whether it runs."""

    def __init__(self, root):
        self.path = os.path.join(root, FILE)
        self.entries, self.running, self.go_on = [], False, False
        try:
            d = json.load(open(self.path))
            self.entries, self.go_on = d.get("entries", []), bool(d.get("go_on", False))
        except (OSError, ValueError):
            pass
        for e in self.entries:  # a GUI restart: what was running is not rerun blindly
            if e["status"] == "running":
                e["status"], e["note"] = "interrupted", "the GUI restarted while it ran"
        self._next_id = max([e["id"] for e in self.entries] + [0]) + 1

    def save(self):
        tmp = self.path + ".part"
        json.dump({"entries": self.entries, "go_on": self.go_on}, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)

    def add(self, project, stage, label, settings):
        e = {"id": self._next_id, "project": os.path.abspath(project), "stage": stage, "label": label,
             "settings": settings, "status": "pending", "added": time.time()}
        self._next_id += 1
        self.entries.append(e)
        self.save()
        return e

    def next_pending(self):
        return next((e for e in self.entries if e["status"] == "pending"), None)

    def find(self, k):
        """Entry by its position in the list (1-based, as shown)."""
        return self.entries[k - 1] if 1 <= k <= len(self.entries) else None

    def remove(self, k):
        e = self.find(k)
        if e is not None and e["status"] != "running":
            self.entries.remove(e)
            self.save()

    def up(self, k):
        if 2 <= k <= len(self.entries) and "running" not in (self.entries[k - 1]["status"],
                                                             self.entries[k - 2]["status"]):
            self.entries[k - 2], self.entries[k - 1] = self.entries[k - 1], self.entries[k - 2]
            self.save()

    def retry(self, k):
        e = self.find(k)
        if e is not None and e["status"] not in ("running", "pending"):
            e["status"] = "pending"
            e.pop("note", None)
            self.save()

    def clear_finished(self):
        self.entries = [e for e in self.entries if e["status"] in ("pending", "running")]
        self.save()

    def finish(self, e, status, note=""):
        """An entry ended: done / failed / stopped. Returns what the queue does next: "go" | "pause" | "stop"."""
        e.update(status=status, ended=time.time(), note=note)
        if status == "failed" and self.go_on:  # skip the rest of that project
            for o in self.entries:
                if o["status"] == "pending" and o["project"] == e["project"]:
                    o["status"], o["note"] = "skipped", f"entry {e['id']} of this project failed"
        self.save()
        if status == "stopped":
            return "pause"
        if status == "failed" and not self.go_on:
            return "stop"
        return "go"

    def markdown(self):
        if not self.entries:
            return "_empty: pick a stage and **Add to queue** (the open project, the tabs' settings as they are now)_"
        out = []
        for k, e in enumerate(self.entries, 1):
            took = ""
            if e.get("started") and e.get("ended"):
                took = f" · {_fmt(e['ended'] - e['started'])}"
            note = f" · _{e['note']}_" if e.get("note") else ""
            out.append(f"{k}. {ICONS.get(e['status'], '')} **{os.path.basename(e['project'])}** · {e['label']}"
                       f" · {e['status']}{took}{note}")
        return "  \n".join(out)


def memory_need(e):
    """(GB the entry needs available before it starts, why): MIN_FREE_GB, the peak its project's last such run
    recorded, or for a refine with BA mode global the estimate of pipeline.global_ba_need_gb."""
    need, why = MIN_FREE_GB, f"at least {MIN_FREE_GB:.0f} GB"
    try:
        rep = json.load(open(os.path.join(e["project"], "report.json")))
    except (OSError, ValueError):
        rep = {}
    key = PEAK_OF.get(e["stage"])
    peak = (rep.get(key) or {}).get("peak_rss_gb") if key else None
    if peak and peak * 1.1 > need:
        need, why = peak * 1.1, f"its last {key} peaked at {peak:.0f} GB"
    if key == "refine" and (e.get("settings") or {}).get("g_ba_mode") == "global":
        inl = (rep.get("match") or {}).get("inlier_matches")
        if inl:
            from modelforge import pipeline
            g = pipeline.global_ba_need_gb(inl)
            if g > need:
                need, why = g, f"a global BA of {inl / 1e6:.0f}M inlier matches needs ~{g:.0f} GB"
    return need, why


def _available_gb():
    from modelforge import pipeline
    return pipeline.ram_available_gb()


def _fmt(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


class QueuePanel:
    """The Queue folder of the right panel; a runner thread starts the entries when the GUI is idle."""

    def __init__(self, app, gui):
        import viser
        self.app = app
        self.q = JobQueue(app.root)
        self.current, self.current_run = None, None
        self.g_md = gui.add_markdown(self.q.markdown())
        self.g_stage = gui.add_dropdown("Stage", options=tuple(STAGES), initial_value="Run full pipeline",
                                        hint="What the entry runs, as the button of that name does.")
        b = gui.add_button("Add to queue", icon=viser.Icon.PLUS,
                           hint="Queue the stage for the open project with the tabs' settings as they are now "
                                "(Sparse, Sparse Mesh, Splat, Splat mesh, Dense); later changes to the tabs do not "
                                "change it.")
        b.on_click(lambda _: self.add())
        self.g_start = gui.add_button("Start queue", icon=viser.Icon.PLAYER_PLAY, color="green",
                                      hint="Run the pending entries one after the other (each opens its project). "
                                           "A browser notification tells when an entry ends (allow notifications "
                                           "when the browser asks, after your first click in the page).")
        self.g_start.on_click(lambda _: self.start())
        self.g_pause = gui.add_button("Pause queue", icon=viser.Icon.PLAYER_PAUSE, disabled=True,
                                      hint="Start no further entry; the running one finishes (Stop ends it now).")
        self.g_pause.on_click(lambda _: self.pause())
        self.g_go_on = gui.add_checkbox("On failure: go on", initial_value=self.q.go_on,
                                        hint="A failed entry skips the other entries of its project and the queue "
                                             "goes on with the other projects. Off: the queue stops.")
        self.g_go_on.on_update(lambda _: self._set_go_on())
        self.g_entry = gui.add_number("Entry", initial_value=1, min=1, step=1, hint="The entry the buttons below act on.")
        g = gui.add_button_group("Entry", ("Up", "Remove", "Retry", "Clear finished"))
        g.on_click(lambda e: self.edit(e.target.value))
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- editing -------------------------------------------------------------------------------------
    def refresh(self):
        self.g_md.content = self.q.markdown()
        self.g_start.disabled = self.q.running or self.q.next_pending() is None
        self.g_pause.disabled = not self.q.running

    def add(self):
        if not self.app.path:
            return
        label = self.g_stage.value
        self.q.add(self.app.path, STAGES[label], label, self.app._settings())
        self.refresh()

    def edit(self, what):
        k = int(self.g_entry.value)
        {"Up": self.q.up, "Remove": self.q.remove, "Retry": self.q.retry}.get(what, lambda _: self.q.clear_finished())(k)
        self.refresh()

    def _set_go_on(self):
        self.q.go_on = bool(self.g_go_on.value)
        self.q.save()

    def start(self):
        self.q.running = True
        self.refresh()

    def pause(self):
        self.q.running = False
        self.refresh()

    def stop_thread(self):
        self._stop.set()

    # --- running -------------------------------------------------------------------------------------
    def _loop(self):
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the queue never takes the GUI down
                import traceback
                traceback.print_exc()

    def tick(self):
        """One step of the runner: notice the end of the running entry, start the next one."""
        app = self.app
        if self.current is not None:
            if app.running and app._run_id == self.current_run:
                return
            if app.running:  # another job runs (the run id moved on): wait
                return
            self._ended(self.current)
            self.current = None
        if not self.q.running or app.running or app.ingest.running:
            return
        e = self.q.next_pending()
        if e is None:
            self.q.running = False
            self.refresh()
            app._notice("queue finished: " + ", ".join(
                f"{os.path.basename(x['project'])} {x['label']} {x['status']}" for x in self.q.entries
                if x.get("ended", 0) >= (getattr(self, "_started_at", None) or 0)))
            self._started_at = None
            return
        need, why = memory_need(e)
        avail = _available_gb()
        if avail is not None and avail < need:  # wait: something else holds the memory (another job, a test run)
            note = f"waiting for memory: {avail:.1f} GB available, needs {need:.0f} GB ({why})"
            if e.get("note") != note:
                if not (e.get("note") or "").startswith("waiting for memory"):
                    app._notice(f"{os.path.basename(e['project'])}: {e['label']} waits for memory ({why})")
                e["note"] = note
                self.q.save()
                self.refresh()
            return
        if (e.get("note") or "").startswith("waiting for memory"):
            e.pop("note")
        self._started_at = getattr(self, "_started_at", None) or time.time()
        try:
            if os.path.abspath(app.path or "") != e["project"]:
                if not os.path.isdir(e["project"]):
                    raise RuntimeError(f"{e['project']} is gone")
                app.open_project(e["project"])
            app._apply_settings(e["settings"])
        except Exception as err:  # noqa: BLE001
            e["started"] = time.time()
            nxt = self.q.finish(e, "failed", f"could not start: {err}")
            if nxt != "go":
                self.q.running = False
            self.refresh()
            return
        e["status"], e["started"] = "running", time.time()
        self.q.save()
        self.refresh()
        app.run_stage(e["stage"])
        self.current, self.current_run = e, app._run_id

    def _ended(self, e):
        content = self.app.g_state.content
        status = ("done" if content.startswith("✅") else "stopped" if content.startswith(("⏹", "⏸"))
                  else "failed")
        note = "" if status == "done" else content.split("\n")[0][:200]
        nxt = self.q.finish(e, status, note)
        if nxt in ("pause", "stop"):
            self.q.running = False
        self.refresh()
        self.app._notice(f"{os.path.basename(e['project'])}: {e['label']} {status}"
                         + (f" after {_fmt(e['ended'] - e['started'])}" if e.get("started") else ""))
