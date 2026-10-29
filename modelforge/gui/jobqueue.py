"""Job queue (right panel, folder "Queue"; docs/todo_job_queue.md): stages of any project with the settings the tabs
had when they were queued, run one after the other; a browser notification when an entry or the queue ends.

An entry = project + stage (as App.run_stage takes it: "all", "sfm", "refine", "splat", ...) + the values of the right
panel's inputs at queueing time (App._settings: Sparse, Sparse Mesh, Splat, Splat mesh, Dense tabs; changing a tab
later does not change queued work). An entry of the open project runs in the GUI: its settings are put back into the
tabs and the stage runs as if its button was pressed, shown as any job (state line, progress, Stop). An entry of
another project runs in a process of its own (gui/headless.py: an App on that project without a browser, the same
stages), so the GUI keeps its project and view; its line shows the progress, "Stop entry" stops it, the open project's
own jobs are refused while it runs there, and it outlives a GUI restart (the restarted queue follows it). Stop pauses the queue; a failure stops it, or (On failure: go on) skips that project's other entries.
An entry starts only when enough RAM is available (memory_need: MIN_FREE_GB or modelforge.memory's estimate for its
stage); it waits meanwhile, with a note. Kept in ROOT/.modelforge_queue.json
(+ ROOT/.modelforge_queue.d/: the external entries' files): after a GUI restart a running entry of the GUI is marked
interrupted (not rerun) and the queue waits to be started again.
"""

import json
import os
import subprocess
import sys
import threading
import time

FILE = ".modelforge_queue.json"
DIR = ".modelforge_queue.d"  # an external entry's <id>.json / .status.json / .cancel / .log
STAGES = {  # label -> App.run_stage stage, in the order the full pipeline runs them (2026-10-26, the user asked)
    "Run full pipeline": "all",
    "Run SfM (all: features … export, register more photos)": "sfm",
    "Overlap analysis of the extracted images": "analyze",
    "Images used: apply the selection": "images",
    "Features": "features",
    "Pairs": "pairs",
    "Match": "match",
    "Refine (videos: reconstruct)": "refine",
    "Resume refine": "refine-resume",
    "Export": "export",
    "Register more photos": "register",
    "SfM mesh (all: mesh, patch holes, colour)": "sfmmesh-all",
    "SfM mesh: build": "sfmmesh",
    "SfM mesh: patch holes": "sfmmesh-fill",
    "SfM mesh: colour from the photos": "sfmmesh-colour",
    "Set up tiles": "splat-tiles",
    "Train splat": "splat",
    "Resume splat": "splat-resume",
    "Semantic features": "semantic",
    "Difix: suggest target views": "difix-suggest",
    "Difix: refine the splat": "difix",
    "Splat mesh: build": "splatmesh",
    "Splat mesh: patch holes": "splatmesh-fill",
    "COLMAP dense (all)": "dense",
}
MIN_FREE_GB = 6.0  # an entry starts only with this much RAM available (an OOM kill once took a 5 h refine), or what
#                   modelforge.memory estimates for it
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
            if e["status"] == "running" and external_alive(e):
                continue  # its own process goes on: the queue follows it
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


def external_alive(e):
    from modelforge import worker
    x = e.get("external")
    return bool(x) and worker.alive(x["pid"])


def external_status(e):
    try:
        return json.load(open(e["external"]["base"] + ".status.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def memory_need(e):
    """(GB the entry needs available before it starts, why): MIN_FREE_GB, or modelforge.memory's estimate for its
    stage with its settings (its project's last measured peak, the machine's records, else a model)."""
    need, why = MIN_FREE_GB, f"at least {MIN_FREE_GB:.0f} GB"
    try:
        from modelforge import memory
        g, gwhy = memory.estimate(e["project"], e["stage"], memory.settings_from(e.get("settings") or {}))
    except Exception:  # noqa: BLE001 - no estimate: the floor
        return need, why
    return (g, gwhy) if g > need else (need, why)


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
        self._proc = None  # the external entry's Popen (None after a GUI restart: followed by its pid)
        self.current = next((e for e in self.q.entries if e["status"] == "running" and external_alive(e)), None)
        if self.current is not None:
            self.q.running = True  # it was running when the GUI went: go on after it
        self.g_md = gui.add_markdown(self.q.markdown())
        self.g_stage = gui.add_dropdown("Stage", options=tuple(STAGES), initial_value="Run full pipeline",
                                        hint="What the entry runs, as the button of that name does.")
        b = gui.add_button("Add to queue", icon=viser.Icon.PLUS,
                           hint="Queue the stage for the open project with the tabs' settings as they are now "
                                "(Sparse, Sparse Mesh, Splat, Splat mesh, Dense); later changes to the tabs do not "
                                "change it.")
        b.on_click(lambda _: self.add())
        self.g_start = gui.add_button("Start queue", icon=viser.Icon.PLAYER_PLAY, color="green",
                                      hint="Run the pending entries one after the other: those of the open project here, "
                                           "the others in a process of their own (the view stays). "
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
        self.g_stop_entry = gui.add_button("Stop entry", icon=viser.Icon.PLAYER_STOP, color="red",
                                           disabled=self.current is None,
                                           hint="Stop the entry running in its own process (an entry of another "
                                                "project); the queue pauses. An entry of the open project stops with "
                                                "the Stop button at the top.")
        self.g_stop_entry.on_click(lambda _: self.stop_external())
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- editing -------------------------------------------------------------------------------------
    def refresh(self):
        self.g_md.content = self.q.markdown()
        self.g_start.disabled = self.q.running or self.q.next_pending() is None
        self.g_pause.disabled = not self.q.running
        if hasattr(self, "g_stop_entry"):
            self.g_stop_entry.disabled = not (self.current is not None and self.current.get("external"))

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

    def busy(self, project):
        """The external entry running in this project, or None."""
        e = self.current
        if e is not None and e.get("external") and os.path.abspath(project or "") == e["project"]:
            return e
        return None

    def stop_external(self):
        e = self.current
        if e is not None and e.get("external"):
            open(e["external"]["base"] + ".cancel", "w").close()
            self.app._on_log(f"--- stop requested for queue entry {e['id']} ({os.path.basename(e['project'])})")

    def _start_external(self, e):
        """Run the entry in gui/headless.py, a process of its own (its own session: it outlives the GUI)."""
        d = os.path.join(self.app.root, DIR)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, str(e["id"]))
        for ext in (".status.json", ".cancel"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        json.dump(e, open(base + ".json", "w"))
        log = open(base + ".log", "w")
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (repo, os.environ.get("PYTHONPATH")) if p))
        self._proc = subprocess.Popen([sys.executable, "-u", "-m", "modelforge.gui.headless", base + ".json"],
                                      cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        e["external"] = {"pid": self._proc.pid, "base": base}

    def _follow_external(self, e):
        """The external entry: its progress on its line while it runs; True once it ended (recorded)."""
        alive = (self._proc.poll() is None) if self._proc is not None else external_alive(e)
        st = external_status(e)
        if alive:
            line = (st.get("state") or "starting").split("\n")[0].replace("*", "").strip()
            pct = st.get("progress")
            note = (f"{pct:.0f} % · " if pct else "") + line[:160]
            if e.get("note") != note:
                e["note"] = note
                self.refresh()
            return False
        self._proc = None
        status = st.get("result") or "failed"
        note = "" if status == "done" else (st.get("state") or "its process ended without a result (killed: out of "
                                                                "memory?)").split("\n")[0][:200]
        self._record_end(e, status, note)
        app = self.app
        if os.path.abspath(app.path or "") == e["project"] and not app.running:  # its results: into the view
            app._after_reattached({"refine": "refine", "refine-resume": "refine", "sfm": "export",
                                   "all": "export", "export": "export"}.get(e["stage"], e["stage"]))
            app.g_state.content = st.get("state") or app.g_state.content
        return True

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
        if self.current is not None and self.current.get("external"):
            if not self._follow_external(self.current):
                return
            self.current = None
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
        here = os.path.abspath(app.path or "") == e["project"]
        try:
            if not os.path.isdir(e["project"]):
                raise RuntimeError(f"{e['project']} is gone")
            if here:
                app._apply_settings(e["settings"])
            else:  # another project: its own process, the GUI keeps its view
                e["status"], e["started"], e["note"] = "running", time.time(), "starting in its own process"
                self._start_external(e)
        except Exception as err:  # noqa: BLE001
            e["started"] = time.time()
            nxt = self.q.finish(e, "failed", f"could not start: {err}")
            if nxt != "go":
                self.q.running = False
            self.refresh()
            return
        if not here:
            self.current = e
            self.q.save()
            self.refresh()
            return
        e["status"], e["started"] = "running", time.time()
        self.q.save()
        self.refresh()
        app.run_stage(e["stage"], check_memory=False)  # waited for its memory already
        self.current, self.current_run = e, app._run_id

    def _ended(self, e):
        content = self.app.g_state.content
        status = ("done" if content.startswith("✅") else "stopped" if content.startswith(("⏹", "⏸"))
                  else "failed")
        note = "" if status == "done" else content.split("\n")[0][:200]
        self._record_end(e, status, note)

    def _record_end(self, e, status, note):
        self.current = None
        nxt = self.q.finish(e, status, note)
        if nxt in ("pause", "stop"):
            self.q.running = False
        self.refresh()
        self.app._notice(f"{os.path.basename(e['project'])}: {e['label']} {status}"
                         + (f" after {_fmt(e['ended'] - e['started'])}" if e.get("started") else ""))
