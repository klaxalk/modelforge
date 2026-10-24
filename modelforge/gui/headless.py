"""A queue entry of a project other than the one the GUI shows (gui/jobqueue.py), run in a process of its own: an App on
that project without a browser (its viser server on a free local port), the entry's settings put into its tabs, its
stage run as the button would. So every stage runs exactly as in the GUI, and the GUI keeps its project and view.

`python -m modelforge.gui.headless ENTRY_JSON` (written by the queue). Every second it writes ENTRY.status.json (pid,
the state line, progress, and at the end the result: done / stopped / failed + the line); ENTRY.cancel asks it to stop
(the GUI's Stop entry). Started in a session of its own, so it outlives a GUI restart; the restarted queue follows it.
"""
import json
import os
import socket
import sys
import time


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write(path, d):
    with open(path + ".part", "w") as f:
        json.dump(d, f)
    os.replace(path + ".part", path)


def result_of(state_line):
    """done / stopped / failed from a final state line (as the queue reads the GUI's)."""
    return ("done" if state_line.startswith("✅") else "stopped" if state_line.startswith(("⏹", "⏸"))
            else "failed")


def main(entry_path):
    base = entry_path[:-len(".json")]
    status_path, cancel_path = base + ".status.json", base + ".cancel"
    e = json.load(open(entry_path))
    st = {"pid": os.getpid(), "entry": e["id"], "project": e["project"], "label": e["label"], "started": time.time(),
          "state": "starting", "progress": None}
    _write(status_path, st)
    app = None
    try:
        from modelforge.gui.app import App
        app = App(e["project"], port=_free_port(), root=os.path.dirname(os.path.dirname(base)))
        app.queue.stop_thread()  # this process runs one entry, never the queue ...
        app.queue.current = None  # ... and is not refused as busy with itself
        app._apply_settings(e.get("settings") or {})
        app.run_stage(e["stage"], check_memory=False)  # the queue waited for its memory
        stop_sent = False
        while app.running:
            if not stop_sent and os.path.exists(cancel_path):
                app.stop()
                stop_sent = True
            st.update(state=app.g_state.content, progress=app.g_progress.value if app.g_progress.visible else None)
            _write(status_path, st)
            time.sleep(1.0)
        line = app.g_state.content
        st.update(state=line, progress=None, result=result_of(line), ended=time.time())
    except BaseException as err:  # noqa: BLE001 - the queue must learn the end
        st.update(state=f"❌ **could not run**: `{type(err).__name__}: {err}`", progress=None, result="failed",
                  ended=time.time())
    _write(status_path, st)
    if app is not None:
        try:
            app.lod.stop()
            app.server.stop()
        except Exception:  # noqa: BLE001
            pass
    os._exit(0)  # viser / worker threads: do not wait for them


if __name__ == "__main__":
    main(sys.argv[1])
