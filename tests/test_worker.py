"""modelforge/worker.py: stages in a job process of their own."""
import os
import threading
import time

import pytest

from modelforge import jobs, worker


def test_callbacks_and_log_lines():
    seen, lines = [], []
    worker.run("tests.worker_targets:talk", {"project": "p", "token": worker.TOKEN},
               callbacks={"progress": seen.append}, on_log=lines.append)
    assert [e["k"] for e in seen] == [0, 1, 2] and seen[0]["project"] == "p"
    assert any("hello from the job" in ln for ln in lines)


def test_stop_reaches_the_job():
    tok = jobs.Token()
    threading.Timer(1.0, tok.cancel).start()
    t0 = time.time()
    with pytest.raises(jobs.Cancelled):
        worker.run("tests.worker_targets:wait_for_cancel", {"project": "p", "token": worker.TOKEN}, token=tok)
    assert time.time() - t0 < 10


def test_skip_event_reaches_the_job():
    ev = threading.Event()
    threading.Timer(1.0, ev.set).start()
    worker.run("tests.worker_targets:wait_for_cancel", {"project": "p", "token": worker.TOKEN},
               events={"skip": ev})  # returns normally once the job saw Skip


def test_a_stuck_job_is_terminated(monkeypatch):
    monkeypatch.setattr(worker, "STOP_GRACE", 2.0)
    tok = jobs.Token()
    threading.Timer(1.0, tok.cancel).start()
    t0 = time.time()
    with pytest.raises(jobs.Cancelled):
        worker.run("tests.worker_targets:stuck", {"project": "p", "token": worker.TOKEN}, token=tok)
    assert time.time() - t0 < 20


def test_killed_job_is_reported():
    with pytest.raises(worker.Killed, match="SIGKILL"):
        worker.run("tests.worker_targets:die", {"project": "p"})


def test_errors_come_back():
    with pytest.raises(RuntimeError, match="ValueError: no images"):
        worker.run("tests.worker_targets:fail", {"project": "p"})


def _gui(project, seconds):
    import subprocess
    import sys
    return subprocess.Popen([sys.executable, "-c", "import sys; from tests.worker_targets import gui_that_dies; "
                             f"gui_that_dies({str(project)!r}, {seconds})"], cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))))


def _wait(cond, timeout=30.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.1)
    return cond()


def test_job_outlives_the_gui(tmp_path):
    """The GUI killed (SIGTERM, as a restart) while its job runs: the job finishes on its own, job.json / job.log tell
    a new GUI about it, and its result."""
    gui = _gui(tmp_path, 4.0)
    assert _wait(lambda: (worker.detached(str(tmp_path)) or {}).get("state") == "running")
    gui.terminate()
    gui.wait(10)
    d = worker.detached(str(tmp_path))
    assert d["state"] == "running" and d["label"] == "long job" and worker.alive(d["pid"])
    assert _wait(lambda: worker.detached(str(tmp_path))["state"] == "finished")
    d = worker.detached(str(tmp_path))
    assert d["result"] == "done" and os.path.exists(tmp_path / "long_job.done")
    assert "long job done" in open(tmp_path / worker.LOG).read()
    worker.clear(str(tmp_path))
    assert worker.detached(str(tmp_path)) is None


def test_a_detached_job_stops_by_file(tmp_path):
    gui = _gui(tmp_path, 60.0)
    assert _wait(lambda: (worker.detached(str(tmp_path)) or {}).get("state") == "running")
    gui.terminate()
    gui.wait(10)
    worker.FileToken(str(tmp_path)).cancel()
    assert _wait(lambda: worker.detached(str(tmp_path))["state"] == "finished")
    assert worker.detached(str(tmp_path))["result"] == "cancelled" and not os.path.exists(tmp_path / "long_job.done")


def test_a_finished_job_leaves_no_job_file(tmp_path):
    """Seen to its end by the GUI that started it: nothing to re-attach to."""
    worker.run("tests.worker_targets:long_job", {"project": str(tmp_path), "token": worker.TOKEN, "seconds": 0.2})
    assert worker.detached(str(tmp_path)) is None and os.path.exists(tmp_path / "long_job.done")


REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "poses.json")), reason="needs a project")
def test_gui_reattaches_after_a_restart(tmp_path):
    """A new GUI finds the job its predecessor started: running (log lines, Stop by file), then its end."""
    import shutil
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path / "p")
    os.makedirs(p)
    for f in ("poses.json", "images"):
        os.symlink(os.path.abspath(os.path.join(REAL, f)), os.path.join(p, f))
    if os.path.exists(os.path.join(REAL, "ingest.json")):
        shutil.copyfile(os.path.join(REAL, "ingest.json"), os.path.join(p, "ingest.json"))
    gui = _gui(p, 3.0)
    assert _wait(lambda: (worker.detached(p) or {}).get("state") == "running")
    gui.terminate()
    gui.wait(10)
    app = App(p, port=free_port(), root=str(tmp_path))
    try:
        app._reattach()
        assert app.running and "re-attached" in app._job["label"]
        assert _wait(lambda: not app.running)
        assert app.g_state.content.startswith("✅ **long job finished**"), app.g_state.content
        assert any("long job" in h and "re-attached" in h for h in app._history_lines)
        assert any("long job done" in line for line in app._log_lines)
        assert worker.detached(p) is None
        gui = _gui(p, 60.0)  # Stop reaches a re-attached job
        assert _wait(lambda: (worker.detached(p) or {}).get("state") == "running")
        gui.terminate()
        gui.wait(10)
        app._reattach()
        app.stop()
        assert _wait(lambda: not app.running)
        assert app.g_state.content.startswith("⏹ **stopped**"), app.g_state.content
    finally:
        app.queue.stop_thread()
        app.server.stop()
