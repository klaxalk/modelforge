"""modelforge/worker.py: stages in a job process of their own."""
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
