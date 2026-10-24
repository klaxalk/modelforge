"""Job queue (modelforge/gui/jobqueue.py)."""
import json
import os
import shutil
import time

import pytest

from modelforge.gui import jobqueue
from modelforge.gui.jobqueue import JobQueue

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


def test_queue_file_and_policies(tmp_path):
    q = JobQueue(str(tmp_path))
    a = q.add(str(tmp_path / "a"), "refine", "Refine", {"g_ba_mode": "chunked"})
    b = q.add(str(tmp_path / "a"), "export", "Export", {})
    c = q.add(str(tmp_path / "b"), "splat", "Train splat", {})
    assert q.next_pending() is a
    a["status"] = "running"
    q.save()
    r = JobQueue(str(tmp_path))  # a GUI restart: the running entry is not rerun
    assert [e["status"] for e in r.entries] == ["interrupted", "pending", "pending"]
    assert r.entries[0]["settings"] == {"g_ba_mode": "chunked"} and r.next_pending()["id"] == b["id"]
    assert q.finish(a, "failed", "boom") == "stop"  # default: a failure stops the queue
    q.go_on = True
    a["status"] = "pending"
    assert q.finish(a, "failed") == "go" and b["status"] == "skipped" and c["status"] == "pending"
    assert q.finish(c, "stopped") == "pause"
    q.retry(2)
    assert b["status"] == "pending"
    q.up(2)
    assert q.entries[0] is b
    q.remove(1)
    assert b not in q.entries
    q.clear_finished()
    assert q.entries == []


def test_entry_waits_for_memory(tmp_path, monkeypatch):
    """An entry starts only when the RAM it needs is available: MIN_FREE_GB, its project's last recorded peak, a
    forced-global BA's estimate; meanwhile it waits with a note (one notice)."""
    from types import SimpleNamespace
    proj = tmp_path / "p"
    proj.mkdir()
    json.dump({"refine": {"peak_rss_gb": 14.6}, "match": {"inlier_matches": 82.7e6}}, open(proj / "report.json", "w"))
    q = JobQueue(str(tmp_path))
    e = q.add(str(proj), "refine", "Refine", {"g_ba_mode": "chunked"})
    need, why = jobqueue.memory_need(e)
    assert abs(need - 16.06) < 0.01 and "peaked at 15 GB" in why
    g = q.add(str(proj), "refine", "Refine", {"g_ba_mode": "global"})
    assert jobqueue.memory_need(g)[0] > 50  # 82.7M inlier matches
    assert jobqueue.memory_need(q.add(str(proj), "splat", "Train splat", {}))[0] == jobqueue.MIN_FREE_GB
    started, notices = [], []
    app = SimpleNamespace(running=False, ingest=SimpleNamespace(running=False), path=str(proj), _run_id=0,
                          run_stage=started.append, _notice=notices.append, _apply_settings=lambda s: None,
                          open_project=lambda p: None)
    panel = jobqueue.QueuePanel.__new__(jobqueue.QueuePanel)
    panel.app, panel.q, panel.current, panel.current_run, panel.refresh = app, q, None, None, lambda: None
    q.entries = [e]
    q.running = True
    monkeypatch.setattr(jobqueue, "_available_gb", lambda: 10.0)
    panel.tick()
    panel.tick()
    assert started == [] and e["note"].startswith("waiting for memory: 10.0 GB") and len(notices) == 1
    monkeypatch.setattr(jobqueue, "_available_gb", lambda: 20.0)
    panel.tick()
    assert started == ["refine"] and "note" not in e


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")), reason="needs a matched project")
def test_queued_entries_run_with_their_settings(tmp_path, monkeypatch):
    """Two projects queued with different Pairs settings: each entry opens its project and runs with the settings the
    tab had when it was queued; a notice per entry and one for the queue."""
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    root = str(tmp_path)
    for name in ("a", "b"):
        p = os.path.join(root, name)
        os.makedirs(p)
        for f in ("poses.json", "images", "database.db"):
            os.symlink(os.path.abspath(os.path.join(REAL, f)), os.path.join(p, f))
        for f in ("ingest.json", "camera.json"):
            if os.path.exists(os.path.join(REAL, f)):
                shutil.copyfile(os.path.join(REAL, f), os.path.join(p, f))
    app = App(os.path.join(root, "a"), port=free_port(), root=root)
    try:
        app.g_pairs.value = "sequential"
        app.g_neighbors.value = 7
        app.queue.g_stage.value = "Run SfM (all)"
        monkeypatch.setitem(jobqueue.STAGES, "Pairs", "pairs")
        app.queue.g_stage.options = tuple(jobqueue.STAGES)
        app.queue.g_stage.value = "Pairs"
        app.queue.add()
        app.open_project(os.path.join(root, "b"))
        app.g_pairs.value = "auto"
        app.queue.add()
        app.open_project(os.path.join(root, "a"))
        app.g_pairs.value = "exhaustive"  # a later change of the tab does not change the queued entries
        app.queue.start()
        t0 = time.time()
        while (app.queue.q.running or app.running) and time.time() - t0 < 300:
            time.sleep(0.2)
        assert [e["status"] for e in app.queue.q.entries] == ["done", "done"], app.queue.q.entries
        ra = json.load(open(os.path.join(root, "a", "report.json")))["pairs"]
        rb = json.load(open(os.path.join(root, "b", "report.json")))["pairs"]
        assert ra["mode"] == "sequential" and ra["neighbors"] == 7 and rb["mode"] == "auto"
        assert os.path.abspath(app.path) == os.path.join(root, "b")  # the last entry's project stays open
        assert "queue finished" in app._notice_last and app._notice_n == 3
        assert 'data-notice="3|queue finished' in app._status_html()
        assert json.load(open(os.path.join(root, jobqueue.FILE)))["entries"][0]["status"] == "done"
    finally:
        app.queue.stop_thread()
        app.server.stop()
