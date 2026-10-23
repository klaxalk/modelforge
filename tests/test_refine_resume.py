"""Resumable refine (modelforge/sfm/checkpoint.py): stopped after a step, resumed from there."""
import json
import os
import shutil

import pytest

from modelforge import jobs, pipeline
from modelforge.sfm import checkpoint, chunked, robust
from modelforge.sfm import priors as priors_mod

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))
needs = pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "database.db")),
                           reason="needs a matched project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")


def workspace(tmp_path):
    p = str(tmp_path)
    for name in ("poses.json", "images", "database.db"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    for name in ("ingest.json", "camera.json"):
        if os.path.exists(os.path.join(REAL, name)):
            shutil.copyfile(os.path.join(REAL, name), os.path.join(p, name))
    return p


def run(p, stop_when=None, resume=False, **kw):
    """stage_refine; stop_when(event) True cancels the job there. Returns (events, cancelled)."""
    pri, _ = priors_mod.load(p)
    token, events = jobs.Token(), []

    def progress(ev):
        events.append({k: v for k, v in ev.items() if k not in ("rec", "eval")})
        if stop_when and stop_when(ev):
            token.cancel()
    try:
        pipeline.stage_refine(p, pipeline.Report(p), pri, progress=progress, token=token, model=robust.PriorModel(),
                              resume=resume, **kw)
    except jobs.Cancelled:
        return events, True
    return events, False


@needs
def test_global_refine_resumes_after_the_last_round(tmp_path):
    p = workspace(tmp_path)
    _, stopped = run(p, lambda ev: ev["phase"] == "round_done" and ev.get("name") == "round_1", ba_mode="global")
    assert stopped and os.path.exists(os.path.join(p, checkpoint.FILE))
    assert not os.path.exists(os.path.join(p, "sparse", "refined"))
    events, stopped = run(p, resume=True, ba_mode="global")
    assert not stopped
    done = [ev["name"] for ev in events if ev["phase"] == "round_done"]
    assert done == ["round_2"], done  # rounds 0 and 1 came from the checkpoint
    r = json.load(open(os.path.join(p, "report.json")))["refine"]
    assert r["resumed_after"] == ["round_0", "round_1"] and {"round_0", "round_1", "round_2"} <= set(r)
    assert r["final"]["point_error_px"]["median"] < 1.0
    assert not os.path.exists(os.path.join(p, checkpoint.FILE))  # a finished refine removes it


@needs
def test_chunked_refine_resumes_at_the_next_chunk(tmp_path):
    p = workspace(tmp_path)
    cfg = dict(ba_mode="chunked", chunks=chunked.Chunked(size=30, overlap=10))
    _, stopped = run(p, lambda ev: ev["phase"] == "chunk_done" and ev["chunk"] == 1, **cfg)
    assert stopped
    events, stopped = run(p, resume=True, **cfg)
    assert not stopped
    chunks = [ev["chunk"] for ev in events if ev["phase"] == "chunk_done"]
    assert chunks and min(chunks) == 2, chunks  # chunks 0 and 1 came from the checkpoint
    assert not any(ev.get("stage") == "calib" for ev in events)  # so did the calibration
    r = json.load(open(os.path.join(p, "report.json")))["refine"]
    assert r["resumed_after"][:2] == ["calib_0", "calib_1"] and r["final"]["point_error_px"]["median"] < 1.0
    assert len(r["chunked"]["chunks"]) == max(chunks) + 1  # the resumed chunks' stats were kept


@needs
def test_other_settings_start_afresh(tmp_path):
    p = workspace(tmp_path)
    run(p, lambda ev: ev["phase"] == "round_done" and ev.get("name") == "round_0", ba_mode="global")
    events, _ = run(p, lambda ev: ev["phase"] == "round_done", resume=True, ba_mode="global", max_iterations=100)
    assert [ev["name"] for ev in events if ev["phase"] == "round_done"] == ["round_0"]  # not resumed
