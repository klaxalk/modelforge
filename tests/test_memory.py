"""Memory estimates before heavy stages (modelforge/memory.py): models, measured peaks (this project, scaled; the
machine's records), refine by BA mode, the sampler on a child process."""
import json
import os
import subprocess
import sys
import time

import pytest

from modelforge import memory


def _project(root, name, report):
    p = os.path.join(root, name)
    os.makedirs(p)
    json.dump(report, open(os.path.join(p, "report.json"), "w"))
    return p


def test_estimates_follow_measured_peaks(tmp_path):
    a = _project(str(tmp_path), "a", {"match": {"inlier_matches": 10e6}, "sfm_mesh": {"vertices": 2e6}})
    need, why = memory.estimate(a, "export")  # the model: 1.5 + 0.15 GB per M inlier matches
    assert abs(need - 3.0) < 1e-6 and "10.0 M inlier matches" in why
    memory.record(a, "export", 4.0)
    need, why = memory.estimate(a, "export")
    assert abs(need - 4.4) < 1e-6 and "peaked at 4.0 GB" in why  # its own peak + 10 %
    rep = json.load(open(os.path.join(a, "report.json")))
    assert rep["memory"]["export"] == {"peak_gb": 4.0, "driver": 10.0, "at": rep["memory"]["export"]["at"]}
    rep["match"]["inlier_matches"] = 20e6  # twice the matches: the part above the base grows with them
    json.dump(rep, open(os.path.join(a, "report.json"), "w"))
    need, why = memory.estimate(a, "export")
    assert abs(need - 1.1 * (1.5 + 2.5 * 2)) < 1e-6 and "2.0x" in why
    # another project on this machine learns from a's peak (GB per driver unit), not from the model
    b = _project(str(tmp_path), "b", {"match": {"inlier_matches": 30e6}})
    need, why = memory.estimate(b, "export")
    assert abs(need - 1.1 * (1.5 + 0.25 * 30)) < 1e-6 and "measured on this machine" in why
    # a group: its largest stage
    need, why = memory.estimate(b, "sfm", {"ba_mode": "chunked"})
    assert why.startswith("export:")


def test_chunked_refine_grows_at_the_model_rate(tmp_path):
    """Only a chunked refine's triangulations of all photos grow with the matches: a recorded peak grows by the
    model's GB per M matches, not in proportion (berlin_all: 16.3 GB at 31.1M -> 36 GB at 68.4M before)."""
    p = _project(str(tmp_path), "p", {"match": {"inlier_matches": 31.1e6}, "features": {"images": 7753}})
    memory.record(p, "refine", 16.32, {"ba_mode": "chunked"})
    rep = json.load(open(os.path.join(p, "report.json")))
    rep["match"]["inlier_matches"] = 68.4e6
    json.dump(rep, open(os.path.join(p, "report.json"), "w"))
    need, why = memory.estimate(p, "refine", {"ba_mode": "chunked"})
    assert abs(need - 1.1 * (16.32 + 0.16 * (68.4 - 31.1))) < 1e-6 and "2.2x" in why


def test_refine_by_ba_mode(tmp_path):
    p = _project(str(tmp_path), "p", {"match": {"inlier_matches": 80e6}, "features": {"images": 5000}})
    g = memory.estimate(p, "refine", {"ba_mode": "global"})[0]
    c = memory.estimate(p, "refine", {"ba_mode": "chunked"})[0]
    assert g > 50 and c < 20
    rep = json.load(open(os.path.join(p, "report.json")))
    rep["refine"] = {"ba": {"mode": "chunked"}}
    json.dump(rep, open(os.path.join(p, "report.json"), "w"))
    memory.record(p, "refine", 30.0, {"ba_mode": "auto"})  # filed under the mode it ran
    assert "refine:chunked" in json.load(open(os.path.join(p, "report.json")))["memory"]
    assert memory.estimate(p, "refine", {"ba_mode": "chunked"})[0] == 33.0
    assert memory.estimate(p, "refine", {"ba_mode": "global"})[0] > 50  # not from the chunked peak


def test_sampler_measures_a_child():
    s = memory.Sampler(every=0.1)
    s.begin()
    p = subprocess.Popen([sys.executable, "-c", "import time; b = bytearray(600 * 2 ** 20); b[::4096] = b'x' * "
                          "len(b[::4096]); time.sleep(1.5)"])
    p.wait()
    time.sleep(0.2)
    peak = s.peak()
    s.stop()
    assert 0.55 < peak < 1.5, peak


REAL = os.environ.get("MODELFORGE_TEST_PROJECT",
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_gui_checks_and_records(tmp_path, monkeypatch):
    """A job whose estimate exceeds the available RAM starts anyway without a browser (with a log line); the
    finished stage's measured peak lands in report.json."""
    import shutil
    from modelforge.gui.app import App
    from tests.test_gui_dense import free_port
    p = str(tmp_path / "proj")
    os.makedirs(p)
    for name in ("poses.json", "images", "sparse", "features.db", "database.db", "ingest.json"):
        if os.path.exists(os.path.join(REAL, name)):
            os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(p, name))
    shutil.copyfile(os.path.join(REAL, "report.json"), os.path.join(p, "report.json"))
    monkeypatch.setattr(memory, "estimate", lambda *a, **k: (1e6, "a test"))
    app = App(p, port=free_port())
    try:
        app.run_stage("pairs")
        t0 = time.time()
        while app.running and time.time() - t0 < 120:
            time.sleep(0.2)
        assert "finished" in app.g_state.content, app.g_state.content
        assert any("needs ~1000000.0 GB" in line for line in app._log_lines)
        rec = json.load(open(os.path.join(p, "report.json")))["memory"]["pairs"]
        assert rec["peak_gb"] > 0
    finally:
        app.lod.stop()
        app._mem.stop()
        app.server.stop()
