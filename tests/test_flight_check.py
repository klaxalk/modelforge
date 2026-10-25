"""Check flight alignment (dji/flight_check.py): photos chosen where the flights look at the same ground, in short
runs per spot; flights over different ground are told apart."""
import numpy as np

from modelforge.dji import flight_check as fc


class Clip:
    def __init__(self, x0, n=600):
        a = np.zeros((n, 10))
        a[:, 0] = 50.0
        a[:, 1] = 14.0 + (x0 + np.linspace(0, 200, n)) / 71_500  # ~1 m per 1/71500 deg of longitude at 50 deg
        a[:, 2] = 20.0  # rel_alt
        self._t = {"a": a, "yaw": np.full(n, 90.0), "pitch": np.full(n, -60.0)}  # east, looking down

    def telemetry(self, yaw_source="auto"):
        return self._t


class SP:
    def __init__(self, index, x0, alt_offset=0.0, n=600):
        self.index, self.clip, self.alt_offset = index, Clip(x0, n), alt_offset
        self.valid = np.ones(n, bool)


class Plan:
    def __init__(self, sources):
        self.sources = sources


def test_choose_shared_ground():
    plan = Plan([SP(0, 0.0), SP(1, 100.0, alt_offset=-240.0), SP(2, 5000.0)])  # 1 overlaps 0 by half; 2 elsewhere
    chosen, shared = fc.choose(plan)
    assert shared.get((0, 1), 0) > 0 and (0, 2) not in shared and (1, 2) not in shared
    assert chosen[2] == []  # no ground shared: the check says so
    for s in (0, 1):
        fr = chosen[s]
        assert 0 < len(fr) <= fc.PER_FLIGHT + fc.RUN
        d = np.diff(fr)
        assert (d == fc.STEP).sum() >= len(fr) // 2  # runs of frames STEP apart (a baseline per spot)


def test_ingest_panel_shows_and_applies_the_check(tmp_path):
    """Ingest tab → Flight alignment: shown for several DJI videos; the stored result listed; Use the measured
    altitude offsets puts the trusted ones into the sources (all pieces of that video), the others stay auto."""
    import json
    import os
    import pytest
    from modelforge.dji import ingest
    from modelforge.gui.app import App
    from tests.test_projects import free_port
    vids = ["data/videos/karlak/karlak_1.mp4", "data/videos/karlak/karlak_2.mp4", "data/videos/karlak/karlak3.mp4"]
    if not all(os.path.exists(v) for v in vids):
        pytest.skip("karlak clips not here")
    p = tmp_path / "proj"
    p.mkdir()
    ingest.save_plan(str(p), [ingest.Source(v) for v in vids] + [ingest.Source(vids[2], 10.0, 20.0)], ingest.Settings())
    os.makedirs(p / fc.DIR)
    flights = [{"source": k, "video": v, "name": os.path.basename(v), "photos": 42, "offset_used_m": 0.0,
                "correction_m": [0, 0, 0], "alt_offset_m": a, "merged_points": 3000, "last_step_m": [0, 0, 0.1],
                "trusted": t} for k, (v, a, t) in enumerate(zip(vids, (0.0, 0.78, 1.59), (True, False, True)))]
    json.dump({"flights": flights, "apart": {"karlak_1.mp4 – karlak3.mp4": {"merged": 1, "split": 1,
                                                                            "apart_m": [0.3, 0.4, 0.1]}},
               "registered": 126, "photos": 126, "seconds": 300}, open(p / fc.DIR / "result.json", "w"))
    app = App(str(p), port=free_port(), root=str(tmp_path))
    try:
        ing = app.ingest
        assert ing.g_fc.visible and "**+1.6 m**" in ing.g_fc_md.content and "not trusted" in ing.g_fc_md.content
        assert "0.5 m apart horizontally" in ing.g_fc_md.content and not ing.g_fc_use.disabled
        assert ing.flight_check_wanted()
        done = ing.apply_flight_check()
        assert done == {"karlak_1.mp4": 0.0, "karlak3.mp4": 1.59}
        assert [s.alt_offset for s in ing.sources] == [0.0, None, 1.59, 1.59]  # karlak_2 not trusted: auto
        assert [s.get("alt_offset") for s in json.load(open(p / "ingest.json"))["sources"]] == [0.0, None, 1.59, 1.59]
    finally:
        app.server.stop()
