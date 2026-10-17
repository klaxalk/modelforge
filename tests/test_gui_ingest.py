"""GUI Ingest tab, headless: start on an empty project, add two pieces of the sample clip,
plan, preview a frame, extract, then run the first SfM stages (skipped without the video)."""

import os
import socket
import time

import numpy as np
import pytest

VIDEO = os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1.MP4")
pytestmark = pytest.mark.skipif(not os.path.exists(VIDEO), reason="needs data/berlin_3_1.MP4")


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait(cond, timeout=120):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.1)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from modelforge.gui.app import App
    a = App(str(tmp_path_factory.mktemp("site")), port=free_port())
    yield a
    a.server.stop()


def test_empty_project_starts(app):
    assert app.project is None and "no images yet" in app.g_header.content
    app.run_stage("pairs")
    assert "Ingest tab" in app.g_state.content


def test_plan_two_pieces(app):
    p = app.ingest
    p.g_video_path.value = VIDEO
    p.add_source()
    wait(lambda: p._source_widgets and p._source_widgets[0]["ready"])
    p._source_action(0, "Add piece")
    wait(lambda: len(p._source_widgets) == 2 and p._source_widgets[1]["ready"])
    p._source_widgets[0]["range"].value = (10.0, 40.0)
    p._source_widgets[1]["range"].value = (60.0, 90.0)
    p.g_num.value = 6
    wait(lambda: p.plan is not None and p.plan.num_images == 6 and len(p.plan.sources) == 2)
    assert "plan: 6 images" in p.g_plan_md.content
    assert os.path.exists(os.path.join(app.path, "ingest.json"))
    assert p.prev and len(p.prev["sources"][1]["samples"]) == len(p.plan.sources[1].windows)
    p._show_preview(1, 75.0)
    assert p.g_pv_image.image.shape == (360, 640, 3) and "inside" in p.g_pv_md.content


def test_extract_and_run_first_stages(app):
    p = app.ingest
    p.extract()
    wait(lambda: not p.running, timeout=300)
    assert app.project is not None, p.g_status.content
    assert len(app.project.names) == 6 and set(app.project.source.tolist()) == {0, 1}
    assert "from 2 sources" in app.g_header.content
    app.g_color.value = "source"
    app._apply_colors()
    assert "berlin_3_1" in app.g_legend.content
    for stage in ("features", "pairs"):
        app.run_stage(stage)
        wait(lambda: not app.running, timeout=300)
        assert "finished" in app.g_state.content, app.g_state.content
    pairs = open(os.path.join(app.path, "pairs.txt")).read().split("\n")
    assert len([x for x in pairs if x]) > 0
    p.extract()  # project has images now: needs the overwrite box
    assert "Overwrite" in p.g_status.content
    assert np.all(np.isfinite(app.project.prior_cams.centers))
