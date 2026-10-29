"""GUI Ingest tab, headless: start on an empty project, add two pieces of the sample clip,
plan, preview a frame, extract, then run the first SfM stages (skipped without the video)."""

import os
import socket
import time

import numpy as np
import pytest

from tests.sample_clip import sample_video

VIDEO = sample_video()
pytestmark = pytest.mark.skipif(not os.path.exists(VIDEO), reason="needs the sample clip berlin_3_1.MP4 in data/")


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


def test_images_of_an_interrupted_extraction(app):
    """Images without poses.json (an extraction that stopped): the Overwrite box shows and extraction asks for it
    (before, the box stayed hidden and extraction refused: no way out in the GUI)."""
    p = app.ingest
    img = os.path.join(app.path, "images")
    os.makedirs(img, exist_ok=True)
    open(os.path.join(img, "x_00001.jpg"), "wb").close()
    try:
        p._update_status()
        assert p.g_overwrite.visible and "interrupted extraction" in p.g_md.content
        p.extract()
        assert "tick **Overwrite**" in p.g_status.content and not p.running
    finally:
        os.remove(os.path.join(img, "x_00001.jpg"))
        p._update_status()
    assert not p.g_overwrite.visible


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


def test_ingest_folder_filter(tmp_path):
    """The Ingest tab's Folder narrows the Video list to one folder's clips (by file name); a rescan keeps the
    folder (its count updated); Add as source resolves the name to the clip's path."""
    from modelforge.dji import ingest
    from modelforge.gui.ingest import ALL_FOLDERS, IngestPanel

    class W:
        def __init__(self, value=""):
            self.value, self.options, self.content = value, (), ""
    root = tmp_path / "data"
    for rel in ("videos/a/one.MP4", "videos/a/two.MP4", "videos/b/three.MP4"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    panel = IngestPanel.__new__(IngestPanel)
    panel.app = type("A", (), {"root": str(root)})()
    panel.path, panel.kind = str(root / "proj"), "dji"
    panel.g_folder, panel.g_video, panel.g_video_path, panel.g_status = W(ALL_FOLDERS), W(), W(), W()
    panel.rescan()
    folders = panel.g_folder.options  # (rescan also lists the working directory's data/: the real library)
    mine = [o for o in folders if str(tmp_path.name) in o]
    assert folders[0] == ALL_FOLDERS and len(mine) == 2 and any(o.endswith("a (2)") for o in mine)
    assert sum(str(tmp_path.name) in v for v in panel.g_video.options) == 3  # all folders: paths
    panel.g_folder.value = next(o for o in mine if o.endswith("a (2)"))
    panel._filter_videos()
    assert panel.g_video.options == ("one.MP4", "two.MP4")
    (root / "videos/a/four.MP4").write_bytes(b"")
    panel.rescan()
    assert panel.g_folder.value.endswith("a (3)") and len(panel.g_video.options) == 3
    added = []
    panel.sources = []
    panel.add_source = lambda video=None, _orig=IngestPanel.add_source: added.append(
        video or panel._video_paths.get(panel.g_video.value))
    panel.g_video.value = "two.MP4"
    panel.add_source()
    assert added[0].endswith(os.path.join("videos", "a", "two.MP4"))

    # Add all in folder: the folder's clips, those already sources skipped; refused on "all folders"
    del panel.add_source
    panel._sources_added = lambda n: added.append(n)
    panel.sources = [ingest.Source(panel._video_paths["two.MP4"])]
    panel.add_folder()
    assert sorted(os.path.basename(s.video) for s in panel.sources) == ["four.MP4", "one.MP4", "two.MP4"]
    assert added[-1] == 2 and "added 2" in panel.g_status.content and "1 skipped" in panel.g_status.content
    panel.g_folder.value = ALL_FOLDERS
    panel._filter_videos()
    panel.add_folder()
    assert len(panel.sources) == 3 and "pick a Folder" in panel.g_status.content
