"""Projects: create / list / status / clean on a temporary root, and GUI project switching
(the real project part is skipped without data/berlin_3_1_frames)."""

import json
import os
import socket

import pytest

from modelforge import projects
from modelforge.dji import ingest

REAL = os.environ.get("MODELFORGE_TEST_PROJECT",  # any refined project
                      os.path.join(os.path.dirname(__file__), "..", "data", "berlin_3_1_frames"))


def touch(path, nbytes=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * nbytes)


def fake_project(root, name):
    p = projects.create(root, name)
    ingest.save_plan(p, [ingest.Source("data/a.MP4", 5.0, 50.0)], ingest.Settings(num_images=10))
    touch(os.path.join(p, "images", "a_00001.jpg"))
    json.dump({"images": [{"image": "a_00001.jpg"}]}, open(os.path.join(p, "poses.json"), "w"))
    json.dump({"features": {}, "pairs": {}, "dense": {"undistort": {}}}, open(os.path.join(p, "report.json"), "w"))
    touch(os.path.join(p, "features.db"), 1000)
    touch(os.path.join(p, "sparse", "round_0", "images.bin"), 100)
    touch(os.path.join(p, "sparse", "refined", "images.bin"), 100)
    touch(os.path.join(p, "dense", "stereo", "depth_maps", "a.bin"), 5000)
    touch(os.path.join(p, "dense", "meshed-delaunay.ply"), 300)
    touch(os.path.join(p, "transforms.json"))
    return p


def test_create_list_status_clean(tmp_path):
    root = str(tmp_path)
    a = fake_project(root, "site_a")
    b = projects.create(root, "site_b", copy_plan_from=a)
    assert ingest.load_plan(b)[0][0].start == 5.0  # sources copied, nothing else
    assert not os.path.exists(os.path.join(b, "images"))
    with pytest.raises(FileExistsError):
        projects.create(root, "site_b")
    with pytest.raises(ValueError):
        projects.create(root, "../evil")
    os.makedirs(os.path.join(root, "not_a_project"))
    assert {os.path.basename(p) for p in projects.list_projects(root)} == {"site_a", "site_b"}

    st = projects.status(a)
    assert st["images"] == 1 and len(st["sources"]) == 1
    assert st["stages"] == ["features", "pairs", "dense:undistort"]
    final = {e["path"] for e in st["final"]}
    inter = {e["path"]: e["bytes"] for e in st["intermediate"]}
    assert {"transforms.json", "sparse/refined", "dense/meshed-delaunay.ply"} <= final
    assert inter["dense/stereo"] == 5000 and "sparse/round_0" in inter and "images" in inter

    freed = projects.clean(a)
    assert freed == 1000 + 100 + 5000
    for kept in ("images", "ingest.json", "poses.json", "transforms.json", "sparse/refined",
                 "dense/meshed-delaunay.ply"):
        assert os.path.exists(os.path.join(a, kept)), kept
    for gone in ("features.db", "sparse/round_0", "dense/stereo"):
        assert not os.path.exists(os.path.join(a, gone)), gone


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "sparse", "refined", "images.bin")),
                    reason="needs a refined project (data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_gui_switches_projects(tmp_path):
    from modelforge.gui.app import App
    root = str(tmp_path)
    real = os.path.join(root, "real")
    os.makedirs(real)
    for name in ("poses.json", "images", "sparse"):
        os.symlink(os.path.abspath(os.path.join(REAL, name)), os.path.join(real, name))
    app = App(os.path.join(root, "empty"), port=free_port(), root=root)
    try:
        assert app.project is None
        app.projects.g_name.value = "fresh"
        app.projects.create()
        assert app.path.endswith("fresh") and os.path.exists(os.path.join(root, "fresh", "ingest.json"))
        assert "fresh" in app.projects.g_list.options

        app.projects.refresh()
        app.projects.g_list.value = "real"
        app.projects.open_selected()
        assert app.project is not None and app.snap.name == "refined"
        n = len(os.listdir(os.path.join(REAL, "images")))
        assert len(app.cam_handles) == len(app.project.names) == n
        assert f"{n} images" in app.g_header.content

        app.open_project(os.path.join(root, "fresh"))
        assert app.project is None and not app.cam_handles and "no images yet" in app.g_header.content
        assert app.ingest.path.endswith("fresh") and app.ingest.sources == []

        app.running = True  # a running job blocks switching
        with pytest.raises(RuntimeError):
            app.open_project(real)
        app.running = False
    finally:
        app.server.stop()
