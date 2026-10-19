"""Projects: create / list / status / clean on a temporary root, and GUI project switching
(the real project part is skipped without data/berlin_3_1_frames)."""

import json
import os
import shutil
import socket
import time

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


def test_rename_and_delete(tmp_path):
    root = str(tmp_path)
    a = fake_project(root, "site_a")
    fake_project(root, "site_b")
    new = projects.rename(root, a, "site_c")
    assert new == os.path.join(root, "site_c") and os.path.exists(os.path.join(new, "images", "a_00001.jpg"))
    assert not os.path.exists(a)
    with pytest.raises(FileExistsError):
        projects.rename(root, new, "site_b")
    with pytest.raises(ValueError):
        projects.rename(root, new, "../evil")
    os.makedirs(os.path.join(root, "plain"))
    with pytest.raises(ValueError):  # not a project
        projects.delete(root, os.path.join(root, "plain"))
    with pytest.raises(ValueError):  # not directly under root
        projects.delete(root, os.path.join(new, "images"))
    assert projects.delete(root, new) > 0 and not os.path.exists(new)
    assert [os.path.basename(p) for p in projects.list_projects(root)] == ["site_b"]


def test_gui_rename_and_delete(tmp_path):
    from modelforge.gui.app import App
    root = str(tmp_path)
    a, b = projects.create(root, "one"), projects.create(root, "two")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    app = App(a, port=port, root=root)
    try:
        p = app.projects
        p.refresh()
        p.g_list.value = "one"  # the open project
        p.g_new_name.value = "uno"
        p.rename_selected()
        assert app.path == os.path.join(root, "uno") and "renamed" in p.g_msg.content
        assert "uno" in p.g_list.options and "one" not in p.g_list.options
        p.g_del_confirm.value = True
        p.delete_selected()  # the open project: refused
        assert os.path.exists(app.path) and "open project" in p.g_msg.content
        p.g_list.value = "two"
        p.g_del_confirm.value = False
        p.delete_selected()  # no confirmation: refused
        assert os.path.exists(b)
        p.g_del_confirm.value = True
        p.delete_selected()
        assert not os.path.exists(b) and "deleted" in p.g_msg.content and not p.g_del_confirm.value
        assert p.g_list.options == ("uno",)
    finally:
        app.server.stop()


def test_subsample(tmp_path):
    """A subset of the extracted images in the same project: every N-th per source in a time window,
    hard links into the pool, old results archived, every 1 over everything restores all images."""
    root = str(tmp_path)
    p = projects.create(root, "big")
    images = []
    for source, (t0, n) in enumerate(((0.0, 10), (100.0, 6))):
        for i in range(n):
            name = f"v{source}_{i:05d}.jpg"
            touch(os.path.join(p, "images", name))
            images.append({"image": name, "source": source, "time_s": t0 + i, "enu_x": float(i)})
    json.dump({"sources": [{}, {}], "ground_z": -5.0, "images": images}, open(os.path.join(p, "poses.json"), "w"))
    with open(os.path.join(p, "poses.csv"), "w") as f:
        f.write("image,source,time_s\n" + "".join(f"{im['image']},{im['source']},{im['time_s']}\n" for im in images))
    touch(os.path.join(p, "features.db"))
    touch(os.path.join(p, "sparse", "refined", "images.bin"))

    keep = projects.subsample_selection(images, every=3)
    assert [images[k]["image"] for k in keep] == ["v0_00000.jpg", "v0_00003.jpg", "v0_00006.jpg", "v0_00009.jpg",
                                                  "v1_00000.jpg", "v1_00003.jpg"]  # per source
    assert len(projects.subsample_selection(images, every=2, offset=1)) == 5 + 3
    assert [images[k]["time_s"] for k in projects.subsample_selection(images, every=2, start=3, end=7)] == [3, 5, 7]

    n, total, archived = projects.apply_subsample(p, every=2, offset=1)
    assert (n, total) == (8, 16) and archived and os.path.exists(os.path.join(archived, "features.db"))
    assert not os.path.exists(os.path.join(p, "features.db")) and not os.path.exists(os.path.join(p, "sparse"))
    poses = json.load(open(os.path.join(p, "poses.json")))
    assert [im["image"] for im in poses["images"]][:2] == ["v0_00001.jpg", "v0_00003.jpg"]
    assert poses["ground_z"] == -5.0 and poses["subsample"]["every"] == 2
    assert sorted(os.listdir(os.path.join(p, "images"))) == sorted(im["image"] for im in poses["images"])
    a, b = os.path.join(p, "extracted", "images", "v0_00001.jpg"), os.path.join(p, "images", "v0_00001.jpg")
    assert os.stat(a).st_ino == os.stat(b).st_ino  # hard link into the pool
    assert len(open(os.path.join(p, "poses.csv")).read().splitlines()) == 1 + 8
    assert projects.subsample_info(p) == {"extracted": 16, "used": 8, "setting": poses["subsample"]}

    n, _, archived = projects.apply_subsample(p, every=3, start=2, end=8)  # another subset of the pool
    assert n == 3 and archived is None  # nothing computed since
    with pytest.raises(ValueError):
        projects.apply_subsample(p, start=50, end=60)  # nothing in that window
    n, _, _ = projects.apply_subsample(p)  # all again
    assert n == 16 and "subsample" not in json.load(open(os.path.join(p, "poses.json")))
    assert len(os.listdir(os.path.join(p, "images"))) == 16 and len(projects.extracted_images(p)) == 16
    assert "extracted" in projects.KEEP_ON_CLEAN and "extracted" in ingest.OUTPUTS  # re-extraction archives it


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL, "images")), reason="needs a project with images "
                    "(data/berlin_3_1_frames or MODELFORGE_TEST_PROJECT)")
def test_gui_subsample(tmp_path):
    from modelforge.gui.app import App
    root = str(tmp_path)
    real = os.path.join(root, "real")
    shutil.copytree(os.path.join(REAL, "images"), os.path.join(real, "images"))  # applying moves images/
    shutil.copy2(os.path.join(REAL, "poses.json"), os.path.join(real, "poses.json"))
    os.symlink(os.path.abspath(os.path.join(REAL, "sparse")), os.path.join(real, "sparse"))
    app = App(real, port=free_port(), root=root)
    try:
        n = len(app.project.names)
        s = app.subset
        assert f"all **{n}**" in s.g_md.content
        s.g_every.value = 4
        assert f"of {n} images" in s.g_preview.content
        s.apply()
        t0 = time.time()
        while "now using" not in s.g_preview.content and time.time() - t0 < 60:
            time.sleep(0.1)
        k = len(range(0, n, 4))
        assert f"now using {k} of {n}" in s.g_preview.content, s.g_preview.content
        assert len(app.project.names) == k and f"**{k}** of {n}" in s.g_md.content
        assert not os.path.exists(os.path.join(real, "sparse"))  # the old results moved to previous/
    finally:
        app.server.stop()
