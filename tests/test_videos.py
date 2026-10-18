"""Shared video library (modelforge.videos) and the GUI Videos tab."""

import json
import os
import socket
import subprocess
import time

import pytest

from modelforge import videos
from modelforge.dji import ingest


def make_clip(path, seconds=1):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={seconds}",
                    "-pix_fmt", "yuv420p", path], check=True)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "data"
    (r / "videos").mkdir(parents=True)
    make_clip(str(r / "videos" / "a.MP4"))
    (r / "videos" / "a.SRT").write_text("1\n")
    make_clip(str(r / "b.mp4"))
    proj = r / "site"
    proj.mkdir()
    ingest.save_plan(str(proj), [ingest.Source(str(r / "b.mp4"))], ingest.Settings())
    make_clip(str(proj / "inside.mp4"))  # inside a project folder: not part of the library
    return str(r)


def test_list_probe_and_usage(root):
    e = {x["name"]: x for x in videos.list_videos(root)}
    assert set(e) == {"a.MP4", "b.mp4"}
    assert e["a.MP4"]["srt"].endswith("a.SRT") and e["b.mp4"]["srt"] is None
    assert e["a.MP4"]["width"] == 64 and e["a.MP4"]["height"] == 48 and e["a.MP4"]["fps"] == 10
    assert abs(e["a.MP4"]["duration_s"] - 1.0) < 0.2
    assert e["b.mp4"]["used_by"] == ["site"] and e["a.MP4"]["used_by"] == []


def test_upload(root):
    p = videos.save_upload(root, "C:\\clips\\c.MOV", b"x")
    assert p == os.path.join(root, "videos", "c.MOV")
    videos.save_upload(root, "c.srt", b"1")
    with pytest.raises(FileExistsError):
        videos.save_upload(root, "c.MOV", b"y")
    for bad in ("notes.txt", "", ".x.mp4"):
        with pytest.raises(ValueError):
            videos.save_upload(root, bad, b"")
    assert not [f for f in os.listdir(os.path.join(root, "videos")) if ".part" in f]


def test_rename_moves_srt_and_refuses_used(root):
    new = videos.rename(root, os.path.join(root, "videos", "a.MP4"), "flight1.mp4")
    assert new == os.path.join(root, "videos", "flight1.MP4")  # extension kept
    assert os.path.exists(os.path.join(root, "videos", "flight1.SRT"))
    assert not os.path.exists(os.path.join(root, "videos", "a.SRT"))
    with pytest.raises(ValueError, match="used by project"):
        videos.rename(root, os.path.join(root, "b.mp4"), "x")
    with pytest.raises(ValueError):
        videos.rename(root, os.path.join(root, "site", "inside.mp4"), "x")  # project folder
    make_clip(os.path.join(root, "videos", "z.MP4"))
    with pytest.raises(FileExistsError):
        videos.rename(root, os.path.join(root, "videos", "z.MP4"), "flight1")


def test_delete(root):
    gone = videos.delete(root, os.path.join(root, "videos", "a.MP4"))
    assert sorted(os.path.basename(g) for g in gone) == ["a.MP4", "a.SRT"]
    with pytest.raises(ValueError, match="used by project"):
        videos.delete(root, os.path.join(root, "b.mp4"))
    with pytest.raises(ValueError):
        videos.delete(root, os.path.join(root, "..", "elsewhere.mp4"))
    assert os.path.exists(os.path.join(root, "b.mp4"))


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for(cond, timeout=10):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.05)
    return cond()


@pytest.fixture
def app(root):
    from modelforge.gui.app import App
    a = App(os.path.join(root, "site"), port=free_port(), root=root)
    yield a
    a.server.stop()


def test_videos_tab(app, root):
    from viser import UploadedFile
    v = app.videos
    assert wait_for(lambda: len(v._entries) == 2)
    assert "used by" in v.g_list.content and "site" in v.g_list.content
    v._upload(UploadedFile("new.MP4", open(os.path.join(root, "b.mp4"), "rb").read()))
    assert wait_for(lambda: v.selected() and v.selected()["name"] == "new.MP4"), v.g_msg.content
    assert "SRT: none (optional" in v.g_info.content  # no SRT: fine
    assert any(o.endswith("new.MP4") for o in app.ingest.g_video.options)
    n = len(app.ingest.sources)
    v.add_to_project()
    assert len(app.ingest.sources) == n + 1 and app.ingest.sources[-1].video.endswith("new.MP4")
    v.g_new_name.value = "renamed"
    v.rename()  # a source of the open project now (even before ingest.json is saved): refused
    assert "source of the open project" in v.g_msg.content
    app.ingest.sources.pop()
    ingest.save_plan(app.path, app.ingest.sources, app.ingest.settings)
    v.rename()
    assert "renamed to `renamed.MP4`" in v.g_msg.content
    assert wait_for(lambda: v.selected() and v.selected()["name"] == "renamed.MP4")
    v.delete()  # no confirmation
    assert os.path.exists(os.path.join(root, "videos", "renamed.MP4"))
    v.g_confirm.value = True
    v.delete()
    assert not os.path.exists(os.path.join(root, "videos", "renamed.MP4")) and "deleted" in v.g_msg.content
    sent = []

    class Client:
        def send_file_download(self, name, data):
            sent.append((name, len(data)))

    assert wait_for(lambda: v.selected() is not None)
    v.download(Client())
    assert wait_for(lambda: sent) and sent[0][0] == v.selected()["name"]
