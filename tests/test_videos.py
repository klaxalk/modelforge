"""Shared video library (modelforge.videos) and the GUI Videos tab."""

import json
import os
import shutil
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


def test_folders(root):
    l0 = videos.listdir(root)
    assert [f["name"] for f in l0["folders"]] == ["videos"]  # the project folder "site" is not listed
    assert [v["name"] for v in l0["videos"]] == ["b.mp4"] and l0["videos"][0]["used_by"] == ["site"]
    p = videos.make_folder(root, "", "Flights 2026")
    assert os.path.isdir(p)
    with pytest.raises(FileExistsError):
        videos.make_folder(root, "", "Flights 2026")
    for bad in ("../x", ".hidden", ""):
        with pytest.raises(ValueError):
            videos.make_folder(root, "", bad)
    for bad in ("site", "..", "videos/../site"):  # projects and outside the root are not library folders
        with pytest.raises(ValueError):
            videos.resolve_dir(root, bad)
    up = videos.save_upload(root, "c.MP4", b"x", rel_dir="Flights 2026")
    videos.save_upload(root, "orphan.SRT", b"1", rel_dir="Flights 2026")
    l1 = videos.listdir(root, "Flights 2026", with_probe=False)
    assert [v["name"] for v in l1["videos"]] == ["c.MP4"] and [f["name"] for f in l1["files"]] == ["orphan.SRT"]
    videos.delete_file(root, os.path.join(p, "orphan.SRT"))
    with pytest.raises(ValueError):
        videos.delete_file(root, up)  # only stray .SRT files
    assert videos.delete_folder(root, "Flights 2026") == (1, 1) and not os.path.exists(p)
    with pytest.raises(ValueError):
        videos.delete_folder(root, "")  # the root
    q = videos.make_folder(root, "", "keep")
    shutil.move(os.path.join(root, "b.mp4"), os.path.join(q, "b.mp4"))  # a used clip inside
    ingest.save_plan(os.path.join(root, "site"), [ingest.Source(os.path.join(q, "b.mp4"))], ingest.Settings())
    with pytest.raises(ValueError, match="used by project"):
        videos.delete_folder(root, "keep")


def test_videos_tab(app, root):
    from viser import UploadedFile
    v = app.videos
    assert wait_for(lambda: len(v._entries) == 2)  # the videos/ folder and b.mp4 (the project is hidden)
    assert "used by" in v.g_list.content and "site" in v.g_list.content
    # new folder, enter it, upload there
    v.g_folder_name.value = "flights"
    v.make_folder()
    assert v.selected()["kind"] == "folder" and v.selected()["name"] == "flights"
    v.open_selected()
    assert v.cwd == "flights" and "empty folder" in v.g_list.content
    v._upload(UploadedFile("new.MP4", open(os.path.join(root, "b.mp4"), "rb").read()))
    assert wait_for(lambda: v.selected() and v.selected()["name"] == "new.MP4"), v.g_msg.content
    assert os.path.exists(os.path.join(root, "flights", "new.MP4"))
    assert "SRT: none (optional" in v.g_info.content
    assert any(o.endswith("new.MP4") for o in app.ingest.g_video.options)
    # add to the open project: then rename / delete are refused
    n = len(app.ingest.sources)
    v.add_to_project()
    assert len(app.ingest.sources) == n + 1 and app.ingest.sources[-1].video.endswith("new.MP4")
    v.g_new_name.value = "renamed"
    v.rename()
    assert "source of the open project" in v.g_msg.content
    app.ingest.sources.pop()
    ingest.save_plan(app.path, app.ingest.sources, app.ingest.settings)
    v.rename()
    assert "renamed to `renamed.MP4`" in v.g_msg.content and v.selected()["name"] == "renamed.MP4"
    sent = []

    class Client:
        def send_file_download(self, name, data):
            sent.append((name, len(data)))

    v.download(Client())
    assert wait_for(lambda: sent) and sent[0][0] == "renamed.MP4"
    v.delete()  # no confirmation
    assert os.path.exists(os.path.join(root, "flights", "renamed.MP4"))
    v.g_confirm.value = True
    v.delete()
    assert not os.path.exists(os.path.join(root, "flights", "renamed.MP4")) and "deleted" in v.g_msg.content
    # back up, delete the (now empty) folder
    v.up()
    assert v.cwd == ""
    v.g_entry.value = next(k for k, e in v._entries.items() if e["kind"] == "folder" and e["name"] == "flights")
    v.g_confirm.value = True
    v.delete()
    assert not os.path.exists(os.path.join(root, "flights"))
