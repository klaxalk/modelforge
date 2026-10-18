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
    make_clip(str(r / "videos" / "b.mp4"))
    proj = r / "site"
    proj.mkdir()
    ingest.save_plan(str(proj), [ingest.Source(str(r / "videos" / "b.mp4"))], ingest.Settings())
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


def test_rename_moves_srt_and_updates_projects(root):
    new = videos.rename(root, os.path.join(root, "videos", "a.MP4"), "flight1.mp4")
    assert new == os.path.join(root, "videos", "flight1.MP4")  # extension kept
    assert os.path.exists(os.path.join(root, "videos", "flight1.SRT"))
    assert not os.path.exists(os.path.join(root, "videos", "a.SRT"))
    new_b = videos.rename(root, os.path.join(root, "videos", "b.mp4"), "x")  # used by "site": its plan follows
    assert [s.video for s in ingest.load_plan(os.path.join(root, "site"))[0]] == [new_b]
    with pytest.raises(ValueError):
        videos.rename(root, os.path.join(root, "site", "inside.mp4"), "y")  # project folder
    make_clip(os.path.join(root, "videos", "z.MP4"))
    with pytest.raises(FileExistsError):
        videos.rename(root, os.path.join(root, "videos", "z.MP4"), "flight1")


def test_delete(root):
    gone = videos.delete(root, os.path.join(root, "videos", "a.MP4"))
    assert sorted(os.path.basename(g) for g in gone) == ["a.MP4", "a.SRT"]
    with pytest.raises(ValueError, match="used by project"):
        videos.delete(root, os.path.join(root, "videos", "b.mp4"))
    with pytest.raises(ValueError):
        videos.delete(root, os.path.join(root, "..", "elsewhere.mp4"))
    assert os.path.exists(os.path.join(root, "videos", "b.mp4"))


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
    l0 = videos.listdir(root)  # the library: data/videos/ (the project "site" lives next to it, not inside)
    assert l0["folders"] == [] and [v["name"] for v in l0["videos"]] == ["a.MP4", "b.mp4"]
    assert l0["videos"][1]["used_by"] == ["site"]
    assert os.path.dirname(videos.make_folder(root, "", "probe")) == os.path.join(os.path.realpath(root), "videos")
    videos.delete_folder(root, "probe")
    p = videos.make_folder(root, "", "Flights 2026")
    assert os.path.isdir(p)
    with pytest.raises(FileExistsError):
        videos.make_folder(root, "", "Flights 2026")
    for bad in ("../x", ".hidden", ""):
        with pytest.raises(ValueError):
            videos.make_folder(root, "", bad)
    for bad in ("../site", "..", "../videos/../site"):  # projects and outside the library are not library folders
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
    shutil.move(os.path.join(root, "videos", "b.mp4"), os.path.join(q, "b.mp4"))  # a used clip inside
    ingest.save_plan(os.path.join(root, "site"), [ingest.Source(os.path.join(q, "b.mp4"))], ingest.Settings())
    with pytest.raises(ValueError, match="used by project"):
        videos.delete_folder(root, "keep")
    moved = videos.rename_folder(root, "keep", "kept")  # its used clip moves with it: the plan follows
    assert ingest.load_plan(os.path.join(root, "site"))[0][0].video == os.path.join(moved, "b.mp4")
    r = videos.make_folder(root, "", "old name")
    assert videos.rename_folder(root, "old name", "new name") == os.path.join(os.path.join(os.path.realpath(root), "videos"), "new name")
    with pytest.raises(FileExistsError):
        videos.rename_folder(root, "new name", "kept")
    assert not os.path.exists(r)


def test_videos_tab(app, root):
    """The file manager rows: open / select by name, rename and delete through the context actions."""
    from viser import UploadedFile
    v = app.videos

    def entry(name):
        return next(e for e in v.entries if e["name"] == name)

    assert wait_for(lambda: len(getattr(v, "entries", [])) == 2)  # a.MP4 and b.mp4 in data/videos/
    assert len(v._rows) == 2 and entry("b.mp4")["used_by"] == ["site"]
    v.make_folder("flights")
    assert any(e["name"] == "flights" and e["kind"] == "folder" for e in v.entries) and len(v._rows) == 3
    e = entry("flights")
    v._row_click(e, v._label(e), v._label(e), None)  # click the folder name: open it
    assert v.cwd == "flights" and v.entries == [] and len(v._rows) == 1  # the "empty folder" note
    v._upload(UploadedFile("new.MP4", open(os.path.join(root, "videos", "b.mp4"), "rb").read()))
    assert wait_for(lambda: v.current is not None and v.current["name"] == "new.MP4"), v.g_msg.content
    assert "SRT: none (optional" in v.g_info.content and len(v._rows) == 1
    n = len(app.ingest.sources)
    v.add_to_project()
    assert len(app.ingest.sources) == n + 1
    v.rename_entry(entry("new.MP4"), "renamed")  # a source of the open project (maybe unsaved): it follows
    assert "renamed `new.MP4` to `renamed.MP4`" in v.g_msg.content and v.current["name"] == "renamed.MP4"
    assert os.path.realpath(app.ingest.sources[-1].video) == os.path.realpath(os.path.join(root, "videos", "flights",
                                                                                         "renamed.MP4"))
    v.move_entry(entry("renamed.MP4"), "")  # ➜ up to the library root: the open project follows again
    assert "moved `renamed.MP4`" in v.g_msg.content and v.current is None
    moved = os.path.join(root, "videos", "renamed.MP4")
    assert os.path.exists(moved) and os.path.realpath(app.ingest.sources[-1].video) == os.path.realpath(moved)
    assert os.path.realpath(ingest.load_plan(app.path)[0][-1].video) == os.path.realpath(moved)  # and on disk
    v.move_entry({"kind": "video", "path": moved, "name": "renamed.MP4", "srt": None}, "flights")  # and back
    app.ingest.sources.pop()
    ingest.save_plan(app.path, app.ingest.sources, app.ingest.settings)
    v._refresh()
    v.current = entry("renamed.MP4")
    sent = []

    class Client:
        def send_file_download(self, name, data):
            sent.append(name)

    v.download(Client())
    assert wait_for(lambda: sent) and sent == ["renamed.MP4"]
    v.delete_entry(entry("renamed.MP4"))
    assert not os.path.exists(os.path.join(root, "videos", "flights", "renamed.MP4")) and v.current is None
    v.up()
    v.rename_entry(entry("flights"), "flights 2026")
    assert os.path.isdir(os.path.join(root, "videos", "flights 2026"))
    v.delete_entry(entry("flights 2026"))
    assert not os.path.exists(os.path.join(root, "videos", "flights 2026")) and len(v._rows) == 2


    class Browser:  # the dialogs are viser modals; the server-wide GUI builds the same widgets headless
        gui = app.server.gui

    v.ask_new_folder(Browser())
    v.ask_rename(entry("b.mp4"), Browser())
    v.ask_delete(entry("a.MP4"), Browser())  # a clip (this crashed before: folder-only field)
    v.make_folder("empty")
    v.ask_delete(entry("empty"), Browser())
    v.ask_move(entry("a.MP4"), Browser())
    v.ask_delete(entry("b.mp4"), Browser())  # used by site: the dialog explains, nothing is deleted
    assert "used by site" in v.g_msg.content and os.path.exists(os.path.join(root, "videos", "b.mp4"))


def test_move(root):
    site = os.path.join(root, "site")
    poses = os.path.join(site, "poses.json")
    b = os.path.join(root, "videos", "b.mp4")
    json.dump({"sources": [{"video": os.path.relpath(b), "prefix": "b"}], "images": [{"video": "b"}]}, open(poses, "w"))
    dst = videos.make_folder(root, "", "flights")
    assert dst == os.path.join(os.path.realpath(root), "videos", "flights")
    new = videos.move(root, b, "flights")  # used by site
    assert new == os.path.join(dst, "b.mp4") and os.path.exists(new) and not os.path.exists(b)
    assert os.path.realpath(ingest.load_plan(site)[0][0].video) == os.path.realpath(new)
    p = json.load(open(poses))
    assert os.path.realpath(p["sources"][0]["video"]) == os.path.realpath(new) and p["images"] == [{"video": "b"}]
    assert not os.path.isabs(p["sources"][0]["video"])  # stored the way it was (relative)
    a = videos.move(root, os.path.join(root, "videos", "a.MP4"), "flights")  # the .SRT follows
    assert os.path.exists(os.path.join(dst, "a.SRT")) and not os.path.exists(os.path.join(root, "videos", "a.SRT"))
    with pytest.raises(ValueError, match="already there"):
        videos.move(root, a, "flights")
    make_clip(os.path.join(root, "videos", "a.MP4"))
    with pytest.raises(FileExistsError):
        videos.move(root, os.path.join(root, "videos", "a.MP4"), "flights")
    sub = videos.make_folder(root, "flights", "sub")
    with pytest.raises(ValueError, match="into itself"):
        videos.move(root, dst, "flights/sub")
    with pytest.raises(ValueError):
        videos.move(root, new, "../site")  # a project folder is no destination
    other = videos.make_folder(root, "", "archive")
    moved = videos.move(root, dst, "archive")  # a folder, with the used clip inside
    assert moved == os.path.join(other, "flights") and not os.path.exists(sub)
    assert os.path.realpath(ingest.load_plan(site)[0][0].video) == os.path.realpath(os.path.join(moved, "b.mp4"))
    assert videos.all_folders(root) == ["", "archive", "archive/flights", "archive/flights/sub"]


def test_videos_is_not_a_project_name(root):
    from modelforge import projects
    for bad in ("videos", "Videos"):
        with pytest.raises(ValueError, match="video library"):
            projects.create(root, bad)
    p = projects.create(root, "karlak")
    with pytest.raises(ValueError, match="video library"):
        projects.rename(root, p, "videos")
    videos.make_folder(root, "", "karlak")  # a library folder of the same name: no clash any more


# --- ROS 2 bags in the library --------------------------------------------------------------------
def _bag_dir(tmp, name="run1", mcap=False):
    from tests.test_rosbag_kind import write_small_bag
    path = os.path.join(str(tmp), "src", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_small_bag(path, n=6, mcap=mcap)
    return path


def _zip(folder, loose=False):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in os.listdir(folder):
            z.write(os.path.join(folder, n), n if loose else os.path.join(os.path.basename(folder), n))
    return buf.getvalue()


def test_bag_uploads(root, tmp_path):
    import io
    import tarfile
    import zipfile
    src = _bag_dir(tmp_path)
    lib = os.path.join(root, "videos")
    p = videos.save_upload(root, "run1.zip", _zip(src))  # a zipped bag folder
    assert p == os.path.join(lib, "run1") and videos.is_bag(p)
    with pytest.raises(FileExistsError):
        videos.save_upload(root, "run1.zip", _zip(src))
    p = videos.save_upload(root, "loose.zip", _zip(src, loose=True))  # the folder's files at the top
    assert p == os.path.join(lib, "loose") and videos.is_bag(p)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(src, arcname="tarred")
    assert videos.save_upload(root, "x.tar.gz", buf.getvalue(), rel_dir="robots") == os.path.join(lib, "robots",
                                                                                                    "tarred")
    m = _bag_dir(tmp_path, "m", mcap=True)
    with open(os.path.join(m, "m.mcap"), "rb") as f:
        assert videos.is_bag(videos.save_upload(root, "m.mcap", f.read()))  # a single-file bag
    for parts in ("metadata.yaml", "run1.db3"):  # the loose files of a bag folder, one by one
        with open(os.path.join(src, parts), "rb") as f:
            videos.save_upload(root, parts, f.read(), rel_dir="byhand")
    assert videos.is_bag(os.path.join(lib, "byhand"))
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../evil/metadata.yaml", "x")
    with pytest.raises(ValueError):
        videos.save_upload(root, "evil.zip", bad.getvalue())
    nobag = io.BytesIO()
    with zipfile.ZipFile(nobag, "w") as z:
        z.writestr("notes/readme.txt", "x")
    with pytest.raises(ValueError, match="no ROS 2 bag"):
        videos.save_upload(root, "nobag.zip", nobag.getvalue())
    with pytest.raises(ValueError):
        videos.save_upload(root, "other.yaml", b"x")
    assert not os.path.exists(os.path.join(root, "evil")) and not os.path.exists(os.path.join(lib, "notes"))
    assert not [n for n in os.listdir(lib) if n.startswith(".upload")]
    listing = videos.listdir(root)
    bags = {b["name"]: b for b in listing["bags"]}
    assert set(bags) == {"run1", "loose", "m.mcap", "byhand"} and bags["run1"]["folder"] and not bags["m.mcap"]["folder"]
    b = bags["run1"]
    assert b["images"] == ["/cam/image/compressed"] and b["clouds"] == ["/scan"] and b["tf"] and not b["error"]
    assert abs(b["duration_s"] - 1.0) < 1e-6 and b["odometry"] == ["/odom"]
    folders = {f["name"]: f for f in listing["folders"]}
    assert folders["robots"]["videos"] == 1 and "byhand" not in folders
    assert videos.listdir(root, "byhand")["inside_bag"]
    assert videos.listdir(root, "byhand")["bags"] == []  # its own .db3 is not listed as another bag


def test_bag_rename_move_delete_follow_projects(root, tmp_path):
    from modelforge import projects
    lib = os.path.join(root, "videos")
    bag_path = videos.save_upload(root, "run1.zip", _zip(_bag_dir(tmp_path)))
    proj = projects.create(root, "robot", kind="rosbag")
    ingest.save_plan(proj, [ingest.Source(os.path.relpath(bag_path))], ingest.Settings())
    used = lambda: ingest.load_plan(proj)[0][0].video
    with pytest.raises(ValueError, match="used by"):
        videos.delete(root, bag_path)
    new = videos.rename(root, bag_path, "warehouse")
    assert new == os.path.join(lib, "warehouse") and os.path.realpath(used()) == os.path.realpath(new)
    assert not os.path.isabs(used())  # stored the way it was (relative)
    videos.make_folder(root, "", "robots")
    moved = videos.move(root, new, "robots")
    assert moved == os.path.join(lib, "robots", "warehouse") and os.path.realpath(used()) == os.path.realpath(moved)
    with pytest.raises(ValueError, match="used by"):
        videos.delete_folder(root, "robots")
    videos.rename_folder(root, "robots", "fleet")  # the folder holding the bag
    assert os.path.realpath(used()) == os.path.realpath(os.path.join(lib, "fleet", "warehouse"))
    ingest.save_plan(proj, [], ingest.Settings())
    assert videos.delete(root, os.path.join(lib, "fleet", "warehouse")) == [os.path.join(lib, "fleet", "warehouse")]
    data = videos.bag_zip(videos.save_upload(root, "again.zip", _zip(_bag_dir(tmp_path, "again"))))
    import io
    import zipfile
    assert sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == ["again/again.db3", "again/metadata.yaml"]
