"""Project file manager back end (modelforge.files) and the GUI Files tab / video upload."""

import io
import os
import socket
import time
import zipfile

import pytest

from modelforge import files


def make_project(tmp_path):
    p = tmp_path / "proj"
    (p / "dense" / "stereo").mkdir(parents=True)
    (p / "ingest.json").write_text('{"sources": [], "settings": {}}')
    (p / "dense" / "fused.ply").write_bytes(b"x" * 1000)
    (p / "dense" / "stereo" / "a.bin").write_bytes(b"y" * 10)
    (tmp_path / "outside.txt").write_text("keep")
    os.symlink(tmp_path / "outside.txt", p / "link.txt")
    return str(p)


def test_listing_and_sizes(tmp_path):
    p = make_project(tmp_path)
    e = files.listdir(p)
    assert [x["name"] for x in e] == ["dense", "ingest.json", "link.txt"]  # folders first
    assert e[0]["dir"] and e[0]["bytes"] == 1010 and e[2]["link"]
    assert [x["rel"] for x in files.listdir(p, "dense")] == ["dense/stereo", "dense/fused.ply"]


@pytest.mark.parametrize("rel", ["..", "../outside.txt", "/etc", "dense/../../outside.txt"])
def test_paths_stay_inside(tmp_path, rel):
    p = make_project(tmp_path)
    with pytest.raises(ValueError):
        files.resolve(p, rel)


def test_delete(tmp_path):
    p = make_project(tmp_path)
    files.delete(p, "link.txt")  # the link, not its target
    assert (tmp_path / "outside.txt").exists() and not os.path.lexists(os.path.join(p, "link.txt"))
    files.delete(p, "dense")
    assert not os.path.exists(os.path.join(p, "dense"))
    with pytest.raises(ValueError):
        files.delete(p, "")
    with pytest.raises(ValueError):
        files.delete(p, "..")


def test_uploads(tmp_path):
    p = make_project(tmp_path)
    path = files.save_video_upload(p, "clip.MP4", b"video")
    assert path == os.path.join(p, "videos", "clip.MP4") and open(path, "rb").read() == b"video"
    files.save_video_upload(p, "C:\\Users\\me\\clip.SRT", b"srt")  # browser paths reduced to the name
    assert os.path.exists(os.path.join(p, "videos", "clip.SRT"))
    with pytest.raises(FileExistsError):
        files.save_video_upload(p, "clip.MP4", b"other")
    with pytest.raises(ValueError):
        files.save_video_upload(p, "notes.txt", b"")
    for bad in ("", ".hidden", "../x.mp4"):
        if bad == "../x.mp4":
            assert files.safe_name(bad) == "x.mp4"
        else:
            with pytest.raises(ValueError):
                files.safe_name(bad)
    assert files.save_upload(p, "dense", "n.txt", b"1") == os.path.join(p, "dense", "n.txt")
    assert not [f for f in os.listdir(os.path.join(p, "videos")) if ".part" in f]


def test_download_payload(tmp_path):
    p = make_project(tmp_path)
    assert files.download_payload(p, "dense/fused.ply") == ("fused.ply", b"x" * 1000)
    name, data = files.download_payload(p, "dense")
    assert name == "dense.zip"
    assert sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == ["dense/fused.ply", "dense/stereo/a.bin"]


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def app(tmp_path):
    from modelforge.gui.app import App
    a = App(make_project(tmp_path), port=free_port())
    yield a
    a.server.stop()


def wait_for(cond, timeout=10):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.05)
    return cond()


def test_files_tab(app):
    from viser import UploadedFile
    f = app.files
    assert any(k.startswith("📁 dense/") for k in f.g_entry.options)
    f.g_entry.value = next(k for k in f.g_entry.options if k.startswith("📁 dense/"))
    f.open_selected()
    assert f.cwd == "dense" and "fused.ply" in f.g_list.content
    sent = []

    class Client:
        def send_file_download(self, name, data):
            sent.append((name, len(data)))

    f.g_entry.value = next(k for k in f.g_entry.options if "fused.ply" in k)
    f.download(Client())
    assert wait_for(lambda: sent) and sent[0] == ("fused.ply", 1000)
    f._upload(UploadedFile("new.txt", b"abc"))
    assert wait_for(lambda: os.path.exists(os.path.join(app.path, "dense", "new.txt")))
    f.g_entry.value = next(k for k in f.g_entry.options if "fused.ply" in k)
    f.delete()  # no confirmation: nothing happens
    assert os.path.exists(os.path.join(app.path, "dense", "fused.ply")) and "confirmation" in f.g_msg.content
    f.g_confirm.value = True
    f.delete()
    assert not os.path.exists(os.path.join(app.path, "dense", "fused.ply")) and not f.g_confirm.value
    f.up()
    assert f.cwd == ""


def test_ingest_video_upload(app):
    from viser import UploadedFile
    app.ingest._on_upload(UploadedFile("flight.MP4", b"\x00" * 100))
    assert wait_for(lambda: "uploaded" in app.ingest.g_upload_md.content), app.ingest.g_upload_md.content
    rel = os.path.relpath(os.path.join(app.path, "videos", "flight.MP4"))
    assert rel in app.ingest.g_video.options and app.ingest.g_video.value == rel
    assert "upload it too" in app.ingest.g_upload_md.content  # no .SRT yet
    app.ingest._on_upload(UploadedFile("flight.txt", b""))
    assert wait_for(lambda: "failed" in app.ingest.g_upload_md.content)
