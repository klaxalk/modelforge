"""Streaming, resumable uploads (modelforge/gui/upload.py) with a stand-in browser: parts out of order, a dropped
transfer resumed, a server restart resumed from the saved length, a refused name, a bag archive unpacked."""
import io
import os
import tarfile
import time
from types import SimpleNamespace

import numpy as np

from modelforge.gui import upload


class Fake:
    """An App with what StreamUploads uses; acks collects what the browser would receive."""

    def __init__(self, root):
        self.root, self.handlers, self.acks, self.done, self.logs = str(root), {}, [], [], []
        conn = SimpleNamespace(queue_message=self.acks.append)
        self.server = SimpleNamespace(
            _websock_server=SimpleNamespace(register_handler=lambda cls, cb: self.handlers.setdefault(cls.__name__, cb)),
            get_clients=lambda: {7: SimpleNamespace(_websock_connection=conn)})
        self.videos = SimpleNamespace(g_msg=SimpleNamespace(content=""),
                                      uploaded=lambda path, name, size: self.done.append((path, name, size)))
        self._on_log = self.logs.append

    def start(self, transfer, name, size, stamp="1700000000", target="videos:"):
        self.handlers["FileTransferStartUpload"](7, SimpleNamespace(
            source_component_uuid=f"mf-upload|{target}|{stamp}", transfer_uuid=transfer, filename=name,
            mime_type="", part_count=-(-size // upload.CHUNK), size_bytes=size))

    def part(self, transfer, data, i):
        self.handlers["FileTransferPart"](7, SimpleNamespace(
            source_component_uuid="mf-upload", transfer_uuid=transfer, part_index=i,
            content=data[i * upload.CHUNK:(i + 1) * upload.CHUNK]))

    def last(self, transfer, until=None, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            got = [a.transferred_bytes for a in self.acks if a.transfer_uuid == transfer]
            if got and (until is None or got[-1] == until):
                return got[-1]
            time.sleep(0.02)
        return got[-1] if got else None


def test_stream_resume_and_restart(tmp_path):
    data = np.random.default_rng(0).integers(0, 256, int(2.5 * upload.CHUNK), dtype=np.uint8).tobytes()
    app = Fake(tmp_path)
    ups = upload.StreamUploads(app)
    app.start("A", "clip.MP4", len(data))
    assert app.last("A") == 0
    app.part("A", data, 0)
    app.part("A", data, 2)  # out of order: written, but the gap-free length stays at part 0
    assert app.last("A", upload.CHUNK) == upload.CHUNK
    time.sleep(0.2)
    assert app.last("A") == upload.CHUNK and ups.progress()
    # the connection drops; the browser starts again with the same file: the server answers what it has
    app.start("B", "clip.MP4", len(data))
    assert app.last("B") == upload.CHUNK
    # the GUI restarts meanwhile: a new server resumes from the saved length
    app2 = Fake(tmp_path)
    upload.StreamUploads(app2)
    app2.start("C", "clip.MP4", len(data))
    assert app2.last("C") == upload.CHUNK
    app2.part("C", data, 1)
    app2.part("C", data, 2)
    assert app2.last("C", len(data)) == len(data)
    t0 = time.time()
    while not app2.done and time.time() - t0 < 5:
        time.sleep(0.02)
    path = os.path.join(tmp_path, "videos", "clip.MP4")
    assert app2.done and app2.done[0][0] == path and open(path, "rb").read() == data
    assert not upload.unfinished(os.path.join(tmp_path, "videos"))  # the part and its state are gone
    # an existing file and a name that may not be uploaded are refused
    app2.start("D", "clip.MP4", 10)
    app2.start("E", "notes.txt", 10)
    assert app2.last("D") == -1 and app2.last("E") == -1 and "refused" in app2.videos.g_msg.content


def test_interrupted_listed_and_discarded(tmp_path):
    data = bytes(3 * upload.CHUNK)
    app = Fake(tmp_path)
    ups = upload.StreamUploads(app)
    app.start("A", "big.MP4", len(data))
    app.part("A", data, 0)
    assert app.last("A", upload.CHUNK) == upload.CHUNK
    ups.by_transfer["A"].stop()
    folder = os.path.join(tmp_path, "videos")
    st = upload.unfinished(folder)
    assert len(st) == 1 and st[0]["name"] == "big.MP4" and st[0]["done"] == upload.CHUNK
    upload.discard(folder, "big.MP4")
    assert not upload.unfinished(folder) and not [n for n in os.listdir(folder) if "mfupload" in n]


def test_bag_archive_unpacked(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in (("metadata.yaml", b"rosbag2_bagfile_information: {}\n"), ("bag_0.db3", b"x" * 100)):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            t.addfile(info, io.BytesIO(body))
    data = buf.getvalue()
    app = Fake(tmp_path)
    upload.StreamUploads(app)
    app.start("A", "flight.tar.gz", len(data), target="videos:bags")
    app.part("A", data, 0)
    t0 = time.time()
    while not app.done and time.time() - t0 < 5:
        time.sleep(0.02)
    assert app.done, app.videos.g_msg.content
    assert app.done[0][0] == os.path.join(tmp_path, "videos", "bags", "flight")
    assert sorted(os.listdir(app.done[0][0])) == ["bag_0.db3", "metadata.yaml"]
