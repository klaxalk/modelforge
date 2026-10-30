"""Streaming, resumable uploads into the video library (the user asked, 2026-10-25: docs/todo_upload.md).

viser's own upload widget keeps a file in server memory until it is complete (a 1.2 GB clip ~2.4 GB of RAM), its
browser side sends every part at once (the browser queues the whole file), and an interrupted upload starts over.
These uploads reuse viser's file transfer messages on the GUI's websocket (no other port), but with their own browser
side (docker/patch_viser_page.py, UPLOADER) and server side (here):

- The browser sends FileTransferStartUpload with source_component_uuid "mf-upload|<target>|<lastModified>" (target:
  a library folder, "videos:<rel>"), then FileTransferPart messages of CHUNK bytes (part i = bytes i x CHUNK ...),
  at most WINDOW bytes beyond what the server has acknowledged; viser's own handlers ignore them (no such widget).
- Every part is written straight to `.<name>.mfupload` in the target folder (a writer thread per upload), and the
  length received without gaps is acknowledged (FileTransferPartAck, to that browser only; the patched page routes
  it to the uploader). Every DURABLE_EVERY bytes the file is fsynced and that length saved next to it
  (`.<name>.mfupload.json`, keyed by name + size + the file's lastModified).
- Resume: a start for a key that has a saved length is answered with that length, and the browser continues from
  there: after a dropped connection on its own, after a reload or a GUI restart when the same file is picked again.
- Complete: renamed into place by videos.finish_upload (archives of bag folders unpacked), as viser's uploads.
- Refused (a name that may not be uploaded, an existing file): acknowledged with -1, the reason in the tab.
- Stopped (the upload box's Stop button): the browser sends no more parts and one FileTransferPart with part_index
  -1; what arrived is saved as for a dropped connection (Videos tab: interrupted uploads, resumed by picking the same
  file again, or 🗑).
"""
import json
import os
import queue
import threading
import time

from viser import _messages

from modelforge import videos

TAG = "mf-upload"
CHUNK = 1 << 20  # bytes per part (the browser uses the same)
DURABLE_EVERY = 64 << 20  # fsync + saved length every this many bytes


def part_paths(folder, name):
    base = os.path.join(folder, f".{name}.mfupload")
    return base, base + ".json"


class Upload:
    """One file arriving: its part file, the length received without gaps, a writer thread."""

    def __init__(self, owner, client_id, transfer, key, name, size, folder, rel_dir):
        self.owner, self.client_id, self.transfer, self.key = owner, client_id, transfer, key
        self.name, self.size, self.folder, self.rel_dir = name, int(size), folder, rel_dir
        self.part, self.info = part_paths(folder, name)
        saved = None
        try:
            saved = json.load(open(self.info))
        except (OSError, ValueError):
            pass
        self.done = 0
        if saved and saved.get("key") == key and os.path.exists(self.part):
            self.done = min(int(saved.get("done", 0)), os.path.getsize(self.part))
        else:  # another file under this name, or nothing yet: start over
            open(self.part, "wb").close()
        self.durable, self.pending = self.done, {}  # offset -> length of parts beyond the gap-free length
        self.f = open(self.part, "r+b")
        self.q, self.started, self.finished, self.error = queue.Queue(), time.time(), False, None
        self.t_last = time.time()
        threading.Thread(target=self._writer, daemon=True).start()

    def _save(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.durable = self.done
        tmp = self.info + ".tmp"
        json.dump({"key": self.key, "name": self.name, "size": self.size, "done": self.done,
                   "rel_dir": self.rel_dir, "time": time.time()}, open(tmp, "w"))
        os.replace(tmp, self.info)

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if item == "stop":  # after the parts queued before it
                self.stop()
                self.owner.stopped(self)
                return
            index, content = item
            off = index * CHUNK
            try:
                if off + len(content) > self.size or (len(content) != CHUNK and off + len(content) != self.size):
                    raise ValueError(f"part {index} does not fit the file ({len(content)} bytes)")
                if off + len(content) > self.done:  # else: sent again after a resume, already here
                    self.f.seek(off)
                    self.f.write(content)
                    self.pending[off] = len(content)
                    while self.done in self.pending:
                        self.done += self.pending.pop(self.done)
                if self.done - self.durable >= DURABLE_EVERY or self.done == self.size:
                    self._save()
                self.t_last = time.time()
                self.owner.ack(self, self.done)
                if self.done == self.size:
                    self.f.close()
                    self.owner.complete(self)
                    return
            except Exception as e:  # noqa: BLE001 - the upload fails, the GUI goes on
                self.error = str(e)
                self.owner.refuse(self, self.error)
                return

    def stop(self):
        """This transfer is superseded (a resume) or the browser is gone: what arrived stays for a resume."""
        self.q.put(None)
        try:
            if not self.f.closed:
                self._save()
                self.f.close()
        except (OSError, ValueError):
            pass


class StreamUploads:
    """The server side; App owns one. on_done(path, upload) / on_error(text, upload) report to the tabs."""

    def __init__(self, app):
        self.app = app
        self.by_transfer, self.by_key = {}, {}
        self.lock = threading.Lock()
        ws = app.server._websock_server
        ws.register_handler(_messages.FileTransferStartUpload, self._start)
        ws.register_handler(_messages.FileTransferPart, self._part)

    # --- messages (viser's event loop: keep them short) -------------------------------------------------------
    def _start(self, client_id, m):
        if not str(m.source_component_uuid).startswith(TAG + "|"):
            return  # one of viser's own upload widgets
        _, target, stamp = (m.source_component_uuid.split("|", 2) + ["", ""])[:3]
        try:
            if not target.startswith("videos:"):
                raise ValueError(f"unknown upload target {target!r}")
            rel_dir = target[len("videos:"):]
            name = videos.upload_name(m.filename)
            folder = videos.resolve_dir(self.app.root, rel_dir, create=True)
            if not name.lower().endswith(videos.ARCHIVE_EXTS) and os.path.lexists(os.path.join(folder, name)):
                raise FileExistsError(f"{os.path.join(rel_dir, name) or name} already exists in the library")
            key = f"{name}|{int(m.size_bytes)}|{stamp}"
            with self.lock:
                old = self.by_key.get((folder, key))
                if old is not None:  # the same file again (a resume): the new transfer takes over
                    self.by_transfer.pop(old.transfer, None)
                    old.stop()
                up = Upload(self, client_id, m.transfer_uuid, key, name, m.size_bytes, folder, rel_dir)
                self.by_transfer[m.transfer_uuid] = up
                self.by_key[(folder, key)] = up
            self.app._on_log(f"upload: {name} ({m.size_bytes / 1e6:.0f} MB) into {target}"
                             + (f", resuming at {up.done / 1e6:.0f} MB" if up.done else ""))
            self.ack(up, up.done)
            if up.done == up.size:  # all here already (a finished upload whose rename failed): finish now
                up.q.put(None)
                up.f.close()
                self.complete(up)
        except Exception as e:  # noqa: BLE001
            self._send(client_id, m.transfer_uuid, -1, -1)
            self.app.videos.g_msg.content = f"❌ upload of `{m.filename}` refused: `{e}`"

    def _part(self, client_id, m):
        up = self.by_transfer.get(m.transfer_uuid)
        if up is None or m.source_component_uuid != TAG:
            return
        if int(m.part_index) < 0:  # the user stopped it: keep what arrived (the writer saves when it gets there)
            self._forget(up)
            up.q.put("stop")
            return
        up.q.put((int(m.part_index), bytes(m.content)))

    # --- to the browser and the tabs ----------------------------------------------------------------------------
    def _send(self, client_id, transfer, done, total):
        msg = _messages.FileTransferPartAck(source_component_uuid=TAG, transfer_uuid=transfer,
                                            transferred_bytes=int(done), total_bytes=int(total))
        client = self.app.server.get_clients().get(client_id)
        if client is not None:
            client._websock_connection.queue_message(msg)

    def ack(self, up, done):
        self._send(up.client_id, up.transfer, done, up.size)

    def stopped(self, up):
        self.app._on_log(f"upload: {up.name} stopped at {up.done / 1e6:.0f} of {up.size / 1e6:.0f} MB (kept: pick the "
                         "same file again to resume, or 🗑 in the Videos tab)")
        self.app.videos.g_msg.content = (f"⏹ upload of `{up.name}` stopped at {up.done / 1e6:.0f} / "
                                         f"{up.size / 1e6:.0f} MB: kept as an interrupted upload")
        try:
            self.app.videos._refresh()  # the interrupted uploads list
        except Exception:  # noqa: BLE001 - only the list
            pass

    def refuse(self, up, why):
        self._send(up.client_id, up.transfer, -1, -1)
        self.app.videos.g_msg.content = f"❌ upload of `{up.name}` failed: `{why}`"
        self._forget(up)

    def _forget(self, up):
        with self.lock:
            if self.by_transfer.get(up.transfer) is up:
                self.by_transfer.pop(up.transfer, None)
            if self.by_key.get((up.folder, up.key)) is up:
                self.by_key.pop((up.folder, up.key), None)

    def complete(self, up):
        self._forget(up)

        def work():
            try:
                path = videos.finish_upload(self.app.root, up.name, up.part, up.rel_dir)
                if os.path.exists(up.info):
                    os.remove(up.info)
                secs = time.time() - up.started
                self.app._on_log(f"upload: {up.name} complete ({up.size / 1e6:.0f} MB in {secs:.0f} s)")
                self.app.videos.uploaded(path, up.name, up.size)
            except (OSError, ValueError) as e:
                self.app.videos.g_msg.content = f"❌ upload of `{up.name}` could not be finished: `{e}` (the data " \
                                                f"is kept in `{os.path.relpath(up.part)}`)"
        threading.Thread(target=work, daemon=True).start()

    def progress(self):
        """'name 412 / 1200 MB (34 %)' per upload arriving (for the tabs' status lines)."""
        with self.lock:
            ups = list(self.by_transfer.values())
        return [f"{u.name} {u.done / 1e6:.0f} / {u.size / 1e6:.0f} MB ({100 * u.done / max(u.size, 1):.0f} %)"
                for u in ups if u.done < u.size and time.time() - u.t_last < 30]


def unfinished(folder):
    """Saved states of interrupted uploads in a library folder: [{"name", "size", "done", "time"}]."""
    out = []
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for n in names:
        if n.startswith(".") and n.endswith(".mfupload.json"):
            try:
                out.append(json.load(open(os.path.join(folder, n))))
            except (OSError, ValueError):
                pass
    return out


def discard(folder, name):
    """Delete an interrupted upload's data."""
    for p in part_paths(folder, name):
        if os.path.exists(p):
            os.remove(p)


def widget(target, label="＋ Upload video / SRT / ROS bag", hint=""):
    """HTML for the tabs: a file picker the patched page's uploader watches (data-mf-upload = the target)."""
    import html
    return (f'<div data-mf-upload-box="1" style="margin:4px 0" title="{html.escape(hint, quote=True)}">'
            f'<label style="display:block;text-align:center;padding:6px 10px;border-radius:6px;cursor:pointer;'
            f'border:1px solid rgba(128,128,128,.5);font-size:0.85em">{label}'
            f'<input type="file" multiple data-mf-upload="{target}" style="display:none"></label>'
            f'<div style="display:flex;align-items:center;gap:6px;margin-top:4px">'
            f'<div class="mf-upload-status" style="font-size:0.8em;opacity:.85;flex:1"></div>'
            f'<button type="button" class="mf-upload-stop" title="Stop uploading (this file and the ones queued after '
            f'it). What arrived is kept: pick the same file again to resume, or delete it in the Videos tab." '
            f'style="display:none;font-size:0.8em;padding:2px 8px;border-radius:5px;cursor:pointer;'
            f'border:1px solid #ff4d4f;color:#ff4d4f;background:transparent">■ Stop</button></div></div>')
