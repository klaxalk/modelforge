"""GUI Videos tab: a file manager over the shared video library (back end: modelforge.videos).

A toolbar (Up, New folder, Refresh) and an upload button on top; below, one row per folder / clip /
stray .SRT with its context buttons (✏ rename, 🗑 delete) next to it. Clicking a folder opens it,
clicking a clip shows its details with Add to open project / Download. Rename, delete and new folder
ask in a dialog (viser modal). Project folders and hidden files are left out; clips used by a project
cannot be renamed or deleted, and neither can folders that hold them."""

import os
import threading
import traceback

import viser

from modelforge import projects, videos

MAX_ROWS = 300


def upload_progress(server):
    """'name 412 / 1200 MB (34 %)' for uploads still arriving (viser keeps them in memory until complete)."""
    states = getattr(server.gui, "_current_file_upload_states", {}) or {}
    out = []
    for st in list(states.values()):
        total = st.get("total_bytes") or 0
        done = st.get("transferred_bytes") or 0
        if total:
            out.append(f"{st.get('filename', '?')} {done / 1e6:.0f} / {total / 1e6:.0f} MB "
                       f"({100 * done / total:.0f} %)")
    return out


def _dur(s):
    if not s:
        return "?"
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


class VideosPanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self.cwd = ""  # current folder, relative to the data root
        self.current = None  # the clip whose details are shown
        self._rows, self._lock = [], threading.RLock()
        self.g_head = gui.add_markdown("")
        self.g_path = gui.add_markdown("")
        bar = gui.add_button_group("", options=("⬆ Up", "📁＋ New folder", "⟳ Refresh"),
                                   hint="Up: parent folder. New folder: create one here. Refresh: rescan.")
        bar.on_click(lambda e: {"⬆ Up": self.up, "📁＋ New folder": lambda: self.ask_new_folder(e.client),
                                "⟳ Refresh": lambda: self.refresh()}[e.target.value]())
        self.g_upload = gui.add_upload_button(
            "＋ Upload video / SRT", icon=viser.Icon.PLUS,
            mime_type="video/mp4,video/quicktime,.mp4,.MP4,.mov,.MOV,.srt,.SRT",
            hint="Upload a DJI clip (.MP4 / .MOV), and optionally its .SRT sidecar (same name), into the current "
                 "folder. The video carries all the metadata; the .SRT only gives a quicker, firmer telemetry "
                 "alignment. Large clips take a while; progress shows below. An existing file is not overwritten.")
        self.g_upload.on_upload(lambda e: self._upload(e.target.value))
        self.g_msg = gui.add_markdown("")
        self.g_files = gui.add_folder("Files")
        with gui.add_folder("Selected clip"):
            self.g_info = gui.add_markdown("_click a clip above_")
            row = gui.add_button_group("", options=("Add to open project", "Download"),
                                       hint="Add to open project: the whole clip becomes a new source (Ingest tab: "
                                            "narrow its range, then extract). Download: send the clip to this "
                                            "browser (read into memory first).")
            row.on_click(lambda e: self.add_to_project() if e.target.value == "Add to open project"
                         else self.download(e.client))
        self.refresh()

    # ------------------------------------------------------------------ listing
    def refresh(self, select=None):
        threading.Thread(target=self._refresh, args=(select,), daemon=True).start()

    def _refresh(self, select=None):
        root = self.app.root
        try:
            listing = videos.listdir(root, self.cwd)
        except ValueError:  # the folder went away (deleted / renamed elsewhere): back to the root
            self.cwd = ""
            listing = videos.listdir(root, self.cwd)
        except Exception as e:
            traceback.print_exc()
            self.g_msg.content = f"❌ cannot read the library: `{e}`"
            return
        entries = ([{"kind": "folder", **f} for f in listing["folders"]] +
                   [{"kind": "video", **v} for v in listing["videos"]] +
                   [{"kind": "srt", **f} for f in listing["files"]])
        self.entries = entries
        if select:
            self.current = next((e for e in entries if e["kind"] == "video" and
                                 os.path.realpath(e["path"]) == os.path.realpath(select)), self.current)
        if self.current is not None:  # keep the details in step (renamed, deleted, new usage)
            self.current = next((e for e in entries if e["kind"] == "video" and
                                 os.path.realpath(e["path"]) == os.path.realpath(self.current["path"])), None)
        n_vid = len(listing["videos"]) + sum(f["videos"] for f in listing["folders"])
        self.g_head.content = (f"Video library under `{root}` (project folders are in the Projects tab). "
                               "Click a folder to open it, a clip for its details; ✏ rename, 🗑 delete.")
        self.g_path.content = (f"📂 `{os.path.basename(os.path.realpath(root))}/{self.cwd + '/' if self.cwd else ''}`"
                               f" · {n_vid} clips here and below")
        with self._lock:
            for r in self._rows:
                r.remove()
            self._rows = []
            with self.g_files:
                if not entries:
                    self._rows.append(self.gui.add_markdown("_empty folder: ＋ upload a clip or 📁＋ create a "
                                                            "folder_"))
                for k, e in enumerate(entries[:MAX_ROWS]):
                    self._rows.append(self._row(k, e))
                if len(entries) > MAX_ROWS:
                    self._rows.append(self.gui.add_markdown(f"_… {len(entries) - MAX_ROWS} more_"))
        self.show_current()

    def _label(self, e):
        if e["kind"] == "folder":
            return f"📁 {e['name']}/ · {e['videos']} clips · {projects.human(e['bytes'])}"
        if e["kind"] == "video":
            used = f" · used by {', '.join(e['used_by'])}" if e["used_by"] else ""
            return (f"🎞 {e['name']} · {projects.human(e['bytes'])} · {_dur(e['duration_s'])}"
                    f"{' · SRT' if e['srt'] else ''}{used}")
        return f"📄 {e['name']} (no video)"

    def _row(self, k, e):
        """One entry: its name (click: open / select) and context buttons."""
        name = self._label(e)
        opts = (name, "✏", "🗑") if e["kind"] != "srt" else (name, "🗑")
        hint = {"folder": "Click the name to open the folder. ✏ rename, 🗑 delete (with everything in it).",
                "video": "Click the name for details (add to project, download). ✏ rename (the .SRT follows), "
                         "🗑 delete (the .SRT too).",
                "srt": "An .SRT without a video of the same name. 🗑 delete."}[e["kind"]]
        g = self.gui.add_button_group("", options=opts, hint=hint, order=float(k))
        g.on_click(lambda ev, e=e, name=name: self._row_click(e, ev.target.value, name, ev.client))
        return g

    def _row_click(self, e, value, name, client):
        if value == name:
            if e["kind"] == "folder":
                self.cwd, self.g_msg.content = e["rel"], ""
                self._refresh()
            elif e["kind"] == "video":
                self.current = e
                self.show_current()
        elif value == "✏":
            self.ask_rename(e, client)
        elif value == "🗑":
            self.ask_delete(e, client)

    def show_current(self):
        e = self.current
        if e is None:
            self.g_info.content = "_click a clip above_"
            return
        res = f"{e['width']}×{e['height']}" if e["width"] else "?"
        fps = f" @ {e['fps']:g} fps" if e["fps"] else ""
        lines = [f"**🎞 {e['name']}** · `{os.path.relpath(e['path'])}`",
                 f"{projects.human(e['bytes'])} · {_dur(e['duration_s'])} · {res}{fps}" +
                 (f" · recorded {e['created'][:19].replace('T', ' ')}" if e["created"] else ""),
                 "SRT: " + (f"`{os.path.basename(e['srt'])}` ✓" if e["srt"] else
                            "none (optional: the telemetry alignment is then estimated from the video, about a "
                            "minute once per clip)"),
                 "used by: " + (", ".join(f"**{p}**" for p in e["used_by"]) if e["used_by"] else "no project")]
        self.g_info.content = "  \n".join(lines)

    def up(self):
        self.cwd, self.g_msg.content = os.path.dirname(self.cwd), ""
        self._refresh()

    # ------------------------------------------------------------------ dialogs
    def ask_new_folder(self, client):
        if client is None:
            return
        with client.gui.add_modal("New folder") as m:
            name = client.gui.add_text("Name", initial_value="", hint="Letters, digits, spaces and '_ . ( ) -'.")
            row = client.gui.add_button_group("", options=("Create", "Cancel"))
        row.on_click(lambda ev: (m.close(), self.make_folder(name.value) if ev.target.value == "Create" else None))

    def ask_rename(self, e, client):
        if client is None:
            return
        stem = os.path.splitext(e["name"])[0] if e["kind"] == "video" else e["name"]
        with client.gui.add_modal(f"Rename {e['name']}") as m:
            client.gui.add_markdown("The extension is kept and the .SRT is renamed too." if e["kind"] == "video"
                                    else "Clips inside keep their names.")
            name = client.gui.add_text("New name", initial_value=stem)
            row = client.gui.add_button_group("", options=("Rename", "Cancel"))
        row.on_click(lambda ev: (m.close(), self.rename_entry(e, name.value) if ev.target.value == "Rename" else None))

    def ask_delete(self, e, client):
        if client is None:
            return
        what = {"folder": f"the folder **{e['name']}/** with everything in it ({e['videos']} clips)",
                "video": f"**{e['name']}**" + (" and its .SRT" if e.get("srt") else ""),
                "srt": f"**{e['name']}**"}[e["kind"]]
        with client.gui.add_modal("Delete?") as m:
            client.gui.add_markdown(f"Delete {what}? There is no undo.")
            row = client.gui.add_button_group("", options=("Delete", "Cancel"))
        row.on_click(lambda ev: (m.close(), self.delete_entry(e) if ev.target.value == "Delete" else None))

    # ------------------------------------------------------------------ actions
    def make_folder(self, name):
        try:
            path = videos.make_folder(self.app.root, self.cwd, name)
            self.g_msg.content = f"✅ created `{os.path.basename(path)}/`"
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"
        self._refresh()

    def _in_open_project(self, path):
        """The open project's sources as edited in the Ingest tab (its ingest.json is saved asynchronously)."""
        p = os.path.realpath(path)
        hit = next((s.video for s in self.app.ingest.sources
                    if os.path.realpath(s.video) == p or os.path.realpath(s.video).startswith(p + os.sep)), None)
        if hit:
            self.g_msg.content = (f"❌ `{os.path.basename(hit)}` is a source of the open project "
                                  f"**{os.path.basename(self.app.path)}**: remove it in the Ingest tab first")
            return True
        return False

    def rename_entry(self, e, new_name):
        if e["kind"] == "srt" or self._in_open_project(e["path"]):
            return
        try:
            if e["kind"] == "folder":
                new = videos.rename_folder(self.app.root, e["rel"], new_name)
            else:
                new = videos.rename(self.app.root, e["path"], new_name)
            self.g_msg.content = f"✅ renamed `{e['name']}` to `{os.path.basename(new)}`" + (
                " (and its .SRT)" if e.get("srt") else "")
            self._refresh(select=new if e["kind"] == "video" else None)
            self.app.ingest.rescan()
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"

    def delete_entry(self, e):
        if e["kind"] != "srt" and self._in_open_project(e["path"]):
            return
        try:
            if e["kind"] == "folder":
                n, nbytes = videos.delete_folder(self.app.root, e["rel"])
                self.g_msg.content = f"✅ deleted `{e['name']}/` ({n} files, {projects.human(nbytes)})"
            elif e["kind"] == "video":
                gone = videos.delete(self.app.root, e["path"])
                self.g_msg.content = "✅ deleted " + ", ".join(f"`{os.path.basename(g)}`" for g in gone)
            else:
                videos.delete_file(self.app.root, e["path"])
                self.g_msg.content = f"✅ deleted `{e['name']}`"
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"
        self._refresh()
        self.app.ingest.rescan()

    def add_to_project(self):
        e = self.current
        if e is None:
            self.g_msg.content = "⚪ click a clip first"
            return
        self.app.ingest.add_source(os.path.relpath(e["path"]))
        self.g_msg.content = f"✅ added `{e['name']}` as a source: Ingest tab to set its range and extract"
        self._refresh()

    def download(self, client):
        e = self.current
        if e is None or client is None:
            self.g_msg.content = "⚪ click a clip first"
            return
        self.g_msg.content = f"⏳ reading `{e['name']}` …"

        def work():
            try:
                with open(e["path"], "rb") as f:
                    data = f.read()
                client.send_file_download(e["name"], data)
                self.g_msg.content = f"✅ sending `{e['name']}` ({projects.human(len(data))}) to the browser"
            except Exception as ex:
                traceback.print_exc()
                self.g_msg.content = f"❌ download failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()

    def _upload(self, up):
        cwd = self.cwd

        def work():  # off viser's message thread: large files take seconds to write
            try:
                path = videos.save_upload(self.app.root, up.name, up.content, rel_dir=cwd)
                self.g_msg.content = f"✅ uploaded `{os.path.relpath(path)}` ({projects.human(len(up.content))})"
                self._refresh(select=path if not path.lower().endswith(".srt") else None)
                self.app.ingest.rescan()
            except (OSError, ValueError) as ex:
                self.g_msg.content = f"❌ upload failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()
