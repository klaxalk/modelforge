"""GUI Videos tab: a file manager over the shared video library (back end: modelforge.videos).

Browse the folders under the data root (project folders and hidden files are left out), create and
delete folders, upload clips and .SRT sidecars into the current folder, rename / delete / download
clips (the .SRT follows), add a clip to the open project, and see which projects use each clip.
Clips used by a project cannot be renamed or deleted, and neither can folders that hold them."""

import os
import threading
import traceback

import viser

from modelforge import projects, videos


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
        self._entries = {}
        self.g_head = gui.add_markdown("")
        self.g_path = gui.add_markdown("")
        self.g_entry = gui.add_dropdown("Entry", options=("(empty folder)",),
                                        hint="A folder, clip or stray .SRT of the current folder; its details "
                                             "are shown below.")
        self.g_entry.on_update(lambda _: self.show_selected())
        row = gui.add_button_group("", options=("Open", "Up", "Refresh"),
                                   hint="Open: enter the selected folder. Up: parent folder. Refresh: rescan.")
        row.on_click(lambda e: {"Open": self.open_selected, "Up": self.up,
                                "Refresh": lambda: self.refresh()}[e.target.value]())
        self.g_info = gui.add_markdown("")
        self.g_upload = gui.add_upload_button(
            "Upload video / SRT here", icon=viser.Icon.UPLOAD,
            mime_type="video/mp4,video/quicktime,.mp4,.MP4,.mov,.MOV,.srt,.SRT",
            hint="Upload a DJI clip (.MP4 / .MOV), and optionally its .SRT sidecar (same name), into the current "
                 "folder. The video carries all the metadata; the .SRT only gives a quicker, firmer telemetry "
                 "alignment. Large clips take a while; progress shows below. An existing file is not overwritten.")
        self.g_upload.on_upload(lambda e: self._upload(e.target.value))
        row = gui.add_button_group("", options=("Add to open project", "Download"),
                                   hint="Add to open project: the whole clip becomes a new source (Ingest tab: "
                                        "narrow its range, then extract). Download: send the clip to this browser "
                                        "(read into memory first).")
        row.on_click(lambda e: self.add_to_project() if e.target.value == "Add to open project"
                     else self.download(e.client))
        with gui.add_folder("New folder", expand_by_default=False):
            self.g_folder_name = gui.add_text("Name", initial_value="",
                                              hint="Letters, digits, spaces and '_ . ( ) -'. Created in the current "
                                                   "folder.")
            gui.add_button("Create folder", icon=viser.Icon.FOLDER_PLUS).on_click(lambda _: self.make_folder())
        with gui.add_folder("Rename", expand_by_default=False):
            self.g_new_name = gui.add_text("New name", initial_value="",
                                           hint="New file name for the selected clip; the extension is kept and the "
                                                ".SRT is renamed too.")
            gui.add_button("Rename", icon=viser.Icon.PENCIL,
                           hint="Only for clips no project uses (their ingest.json points at the old name).").on_click(
                lambda _: self.rename())
        with gui.add_folder("Delete", expand_by_default=False):
            self.g_confirm = gui.add_checkbox("Yes, delete the selected entry", initial_value=False,
                                              hint="A clip goes with its .SRT; a folder with everything in it. There "
                                                   "is no undo. Refused for clips a project uses and for folders "
                                                   "holding them.")
            gui.add_button("Delete", icon=viser.Icon.TRASH, color="red",
                           hint="Needs the confirmation box.").on_click(lambda _: self.delete())
        self.g_msg = gui.add_markdown("")
        self.g_list = gui.add_markdown("")
        self.refresh()

    # ------------------------------------------------------------------ listing
    def selected(self):
        return self._entries.get(self.g_entry.value)

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
        entries = {}
        for f in listing["folders"]:
            entries[f"📁 {f['name']}/ · {f['videos']} clips · {projects.human(f['bytes'])}"] = {"kind": "folder", **f}
        for v in listing["videos"]:
            entries[f"🎞 {v['name']} · {projects.human(v['bytes'])} · {_dur(v['duration_s'])}"] = {"kind": "video", **v}
        for f in listing["files"]:
            entries[f"📄 {f['name']} (no video)"] = {"kind": "srt", **f}
        self._entries = entries
        prev = self.g_entry.value
        self.g_entry.options = tuple(entries) or ("(empty folder)",)
        pick = next((k for k, e in entries.items() if select and os.path.realpath(e.get("path", "")) ==
                     os.path.realpath(select)), prev if prev in entries else None)
        if pick:
            self.g_entry.value = pick
        n_vid = len(listing["videos"]) + sum(f["videos"] for f in listing["folders"])
        self.g_head.content = (f"Video library: the folders and clips under `{root}` (project folders are in the "
                               "Projects tab). The Ingest tab lists every clip in it.")
        self.g_path.content = (f"📂 `{os.path.basename(os.path.realpath(root))}/{self.cwd + '/' if self.cwd else ''}`"
                               f" · {n_vid} clips here and below")
        rows = ["| entry | size | length | SRT | used by |", "|---|---:|---:|:-:|---|"]
        for f in listing["folders"]:
            rows.append(f"| 📁 {f['name']}/ | {projects.human(f['bytes'])} | {f['videos']} clips | | |")
        for v in listing["videos"]:
            rows.append(f"| 🎞 {v['name']} | {projects.human(v['bytes'])} | {_dur(v['duration_s'])} | "
                        f"{'✓' if v['srt'] else '–'} | {', '.join(v['used_by']) or '–'} |")
        for f in listing["files"]:
            rows.append(f"| 📄 {f['name']} (no video) | {projects.human(f['bytes'])} | | | |")
        self.g_list.content = "\n".join(rows) if entries else "_empty folder: upload a clip or create a folder_"
        self.show_selected()

    def show_selected(self):
        e = self.selected()
        if not e:
            self.g_info.content = ""
            return
        if e["kind"] == "folder":
            self.g_info.content = (f"**📁 {e['name']}/** · {e['videos']} clips, {projects.human(e['bytes'])} "
                                   "(with subfolders) · **Open** to enter it")
            return
        if e["kind"] == "srt":
            self.g_info.content = f"**📄 {e['name']}**: an .SRT without a video of the same name; it can be deleted"
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

    def open_selected(self):
        e = self.selected()
        if e and e["kind"] == "folder":
            self.cwd, self.g_msg.content = e["rel"], ""
            self._refresh()
        elif e:
            self.g_msg.content = "⚪ that is a file, not a folder"

    def up(self):
        self.cwd, self.g_msg.content = os.path.dirname(self.cwd), ""
        self._refresh()

    # ------------------------------------------------------------------ actions
    def _video(self):
        e = self.selected()
        if not e or e["kind"] != "video":
            self.g_msg.content = "⚪ select a clip first"
            return None
        return e

    def add_to_project(self):
        e = self._video()
        if not e:
            return
        self.app.ingest.add_source(os.path.relpath(e["path"]))
        self.g_msg.content = f"✅ added `{e['name']}` as a source: Ingest tab to set its range and extract"
        self._refresh()

    def download(self, client):
        e = self._video()
        if not e or client is None:
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

    def make_folder(self):
        try:
            path = videos.make_folder(self.app.root, self.cwd, self.g_folder_name.value)
            self.g_msg.content = f"✅ created `{os.path.basename(path)}/`"
            self.g_folder_name.value = ""
            self._refresh(select=path)
        except (OSError, ValueError) as e:
            self.g_msg.content = f"❌ {e}"

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

    def rename(self):
        e = self._video()
        if not e or self._in_open_project(e["path"]):
            return
        try:
            new = videos.rename(self.app.root, e["path"], self.g_new_name.value)
            self.g_msg.content = f"✅ renamed to `{os.path.basename(new)}`" + (" (and its .SRT)" if e["srt"] else "")
            self.g_new_name.value = ""
            self._refresh(select=new)  # at once, so the selection never points at the old name
            self.app.ingest.rescan()
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"

    def delete(self):
        e = self.selected()
        if not e:
            return
        if not self.g_confirm.value:
            self.g_msg.content = "⚪ tick the confirmation box first"
            return
        path = e.get("path") or os.path.join(os.path.realpath(self.app.root), e["rel"])
        if e["kind"] != "srt" and self._in_open_project(path):
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
        self.g_confirm.value = False
        self._refresh()
        self.app.ingest.rescan()

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
