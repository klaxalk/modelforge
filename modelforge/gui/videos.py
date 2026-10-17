"""GUI Videos tab: the shared video library (back end: modelforge.videos). Upload clips and .SRT
sidecars from the browser, see what each clip is and which projects use it, add one to the open
project, download, rename or delete it (unused clips only)."""

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
        self._entries = {}
        self.g_head = gui.add_markdown("")
        self.g_upload = gui.add_upload_button(
            "Upload video / SRT", icon=viser.Icon.UPLOAD,
            mime_type="video/mp4,video/quicktime,.mp4,.MP4,.mov,.MOV,.srt,.SRT",
            hint="Upload a DJI clip (.MP4 / .MOV) or its .SRT sidecar from this computer into data/videos/. "
                 "Upload the .SRT too, with the same name: it carries per-frame metadata. Large clips take a "
                 "while; progress shows below. An existing file is not overwritten.")
        self.g_upload.on_upload(lambda e: self._upload(e.target.value))
        self.g_video = gui.add_dropdown("Video", options=("(none)",),
                                        hint="A clip of the library; its details are shown below.")
        self.g_video.on_update(lambda _: self.show_selected())
        self.g_info = gui.add_markdown("")
        b = gui.add_button("Add to open project", icon=viser.Icon.PLUS,
                           hint="Add the whole clip as a new source of the open project (Ingest tab: narrow its "
                                "range, then extract).")
        b.on_click(lambda _: self.add_to_project())
        row = gui.add_button_group("", options=("Download", "Refresh"),
                                   hint="Download: send the clip to this browser (the whole file is read into "
                                        "memory first). Refresh: rescan the library.")
        row.on_click(lambda e: self.download(e.client) if e.target.value == "Download" else self.refresh())
        with gui.add_folder("Rename", expand_by_default=False):
            self.g_new_name = gui.add_text("New name", initial_value="",
                                           hint="New file name; the extension is kept and the .SRT is renamed too.")
            gui.add_button("Rename", icon=viser.Icon.PENCIL,
                           hint="Only for clips no project uses (their ingest.json points at the old name).").on_click(
                lambda _: self.rename())
        with gui.add_folder("Delete", expand_by_default=False):
            self.g_confirm = gui.add_checkbox("Yes, delete the selected clip and its .SRT", initial_value=False,
                                              hint="Deletes the files for good. Clips used by a project are refused.")
            gui.add_button("Delete", icon=viser.Icon.TRASH, color="red",
                           hint="Needs the confirmation box.").on_click(lambda _: self.delete())
        self.g_msg = gui.add_markdown("")
        self.g_list = gui.add_markdown("")
        self.refresh()

    def selected(self):
        return self._entries.get(self.g_video.value)

    def refresh(self, select=None):
        threading.Thread(target=self._refresh, args=(select,), daemon=True).start()

    def _refresh(self, select=None):
        try:
            entries = videos.list_videos(self.app.root)
        except Exception as e:
            traceback.print_exc()
            self.g_msg.content = f"❌ cannot read the library: `{e}`"
            return
        self._entries = {}
        for e in entries:
            label = f"{e['rel']} · {projects.human(e['bytes'])} · {_dur(e['duration_s'])}"
            self._entries[label] = e
        prev = self.g_video.value
        self.g_video.options = tuple(self._entries) or ("(none)",)
        pick = next((k for k, e in self._entries.items() if select and os.path.abspath(e["path"]) ==
                     os.path.abspath(select)), prev if prev in self._entries else None)
        if pick:
            self.g_video.value = pick
        total = sum(e["bytes"] for e in entries)
        self.g_head.content = (f"Video library: clips under `{self.app.root}` outside project folders; uploads go to "
                               f"`{videos.library_dir(self.app.root)}`. **{len(entries)} clips**, "
                               f"{projects.human(total)}.")
        rows = ["| clip | size | length | SRT | used by |", "|---|---:|---:|:-:|---|"]
        for e in entries:
            rows.append(f"| {e['rel']} | {projects.human(e['bytes'])} | {_dur(e['duration_s'])} | "
                        f"{'✓' if e['srt'] else '–'} | {', '.join(e['used_by']) or '–'} |")
        self.g_list.content = "\n".join(rows) if entries else "_no clips yet: upload one above_"
        self.show_selected()

    def show_selected(self):
        e = self.selected()
        if not e:
            self.g_info.content = ""
            return
        res = f"{e['width']}×{e['height']}" if e["width"] else "?"
        fps = f" @ {e['fps']:g} fps" if e["fps"] else ""
        lines = [f"**{e['name']}** · `{os.path.relpath(e['path'])}`",
                 f"{projects.human(e['bytes'])} · {_dur(e['duration_s'])} · {res}{fps}" +
                 (f" · recorded {e['created'][:19].replace('T', ' ')}" if e["created"] else ""),
                 "SRT: " + (f"`{os.path.basename(e['srt'])}` ✓" if e["srt"] else
                            "**missing**: upload it (same name) for per-frame metadata"),
                 "used by: " + (", ".join(f"**{p}**" for p in e["used_by"]) if e["used_by"] else "no project")]
        self.g_info.content = "  \n".join(lines)

    def add_to_project(self):
        e = self.selected()
        if not e:
            return
        self.app.ingest.add_source(os.path.relpath(e["path"]))
        self.g_msg.content = f"✅ added `{e['name']}` as a source: Ingest tab to set its range and extract"
        self._refresh()

    def download(self, client):
        e = self.selected()
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

    def _in_open_project(self, e):
        """The open project's sources as edited in the Ingest tab (its ingest.json is saved asynchronously)."""
        if any(os.path.realpath(s.video) == os.path.realpath(e["path"]) for s in self.app.ingest.sources):
            self.g_msg.content = (f"❌ `{e['name']}` is a source of the open project "
                                  f"**{os.path.basename(self.app.path)}**: remove it in the Ingest tab first")
            return True
        return False

    def rename(self):
        e = self.selected()
        if not e or self._in_open_project(e):
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
        if self._in_open_project(e):
            return
        try:
            gone = videos.delete(self.app.root, e["path"])
            self.g_msg.content = "✅ deleted " + ", ".join(f"`{os.path.basename(g)}`" for g in gone)
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"
        self.g_confirm.value = False
        self._refresh()
        self.app.ingest.rescan()

    def _upload(self, up):
        def work():  # off viser's message thread: large files take seconds to write
            try:
                path = videos.save_upload(self.app.root, up.name, up.content)
                self.g_msg.content = f"✅ uploaded `{os.path.relpath(path)}` ({projects.human(len(up.content))})"
                self._refresh(select=path if not path.lower().endswith(".srt") else None)
                self.app.ingest.rescan()
            except (OSError, ValueError) as ex:
                self.g_msg.content = f"❌ upload failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()
