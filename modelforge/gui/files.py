"""GUI Files tab: browse the open project's folder, download files (folders as .zip), delete with
confirmation, and upload files from the browser into the current folder. Back end: modelforge.files."""

import os
import threading
import traceback

import viser

from modelforge import files, projects

MAX_ROWS = 200


def upload_progress(server):
    """'name 412 / 1200 MB' for uploads still arriving (viser keeps them in memory until complete)."""
    states = getattr(server.gui, "_current_file_upload_states", {}) or {}
    out = []
    for st in list(states.values()):
        total = st.get("total_bytes") or 0
        done = st.get("transferred_bytes") or 0
        if total:
            out.append(f"{st.get('filename', '?')} {done / 1e6:.0f} / {total / 1e6:.0f} MB "
                       f"({100 * done / total:.0f} %)")
    return out


class FilesPanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self.cwd = ""
        self._entries = {}
        self.g_path = gui.add_markdown("")
        self.g_entry = gui.add_dropdown("Entry", options=("-",),
                                        hint="A file or folder of the current folder (sizes include subfolders).")
        row = gui.add_button_group("", options=("Open", "Up", "Refresh"),
                                   hint="Open: enter the selected folder. Up: parent folder. Refresh: re-read.")
        row.on_click(lambda e: {"Open": self.open_selected, "Up": self.up, "Refresh": self.refresh}[e.target.value]())
        b = gui.add_button("Download", icon=viser.Icon.DOWNLOAD,
                           hint=f"Send the selected file to this browser; a folder is sent as a .zip "
                                f"(up to {projects.human(files.ZIP_LIMIT)}).")
        b.on_click(lambda e: self.download(e.client))
        self.g_upload = gui.add_upload_button(
            "Upload file here", icon=viser.Icon.UPLOAD,
            hint="Upload a file from this computer into the current folder (an existing file is not overwritten). "
                 "Videos for ingest: use the Ingest tab's upload, which puts them in videos/.")
        self.g_upload.on_upload(lambda e: self._upload(e.target.value))
        with gui.add_folder("Delete", expand_by_default=False):
            self.g_confirm = gui.add_checkbox("Yes, delete the selected entry", initial_value=False,
                                              hint="Deletes the selected file, or the folder with everything in it. "
                                                   "There is no undo. A symlink is removed, its target kept.")
            b = gui.add_button("Delete", icon=viser.Icon.TRASH, color="red",
                               hint="Needs the confirmation box; not while a job runs.")
            b.on_click(lambda _: self.delete())
        self.g_msg = gui.add_markdown("")
        self.g_list = gui.add_markdown("")
        self.refresh()

    def reset(self):
        """Back to the project root (the GUI switched projects)."""
        self.cwd = ""
        self.refresh()

    def selected(self):
        return self._entries.get(self.g_entry.value)

    def refresh(self):
        project = self.app.path
        try:
            if not os.path.isdir(files.resolve(project, self.cwd)):
                self.cwd = ""
            entries = files.listdir(project, self.cwd) if os.path.isdir(project) else []
        except (OSError, ValueError) as e:
            self.g_msg.content = f"❌ `{e}`"
            entries = []
        self._entries = {}
        for e in entries:
            icon = "📁" if e["dir"] else ("🔗" if e["link"] else "📄")
            label = f"{icon} {e['name']}" + ("/" if e["dir"] else "") + f" · {projects.human(e['bytes'])}"
            self._entries[label] = e
        prev = self.g_entry.value
        self.g_entry.options = tuple(self._entries) or ("(empty folder)",)
        if prev in self._entries:
            self.g_entry.value = prev
        name = os.path.basename(os.path.abspath(project))
        total = sum(e["bytes"] for e in entries)
        self.g_path.content = (f"`{name}/{self.cwd + '/' if self.cwd else ''}` · {len(entries)} entries · "
                               f"{projects.human(total)}")
        rows = ["| name | size | modified |", "|---|---:|---|"]
        for e in entries[:MAX_ROWS]:
            rows.append(f"| {'📁' if e['dir'] else ('🔗' if e['link'] else '📄')} {e['name']}"
                        f"{'/' if e['dir'] else ''} | {projects.human(e['bytes'])} | {files.fmt_time(e['mtime'])} |")
        if len(entries) > MAX_ROWS:
            rows.append(f"| … {len(entries) - MAX_ROWS} more | | |")
        self.g_list.content = "\n".join(rows) if entries else "_empty folder_"

    def open_selected(self):
        e = self.selected()
        if e and e["dir"]:
            self.cwd = e["rel"]
            self.g_msg.content = ""
            self.refresh()
        elif e:
            self.g_msg.content = "⚪ that is a file: use Download"

    def up(self):
        self.cwd = os.path.dirname(self.cwd)
        self.g_msg.content = ""
        self.refresh()

    def download(self, client):
        e = self.selected()
        if not e or client is None:
            return
        self.g_msg.content = f"⏳ preparing `{e['rel']}` …"

        def work():
            try:
                name, data = files.download_payload(self.app.path, e["rel"])
                client.send_file_download(name, data)
                self.g_msg.content = f"✅ sent `{name}` ({projects.human(len(data))})"
            except Exception as ex:
                traceback.print_exc()
                self.g_msg.content = f"❌ download failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()

    def delete(self):
        e = self.selected()
        if not e:
            return
        if self.app.running or self.app.ingest.running:
            self.g_msg.content = "⚪ a job is running: delete after it finishes"
            return
        if not self.g_confirm.value:
            self.g_msg.content = "⚪ tick the confirmation box first"
            return
        try:
            files.delete(self.app.path, e["rel"])
            self.g_msg.content = f"✅ deleted `{e['rel']}`"
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ `{ex}`"
        self.g_confirm.value = False
        self.refresh()
        self.app.after_files_changed()

    def _upload(self, up):
        cwd = self.cwd

        def work():  # off viser's message thread: large files take seconds to write
            try:
                path = files.save_upload(self.app.path, cwd, up.name, up.content)
                rel = os.path.relpath(path, self.app.path)
                self.g_msg.content = f"✅ uploaded `{rel}` ({projects.human(len(up.content))})"
            except (OSError, ValueError) as ex:
                self.g_msg.content = f"❌ upload failed: `{ex}`"
            self.refresh()
        threading.Thread(target=work, daemon=True).start()
