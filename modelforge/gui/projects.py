"""GUI Projects tab: list, open and create projects under the root folder, show each one's
sources, progress and final / intermediate results, and clean intermediates."""

import os
import threading
import traceback

import viser

from modelforge import projects
from modelforge.dji import ingest

KIND_LABELS = {"DJI drone video": "dji", "Video without metadata (phone, ...)": "video"}  # label -> kind
KIND_LABELS_DEFAULT = "DJI drone video"


class ProjectsPanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        gui.add_markdown(f"Projects live in `{app.root}`: one folder each, with its sources, intermediate and "
                         "final results.")
        self.g_current = gui.add_markdown("")
        self.g_list = gui.add_dropdown("Project", options=("-",),
                                       hint="Projects found in the root folder (newest first). Selecting one shows "
                                            "its status below; Open switches the whole GUI to it.")
        row = gui.add_button_group("", options=("Open", "Refresh"),
                                   hint="Open: switch to the selected project (not while a job runs). "
                                        "Refresh: rescan the root folder.")
        row.on_click(lambda e: self.open_selected() if e.target.value == "Open" else self.refresh())
        self.g_list.on_update(lambda _: self.show_status())
        self.g_status = gui.add_markdown("")
        with gui.add_folder("New project"):
            self.g_name = gui.add_text("Name", initial_value="",
                                       hint="Folder name under the root: letters, digits, '_', '-', '.'.")
            self.g_kind = gui.add_dropdown(
                "Kind", options=tuple(KIND_LABELS), initial_value=KIND_LABELS_DEFAULT,
                hint="DJI drone video: the clip's embedded position and attitude are used as pose priors and "
                     "refined. Video without metadata (phone, action camera, ...): the poses come from the "
                     "images alone (incremental SfM), in an arbitrary frame and scale.")
            self.g_copy = gui.add_checkbox("Copy sources from the open project", initial_value=False,
                                           hint="Start from the open project's videos, ranges and sampling settings "
                                                "(only ingest.json is copied, no images or results).")
            b = gui.add_button("Create and open", icon=viser.Icon.FOLDER_PLUS,
                               hint="Create the folder with an ingest.json and switch to it; then add sources in "
                                    "the Ingest tab.")
            b.on_click(lambda _: self.create())
        with gui.add_folder("Rename / delete", expand_by_default=False):
            self.g_new_name = gui.add_text("New name", initial_value="",
                                           hint="New folder name for the project selected above: letters, digits, "
                                                "'_', '-', '.'.")
            gui.add_button("Rename selected", icon=viser.Icon.PENCIL,
                           hint="Rename the selected project's folder. The open project is reopened under the new "
                                "name; not while a job runs in it.").on_click(lambda _: self.rename_selected())
            self.g_del_confirm = gui.add_checkbox("Yes, delete the selected project and everything in it",
                                                  initial_value=False,
                                                  hint="Deletes the folder: images, models, mesh, splat, exports. "
                                                       "There is no undo. Videos in the library are not touched.")
            gui.add_button("Delete selected", icon=viser.Icon.TRASH, color="red",
                           hint="Needs the confirmation box. The open project cannot be deleted: open another one "
                                "first.").on_click(lambda _: self.delete_selected())
        with gui.add_folder("Disk", expand_by_default=False):
            self.g_disk = gui.add_markdown("")
            self.g_confirm = gui.add_checkbox("Yes, delete the intermediates of the open project",
                                              initial_value=False,
                                              hint="Deletes features/matches databases, refinement rounds, the dense "
                                                   "workspace (depth maps), training runs and previous/. Keeps the "
                                                   "images, sources and final results. Re-running a stage then "
                                                   "recomputes what it needs.")
            b = gui.add_button("Clean intermediates", icon=viser.Icon.TRASH,
                               hint="Free disk space; needs the confirmation box above.")
            b.on_click(lambda _: self.clean())
        self.g_msg = gui.add_markdown("")
        self.refresh()

    def refresh(self):
        paths = projects.list_projects(self.app.root)
        current = os.path.abspath(self.app.path)
        if os.path.abspath(current) not in [os.path.abspath(p) for p in paths]:
            paths = [current] + paths  # the open project may not have a marker file yet
        self._paths = {os.path.basename(os.path.normpath(p)): p for p in paths}
        self.g_list.options = tuple(self._paths) or ("-",)
        name = os.path.basename(os.path.normpath(current))
        if name in self._paths:
            self.g_list.value = name
        self.g_current.content = f"open: **{name}**"
        self.show_status()

    def selected_path(self):
        return self._paths.get(self.g_list.value)

    def show_status(self):
        path = self.selected_path()
        if not path:
            self.g_status.content = ""
            return
        threading.Thread(target=self._show_status, args=(path,), daemon=True).start()

    def _show_status(self, path):
        try:
            st = projects.status(path)
        except Exception as e:
            traceback.print_exc()
            self.g_status.content = f"❌ cannot read `{path}`: `{e}`"
            return
        src = st["sources"]
        lines = [f"**{st['name']}** · {ingest.KINDS[st['kind']].split(' (')[0]} · {len(src)} source(s) · "
                 f"{st['images']} images · "
                 f"stages: {', '.join(st['stages']) or 'none yet'}"]
        for k, s in enumerate(src):
            end = f"{s.end:g}" if s.end is not None else "end"
            lines.append(f"- {k + 1}. `{s.video}` [{s.start:g}–{end} s]" + ("" if s.enabled else " (disabled)"))
        if st["final"]:
            lines.append("\n**Final results**")
            lines += [f"- `{e['path']}` {e['desc']} · {projects.human(e['bytes'])}" for e in st["final"]]
        if st["intermediate"]:
            total = sum(e["bytes"] for e in st["intermediate"])
            lines.append(f"\n**Intermediate** · {projects.human(total)}")
            lines += [f"- `{e['path']}` {e['desc']} · {projects.human(e['bytes'])}" for e in st["intermediate"]]
        if path == self.selected_path():
            self.g_status.content = "\n".join(lines)
        if os.path.abspath(path) == os.path.abspath(self.app.path):
            cleanable = sum(e["bytes"] for e in st["intermediate"] if e["path"] not in projects.KEEP_ON_CLEAN)
            self.g_disk.content = f"open project: {projects.human(cleanable)} of intermediates can be deleted"

    def open_selected(self):
        path = self.selected_path()
        if not path:
            return
        try:
            self.app.open_project(path)
            self.g_msg.content = f"✅ opened **{os.path.basename(path)}**"
        except RuntimeError as e:
            self.g_msg.content = f"⚪ {e}"

    def create(self):
        name = self.g_name.value.strip()
        try:
            path = projects.create(self.app.root, name, copy_plan_from=self.app.path if self.g_copy.value else None,
                                   kind=KIND_LABELS[self.g_kind.value])
            self.app.open_project(path)
            self.g_name.value = ""
            self.g_msg.content = f"✅ created and opened **{name}**: add sources in the Ingest tab"
        except (ValueError, FileExistsError, RuntimeError) as e:
            self.g_msg.content = f"❌ {e}"

    def _is_open(self, path):
        return os.path.abspath(path) == os.path.abspath(self.app.path)

    def rename_selected(self):
        path = self.selected_path()
        if not path:
            return
        busy = self.app.running or self.app.ingest.running
        if self._is_open(path) and busy:
            self.g_msg.content = "⚪ a job is running in this project: rename it after the job finishes"
            return
        try:
            new = projects.rename(self.app.root, path, self.g_new_name.value.strip())
        except (ValueError, OSError) as e:
            self.g_msg.content = f"❌ {e}"
            return
        self.g_new_name.value = ""
        name = os.path.basename(new)
        if self._is_open(path):
            self.app.open_project(new)  # reloads everything under the new path
        self.refresh()
        self.g_list.value = name
        self.g_msg.content = f"✅ renamed **{os.path.basename(path)}** to **{name}**"
        self.app.videos.refresh()  # its "used by" column names projects

    def delete_selected(self):
        path = self.selected_path()
        if not path:
            return
        if self._is_open(path):
            self.g_msg.content = "⚪ this is the open project: open another one first, then delete it"
            return
        if not self.g_del_confirm.value:
            self.g_msg.content = "⚪ tick the confirmation box first"
            return
        try:
            freed = projects.delete(self.app.root, path)
            self.g_msg.content = f"✅ deleted **{os.path.basename(path)}** ({projects.human(freed)} freed)"
        except (ValueError, OSError) as e:
            self.g_msg.content = f"❌ {e}"
        self.g_del_confirm.value = False
        self.refresh()
        self.app.videos.refresh()

    def clean(self):
        if self.app.running or self.app.ingest.running:
            self.g_msg.content = "⚪ a job is running"
            return
        if not self.g_confirm.value:
            self.g_msg.content = "⚪ tick the confirmation box first"
            return
        freed = projects.clean(self.app.path)
        self.g_confirm.value = False
        self.g_msg.content = f"✅ deleted {projects.human(freed)} of intermediates from **{os.path.basename(self.app.path)}**"
        self.app.open_project(self.app.path)  # snapshots / dense workspace changed
