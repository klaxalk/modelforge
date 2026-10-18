"""GUI Videos tab: a file manager over the shared video library (back end: modelforge.videos).

A toolbar (Up, New folder, Refresh) and an upload button on top; below, one row per folder / clip /
ROS 2 bag / stray .SRT with its context buttons (✏ rename, 🗑 delete) next to it. Clicking a folder opens it,
clicking a clip shows its details with Add to open project / Download. Rename, delete and new folder
ask in a dialog (viser modal); ➜ moves a clip or folder to another folder. Project folders and hidden
files are left out. Renames and moves keep projects pointed at their clips (videos.update_references);
clips used by a project cannot be deleted, and neither can folders that hold them."""

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
            "＋ Upload video / SRT / ROS 2 bag", icon=viser.Icon.PLUS,
            mime_type="video/mp4,video/quicktime,.mp4,.MP4,.mov,.MOV,.srt,.SRT,.mcap,.zip,.tar,.tgz,.gz,.db3,.yaml",
            hint="Into the current folder: a clip (.MP4 / .MOV), optionally with its .SRT sidecar (same name; the "
                 "video carries all the metadata, the .SRT only gives a quicker, firmer telemetry alignment), or a "
                 "ROS 2 bag: an .mcap file, or a bag folder packed as .zip / .tar / .tar.gz (unpacked here). The "
                 "files of a bag folder (metadata.yaml + .db3 / .mcap) can also be uploaded one by one into a new "
                 "folder. The browser sends one file at a time and the server holds it in memory until it is "
                 "complete (about twice its size); progress shows below. Existing files are not overwritten.")
        self.g_upload.on_upload(lambda e: self._upload(e.target.value))
        self.g_msg = gui.add_markdown("")
        self.g_files = gui.add_folder("Files")
        with gui.add_folder("Selected clip / bag"):
            self.g_info = gui.add_markdown("_click a clip or bag above_")
            row = gui.add_button_group("", options=("Add to open project", "Download"),
                                       hint="Add to open project: the whole clip or bag becomes a new source "
                                            "(Ingest tab: narrow its range, then extract; bags go to ROS 2 bag "
                                            "projects, clips to the others). Download: send it to this browser "
                                            "(read into memory first; a bag folder as a .zip).")
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
                   [{"kind": "bag", **b} for b in listing["bags"]] +
                   [{"kind": "srt", **f} for f in listing["files"]])
        self.entries = entries
        if select:
            self.current = next((e for e in entries if e["kind"] in ("video", "bag") and
                                 os.path.realpath(e["path"]) == os.path.realpath(select)), self.current)
        if self.current is not None:  # keep the details in step (renamed, deleted, new usage)
            self.current = next((e for e in entries if e["kind"] in ("video", "bag") and
                                 os.path.realpath(e["path"]) == os.path.realpath(self.current["path"])), None)
        n_vid = len(listing["videos"]) + len(listing["bags"]) + sum(f["videos"] for f in listing["folders"])
        self.g_head.content = (f"Video library `{videos.library_dir(root)}` (projects are in the Projects tab). "
                               "Click a folder to open it, a clip or 🤖 ROS 2 bag for its details; ✏ rename, 🗑 delete.")
        self.g_path.content = (f"📂 `{videos.LIBRARY_DIR}/{self.cwd + '/' if self.cwd else ''}`"
                               f" · {n_vid} clips / bags here and below")
        with self._lock:
            for r in self._rows:
                r.remove()
            self._rows = []
            with self.g_files:
                if listing.get("inside_bag"):
                    self._rows.append(self.gui.add_markdown("_this folder is a ROS 2 bag (metadata.yaml found): "
                                                            "upload any remaining .db3 / .mcap files here, then "
                                                            "⬆ Up to see the bag_"))
                elif not entries:
                    self._rows.append(self.gui.add_markdown("_empty folder: ＋ upload a clip or a bag, or 📁＋ "
                                                            "create a folder_"))
                for k, e in enumerate(entries[:MAX_ROWS]):
                    self._rows.append(self._row(k, e))
                if len(entries) > MAX_ROWS:
                    self._rows.append(self.gui.add_markdown(f"_… {len(entries) - MAX_ROWS} more_"))
        self.show_current()

    def _label(self, e):
        if e["kind"] == "folder":
            return f"📁 {e['name']}/ · {e['videos']} clips / bags · {projects.human(e['bytes'])}"
        if e["kind"] == "video":
            used = f" · used by {', '.join(e['used_by'])}" if e["used_by"] else ""
            return (f"🎞 {e['name']} · {projects.human(e['bytes'])} · {_dur(e['duration_s'])}"
                    f"{' · SRT' if e['srt'] else ''}{used}")
        if e["kind"] == "bag":
            used = f" · used by {', '.join(e['used_by'])}" if e["used_by"] else ""
            what = "unreadable" if e.get("error") else f"{_dur(e['duration_s'])} · {e['topics']} topics"
            return f"🤖 {e['name']}{'/' if e['folder'] else ''} · {projects.human(e['bytes'])} · {what}{used}"
        return f"📄 {e['name']} (no video)"

    def _row(self, k, e):
        """One entry: its name (click: open / select) and context buttons."""
        name = self._label(e)
        opts = (name, "✏", "➜", "🗑") if e["kind"] != "srt" else (name, "🗑")
        hint = {"folder": "Click the name to open the folder. ✏ rename, ➜ move to another folder, 🗑 delete (with "
                          "everything in it). Projects using clips inside follow a rename or move.",
                "video": "Click the name for details (add to project, download). ✏ rename, ➜ move to another folder "
                         "(the .SRT follows, and so do the projects using it), 🗑 delete (the .SRT too).",
                "bag": "A ROS 2 bag. Click the name for details (topics, add to project, download). ✏ rename, ➜ move "
                       "to another folder (projects using it follow), 🗑 delete.",
                "srt": "An .SRT without a video of the same name. 🗑 delete."}[e["kind"]]
        g = self.gui.add_button_group("", options=opts, hint=hint, order=float(k))
        g.on_click(lambda ev, e=e, name=name: self._row_click(e, ev.target.value, name, ev.client))
        return g

    def _row_click(self, e, value, name, client):
        if value == name:
            if e["kind"] == "folder":
                self.cwd, self.g_msg.content = e["rel"], ""
                self._refresh()
            elif e["kind"] in ("video", "bag"):
                self.current = e
                self.show_current()
        elif value == "✏":
            self.ask_rename(e, client)
        elif value == "➜":
            self.ask_move(e, client)
        elif value == "🗑":
            self.ask_delete(e, client)

    def show_current(self):
        e = self.current
        if e is None:
            self.g_info.content = "_click a clip or bag above_"
            return
        if e["kind"] == "bag":
            self.g_info.content = "  \n".join(self._bag_lines(e))
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

    def _bag_lines(self, e):
        lines = [f"**🤖 {e['name']}** · `{os.path.relpath(e['path'])}` · ROS 2 bag "
                 f"({'folder' if e['folder'] else 'MCAP file'})"]
        if e.get("error"):
            return lines + [f"{projects.human(e['bytes'])} · ❌ cannot read it: `{e['error']}`"]
        def topics(t):
            return ", ".join(f"`{x}`" for x in t) if t else "none"
        lines += [f"{projects.human(e['bytes'])} · {_dur(e['duration_s'])} · {e['topics']} topics" +
                  (f" · recorded {e['created'][:19].replace('T', ' ')} UTC" if e["created"] else ""),
                  f"images: {topics(e['images'])}",
                  f"point clouds: {topics(e['clouds'])}",
                  f"poses: {'TF' if e['tf'] else 'no TF'}" + (f", odometry {topics(e['odometry'])}"
                                                               if e["odometry"] else ""),
                  "used by: " + (", ".join(f"**{p}**" for p in e["used_by"]) if e["used_by"] else "no project")]
        if not e["images"]:
            lines.append("⚠️ no image topic: a ROS 2 bag project needs one")
        return lines

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
        file_like = e["kind"] == "video" or (e["kind"] == "bag" and not e["folder"])
        stem = os.path.splitext(e["name"])[0] if file_like else e["name"]
        note = {"video": "The extension is kept and the .SRT is renamed too; projects using the clip are updated.",
                "bag": "Projects using the bag are updated." + (" The extension is kept." if file_like else ""),
                "folder": "Clips inside keep their names; projects using them are updated."}[e["kind"]]
        with client.gui.add_modal(f"Rename {e['name']}") as m:
            client.gui.add_markdown(note)
            name = client.gui.add_text("New name", initial_value=stem)
            row = client.gui.add_button_group("", options=("Rename", "Cancel"))
        row.on_click(lambda ev: (m.close(), self.rename_entry(e, name.value) if ev.target.value == "Rename" else None))

    def ask_move(self, e, client):
        if client is None:
            return
        here = os.path.dirname(e["rel"]) if e["kind"] == "folder" or e.get("folder") else os.path.relpath(
            os.path.dirname(os.path.realpath(e["path"])), os.path.realpath(videos.library_dir(self.app.root)))
        here = "" if here == "." else here
        own = e["rel"] if e["kind"] == "folder" or e.get("folder") else None
        root = videos.LIBRARY_DIR
        targets = {f"{root}/{d + '/' if d else ''}": d for d in videos.all_folders(self.app.root)
                   if d != here and not (own and (d == own or d.startswith(own + os.sep)))}
        if not targets:
            self.g_msg.content = "⚪ no other folder to move it to: create one first (📁＋ New folder)"
            return
        with client.gui.add_modal(f"Move {e['name']}") as m:
            client.gui.add_markdown("Projects using the moved clips are updated to the new location.")
            dest = client.gui.add_dropdown("To folder", options=tuple(targets))
            row = client.gui.add_button_group("", options=("Move", "Cancel"))
        row.on_click(lambda ev: (m.close(), self.move_entry(e, targets[dest.value]) if ev.target.value == "Move"
                                 else None))

    def ask_delete(self, e, client):
        if client is None:
            return
        if e["kind"] != "srt":  # refused anyway: say why in the dialog, not only in the status line
            users = sorted(set(self._users(e)))
            p = os.path.realpath(e["path"])
            if any(os.path.realpath(s.video) == p or os.path.realpath(s.video).startswith(p + os.sep)
                   for s in self.app.ingest.sources):
                users = sorted(set(users) | {os.path.basename(self.app.path)})
            if users:
                what = {"folder": "something in it is", "bag": "the bag is", "video": "the clip is"}[e["kind"]]
                names = ", ".join(f"**{u}**" for u in users)
                with client.gui.add_modal(f"Cannot delete {e['name']}") as m:
                    client.gui.add_markdown(
                        f"It cannot be deleted: {what} a source of the project(s) {names}.  \n"
                        "Remove it from the sources first (open the project → Ingest tab → **Remove** on the "
                        "source), or delete the project (Projects tab). Extracted images and results stay with "
                        "the project; only re-extraction needs the source.")
                    client.gui.add_button("OK").on_click(lambda _: m.close())
                self.g_msg.content = f"❌ `{e['name']}` is used by {', '.join(users)}: not deleted"
                return
        if e["kind"] == "folder":  # built per kind: only folders carry a clip count
            what = f"the folder **{e['name']}/** with everything in it ({e['videos']} clips / bags)"
        elif e["kind"] == "bag":
            what = f"the ROS 2 bag **{e['name']}** ({projects.human(e['bytes'])})"
        else:
            what = f"**{e['name']}**" + (" and its .SRT" if e.get("srt") else "")
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

    def _busy(self):
        if self.app.running or self.app.ingest.running:
            self.g_msg.content = "⚪ a job is running (it may be reading this clip): wait for it to finish"
            return True
        return False

    def _retarget_open_project(self, old, new):
        """The open project's Ingest sources (maybe not saved yet) follow a moved / renamed clip or folder."""
        old, new = os.path.abspath(old), os.path.abspath(new)
        hit = False
        for src in self.app.ingest.sources:
            a = os.path.abspath(src.video)
            if a == old or a.startswith(old + os.sep):
                src.video = videos._stored_like(src.video, new + a[len(old):])
                hit = True
        if hit:
            from modelforge.dji import ingest
            ingest.save_plan(self.app.path, self.app.ingest.sources, self.app.ingest.settings)
            self.app.ingest.request_plan()
        self.app.ingest.rescan()

    def _changed_note(self, before):
        """' · projects updated: a, b' for the projects whose paths a rename / move rewrote."""
        users = sorted(set(before))
        return f" · projects updated: {', '.join(users)}" if users else ""

    def _users(self, e):
        """Projects using the entry (a clip, or any clip inside a folder)."""
        used, base = videos.usage(self.app.root), os.path.realpath(e["path"])
        if e["kind"] in ("video", "bag"):
            return used.get(base, [])
        return [p for k, ps in used.items() if k.startswith(base + os.sep) for p in ps]

    def rename_entry(self, e, new_name):
        if e["kind"] == "srt" or self._busy():
            return
        users = self._users(e)
        try:
            if e["kind"] == "folder":
                new = videos.rename_folder(self.app.root, e["rel"], new_name)
            else:
                new = videos.rename(self.app.root, e["path"], new_name)
            self._retarget_open_project(e["path"], new)
            self.g_msg.content = f"✅ renamed `{e['name']}` to `{os.path.basename(new)}`" + (
                " (and its .SRT)" if e.get("srt") else "") + self._changed_note(users)
            self._refresh(select=new if e["kind"] in ("video", "bag") else None)
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"

    def move_entry(self, e, dest_rel):
        if e["kind"] == "srt" or self._busy():
            return
        users = self._users(e)
        try:
            new = videos.move(self.app.root, e["path"], dest_rel)
            self._retarget_open_project(e["path"], new)
            where = f"{videos.LIBRARY_DIR}/{dest_rel + '/' if dest_rel else ''}"
            self.g_msg.content = (f"✅ moved `{e['name']}` to `{where}`" + (" (and its .SRT)" if e.get("srt") else "")
                                  + self._changed_note(users))
            if self.current is not None and os.path.realpath(self.current["path"]) == os.path.realpath(e["path"]):
                self.current = None
            self._refresh()
        except (OSError, ValueError) as ex:
            self.g_msg.content = f"❌ {ex}"

    def delete_entry(self, e):
        if e["kind"] != "srt" and self._in_open_project(e["path"]):
            return
        try:
            if e["kind"] == "folder":
                n, nbytes = videos.delete_folder(self.app.root, e["rel"])
                self.g_msg.content = f"✅ deleted `{e['name']}/` ({n} files, {projects.human(nbytes)})"
            elif e["kind"] in ("video", "bag"):
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
            self.g_msg.content = "⚪ click a clip or bag first"
            return
        bag, kind = e["kind"] == "bag", self.app.ingest.kind
        if bag != (kind == "rosbag"):
            self.g_msg.content = ("❌ the open project is a ROS 2 bag project: add a bag" if kind == "rosbag" else
                                  "❌ a ROS 2 bag needs a ROS 2 bag project (Projects tab → Create, kind ROS 2 bag)")
            return
        self.app.ingest.add_source(os.path.relpath(e["path"]))
        self.g_msg.content = f"✅ added `{e['name']}` as a source: Ingest tab to set its range and extract"
        self._refresh()

    def download(self, client):
        e = self.current
        if e is None or client is None:
            self.g_msg.content = "⚪ click a clip or bag first"
            return
        self.g_msg.content = f"⏳ reading `{e['name']}` …"

        def work():
            try:
                if os.path.isdir(e["path"]):  # a bag folder
                    data, name = videos.bag_zip(e["path"]), e["name"] + ".zip"
                else:
                    with open(e["path"], "rb") as f:
                        data = f.read()
                    name = e["name"]
                client.send_file_download(name, data)
                self.g_msg.content = f"✅ sending `{name}` ({projects.human(len(data))}) to the browser"
            except Exception as ex:
                traceback.print_exc()
                self.g_msg.content = f"❌ download failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()

    def _upload(self, up):
        cwd = self.cwd

        def work():  # off viser's message thread: large files take seconds to write
            try:
                path = videos.save_upload(self.app.root, up.name, up.content, rel_dir=cwd)
                note = ""
                if up.name.lower().endswith(videos.ARCHIVE_EXTS):
                    note = " · unpacked: ROS 2 bag"
                elif up.name.lower() == "metadata.yaml" or up.name.lower().endswith(".db3"):
                    folder = os.path.dirname(path)
                    note = (" · this folder is now a ROS 2 bag: ⬆ Up to see it" if videos.is_bag(folder) else
                            " · part of a bag folder: also upload metadata.yaml and every .db3 / .mcap file here")
                    path = folder
                self.g_msg.content = (f"✅ uploaded `{os.path.relpath(path)}` "
                                      f"({projects.human(len(up.content))}){note}")
                self._refresh(select=path if not path.lower().endswith(".srt") else None)
                self.app.ingest.rescan()
            except (OSError, ValueError) as ex:
                self.g_msg.content = f"❌ upload failed: `{ex}`"
        threading.Thread(target=work, daemon=True).start()
