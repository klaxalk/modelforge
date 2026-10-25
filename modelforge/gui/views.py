"""Saved camera views (left panel → Views; the user asked, 2026-10-24): the camera of the browser that presses Save
(position, look-at, up, vertical field of view) under a name, per project in PROJECT/views.json (positions in the
project's world frame, so a renamed project keeps them and Clean leaves them); each listed view glides there, can
take the current camera instead, be renamed or deleted."""
import datetime
import json
import os
import threading

import viser

FILE = "views.json"
MAX_ROWS = 200


def path(project, file=FILE):
    return os.path.join(project, file)


def load(project, file=FILE):
    """The project's saved views, in the order saved: [{"name", "position", "look_at", "up", "fov", "saved"}].
    file: another list (the Difix tab's targets, difix.VIEWS_FILE)."""
    try:
        views = json.load(open(path(project, file))).get("views", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [v for v in views if isinstance(v, dict) and "name" in v and "position" in v and "look_at" in v]


def _save(project, views, file=FILE):
    tmp = path(project, file) + ".part"
    json.dump({"views": views}, open(tmp, "w"), indent=1)
    os.replace(tmp, path(project, file))


def _unique(views, name, skip=None):
    names = {v["name"] for v in views if v is not skip}
    if name not in names:
        return name
    k = 2
    while f"{name} ({k})" in names:
        k += 1
    return f"{name} ({k})"


def pose(position, look_at, up=(0.0, 0.0, 1.0), fov=None):
    """A view record's camera part (plain floats, JSON)."""
    out = {"position": [float(x) for x in position], "look_at": [float(x) for x in look_at],
           "up": [float(x) for x in up]}
    if fov is not None:
        out["fov"] = float(fov)
    return out


def add(project, name, cam, file=FILE):
    """Save `cam` (pose()) as a new view; an empty name becomes "View N", a taken one gets " (2)". Returns its name."""
    views = load(project, file)
    name = _unique(views, (name or "").strip() or f"View {len(views) + 1}")
    views.append({"name": name, **cam, "saved": datetime.datetime.now().isoformat(timespec="seconds")})
    _save(project, views, file)
    return name


def update(project, name, cam, file=FILE):
    """The view `name` takes the camera `cam`."""
    views = load(project, file)
    for v in views:
        if v["name"] == name:
            v.update(cam, saved=datetime.datetime.now().isoformat(timespec="seconds"))
            _save(project, views, file)
            return
    raise KeyError(name)


def rename(project, name, new, file=FILE):
    views = load(project, file)
    v = next((v for v in views if v["name"] == name), None)
    if v is None:
        raise KeyError(name)
    new = (new or "").strip()
    if not new:
        raise ValueError("the name is empty")
    v["name"] = _unique(views, new, skip=v)
    _save(project, views, file)
    return v["name"]


def delete(project, name, file=FILE):
    views = load(project, file)
    keep = [v for v in views if v["name"] != name]
    if len(keep) == len(views):
        raise KeyError(name)
    _save(project, keep, file)


class ViewsPanel:
    """The Views tab: Name + Save current view, then one row per saved view: [▶ name][⟳][✏][🗑].
    file / intro / folder: another list (the Difix tab's targets)."""

    def __init__(self, app, gui, file=FILE, intro=None, folder="Saved views"):
        self.app, self.gui, self.file = app, gui, file
        self._rows, self._lock = [], threading.RLock()
        gui.add_markdown(intro or "Save the camera of this browser under a name and come back to it later. Kept per "
                         f"project (`{file}` in the project folder).")
        self.g_name = gui.add_text("Name", initial_value="",
                                   hint="Name of the view to save; empty: View 1, View 2, …")
        b = gui.add_button("Save current view", icon=viser.Icon.DEVICE_FLOPPY,
                           hint="Store where this browser's camera is, where it looks and its field of view.")
        b.on_click(lambda e: self.save(e.client))
        self.g_msg = gui.add_markdown("")
        self.g_list = gui.add_folder(folder)
        self.refresh()

    def _project(self):
        return self.app.path if self.app.path and os.path.isdir(self.app.path) else None

    def save(self, client):
        if client is None or self._project() is None:
            return
        c = client.camera
        name = add(self._project(), self.g_name.value, pose(c.position, c.look_at, c.up_direction, c.fov), self.file)
        self.g_name.value = ""
        self.g_msg.content = f"✅ saved **{name}**"
        self.refresh()

    def go(self, name, client):
        v = next((v for v in load(self._project() or "", self.file) if v["name"] == name), None)
        if v is None or client is None:
            return
        self.app.glide_to(client, v["position"], v["look_at"], up=v.get("up"), fov=v.get("fov"))
        self.g_msg.content = f"▶ **{name}**"

    def refresh(self):
        """Rebuild the list (on start, a project switch and every change)."""
        project = self._project()
        views = load(project, self.file) if project else []
        with self._lock:
            for r in self._rows:
                r.remove()
            self._rows = []
            with self.g_list:
                if not views:
                    self._rows.append(self.gui.add_markdown("_no saved views in this project yet_"))
                for k, v in enumerate(views[:MAX_ROWS]):
                    label = f"▶ {v['name']}"
                    g = self.gui.add_button_group(
                        "", options=(label, "⟳", "✏", "🗑"), order=float(k),
                        hint="▶ go to this view; ⟳ replace it with the current camera; ✏ rename; 🗑 delete.")
                    g.on_click(lambda ev, name=v["name"], label=label: self._click(name, label, ev.target.value,
                                                                                  ev.client))
                    self._rows.append(g)
                if len(views) > MAX_ROWS:
                    self._rows.append(self.gui.add_markdown(f"_… {len(views) - MAX_ROWS} more_"))

    def _click(self, name, label, value, client):
        project = self._project()
        if project is None:
            return
        try:
            if value == label:
                self.go(name, client)
            elif value == "⟳" and client is not None:
                c = client.camera
                update(project, name, pose(c.position, c.look_at, c.up_direction, c.fov), self.file)
                self.g_msg.content = f"✅ **{name}** now shows the current camera"
            elif value == "✏":
                self.ask_rename(name, client)
            elif value == "🗑":
                self.ask_delete(name, client)
        except (KeyError, ValueError) as e:
            self.g_msg.content = f"❌ {e}"
        self.refresh()

    def ask_rename(self, name, client):
        if client is None:
            return
        with client.gui.add_modal(f"Rename {name}") as m:
            new = client.gui.add_text("New name", initial_value=name)
            row = client.gui.add_button_group("", options=("Rename", "Cancel"))

        def done(ev):
            m.close()
            if ev.target.value == "Rename":
                try:
                    self.g_msg.content = f"✅ renamed to **{rename(self._project(), name, new.value, self.file)}**"
                except (KeyError, ValueError) as e:
                    self.g_msg.content = f"❌ {e}"
                self.refresh()
        row.on_click(done)

    def ask_delete(self, name, client):
        if client is None:
            return
        with client.gui.add_modal("Delete?") as m:
            client.gui.add_markdown(f"Delete the saved view **{name}**?")
            row = client.gui.add_button_group("", options=("Delete", "Cancel"))

        def done(ev):
            m.close()
            if ev.target.value == "Delete":
                try:
                    delete(self._project(), name, self.file)
                    self.g_msg.content = f"🗑 deleted **{name}**"
                except KeyError:
                    pass
                self.refresh()
        row.on_click(done)

    def views(self):
        """This list's views (the open project's)."""
        return load(self._project(), self.file) if self._project() else []

    def add_from(self, file):
        """Copy the views of another list (e.g. the Views tab's) into this one; names already here are skipped."""
        project = self._project()
        if project is None:
            return 0
        have = {v["name"] for v in load(project, self.file)}
        n = 0
        for v in load(project, file):
            if v["name"] not in have:
                add(project, v["name"], {k: v[k] for k in ("position", "look_at", "up", "fov") if k in v}, self.file)
                n += 1
        self.g_msg.content = f"✅ {n} views copied" if n else "⚪ nothing new to copy"
        self.refresh()
        return n
