"""Sparse tab → "Exclude photos (splat and after)": the photos the user leaves out of transforms.json without
extracting or running SfM again (modelforge/exclude.py; the user asked, 2026-10-28). Pick the photo shown in Inspect
(a camera clicked) or in the Photo window (Fly), or a time range of a source; Apply exports again."""
import json
import math
import os

import numpy as np
import viser

from modelforge import exclude

ALL = "all sources"
COLOR = (55, 55, 55)  # excluded cameras in the View


class ExcludePanel:
    def __init__(self, app, gui):
        self.app = app
        self.mask = None  # per project image: excluded (the View greys them)
        with gui.add_folder("Exclude photos (splat and after)", expand_by_default=True):
            gui.add_markdown(
                "Leave photos out of everything after SfM (transforms.json: the splat, tiles, Difix, semantic "
                "features, the SfM mesh), without extracting or running SfM again. The SfM model keeps them; the "
                "export triangulates the points again without them. **Apply** runs the export again; then train "
                "the splat again. Excluded cameras show dark grey in the View.")
            self.g_md = gui.add_markdown("")
            self.g_pick = gui.add_button(
                "Exclude / include the selected photo", icon=viser.Icon.PHOTO_OFF,
                hint="The photo of the camera clicked in the 3D view (Inspect tab), else the one in the Photo window "
                     "(Fly). Clicking again includes it again. With more than 2000 cameras the View draws no "
                     "clickable cameras: use a time range, or the Photo window while flying.")
            self.g_pick.on_click(lambda _: self.pick())
            self.g_source = gui.add_dropdown("Source", options=(ALL,), initial_value=ALL,
                                             hint="The time range below covers this source's photos (all sources: "
                                                  "every photo in it, on the project timeline).")
            self.g_range = gui.add_multi_slider(
                "Time range [s]", min=0.0, max=1.0, step=0.1, initial_value=(0.0, 1.0),
                hint="Project time of the photos to exclude (the Inspect tab and the Photo window show a photo's "
                     "time).")
            self.g_range_md = gui.add_markdown("")
            self.g_add = gui.add_button("Exclude this time range", icon=viser.Icon.CLOCK_X,
                                        hint="Exclude every photo of the source inside the range.")
            self.g_add.on_click(lambda _: self.add_range())
            self.g_edit = gui.add_button_group("", options=("Undo last", "Include all"))
            self.g_edit.on_click(lambda e: self.edit(e.target.value))
            self.g_apply = gui.add_button(
                "Apply: export again", icon=viser.Icon.FILE_EXPORT,
                hint="Runs the export (Sparse tab → Run export): transforms.json and sparse_pc.ply without the excluded "
                     "photos, their points triangulated again from the others (sparse/export). Train the splat again "
                     "afterwards (tiles are set up again by themselves).")
            self.g_apply.on_click(lambda _: self.app.run_stage("export"))
        self.g_source.on_update(lambda _: self._source_range())
        self.g_range.on_update(lambda _: self._preview())

    # ------------------------------------------------------------------ state
    def _times(self):
        p = self.app.project
        if p is None or getattr(p, "time", None) is None:
            return None, None
        return np.asarray(p.time, float), np.asarray(p.source)

    def refresh(self):
        """The open project's sources and time span, the excluded set, the status."""
        p = self.app.project
        names = list(p.source_names) if p is not None else []
        opts = (ALL,) + tuple(f"{k + 1}. {n}" for k, n in enumerate(names))
        if tuple(self.g_source.options) != opts:
            self.g_source.options = opts
            self.g_source.value = ALL
        self._source_range()
        self.status()

    def _source(self):
        v = self.g_source.value
        return None if v == ALL else int(v.split(".", 1)[0]) - 1

    def _source_range(self):
        t, src = self._times()
        if t is None:
            return
        k = self._source()
        sel = np.isfinite(t) & ((src == k) if k is not None else True)
        lo, hi = (math.floor(t[sel].min() * 10) / 10, math.ceil(t[sel].max() * 10) / 10) if sel.any() else (0.0, 1.0)
        hi = max(hi, lo + 0.1)
        self.g_range.min, self.g_range.max = float(lo), float(hi)
        self.g_range.value = (float(lo), float(lo))
        self._preview()

    def _in_range(self):
        t, src = self._times()
        if t is None:
            return np.zeros(0, bool)
        lo, hi = self.g_range.value
        k = self._source()
        return np.isfinite(t) & (t >= lo) & (t <= hi) & ((src == k) if k is not None else True)

    def _preview(self):
        n = int(self._in_range().sum())
        lo, hi = self.g_range.value
        self.g_range_md.content = f"→ {n} photo{'s' if n != 1 else ''} in {lo:g}–{hi:g} s"
        self.g_add.disabled = n == 0 or self.app.path is None

    def excluded(self):
        p = self.app.project
        if p is None or self.app.path is None:
            return set()
        return exclude.names(self.app.path)

    def status(self):
        """Counts, whether the last export applied the current choice, and the View's grey cameras."""
        p, path = self.app.project, self.app.path
        if p is None or path is None:
            self.g_md.content, self.mask = "", None
            return
        entries = exclude.load(path)
        names = exclude.names(path, entries=entries)
        self.mask = np.array([n in names for n in p.names], bool)
        picked = sum(1 for e in entries if "name" in e)
        ranges = [e for e in entries if "t0" in e]
        src = list(p.source_names)
        lines = [f"**{int(self.mask.sum())}** of {len(p.names)} photos excluded"
                 + (f" ({picked} picked, {len(ranges)} time range{'s' if len(ranges) != 1 else ''})" if entries else "")]
        for e in ranges[-6:]:
            s = ALL if e.get("source") is None else (src[e["source"]] if e["source"] < len(src) else e["source"])
            lines.append(f"· {e['t0']:g}–{e['t1']:g} s, {s}")
        try:
            rep = json.load(open(os.path.join(path, "report.json"))).get("export") or {}
        except (OSError, ValueError):
            rep = {}
        done = (rep.get("excluded") or {}).get("entries", []) if rep else None
        if rep and done == entries:
            if entries:
                lines.append("✅ applied at the last export")
        elif rep or entries:
            lines.append("⚠ **changed since the last export: Apply**" if rep else "not exported yet")
        self.g_md.content = "  \n".join(lines)
        self.app._apply_colors()

    # ---------------------------------------------------------------- actions
    def _target(self):
        """The photo to toggle: the Inspect selection, else the Photo window's."""
        i = self.app.selected
        if i is None:
            i = getattr(self.app, "_photo_shown", None)
        return i

    def pick(self, i=None):
        i = self._target() if i is None else i
        if i is None or self.app.project is None or i >= len(self.app.project.names):
            self.g_md.content = "❌ no photo selected: click a camera in the 3D view (or fly to one)  \n" + \
                                self.g_md.content
            return None
        now = exclude.toggle(self.app.path, self.app.project.names[i])
        self.status()
        if self.app.selected == i:
            self.app.select(i)  # the Inspect text says excluded / not
        return now

    def add_range(self):
        if self.app.path is None or not self._in_range().any():
            return
        lo, hi = self.g_range.value
        exclude.add_range(self.app.path, lo, hi, self._source())
        self.status()

    def edit(self, what):
        if self.app.path is None:
            return
        if what == "Undo last":
            exclude.undo(self.app.path)
        elif what == "Include all":
            exclude.clear(self.app.path)
        self.status()
