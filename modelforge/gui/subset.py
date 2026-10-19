"""GUI Sparse tab → "Images used": run the pipeline on a subset of the extracted images (every N-th, a
time window) without extracting again. projects.apply_subsample does the work; the results of the
previous image set move to previous/, so the pipeline then runs again on the subset."""

import math
import os
import threading
import traceback

import viser

from modelforge import projects


class SubsetPanel:
    def __init__(self, app, gui):
        self.app = app
        self._images = []
        with gui.add_folder("Images used", expand_by_default=False):
            self.g_md = gui.add_markdown("")
            self.g_every = gui.add_number(
                "Keep every", initial_value=1, min=1, step=1,
                hint="1 = all extracted images (in the time range), 2 = every other one, 3 = one in three, ... "
                     "Counted per source in capture order.")
            self.g_offset = gui.add_number(
                "Starting at", initial_value=0, min=0, step=1,
                hint="Which image of each source to start from (0 = the first): with Keep every 2, starting at 1 "
                     "gives the other half.")
            self.g_range = gui.add_multi_slider(
                "Time range [s]", min=0.0, max=1.0, step=0.5, initial_value=(0.0, 1.0),
                hint="Use only images inside this window of the project timeline (all sources).")
            self.g_preview = gui.add_markdown("")
            self.g_apply = gui.add_button(
                "Use these images", icon=viser.Icon.FILTER,
                hint="From now on every stage runs on this subset of the extracted images (images/ and poses.json "
                     "become the subset; the full extraction stays in extracted/). Results of the current image "
                     "set (features, matches, sparse, dense, splat, ...) move to previous/: run the pipeline "
                     "again. Keep every 1 over the whole range returns to all images.")
            self.g_apply.on_click(lambda _: self.apply())
            for g in (self.g_every, self.g_offset, self.g_range):
                g.on_update(lambda _: self._preview())

    def refresh(self):
        """Load the open project's extracted images and show the current subset in the controls."""
        try:
            self._images = projects.extracted_images(self.app.path)
            info = projects.subsample_info(self.app.path)
        except (OSError, ValueError, KeyError):
            self._images, info = [], None
        times = [im["time_s"] for im in self._images if im.get("time_s") is not None]
        lo, hi = (math.floor(min(times)), math.ceil(max(times))) if times else (0.0, 1.0)
        hi = max(hi, lo + 1)
        self.g_range.min, self.g_range.max = float(lo), float(hi)
        st = (info or {}).get("setting") or {}
        self.g_every.value, self.g_offset.value = int(st.get("every", 1)), int(st.get("offset", 0))
        self.g_range.value = (float(st["start"]) if st.get("start") is not None else float(lo),
                              float(st["end"]) if st.get("end") is not None else float(hi))
        if not info or not info["used"]:
            self.g_md.content = "_no extracted images yet (Ingest tab)_"
        elif st:
            self.g_md.content = (f"using **{info['used']}** of {info['extracted']} extracted images "
                                 f"(every {st['every']}" + (f" from #{st['offset']}" if st["offset"] else "")
                                 + (f", {st['start']:g}–{st['end']:g} s" if st.get("start") is not None else "")
                                 + ")")
        else:
            self.g_md.content = f"using all **{info['used']}** extracted images"
        self._preview()

    def _args(self):
        lo, hi = self.g_range.value
        full = lo <= self.g_range.min and hi >= self.g_range.max
        return dict(every=int(self.g_every.value), offset=int(self.g_offset.value),
                    start=None if full else float(lo), end=None if full else float(hi))

    def _preview(self):
        if not self._images:
            self.g_preview.content = ""
            self.g_apply.disabled = True
            return
        n = len(projects.subsample_selection(self._images, **self._args()))
        self.g_apply.disabled = n == 0
        self.g_preview.content = (f"→ **{n}** of {len(self._images)} images; applying moves the current results "
                                  "to previous/")

    def apply(self):
        app = self.app
        if app.running or app.ingest.running:
            self.g_preview.content = "⚪ a job is running: wait for it to finish"
            return
        self.g_apply.disabled = True
        args = self._args()

        def work():
            try:
                n, total, archived = projects.apply_subsample(app.path, **args)
                app.load_project()
                app.refresh_after_images_changed()
                msg = f"✅ now using {n} of {total} images" + (
                    f"; earlier results moved to `{os.path.relpath(archived, app.path)}`" if archived else "")
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                traceback.print_exc()
                msg = f"❌ {e}"
            self.refresh()
            self.g_preview.content = msg + "  \n" + self.g_preview.content
        threading.Thread(target=work, daemon=True).start()
