"""GUI Sparse tab → "Images used": run the pipeline on a subset of the extracted images without extracting
again: every N-th image (with an offset and a time window), or keyframes chosen from the measured image
overlap (optional "Analyze extracted images" pass, sfm/overlap.py). projects.apply_subsample does the
work; the results of the previous image set move to previous/, so the pipeline then runs again."""

import math
import os
import threading
import traceback

import viser

from modelforge import projects
from modelforge.sfm import overlap

MODES = {"every N-th image": "every", "keyframes (from the analysis)": "keyframes"}


class SubsetPanel:
    def __init__(self, app, gui):
        self.app = app
        self._images, self._overlap = [], None
        with gui.add_folder("Images used", expand_by_default=False):
            self.g_md = gui.add_markdown("")
            self.g_mode = gui.add_dropdown(
                "Selection", options=tuple(MODES), initial_value="every N-th image",
                hint="every N-th image: uniform along each source. keyframes: as few images as keep enough overlap "
                     "between neighbours, measured on the images (needs Analyze below): fewer on straight "
                     "cruising, more in turns and close passes.")
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
            self.g_level = gui.add_dropdown(
                "Keyframe density", options=tuple(overlap.LEVELS), initial_value=overlap.DEFAULT_LEVEL,
                hint="Minimum share of features a keyframe shares with the previous one, and half of that with "
                     "the one before. Expected splat quality (held-out photos, kladno_dopravnik): "
                     + "; ".join(f"{k} ({v:.2f}): {overlap.QUALITY[k]}" for k, v in overlap.LEVELS.items())
                     + ".")
            self.g_analyze = gui.add_button(
                "Analyze extracted images (optional)", icon=viser.Icon.ZOOM_CHECK,
                hint="SIFT features of all extracted images and matches of each with the next "
                     f"{overlap.WINDOW} of its source (a few minutes), in analysis/. Measures how much images "
                     "overlap, for the keyframe selection. Runs as a job (Stop works).")
            self.g_analyze.on_click(lambda _: self.app.run_stage("analyze"))
            self.g_preview = gui.add_markdown("")
            self.g_apply = gui.add_button(
                "Use these images", icon=viser.Icon.FILTER,
                hint="From now on every stage runs on this subset of the extracted images (images/ and poses.json "
                     "become the subset; the full extraction stays in extracted/). Results of the current image "
                     "set (features, matches, sparse, dense, splat, ...) move to previous/: run the pipeline "
                     "again. Keep every 1 over the whole range returns to all images.")
            self.g_apply.on_click(lambda _: self.apply())
            for g in (self.g_mode, self.g_every, self.g_offset, self.g_range, self.g_level):
                g.on_update(lambda _: self._preview())
            self.g_mode.on_update(lambda _: self._show_mode())

    def _mode(self):
        return MODES[self.g_mode.value]

    def _show_mode(self):
        every = self._mode() == "every"
        for g in (self.g_every, self.g_offset, self.g_range):
            g.visible = every
        self.g_level.visible = not every

    def refresh(self):
        """Load the open project's extracted images and show the current subset in the controls."""
        try:
            self._images = projects.extracted_images(self.app.path)
            info = projects.subsample_info(self.app.path)
        except (OSError, ValueError, KeyError):
            self._images, info = [], None
        self._overlap = overlap.load(self.app.path) if self.app.path else None
        times = [im["time_s"] for im in self._images if im.get("time_s") is not None]
        lo, hi = (math.floor(min(times)), math.ceil(max(times))) if times else (0.0, 1.0)
        hi = max(hi, lo + 1)
        self.g_range.min, self.g_range.max = float(lo), float(hi)
        st = (info or {}).get("setting") or {}
        if st.get("method") == "keyframes":
            self.g_mode.value = "keyframes (from the analysis)"
            self.g_level.value = st.get("level", overlap.DEFAULT_LEVEL)
        else:
            self.g_every.value, self.g_offset.value = int(st.get("every", 1)), int(st.get("offset", 0))
            self.g_range.value = (float(st["start"]) if st.get("start") is not None else float(lo),
                                  float(st["end"]) if st.get("end") is not None else float(hi))
        if not info or not info["used"]:
            self.g_md.content = "_no extracted images yet (Ingest tab)_"
        elif st.get("method") == "keyframes":
            self.g_md.content = (f"using **{info['used']}** of {info['extracted']} extracted images "
                                 f"(keyframes, {st.get('level')}: overlap ≥ {st.get('min_overlap')})")
        elif st:
            self.g_md.content = (f"using **{info['used']}** of {info['extracted']} extracted images "
                                 f"(every {st['every']}" + (f" from #{st['offset']}" if st["offset"] else "")
                                 + (f", {st['start']:g}–{st['end']:g} s" if st.get("start") is not None else "")
                                 + ")")
        else:
            self.g_md.content = f"using all **{info['used']}** extracted images"
        if self._overlap:
            self.g_md.content += (f"  \nanalysis: {len(self._overlap['keypoints'])} images, "
                                  f"{len(self._overlap['pairs']):,} verified pairs")
        self._show_mode()
        self._preview()

    def _args(self):
        if self._mode() == "keyframes":
            level = self.g_level.value
            names = overlap.select_keyframes(self._images, self._overlap, overlap.LEVELS[level])
            return dict(names=names, setting={"method": "keyframes", "level": level,
                                              "min_overlap": overlap.LEVELS[level]})
        lo, hi = self.g_range.value
        full = lo <= self.g_range.min and hi >= self.g_range.max
        return dict(every=int(self.g_every.value), offset=int(self.g_offset.value),
                    start=None if full else float(lo), end=None if full else float(hi))

    def _preview(self):
        if not self._images:
            self.g_preview.content = ""
            self.g_apply.disabled = True
            return
        if self._mode() == "keyframes" and not self._overlap:
            self.g_preview.content = "keyframes need the analysis: press **Analyze extracted images** first"
            self.g_apply.disabled = True
            return
        args = self._args()
        n = len(args["names"]) if "names" in args else len(projects.subsample_selection(self._images, **args))
        self.g_apply.disabled = n == 0
        extra = ""
        if self._overlap:
            names = args.get("names") or [self._images[k]["image"] for k in
                                          projects.subsample_selection(self._images, **args)]
            c = overlap.overlap_stats(self._images, self._overlap, names)["consecutive"]
            if c:
                extra = f" · overlap of neighbours: weakest {c['min']:.2f}, 5% {c['p5']:.2f}, median {c['median']:.2f}"
        self.g_preview.content = (f"→ **{n}** of {len(self._images)} images{extra}  \n"
                                  "applying moves the current results to previous/")

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
