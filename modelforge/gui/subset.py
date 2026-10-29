"""GUI Sparse tab → "Images used": run the pipeline on a subset of the extracted images without extracting
again: every N-th image (with an offset and a time window), or keyframes chosen from the measured image
overlap (optional "Analyze extracted images" pass, sfm/overlap.py). projects.apply_subsample does the
work; the results of the previous image set move to previous/, so the pipeline then runs again."""

import json
import math
import os
import threading
import traceback

import viser

from modelforge import projects
from modelforge.sfm import overlap

MODES = {"every N-th image": "every", "keyframes (from the analysis)": "keyframes"}
DEFAULT_MODE = "keyframes (from the analysis)"  # a project that never chose a subset; Run full pipeline analyzes


class Selection:
    """The choice of extracted images: every N-th (with an offset and a time window) or keyframes at a density
    (from the overlap analysis). Images used and Register more photos each have one (the user asked for the same
    options in both, 2026-10-25)."""

    def __init__(self, gui, mode=DEFAULT_MODE, level=overlap.DEFAULT_LEVEL, on_update=None):
        self.g_mode = gui.add_dropdown(
            "Selection", options=tuple(MODES), initial_value=mode,
            hint="every N-th image: uniform along each source. keyframes (default; density sparse): as few "
                 "images as keep enough overlap between neighbours, measured on the images (Analyze below; "
                 "Run full pipeline analyzes and selects by itself): fewer on straight cruising, more in turns "
                 "and close passes. A project that chose every N-th image keeps it.")
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
            "Keyframe density", options=tuple(overlap.LEVELS), initial_value=level,
            hint="Minimum share of features a keyframe shares with the previous one, and half of that with "
                 "the one before. Expected splat quality (held-out photos, kladno_dopravnik): "
                 + "; ".join(f"{k} ({v:.2f}): {overlap.QUALITY[k]}" for k, v in overlap.LEVELS.items())
                 + ".")
        self.g_area = gui.add_checkbox(
            "Only photos looking into the map's area", initial_value=True, visible=False,
            hint="Map window → Click: Draw an area, then Use the area for Images used / Register: of the photos "
                 "chosen above, only those whose look-at point (the optical axis on the terrain) lies inside the "
                 "drawn polygon. Untick to ignore the area.")
        self.area = None  # {"polygon": [[x, y], ...], "names": set of the extracted photos looking into it}
        for g in (self.g_mode, self.g_every, self.g_offset, self.g_range, self.g_level, self.g_area):
            g.on_update(lambda _: on_update and on_update())
        self.g_mode.on_update(lambda _: self.show_mode())

    def mode(self):
        return MODES[self.g_mode.value]

    def set_area(self, polygon, names):
        """The map's drawn area (ENU polygon) and the extracted photos looking into it; None clears it."""
        self.area = None if polygon is None else {"polygon": [[float(x), float(y)] for x, y in polygon],
                                                  "names": set(names)}
        self.g_area.visible = self.area is not None
        self.g_area.value = self.area is not None

    def _area_on(self):
        return self.area is not None and self.g_area.value

    def show_mode(self):
        every = self.mode() == "every"
        for g in (self.g_every, self.g_offset, self.g_range):
            g.visible = every
        self.g_level.visible = not every

    def set_range(self, images):
        """The time slider's bounds from the images; returns them."""
        times = [im["time_s"] for im in images if im.get("time_s") is not None]
        lo, hi = (math.floor(min(times)), math.ceil(max(times))) if times else (0.0, 1.0)
        hi = max(hi, lo + 1)
        self.g_range.min, self.g_range.max = float(lo), float(hi)
        return lo, hi

    def args(self, images, ov):
        """apply_subsample / register keyword arguments: names (keyframes, or any choice inside the map's area) or
        every / offset / start / end."""
        a = self._args(images, ov)
        if not self._area_on():
            return a
        names = a["names"] if "names" in a else [images[k]["image"] for k in projects.subsample_selection(images, **a)]
        setting = a.get("setting") or {"method": "every", "every": a["every"], "offset": a["offset"],
                                       "start": a["start"], "end": a["end"]}
        return dict(names=[n for n in names if n in self.area["names"]],
                    setting={**setting, "area": self.area["polygon"]})

    def _args(self, images, ov):
        if self.mode() == "keyframes":
            level = self.g_level.value
            names = overlap.select_keyframes(images, ov, overlap.LEVELS[level])
            return dict(names=names, setting={"method": "keyframes", "level": level,
                                              "min_overlap": overlap.LEVELS[level]})
        lo, hi = self.g_range.value
        full = lo <= self.g_range.min and hi >= self.g_range.max
        return dict(every=int(self.g_every.value), offset=int(self.g_offset.value),
                    start=None if full else float(lo), end=None if full else float(hi))

    def names(self, images, ov):
        a = self.args(images, ov)
        return a["names"] if "names" in a else [images[k]["image"] for k in projects.subsample_selection(images, **a)]

    def describe(self):
        area = " in the map's area" if self._area_on() else ""
        if self.mode() == "keyframes":
            return f"keyframes ({self.g_level.value}){area}"
        lo, hi = self.g_range.value
        full = lo <= self.g_range.min and hi >= self.g_range.max
        return (f"every {int(self.g_every.value)}" + (f" from #{int(self.g_offset.value)}" if self.g_offset.value else "")
                + ("" if full else f", {lo:g}–{hi:g} s") + area)


class SubsetPanel:
    def __init__(self, app, gui):
        self.app = app
        self._images, self._overlap = [], None
        with gui.add_folder("Images used", expand_by_default=True):
            self.g_md = gui.add_markdown("")
            self.sel = Selection(gui, on_update=self._preview)
            self.g_mode, self.g_every, self.g_offset = self.sel.g_mode, self.sel.g_every, self.sel.g_offset
            self.g_range, self.g_level = self.sel.g_range, self.sel.g_level
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
                     "again. Keep every 1 over the whole range returns to all images. Run full pipeline applies "
                     "the panel's choice (keyframes or every N-th) by itself as its first step, but only while no "
                     "choice was applied yet; after that it keeps the images in use.")
            self.g_apply.on_click(lambda _: self.apply())

    def _mode(self):
        return self.sel.mode()

    def _show_mode(self):
        self.sel.show_mode()

    def refresh(self):
        """Load the open project's extracted images and show the current subset in the controls."""
        try:
            self._images = projects.extracted_images(self.app.path)
            info = projects.subsample_info(self.app.path)
        except (OSError, ValueError, KeyError):
            self._images, info = [], None
        self._overlap = overlap.load(self.app.path) if self.app.path else None
        lo, hi = self.sel.set_range(self._images)
        st = (info or {}).get("setting") or {}
        # every N-th's fields are set whatever the mode: in keyframes mode (the default) they used to keep their
        # placeholders, so switching to every N-th selected the first second of the flight only (moric, 2026-10-22:
        # "now using 1 of 400 images")
        self.g_every.value, self.g_offset.value = int(st.get("every", 1)), int(st.get("offset", 0))
        rng = (float(st["start"]) if st.get("start") is not None else float(lo),
               float(st["end"]) if st.get("end") is not None else float(hi))
        if self._images and len(projects.subsample_selection(self._images, every=1, start=rng[0], end=rng[1])) < 2:
            rng = (float(lo), float(hi))  # a stored range that holds (almost) nothing: the whole flight
        self.g_range.value = rng
        if st.get("method") == "keyframes":
            self.g_mode.value = "keyframes (from the analysis)"
            self.g_level.value = st.get("level", overlap.DEFAULT_LEVEL)
        else:  # every N-th chosen (kept), or none chosen yet: the default
            self.g_mode.value = "every N-th image" if st else DEFAULT_MODE
            self.g_level.value = overlap.DEFAULT_LEVEL
        if not info or not info["used"]:
            self.g_md.content = "_no extracted images yet (Ingest tab)_"
        elif st.get("method") == "keyframes":
            self.g_md.content = (f"using **{info['used']}** of {info['extracted']} extracted images "
                                 f"(keyframes, {st.get('level')}: overlap ≥ {st.get('min_overlap')})")
        elif st and info["used"] == info["extracted"]:
            self.g_md.content = f"using all **{info['used']}** extracted images (every N-th image)"
        elif st:
            self.g_md.content = (f"using **{info['used']}** of {info['extracted']} extracted images "
                                 f"(every {st['every']}" + (f" from #{st['offset']}" if st["offset"] else "")
                                 + (f", {st['start']:g}–{st['end']:g} s" if st.get("start") is not None else "")
                                 + ")")
        else:
            self.g_md.content = f"using all **{info['used']}** extracted images"
        if st.get("area"):
            self.g_md.content += f" · inside an area drawn on the map ({len(st['area'])} corners)"
        if self._overlap:
            self.g_md.content += (f"  \nanalysis: {len(self._overlap['keypoints'])} images, "
                                  f"{len(self._overlap['pairs']):,} verified pairs")
        self._show_mode()
        self._preview()
        if getattr(self.app, "register", None) is not None:  # the same pool and analysis
            self.app.register.refresh()

    def _args(self):
        return self.sel.args(self._images, self._overlap)

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
        self.g_apply.disabled = n < 2  # SfM needs two images; one meant a wrong range
        extra = ""
        if self._overlap:
            names = args.get("names") or [self._images[k]["image"] for k in
                                          projects.subsample_selection(self._images, **args)]
            c = overlap.overlap_stats(self._images, self._overlap, names)["consecutive"]
            if c:
                extra = f" · overlap of neighbours: weakest {c['min']:.2f}, 5% {c['p5']:.2f}, median {c['median']:.2f}"
        self.g_preview.content = (f"→ **{n}** of {len(self._images)} images{extra}  \n"
                                  "applying moves the current results to previous/")

    def selection_made(self):
        """An Images used choice was applied before (poses.json "subsample"): the full pipeline keeps it."""
        try:
            return bool(self.app.path and (projects.subsample_info(self.app.path) or {}).get("setting"))
        except (OSError, ValueError, KeyError):
            return False

    def keyframes_selected(self):
        """Selection = keyframes: the full pipeline then analyzes and selects first."""
        return bool(self._images) and self._mode() == "keyframes"

    def apply_selection(self):
        """In a running job (the full pipeline's first step): use what the panel is set to, keyframes (after the
        analysis) or every N-th image, unless those images are in use already. Before, only keyframes were
        applied, so switching the panel to every N-th left the pipeline on the keyframes (the user asked).
        Returns (images used, images extracted, a description)."""
        if self._mode() == "keyframes":
            n, total = self.apply_keyframes()
            return n, total, f"keyframes ({self.g_level.value})"
        self._images = projects.extracted_images(self.app.path)
        args = self._args()
        names = set(args["names"]) if "names" in args else \
            {self._images[k]["image"] for k in projects.subsample_selection(self._images, **args)}
        if len(names) < 2:
            raise RuntimeError(f"Images used: every N-th selects {len(names)} image(s) (check Starting at / Time range)")
        current = set(self.app.project.names) if self.app.project else set()
        if names != current:
            projects.apply_subsample(self.app.path, **args)
            self.app.load_project()
            self.app.refresh_after_images_changed()
        self.refresh()
        return len(names), len(self._images), self.sel.describe()

    def apply_keyframes(self):
        """In a running job (full pipeline, after the analysis): use the keyframes at the chosen density
        unless they are in use already. Returns (images used, images extracted)."""
        self.refresh_overlap()
        args = self._args()
        names = set(args["names"])
        current = set(self.app.project.names) if self.app.project else set()
        if names != current:
            projects.apply_subsample(self.app.path, **args)
            self.app.load_project()
            self.app.refresh_after_images_changed()
        self.refresh()
        return len(names), len(self._images)

    def refresh_overlap(self):
        self._images = projects.extracted_images(self.app.path)
        self._overlap = overlap.load(self.app.path)
        if not self._overlap:
            raise RuntimeError("no overlap analysis: run Analyze extracted images first")

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


class RegisterPanel:
    """Sparse tab → "Register more photos (for the splat)": after SfM, the photos of a Selection over the extracted
    pool that are not in use are placed into the model (sfm/register.py) and added to transforms.json."""

    def __init__(self, app, gui):
        self.app = app
        self._images, self._overlap = [], None
        with gui.add_folder("Register more photos (for the splat)", expand_by_default=True):
            gui.add_markdown(
                "After SfM: place more of the **extracted** photos into the model, without running SfM again, so the "
                "splat trains on a denser set. Choose them as in **Images used**; those of the choice already in "
                "use are skipped. Each is matched to the model photos nearest in time and located (PnP, the model "
                "unchanged; ~3 s per 4K photo); its pose must agree with the flight's trajectory (the metadata + "
                "the refine's corrections, interpolated).")
            self.sel = Selection(gui, level="dense", on_update=self._preview)
            self.g_preview = gui.add_markdown("")
            self.g_run = gui.add_button("Register more photos", icon=viser.Icon.PHOTO_PLUS,
                                        hint="Place the photos chosen above (those not in use) into the refined model "
                                             "and add them to transforms.json for the splat. A new refine makes them "
                                             "stale: register again.")
            self.g_run.on_click(lambda _: self.app.run_stage("register"))
            self.g_remove = gui.add_button("Remove registered photos", icon=viser.Icon.TRASH,
                                           hint="Back to the model's own photos: the registered ones leave "
                                                "transforms.json.")
            self.g_remove.on_click(lambda _: self.remove())
            self.g_md = gui.add_markdown("")

    def refresh(self):
        """The open project's pool and analysis, the time range, the status."""
        from modelforge.sfm import register
        path = self.app.path
        try:
            self._images = projects.extracted_images(path) if path else []
        except (OSError, ValueError, KeyError):
            self._images = []
        self._overlap = overlap.load(path) if path else None
        lo0, hi0 = self.sel.g_range.value
        lo, hi = self.sel.set_range(self._images)
        if not (lo <= lo0 < hi0 <= hi) or (lo0, hi0) == (0.0, 1.0):  # keep a range chosen in this flight
            self.sel.g_range.value = (float(lo), float(hi))
        self.sel.show_mode()
        self._preview()
        if not path:
            self.g_md.content = ""
            return
        reg = register.load(path)
        stale = os.path.exists(os.path.join(path, register.DIR, "registered.json")) and reg is None
        try:
            mf = json.load(open(os.path.join(path, "transforms.json")))["modelforge"]
            near = len(mf["registered_near_dropped"])
            user = len(mf.get("registered_excluded", ()))
        except (OSError, ValueError, KeyError):
            near = user = 0
        self.g_md.content = (
            (f"{len(reg['images'])} photos registered into this model ({len(reg['images']) - near - user} in "
             f"transforms.json; left out: {near} next to isolated photos the export left out, {user} excluded "
             "below)" if near or user else
             f"{len(reg['images'])} photos registered into this model (in transforms.json)")
            + (f": {reg['stats']['selection']}" if reg.get("stats", {}).get("selection") else "") if reg else
            "⚠ the registered photos belong to an earlier model: register again" if stale else "none registered")

    def _in_use(self):
        try:
            return set(os.listdir(os.path.join(self.app.path, "images")))
        except (OSError, TypeError):
            return set()

    def _preview(self):
        has_model = bool(self.app.path) and os.path.exists(os.path.join(self.app.path, "sparse", "refined",
                                                                        "images.bin"))
        if not self._images or not has_model:
            self.g_preview.content = "_needs extracted photos and an SfM model (Run SfM)_" if self.app.path else ""
            self.g_run.disabled = True
            return
        if self.sel.mode() == "keyframes" and not self._overlap:
            self.g_preview.content = ("keyframes need the analysis: **Images used → Analyze extracted images** "
                                      "first, or choose every N-th")
            self.g_run.disabled = True
            return
        names = self.sel.names(self._images, self._overlap)
        used = self._in_use()
        new = [n for n in names if n not in used]
        self.g_preview.content = (f"→ **{len(new)}** photos to register ({len(names)} chosen, "
                                  f"{len(names) - len(new)} of them in use)")
        self.g_run.disabled = not new

    def pending(self):
        """Photos the selection would register (not in use), 0 when there is no pool or no analysis it needs."""
        from modelforge.sfm import register
        if not self._images or (self.sel.mode() == "keyframes" and not self._overlap):
            return 0
        return len(register.candidates(self.app.path, selection=self.job_kwargs()["selection"]))

    def job_kwargs(self):
        """register.run's selection + describe for the job."""
        a = self.sel.args(self._images, self._overlap)
        a.pop("setting", None)
        return {"selection": a, "describe": self.sel.describe()}

    def remove(self):
        from modelforge.sfm import register
        if self.app.running:
            return
        register.remove(self.app.path)
        self.refresh()
