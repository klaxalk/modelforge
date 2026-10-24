"""Level of detail in the View tab (modelforge/splat_lod.py): a splat larger than the budget is drawn per cell, each
cell its own viser splat object at the level the camera needs (viser sorts all splat objects together, so cells
blend correctly). The cut follows the browser that moved last: ~SETTLE s after its camera stops (every MOVING_EVERY s
while it keeps moving), the changed cells are swapped in one atomic update.

No hierarchy yet for this splat: it is built in a job process of its own (~10 s per million Gaussians, ~0.5 GB per
million), meanwhile the random subset is shown as before.
"""
import os
import threading
import time

import numpy as np

from modelforge import splat_lod

NODE = "/dense/splat_lod"
SETTLE = 0.4  # s after the camera stopped
MOVING_EVERY = 2.0  # s between updates while it keeps moving
MIN_MOVE = 0.02  # of the distance to the nearest cell: smaller moves do not count


class SplatLod:
    def __init__(self, app):
        self.app = app
        self.idx = None  # the index in use
        self.drawn = {}  # cell -> level drawn
        self.cam = None  # (position, forward, fov, height) of the last camera
        self.moved_at = 0.0
        self.cut_at = 0.0
        self.cut_cam = None
        self.building = None  # project being built
        self.lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- state -------------------------------------------------------------------------------------------
    def wanted(self, total):
        """LoD is used for this splat (View → Level of detail, more Gaussians than the budget)."""
        return bool(self.app.g_splat_lod.value) and total > int(self.app.g_lod_budget.value)

    def ready(self):
        """The index of the open project's splat (None: not built yet; starts the build)."""
        path = self.app.path
        idx = splat_lod.index(path)
        if idx is None and self.building != path:
            self._build(path)
        return idx

    def _build(self, path):
        from modelforge import worker
        from modelforge.gui import jobqueue
        need = 1.0 + 0.6 * self.app._splat_total / 1e6
        if jobqueue._available_gb() < need + jobqueue.MIN_FREE_GB:
            self.app._on_log(f"level of detail: not built, needs ~{need:.0f} GB of free RAM")
            return
        self.building = path

        def go():
            t0 = time.time()
            try:
                if os.environ.get("MODELFORGE_INPROCESS") == "1":
                    splat_lod.build_in(path)
                else:
                    worker.run("modelforge.splat_lod:build_in", {"folder": path}, label="level of detail")
                self.app._on_log(f"level of detail built in {time.time() - t0:.0f} s")
            except Exception as e:  # noqa: BLE001 - the subset stays
                self.app._on_log(f"level of detail failed: {e}")
            finally:
                self.building = None
            if os.path.abspath(self.app.path) == os.path.abspath(path):
                self.app._unchanged("splat", None)  # redraw with it
                self.app._draw_splat()
        threading.Thread(target=go, daemon=True).start()

    # --- drawing -----------------------------------------------------------------------------------------
    def show(self, idx):
        """Draw the splat from idx (a new splat or the first time): the cut for the last camera."""
        with self.lock:
            if self.idx is not idx:
                self._clear()
                self.idx = idx
            self._apply(self._cut())

    def hide(self):
        with self.lock:
            self._clear()
            self.idx = None

    def forget(self):
        """The scene was cleared (project switch): nothing is drawn any more."""
        with self.lock:
            self.drawn, self.idx, self.cam, self.cut_cam = {}, None, None, None

    def _clear(self):
        if self.drawn:
            self.app.server.scene.remove_by_name(NODE)
        self.drawn = {}

    def redecorate(self):
        """The semantic search changed: send every drawn cell again."""
        with self.lock:
            if self.idx is not None:
                cut, self.drawn = dict(self.drawn), {}
                self._apply(cut)

    def total(self, cut=None):
        cut = self.drawn if cut is None else cut
        return sum(self.idx["cells"][i]["levels"][k]["n"] for i, k in cut.items()) if self.idx else 0

    def _cut(self):
        idx, budget = self.idx, int(self.app.g_lod_budget.value)
        if self.cam is None:  # no browser yet: the whole scene from above
            lo = np.min([c["box"][0] for c in idx["cells"]], 0)
            hi = np.max([c["box"][1] for c in idx["cells"]], 0)
            mid = 0.5 * (lo + hi)
            return splat_lod.choose(idx, mid + [0, 0, np.linalg.norm(hi - lo)], None, budget=budget)
        pos, fwd, fov, h = self.cam
        return splat_lod.choose(idx, pos, fwd, fov_y=fov, height_px=h, budget=budget)

    def _apply(self, cut):
        project, scene = self.app.path, self.app.server.scene
        changed = [i for i, k in cut.items() if self.drawn.get(i) != k]
        if not changed:
            return
        sem = self.app.semantic
        with self.app.server.atomic():
            for i in changed:
                k = cut[i]
                cell = self.idx["cells"][i]
                c, cov, rgb, op = splat_lod.load_cell(project, k, cell)
                if sem.rel is not None:
                    src, parent = splat_lod.cell_sources(project, k, cell)
                    c, cov, rgb, op = sem.decorate(c, cov, rgb, op, src, parent)
                scene.add_gaussian_splats(f"{NODE}/c{i:04d}", centers=c, covariances=cov, rgbs=rgb, opacities=op)
                self.drawn[i] = k
        self.app._lod_status()

    # --- the camera --------------------------------------------------------------------------------------
    def on_camera(self, client):
        cam = client.camera
        try:
            pos = np.array(cam.position, float)
            fwd = np.array(cam.look_at, float) - pos
            fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
            self.cam = (pos, fwd, float(cam.fov), int(cam.image_height or 1080))
        except Exception:  # noqa: BLE001 - the browser went away
            return
        self.moved_at = time.time()

    def _loop(self):
        while not self._stop.wait(0.1):
            if self.idx is None or self.cam is None or self.cut_cam is self.cam:
                continue
            now = time.time()
            if now - self.moved_at < SETTLE and now - self.cut_at < MOVING_EVERY:
                continue
            if self.cut_cam is not None and not self._moved_enough(self.cut_cam, self.cam):
                self.cut_cam = self.cam
                continue
            try:
                with self.lock:
                    if self.idx is None:
                        continue
                    self.cut_cam, self.cut_at = self.cam, now
                    self._apply(self._cut())
            except Exception as e:  # noqa: BLE001 - never kill the loop
                self.app._on_log(f"level of detail: {e}")

    def _moved_enough(self, a, b):
        d = min(max(splat_lod._box_distance(a[0], c["box"]), 1.0) for c in self.idx["cells"])
        return (np.linalg.norm(a[0] - b[0]) > MIN_MOVE * d or float(np.dot(a[1], b[1])) < 0.999
                or abs(a[2] - b[2]) > 1e-3)

    def stop(self):
        self._stop.set()
