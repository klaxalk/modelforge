"""GUI View tab → "Semantic search": compute C-RADIOv3 features for the splat (a job), then type a text and
see where it is in the splat: matching Gaussians tinted red, or only the matches shown (modelforge.semantic).
The splat file is never changed; only the colours / selection sent to the browser."""

import threading
import time
import traceback

import numpy as np
import viser

from modelforge import semantic

RED = np.array([1.0, 0.1, 0.1], np.float32)
MODES = ("highlight in red", "matches only")


class SemanticPanel:
    def __init__(self, app, gui):
        self.app = app
        self.feats, self.encoder, self.rel, self.query = None, semantic.TextEncoder(), None, ""
        with gui.add_folder("Semantic search", expand_by_default=False):
            self.g_md = gui.add_markdown("")
            self.g_compute = gui.add_button(
                "Compute semantic features", icon=viser.Icon.BRAIN,
                hint="C-RADIOv3 features of every photo (2048 px, text-aligned through SigLIP2), projected onto the "
                     "splat's Gaussians with an occlusion test: splat/semantic/ (the splat itself is unchanged). "
                     "About a minute for ~60 photos; the first run downloads the models (~5 GB).")
            self.g_compute.on_click(lambda _: self.app.run_stage("semantic"))
            self.g_query = gui.add_text("Query", initial_value="",
                                        hint="What to look for, in words: 'a car', 'gravel pile', 'a roof'.")
            self.g_search = gui.add_button("Search", icon=viser.Icon.SEARCH,
                                           hint="Score every Gaussian against the query. The first search loads "
                                                "the text model (~1 min); later ones take a moment.")
            self.g_search.on_click(lambda _: self.search())
            self.g_thr = gui.add_slider(
                "Threshold", min=0.5, max=0.95, step=0.01, initial_value=0.6,
                hint="Relevance (0.5 = no better than generic words like 'object' or 'stuff', 1 = clearly the "
                     "query) above which a Gaussian counts as a match. Raise it when too much lights up.")
            self.g_mode = gui.add_dropdown("Show", options=MODES, initial_value=MODES[0],
                                           hint="highlight in red: the whole splat, matches tinted by relevance. "
                                                "matches only: hide everything else.")
            self.g_clear = gui.add_button("Clear", hint="Back to the normal splat colours.")
            self.g_clear.on_click(lambda _: self.clear())
            self.g_result = gui.add_markdown("")
            for h in (self.g_thr, self.g_mode):
                h.on_update(lambda _: self._redraw())

    # ------------------------------------------------------------ state
    def refresh(self):
        """Status of the project's features (after a project switch, a new splat or a compute)."""
        self.feats, self.rel = None, None
        st = semantic.status(self.app.path) if self.app.path else "none"
        why = semantic.available()
        self.g_compute.disabled = why is not None or not (self.app.dense or {}).get("splat")
        self.g_search.disabled = st != "ready"
        if why:
            self.g_md.content = f"unavailable: {why}"
        elif not (self.app.dense or {}).get("splat"):
            self.g_md.content = "no splat yet (Splat tab)"
        else:
            self.g_md.content = {"none": "no semantic features yet: **Compute semantic features**",
                                 "stale": "⚠️ the splat changed since the features were computed: compute again",
                                 "ready": "features ready: type a query and **Search**"}[st]
        self.g_result.content = ""

    def _features(self):
        if self.feats is None:
            self.feats = semantic.Features(self.app.path)
        return self.feats

    # ------------------------------------------------------------ search
    def search(self):
        q = self.g_query.value.strip()
        if not q:
            self.g_result.content = "type a query first"
            return
        self.g_search.disabled = True

        def work():
            t0 = time.time()
            try:
                if self.encoder.proc is None:
                    self.g_result.content = "⏳ loading the text model (first search, about a minute) …"
                emb = self.encoder.encode([q] + list(semantic.NEGATIVES))
                self.g_result.content = "⏳ scoring …"
                self.rel, self.query = self._features().relevance(emb[0], emb[1:]), q
                self._redraw()
                self.g_result.content += f" · {time.time() - t0:.1f} s"
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                traceback.print_exc()
                self.g_result.content = f"❌ {e}"
            finally:
                self.g_search.disabled = semantic.status(self.app.path) != "ready"
        threading.Thread(target=work, daemon=True).start()

    def clear(self):
        self.rel, self.query = None, ""
        self.g_result.content = ""
        self._redraw()

    def _redraw(self):
        self.app.redraw_splat_decorated()
        if self.rel is not None:
            n = int((self.rel >= self.g_thr.value).sum())
            self.g_result.content = f"**{self.query}**: {n:,} Gaussians above {self.g_thr.value:.2f}"

    def decorate(self, centers, cov, rgb, op, idx):
        """The splat arrays to send, with the current search applied (idx: Gaussians shown of the file)."""
        if self.rel is None or len(self.rel) <= (idx.max() if len(idx) else 0):
            return centers, cov, rgb, op
        r = self.rel[idx]
        thr = float(self.g_thr.value)
        if self.g_mode.value == "matches only":
            keep = r >= thr
            return centers[keep], cov[keep], rgb[keep], op[keep]
        a = np.clip((r - thr) / max(1e-6, 1.0 - thr) * 0.7 + 0.3, 0, 1) * (r >= thr)  # tint 30-100 %
        rgb = rgb * (1 - a[:, None]) + RED * a[:, None]
        return centers, cov, rgb.astype(np.float32), op
