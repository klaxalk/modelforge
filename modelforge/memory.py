"""Memory estimates before heavy stages (docs/todo_robustness.md §3; an OOM kill once took a 5 h refine).

estimate(project, stage, settings) -> (GB the job needs on top of what runs now, why). Per stage a model on its
driver (inlier matches, SfM points, mesh faces / vertices, decoded photos of a cpu image cache, Gaussians, ...), and
measured peaks, which win:
- this project's last peak of that stage (report.json "memory"), scaled by how much the driver grew;
- else the largest GB per driver unit any project on this machine measured (ROOT/.modelforge_memory.json);
- else the model, its coefficients from the peaks in CLAUDE.md.

Sampler measures a running job: the proportional set size (PSS: shared pages split between their processes, so
nerfstudio's two copies of a cpu cache count once each) of every process below the GUI (job processes, ns-train and
its data workers, COLMAP) plus the GUI's own growth since the stage began; queue entries running in their own
process (gui/headless.py) are left out, they measure themselves.
"""
import json
import os
import struct
import threading
import time

MARGIN = 1.1  # on a measured peak
# stages of which only a part grows with the driver: a measured peak grows at the model's rate per driver unit, not in
# proportion. A chunked refine's chunks do not grow with the project, only its triangulations of all photos (COLMAP's
# match cache + correspondence graph: berlin_all 75M ALIKED + RoMa matches 4.8 GB): its 16.3 GB at 31M SIFT matches
# scaled in proportion to 68M asked for 36 GB (2026-10-29).
GROW_AT_MODEL_RATE = {"refine:chunked"}
HISTORY = 20  # machine-wide records kept per stage
MACHINE_FILE = ".modelforge_memory.json"

# stage -> (base GB, GB per driver unit, driver, unit). Drivers: inlier matches (M), images, SfM points (M), faces
# (M), vertices (M), decoded photos (GB), Gaussians (M), depth-map GB (dense fusion).
MODELS = {
    "features": (2.0, 0.0, None, ""),
    "pairs": (1.0, 0.0, None, ""),
    "match": (2.0, 0.0, None, ""),
    "analyze": (3.0, 0.0, None, ""),
    # global: pipeline.global_ba_need_gb (0.49-0.72 GB per M measured); chunked: kladno_dira 82.7M -> 14.6 GB
    "refine:global": (1.0, 0.7, "inlier_matches", "M inlier matches"),
    "refine:chunked": (2.0, 0.16, "inlier_matches", "M inlier matches"),
    "reconstruct": (1.0, 0.7, "inlier_matches", "M inlier matches"),
    # measured on karlak (build/memcal.py, 2026-10-24): export 1.9 GB at 6.9M inlier matches, SfM mesh 1.5 GB at
    # 0.65M points, patching 1.0 GB at 1.4M faces, colouring 1.0 GB at 0.74M vertices, splat mesh coarse 2.9 GB
    # at 1.4M faces, semantic 4.6 GB at 2.3M Gaussians; the slopes are kept above what one project can show.
    # triangulating all images: COLMAP's correspondence graph of every inlier match, then the points
    "export": (1.5, 0.15, "inlier_matches", "M inlier matches"),
    "sfmmesh": (1.0, 1.0, "points", "M SfM points"),
    "sfmmesh-fill": (0.8, 0.2, "sfm_faces", "M mesh faces"),
    "sfmmesh-colour": (0.8, 0.3, "sfm_vertices", "M mesh vertices"),
    # ns-train itself ~4 GB; a cpu image cache holds the decoded photos twice (trainer + data workers)
    "splat": (4.0, 2.0, "cache_gb", "GB of decoded photos in a cpu cache"),
    # splat mesh: the CPU steps after marching cubes, ~90 B per face (berlin fine: 99M faces, 9.1 GB)
    "splatmesh": (3.0, 0.09, "splat_faces", "M mesh faces (estimated)"),
    "splatmesh-fill": (2.0, 0.25, "splat_faces", "M mesh faces"),
    "semantic": (3.5, 0.5, "gaussians", "M Gaussians"),
    "undistort": (2.0, 0.0, None, ""),
    "stereo": (3.0, 0.0, None, ""),
    "fusion": (2.0, 1.0, "depth_gb", "GB of depth + normal maps"),
    "mesh": (4.0, 0.0, None, ""),
    "texture": (4.0, 0.0, None, ""),
}
ALIASES = {"splat-resume": "splat", "splat-tiles": "splat", "refine-resume": "refine"}
# what the full pipeline / the button groups run (for the estimate before they start: the largest stage counts)
GROUPS = {"sfm": ("features", "pairs", "match", "refine", "export"),
          "all": ("features", "pairs", "match", "refine", "export", "sfmmesh", "sfmmesh-fill", "splat", "semantic",
                  "sfmmesh-colour"),
          "dense": ("undistort", "stereo", "fusion", "mesh", "texture"),
          "sfmmesh-all": ("sfmmesh", "sfmmesh-fill", "sfmmesh-colour")}


WIDGETS = {"ba_mode": "g_ba_mode", "cache": "g_splat_cache", "downscale": "g_splat_down",
           "splatmesh_quality": "g_sm_quality"}  # estimate settings <- the GUI widgets (a queue entry's settings)


def settings_from(values):
    """The estimate's settings from widget values by widget name (gui/jobqueue.py entries store them so)."""
    return {k: values[w] for k, w in WIDGETS.items() if w in (values or {})}


def available_gb():
    """MemAvailable (GiB), None if unknown."""
    from modelforge import pipeline
    return pipeline.ram_available_gb()


def _report(project):
    try:
        return json.load(open(os.path.join(project, "report.json")))
    except (OSError, ValueError):
        return {}


def _points(project):
    """Points of the model the export / SfM mesh reads (the count at the head of points3D.bin)."""
    for m in ("export", "refined"):
        try:
            with open(os.path.join(project, "sparse", m, "points3D.bin"), "rb") as f:
                return struct.unpack("<Q", f.read(8))[0]
        except (OSError, struct.error):
            continue
    return None


def key(project, stage, settings=None, rep=None):
    """The record key of a stage: refine splits by BA mode (auto resolved as pipeline.auto_ba_mode would)."""
    stage = ALIASES.get(stage, stage)
    if stage != "refine":
        return stage
    mode = (settings or {}).get("ba_mode", "auto")
    if mode == "auto":
        from modelforge import pipeline
        rep = _report(project) if rep is None else rep
        mode = pipeline.auto_ba_mode(rep, (rep.get("features") or {}).get("images"))[0]
    return f"refine:{mode}"


def driver(project, k, settings=None, rep=None):
    """The driver of stage key k now (in the MODELS unit), None if unknown."""
    settings = settings or {}
    rep = _report(project) if rep is None else rep
    name = MODELS.get(k, (0, 0, None, ""))[2]
    try:
        if name == "inlier_matches":
            v = (rep.get("match") or {}).get("inlier_matches")
            return v / 1e6 if v else None
        if name == "points":
            n = _points(project)
            return n / 1e6 if n else None
        if name in ("sfm_faces", "sfm_vertices"):
            v = (rep.get("sfm_mesh") or {}).get("faces" if name == "sfm_faces" else "vertices")
            return v / 1e6 if v else None
        if name == "gaussians":
            v = (rep.get("splat") or {}).get("num_gaussians")
            return v / 1e6 if v else None
        if name == "cache_gb":
            return _cache_gb(project, settings)
        if name == "splat_faces":
            return _splat_faces(rep, settings)
        if name == "depth_gb":
            return _depth_gb(project, rep)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return None
    return None


def _cache_gb(project, settings):
    """Decoded photos a cpu image cache would hold (0 for gpu / disk), as splat.auto_cache decides."""
    from modelforge import splat, splat_plan
    if not os.path.exists(os.path.join(project, "transforms.json")):
        return None
    t = json.load(open(os.path.join(project, "transforms.json")))
    from modelforge.export import frame_camera
    cams = [frame_camera.camera(t, f) for f in t["frames"]]
    if not cams:
        return 0.0
    down = settings.get("downscale", "auto")
    down = splat_plan.auto_downscale(max(c["w"] for c in cams)) if down in (None, "auto", 0, "0") else int(down)
    cache = settings.get("cache", "auto")
    if cache in (None, "auto"):
        cache, gb = splat.auto_cache(project, down)
    else:
        gb = sum(c["w"] * c["h"] * 3 for c in (frame_camera.train_camera(t, c, down) for c in cams)) / 1e9
    return gb if cache == "cpu" else 0.0


def _splat_faces(rep, settings):
    """Faces of the splat mesh at the chosen quality, from the last one's (faces grow with 1 / voxel^2)."""
    from modelforge import splat_mesh
    sm = rep.get("splat_mesh") or {}
    faces, q = sm.get("faces_raw") or sm.get("faces"), sm.get("quality")
    if not faces or q not in splat_mesh.QUALITY:
        return None
    new = settings.get("splatmesh_quality") or q
    return faces / 1e6 * (splat_mesh.QUALITY[q] / splat_mesh.QUALITY.get(new, splat_mesh.QUALITY[q])) ** 2


def _depth_gb(project, rep):
    """Depth + normal maps (float32, 4 channels) of the dense workspace's images, as fusion caches them."""
    import glob
    from PIL import Image
    files = glob.glob(os.path.join(project, "dense", "images", "*"))
    if not files:
        return None
    w, h = Image.open(files[0]).size
    return len(files) * w * h * 16 / 1e9


def model_gb(k, d):
    base, per, name, _ = MODELS.get(k, (2.0, 0.0, None, ""))
    return base + per * (d or 0.0) if name else base


def estimate(project, stage, settings=None):
    """(GB the job needs available before it starts, why). A group of stages: its largest."""
    rep = _report(project)
    stages = GROUPS.get(stage, (stage,)) if isinstance(stage, str) else tuple(stage)
    best = (0.0, "")
    for st in stages:
        k = key(project, st, settings, rep)
        need, why = _one(project, k, settings, rep)
        if need > best[0]:
            best = (need, why if len(stages) == 1 else f"{st}: {why}")
    return best


def _one(project, k, settings, rep):
    d = driver(project, k, settings, rep)
    base, per, name, unit = MODELS.get(k, (2.0, 0.0, None, ""))
    rec = (rep.get("memory") or {}).get(k)
    old = rep.get("refine") or {}
    if not rec and k.startswith("refine:") and old.get("peak_rss_gb") and (old.get("ba") or {}).get(
            "mode", k[7:]) == k[7:]:  # a refine before these records (its process' peak; the mode maybe unknown):
        m = model_gb(k, d)        # at least that
        if MARGIN * old["peak_rss_gb"] >= m:
            return MARGIN * old["peak_rss_gb"], f"its last refine peaked at {old['peak_rss_gb']:.1f} GB"
        return m, f"{_num(d)} {unit}" if d is not None else "a typical run"
    if rec and rec.get("peak_gb"):
        peak, dr = rec["peak_gb"], rec.get("driver")
        grow = d / dr if d and dr else 1.0
        if grow <= 1:
            need = MARGIN * peak
        elif k in GROW_AT_MODEL_RATE:
            need = MARGIN * (peak + per * (d - dr))
        else:
            need = MARGIN * (base + max(peak - base, 0.0) * grow)
        return need, (f"its last {k} peaked at {peak:.1f} GB" +
                      (f", {grow:.1f}x the {unit} now" if grow > 1.05 else ""))
    per_seen = learned(project, k)
    if per_seen is not None and name and d:
        return MARGIN * (base + per_seen * d), f"{_num(d)} {unit} x {per_seen:.2f} GB (measured on this machine)"
    if name and d is not None:
        return model_gb(k, d), f"{_num(d)} {unit}"
    return model_gb(k, None), "a typical run"


def _num(v):
    return f"{v:.2g}" if v < 1 else f"{v:.1f}"


def learned(project, k):
    """The largest (peak - base) / driver measured on this machine for stage key k, None if none."""
    recs = _machine(project).get(k) or []
    base = MODELS.get(k, (2.0,))[0]
    per = [(r["peak_gb"] - base) / r["driver"] for r in recs if r.get("driver")]
    return max(per) if per else None


def _machine_path(project):
    return os.path.join(os.path.dirname(os.path.abspath(project)), MACHINE_FILE)


def _machine(project):
    try:
        return json.load(open(_machine_path(project)))
    except (OSError, ValueError):
        return {}


def record(project, stage, peak_gb, settings=None):
    """A stage finished with this peak (GB): report.json "memory" + the machine-wide file. A refine is filed under
    the BA mode it ran (report.json refine.ba.mode)."""
    if not peak_gb or peak_gb <= 0:
        return
    from modelforge import pipeline
    rep = pipeline.Report(project)
    ran = ((rep.data.get("refine") or {}).get("ba") or {}).get("mode")
    k = f"refine:{ran}" if ALIASES.get(stage, stage) == "refine" and ran else key(project, stage, settings, rep.data)
    d = driver(project, k, settings, rep.data)
    entry = {"peak_gb": round(peak_gb, 2), "driver": round(d, 4) if d else None, "at": time.strftime("%Y-%m-%d %H:%M")}
    rep.set("memory", dict(rep.data.get("memory") or {}, **{k: entry}))
    m = _machine(project)
    m[k] = (m.get(k) or [])[-(HISTORY - 1):] + [dict(entry, project=os.path.basename(os.path.abspath(project)))]
    try:
        tmp = _machine_path(project) + ".part"
        json.dump(m, open(tmp, "w"), indent=1)
        os.replace(tmp, _machine_path(project))
    except OSError:
        pass


# --- measuring ---------------------------------------------------------------------------------------------------
def _pss_gb(pid):
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Pss:"):
                return int(line.split()[1]) / 2 ** 20
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _children():
    """{ppid: [pid]} of every process."""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            ppid = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        out.setdefault(ppid, []).append(int(p))
    return out


def _excluded(pid):
    try:
        return b"modelforge.gui.headless" in open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return True


def job_gb(root=None, base=0.0):
    """PSS (GB) of every process below root (not queue entries running on their own) + root's growth over base."""
    root = os.getpid() if root is None else root
    kids, total = _children(), 0.0
    todo = list(kids.get(root) or [])
    while todo:
        p = todo.pop()
        if _excluded(p):
            continue
        total += _pss_gb(p)
        todo += kids.get(p) or []
    return total + max(_pss_gb(root) - base, 0.0)


class Sampler:
    """Peak of job_gb() once a second, per stage: begin() starts a stage, peak() reads it."""

    def __init__(self, every=1.0):
        self.every, self.root = every, os.getpid()
        self._base, self._peak = 0.0, 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def begin(self):
        with self._lock:
            self._base, self._peak = _pss_gb(self.root), 0.0

    def peak(self):
        self._sample()
        return self._peak

    def _sample(self):
        v = job_gb(self.root, self._base)
        with self._lock:
            self._peak = max(self._peak, v)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                self._sample()
            except Exception:  # noqa: BLE001 - never break a job over a measurement
                pass

    def stop(self):
        self._stop.set()
