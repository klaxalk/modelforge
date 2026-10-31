"""Refine a trained splat towards chosen views with Difix (the user asked, 2026-10-25; right panel → Difix).

Difix (NVIDIA, nvidia/difix_ref: a single-step diffusion model, non-commercial license accepted by the user) cleans
renders of the splat; export/difix_refine.py fine-tunes the splat on those cleaned views, moved step by step from
the nearest training camera towards each target view, together with the photos. The targets are the Difix tab's own
saved views (PROJECT/difix_views.json, as the Views tab's). On karlak (docs/todo_splat_quality.md) the views it aims
at gained +0.65 dB / LPIPS -0.04; views elsewhere did not change. The refined splat replaces splat/splat_enu.ply;
the one before the first refinement is kept as splat/splat_enu.pre_difix.ply (Undo).

Needs nerfstudio's venv, the Difix packages (diffusers 0.25.1, transformers 4.38, peft 0.9, accelerate 0.27, einops:
/opt/difix-pkgs or build/difix_pkgs) and the Difix3D repository (its pipeline code: /opt/difix3d or
build/difix/Difix3D); the weights (5.2 GB) download into the model cache on first use."""
import json
import os
import re
import shutil
import time

import numpy as np

from modelforge import jobs, semantic, sfm_mesh, splat, splat_mesh

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = next((p for p in (os.environ.get("MODELFORGE_DIFIX_PKGS"), "/opt/difix-pkgs",
                         os.path.join(_REPO, "build", "difix_pkgs")) if p and os.path.isdir(p)), None)
CODE = next((p for p in (os.environ.get("MODELFORGE_DIFIX_REPO"), "/opt/difix3d",
                         os.path.join(_REPO, "build", "difix", "Difix3D"))
             if p and os.path.exists(os.path.join(p, "src", "pipeline_difix.py"))), None)
MODEL = "nvidia/difix_ref"
# NVIDIA Fixer (2026-10-31; nvidia/Fixer, Cosmos-Predict2 0.6B, one step, no reference photo; NVIDIA Open Model
# License, commercial use allowed): its code (github.com/nv-tlabs/Fixer, cosmos-predict2 1.0.9), packages, weights
# (5.2 GB in models/) and build/fixer/run_fixer.py, set up by hand in build/fixer (stand-ins in stubs/ for Transformer
# Engine, Megatron and qwen_vl_utils, which only its multi-GPU / prompt-refiner paths need)
FIXER = next((p for p in (os.environ.get("MODELFORGE_FIXER"), "/opt/fixer", os.path.join(_REPO, "build", "fixer"))
              if p and os.path.exists(os.path.join(p, "run_fixer.py"))
              and os.path.exists(os.path.join(p, "models", "pretrained", "pretrained_fixer.pkl"))), None)
FIXER_PATH = ("stubs", "pkgs", os.path.join("dl", "x"), os.path.join("Fixer", "src"))
CLEANERS = {"Difix": "difix", "Fixer": "fixer"}
VIEWS_FILE = "difix_views.json"
BACKUP = os.path.join("splat", "splat_enu.pre_difix.ply")  # the baseline: the splat before the first Difix run
STATE = os.path.join("splat", "difix_state.json")  # the splat the last Difix run wrote (size + mtime)
WORK = os.path.join("splat", "difix")
STEPS, ROUNDS, SHIFT_M = 3000, 4, 3.0
WIDTHS = {"training size": 0, "1024 px": 1024}  # Difix's input width (it runs at its input size; 15 GB at 1920)
_LINE = re.compile(r"\[difix\] (?:round (\d+) / (\d+): view (\d+) / (\d+)|step (\d+) / (\d+))")


def available(project=None, cleaner=None):
    """None if a Difix refinement can run here (and on project, when given) with `cleaner` ("difix" / "fixer"; None:
    either), else the reason."""
    why = splat.available()
    if why:
        return why
    if cleaner == "fixer" and FIXER is None:
        return "Fixer is not set up (build/fixer: run_fixer.py, its packages and models/)"
    if cleaner == "difix" and (PKGS is None or CODE is None) or cleaner is None and FIXER is None and (
            PKGS is None or CODE is None):
        return ("the Difix packages or its code are missing (/opt/difix-pkgs + /opt/difix3d, or build/difix_pkgs + "
                "build/difix/Difix3D: docs/todo_splat_quality.md)")
    if project is not None and not os.path.exists(splat_mesh.splat_path(project)):
        return "no splat yet: train one in the Splat tab first"
    return None


def c2w(view):
    """OpenCV camera-to-world (4 x 4) of a saved view (gui/views.py: position, look_at, up)."""
    p, la = np.array(view["position"], float), np.array(view["look_at"], float)
    up = np.array(view.get("up") or (0.0, 0.0, 1.0), float)
    z = la - p
    z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross(-up, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight along up
        x = np.cross(np.array([0.0, -1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    out = np.eye(4)
    out[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    out[:3, 3] = p
    return out


def _key(path):
    st = os.stat(path)
    return [st.st_size, int(st.st_mtime_ns)]


def baseline(project):
    """The baseline splat (BACKUP: the splat before the first of the Difix runs, which build on each other), or None.
    Once the project's splat is no longer the one the last Difix run wrote (a new training, resume or tiles merge
    replaced it), the baseline is stale: it is removed with its level of detail and the Difix record, so the next run
    keeps the new splat as the baseline. A baseline without a record (before 2026-10-27) adopts the current splat."""
    from modelforge import splat_lod
    from modelforge.pipeline import Report
    backup, sp, state = (os.path.join(project, f) for f in (BACKUP, splat_lod.SPLAT, STATE))
    if not os.path.exists(backup):
        return None
    try:
        written = json.load(open(state)).get("splat")
    except (OSError, ValueError):
        written = None
    if not os.path.exists(sp):
        return backup
    if written is None:
        json.dump({"splat": _key(sp)}, open(state, "w"))
        return backup
    if written == _key(sp):
        return backup
    os.remove(backup)
    os.remove(state)
    shutil.rmtree(splat_lod.lod_dir(project, BACKUP), ignore_errors=True)
    rep = Report(project)
    if "difix" in rep.data:
        rep.set("difix", None)
    return None


def refined(project):
    """{"backup": bool, "info": the last refinement's record} for the tab's status line."""
    backup = baseline(project) is not None
    rep = os.path.join(project, "report.json")
    info = (json.load(open(rep)).get("difix") if os.path.exists(rep) else None) or None
    return {"backup": backup, "info": info}


def _downscale(project):
    from modelforge import splat_plan
    from modelforge.export import frame_camera
    rep = os.path.join(project, "report.json")
    d = (json.load(open(rep)).get("splat") or {}).get("downscale") if os.path.exists(rep) else None
    if d:
        return int(d)
    t = json.load(open(os.path.join(project, "transforms.json")))
    return splat_plan.auto_downscale(frame_camera.largest(t)["w"])


def write_splat(template, out, g):
    """The ply `template` with its Gaussians' parameters replaced by g (load_splat's arrays, same count), so every
    other property and the layout stay; template None: a new ply in the standard 3DGS layout."""
    import plyfile
    if template is None:  # the standard 3DGS layout (nerfstudio's export)
        k = g["sh"].shape[1] - 1
        names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                 + [f"f_rest_{i}" for i in range(3 * k)] + ["opacity"] + [f"scale_{i}" for i in range(3)]
                 + [f"rot_{i}" for i in range(4)])
        v = np.zeros(len(g["means"]), dtype=[(n, "f4") for n in names])
    else:
        v = plyfile.PlyData.read(template)["vertex"].data.copy()
    if len(v) != len(g["means"]):
        raise ValueError(f"{len(g['means'])} Gaussians for a template of {len(v)}")
    for i, k in enumerate("xyz"):
        v[k] = g["means"][:, i]
    for i in range(4):
        v[f"rot_{i}"] = g["quats"][:, i]
    for i in range(3):
        v[f"scale_{i}"] = g["scales"][:, i]
    v["opacity"] = g["opacities"].reshape(-1)
    sh = g["sh"]
    for i in range(3):
        v[f"f_dc_{i}"] = sh[:, 0, i]
    rest = sorted((n for n in v.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest:  # channel-major, as load_splat reads them
        r = sh[:, 1:, :].transpose(0, 2, 1).reshape(len(v), -1)
        for j, n in enumerate(rest):
            v[n] = r[:, j]
    tmp = out + ".part"
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")], text=False).write(tmp)
    os.replace(tmp, out)


def shape_terms(project):
    """The flatten / no-needles weights the project's splat was trained with (report.json splat.regularize; else the
    defaults), for the fine-tune: without them it turns many Gaussians into needles."""
    from modelforge.pipeline import Report
    reg = (Report(project).data.get("splat") or {}).get("regularize") or {}
    return {"flatten": float(reg.get("flatten", splat.FLATTEN)), "needle": float(reg.get("needle", splat.NEEDLE)),
            "needle_ratio": 3.0}


def run(project, views, steps=STEPS, rounds=ROUNDS, shift_m=SHIFT_M, difix_width=0, on_line=print, job=None,
        on_progress=None, cleaner="difix"):
    """Refine PROJECT's splat towards `views` (gui/views.py records), the renders cleaned by `cleaner` ("difix" or
    "fixer"). on_progress(phase, done, total): "difix" views cleaned in a round, "train" fine-tune steps. Returns the
    stats (also report.json["difix"])."""
    from modelforge.pipeline import Report
    why = available(project, cleaner)
    if why:
        raise RuntimeError(why)
    if not views:
        raise RuntimeError("no target views: save some in the Difix tab first")
    job = job or jobs.Token()
    t0 = time.time()
    sp = splat_mesh.splat_path(project)
    backup = os.path.join(project, BACKUP)
    if baseline(project) is None:  # the splat before the first refinement (Compare, Undo); runs then build on it
        shutil.copy2(sp, backup)
    work = os.path.join(project, WORK)
    os.makedirs(work, exist_ok=True)
    npz_in, npz_out = os.path.join(work, "in.npz"), os.path.join(work, "out.npz")
    on_line(f"difix: reading {sp}")
    np.savez(npz_in, **splat_mesh.load_splat(sp))
    ds = _downscale(project)
    from modelforge import sky
    sky_dir = sky.folder(project, ds)
    spec = {"splat_npz": npz_in, "out_npz": npz_out, "repo": CODE, "model": MODEL,
            "transforms": os.path.join(project, "transforms.json"), "root": project, "downscale": ds,
            "images_dirs": [os.path.join(project, f"images_{ds}"), os.path.join(project, "images")],
            "sky_dir": sky_dir if os.path.isdir(sky_dir) else None, "steps": int(steps), "rounds": int(rounds),
            "shift_m": float(shift_m), "rot_m": 10.0, "novel_prob": 0.3, "novel_weight": 0.3,
            "difix_width": int(difix_width), "targets": [c2w(v).tolist() for v in views], "workdir": work,
            "cache_gb": 4.0, "cleaner": cleaner, "fixer_dir": FIXER, **shape_terms(project)}
    spec_path = os.path.join(work, "spec.json")
    json.dump(spec, open(spec_path, "w"), indent=1)

    def line(text):
        m = _LINE.search(text)
        if m and on_progress:
            if m.group(1):
                on_progress(f"difix round {m.group(1)} / {m.group(2)}: view", int(m.group(3)), int(m.group(4)))
            else:
                on_progress("fine-tune: step", int(m.group(5)), int(m.group(6)))
        if not m or (m.group(5) and int(m.group(5)) % 500 == 0):
            on_line(text)
    env = splat._env()
    first = [os.path.join(FIXER, d) for d in FIXER_PATH] if cleaner == "fixer" else [PKGS]
    env["PYTHONPATH"] = os.pathsep.join(p for p in (*first, env.get("PYTHONPATH")) if p)
    env["HF_HOME"] = os.path.join(semantic.MODEL_CACHE, "hf_cache")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "difix_refine.py")
    job.run([os.path.join(splat.NS_BIN, "python"), script, spec_path], line, env=env)
    on_line("difix: writing the refined splat")
    tmp = sp + ".difix.part"
    write_splat(sp, tmp, {k: v for k, v in np.load(npz_out).items()})
    json.dump({"splat": _key(tmp)}, open(os.path.join(project, STATE), "w"))  # before it replaces the splat (a rename
    os.replace(tmp, sp)  # keeps size + mtime), so a check in between never takes it for a new training
    for f in (npz_in, npz_out):
        os.remove(f)
    st = json.load(open(os.path.join(work, "stats.json")))
    stats = {**st, "views": [v["name"] for v in views], "cleaner": cleaner,
             "difix_width": int(difix_width) or ("1024 px" if cleaner == "fixer" else "training size"),
             "shift_m": float(shift_m), "seconds": round(time.time() - t0, 1),
             "time": time.strftime("%Y-%m-%d %H:%M"), "previews": os.path.relpath(work, project)}
    Report(project).set("difix", stats)
    on_line(f"difix: refined towards {len(views)} views in {stats['seconds']:.0f} s")
    return stats


def undo(project):
    """Back to the splat before the first refinement."""
    from modelforge.pipeline import Report
    backup = os.path.join(project, BACKUP)
    if not os.path.exists(backup):
        raise RuntimeError("no splat from before a Difix refinement")
    os.replace(backup, splat_mesh.splat_path(project))
    if os.path.exists(os.path.join(project, STATE)):
        os.remove(os.path.join(project, STATE))
    from modelforge import splat_lod
    pre = splat_lod.lod_dir(project, BACKUP)  # its level of detail (Difix tab → Compare) is now the splat's
    if os.path.isdir(pre):
        shutil.rmtree(splat_lod.lod_dir(project), ignore_errors=True)
        os.replace(pre, splat_lod.lod_dir(project))
    rep = Report(project)
    if "difix" in rep.data:
        rep.set("difix", None)


# --- suggested target views: where the splat is thin on the SfM mesh (the user's idea, 2026-10-26) ---------------
SUGGEST_N = 8
THIN = 0.2  # a mesh patch is thin when the splat there holds < THIN x the median (summed opacity of Gaussian centres)
MAX_SAMPLES = 400_000
MAX_DOWN_DEG = 75.0  # views look down at most this steeply (a straight-down view has no "up" for the viewer)
MIN_DOWN_DEG = 10.0  # and at least this (a drone's view; level or upward views end up in or under things)
CLEARANCE = 0.9  # a view keeps at least this x the clearance of the closest of the photos nearest to it


def _mesh_samples(project, spacing, rng):
    """(points, normals, area each) spread evenly over the SfM mesh's measured faces (patched holes left out)."""
    v, faces = sfm_mesh.read_mesh(os.path.join(project, "sfmmesh", "mesh.ply"))
    V = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    pf = os.path.join(project, "sfmmesh", "patched.npy")
    if os.path.exists(pf):
        patched = np.load(pf)
        if len(patched) == len(faces):
            faces = faces[~patched.astype(bool)]
    a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    keep = area > 0
    a, b, c, cr, area = a[keep], b[keep], c[keep], cr[keep], area[keep]
    n = int(min(MAX_SAMPLES, max(1000, area.sum() / spacing ** 2)))
    f = rng.choice(len(area), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    p = (1 - r1)[:, None] * a[f] + (r1 * (1 - r2))[:, None] * b[f] + (r1 * r2)[:, None] * c[f]
    return p, cr[f] / (2 * area[f])[:, None], area.sum() / n


def _splat_centres(project):
    """The splat's Gaussian centres (ENU) and opacities."""
    import plyfile
    g = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data
    G = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    return G, 1 / (1 + np.exp(-g["opacity"].astype(np.float64)))


def _splat_density(G, op, p, cell):
    """Summed opacity of the splat's Gaussian centres around each sample (the 3 x 3 x 3 cubes of `cell` around it).
    Low where the splat has few Gaussians on the surface: gaps, and areas drawn with a few large ones (low detail,
    e.g. foliage). Counting each Gaussian over its whole extent instead found nothing on berlin (0.0 %): some
    Gaussian reaches every point of its mesh."""
    lo = p.min(0) - 2 * cell
    hi = p.max(0) + 2 * cell
    inside = np.all((G > lo) & (G < hi), 1)
    kg = np.floor((G[inside] - lo) / cell).astype(np.int64)
    dims = np.floor((hi - lo) / cell).astype(np.int64) + 2
    lin = (kg[:, 0] * dims[1] + kg[:, 1]) * dims[2] + kg[:, 2]
    keys, inv = np.unique(lin, return_inverse=True)
    w = np.bincount(inv.ravel(), weights=op[inside])
    kp = np.floor((p - lo) / cell).astype(np.int64)
    out = np.zeros(len(p))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                q = ((kp[:, 0] + dx) * dims[1] + kp[:, 1] + dy) * dims[2] + kp[:, 2] + dz
                pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
                hit = keys[pos] == q
                out[hit] += w[pos[hit]]
    return out


def _cameras(project):
    """Training camera centres (ENU) and the vertical field of view (rad) from transforms.json."""
    t = json.load(open(os.path.join(project, "transforms.json")))
    C = np.array([np.array(f["transform_matrix"], float)[:3, 3] for f in t["frames"]])
    f0 = t["frames"][0]
    fy, h = f0.get("fl_y", t.get("fl_y")), f0.get("h", t.get("h"))
    return C, 2 * np.arctan(h / (2 * fy))


def _blocked(cam, target, pts, radius):
    """A mesh sample lies within radius of the line of sight (away from both ends)."""
    d = target - cam
    L = np.linalg.norm(d)
    u = d / L
    rel = pts - cam
    t = rel @ u
    m = (t > 0.02 * L) & (t < L - 3 * radius)
    if not m.any():
        return False
    perp = np.linalg.norm(rel[m] - t[m, None] * u, axis=1)
    return bool((perp < radius).any())


def _clearance(q, pts):
    """Distance from each of q to the nearest of pts (exact)."""
    return np.array([np.sqrt(((pts - x.astype(np.float32)) ** 2).sum(1).min()) for x in np.atleast_2d(q)])


def _buried(cam, pts, clear=1.0, reach=3.0):
    """The camera is under the mesh: samples within `reach` horizontally reach above it - clear."""
    m = np.linalg.norm(pts[:, :2] - cam[:2], axis=1) < reach
    return bool(m.any() and pts[m, 2].max() > cam[2] - clear)


def suggest_views(project, n=SUGGEST_N, thin=THIN, on_line=print, seed=0):
    """Target views for Difix where the splat is thin on the SfM mesh: the mesh's surface sampled at its point
    spacing, the splat's summed opacity per sample's neighbourhood (cells of 2 spacings) against the median; samples
    below `thin` x it grouped into connected patches (cells of 8 spacings), the largest n, each seen along its
    outward normal from the photos' typical distance (the median distance to its 10 nearest photo centres), so the
    patch fills the view. A view whose line of sight the mesh blocks, or that would look down more steeply than
    MAX_DOWN_DEG, turns towards the nearest photos' side. A view keeps the clearance a real photo there has: its
    distance to the nearest surface (mesh samples and the splat's opaque Gaussians, which also hold what the mesh
    lacks: trees, machines) at least CLEARANCE x the smallest of its 10 nearest photos'. Returns [{"name", "position", "look_at", "up", "fov",
    "suggested": {...}}] (views.pose layout) and stats."""
    from modelforge import semantic
    t0 = time.time()
    rng = np.random.default_rng(seed)
    st = json.load(open(os.path.join(project, "sfmmesh", "stats.json")))
    spacing = float(st.get("point_spacing") or 0.1)
    p, nrm, a_each = _mesh_samples(project, spacing, rng)
    G, op = _splat_centres(project)
    dens = _splat_density(G, op, p, 2 * spacing)
    med = float(np.median(dens))
    low = dens < thin * med
    on_line(f"suggest views: {len(p):,} mesh samples, splat density median {med:.2f}, "
            f"{low.mean():.1%} thin ({low.sum() * a_each:.0f} m²)")
    C, fov = _cameras(project)
    if not low.any():
        return [], {"samples": len(p), "thin_share": 0.0, "seconds": round(time.time() - t0, 1)}
    lowp, lown = p[low], nrm[low]
    # how far photos usually are from the surface: one view covers about a photo's footprint at that distance;
    # thin surface much farther from every photo is the scene's edge
    probe = p[rng.choice(len(p), min(len(p), 2000), replace=False)]
    typical = float(np.median([np.linalg.norm(C - q, axis=1).min() for q in probe]))
    r = typical * np.tan(fov / 2)  # half the footprint's height
    sub = p[rng.choice(len(p), min(len(p), 100_000), replace=False)]  # occluders
    obstacles = np.concatenate([sub, G[op > 0.5]]).astype(np.float32)  # what a camera keeps clear of (all of them:
    # a subsample misses the nearest)
    photo_clear = {}  # photo index -> its clearance, measured when first needed

    def needed(pos):
        near = np.argsort(np.linalg.norm(C - pos, axis=1))[:10]
        for i in near:
            if i not in photo_clear:
                photo_clear[i] = float(_clearance(C[i], obstacles)[0])
        return CLEARANCE * min(photo_clear[i] for i in near)
    # where views go: greedily the footprint-sized cell holding the most thin surface, its samples then taken
    cellk = np.floor(lowp / r).astype(np.int64)
    keys, inv = np.unique(cellk, axis=0, return_inverse=True)
    inv = inv.ravel()
    left = np.ones(len(lowp), bool)
    views, used, tried, too_close = [], [], 0, 0
    while len(views) < n and left.any() and tried < 20 * n:
        tried += 1
        count = np.bincount(inv[left], minlength=len(keys))
        best = int(np.argmax(count))
        if count[best] == 0:
            break
        cand = np.flatnonzero(left & (inv == best))
        centre = lowp[cand][np.argmin(np.linalg.norm(lowp[cand] - np.median(lowp[cand], 0), axis=1))]
        idx = np.flatnonzero(left & (np.linalg.norm(lowp - centre, axis=1) < r))
        left[idx] = False
        left[cand] = False
        if len(idx) * a_each < 0.1 * r * r:  # specks: the best cell holds hardly any thin surface; none better left
            break
        P = lowp[idx]
        dC = np.linalg.norm(C - centre, axis=1)
        if dC.min() > 2 * typical:  # the scene's edge: seen only from afar
            continue
        near = np.argsort(dC)[:10]
        towards = C[near].mean(0) - centre
        N = lown[idx] * np.sign(np.sum(lown[idx] * (C[near].mean(0) - P), 1))[:, None]  # outward: the photos' side
        N = N.sum(0)
        N = N / np.linalg.norm(N) if np.linalg.norm(N) > 0 else towards / np.linalg.norm(towards)
        dist = max(float(np.median(np.linalg.norm(C[near] - centre, axis=1))), 2.0)
        tw = towards / np.linalg.norm(towards)
        around = np.linalg.norm(C[:, :2] - centre[:2], axis=1) < dist
        top = float(C[around, 2].max() if around.any() else C[near, 2].max())  # the height flown around it
        pos = None
        for w in (0.0, 0.33, 0.66, 1.0):  # orthogonal first, then towards where the photos were taken
            d = (1 - w) * N + w * tw
            d /= np.linalg.norm(d)
            down = np.degrees(np.arcsin(np.clip(d[2], -1, 1)))  # d: from the patch to the camera
            if down > MAX_DOWN_DEG:  # tilt to MAX_DOWN_DEG towards the photos' side (horizontally)
                h = tw[:2] if np.linalg.norm(tw[:2]) > 1e-6 else np.array([0.0, 1.0])
                h = h / np.linalg.norm(h)
                e = np.radians(MAX_DOWN_DEG)
                d = np.array([h[0] * np.cos(e), h[1] * np.cos(e), np.sin(e)])  # from the patch up to the camera
                d = -d if d[2] < 0 else d
            if d[2] < np.sin(np.radians(MIN_DOWN_DEG)):  # from above only, as a drone sees it
                continue
            for k in (1.0, 0.6):
                kk = min(k, (top - centre[2]) / (dist * d[2])) if top > centre[2] else k
                if kk * dist < max(2.0, 0.5 * dist):  # the height cap would bring it too close
                    continue
                cand = centre + d * dist * kk
                if _blocked(cand, centre, sub, 2 * spacing) or _buried(cand, sub):
                    continue
                if _clearance(cand, obstacles)[0] < needed(cand):  # closer to things than any photo there
                    too_close += 1
                    continue
                pos = cand
                break
            if pos is not None:
                break
        if pos is None:
            continue
        used.append((centre, dist))
        views.append({"name": f"thin {len(views) + 1}", "position": pos.tolist(), "look_at": centre.tolist(),
                      "up": [0.0, 0.0, 1.0], "fov": float(fov),
                      "suggested": {"thin_m2": round(float(len(idx) * a_each), 1),
                                    "density": round(float(np.median(dens[low][idx]) / med), 3),
                                    "orthogonal": bool(w == 0.0), "distance_m": round(float(np.linalg.norm(pos - centre)), 1),
                                    "clearance_m": round(float(_clearance(pos, obstacles)[0]), 1)}})
    stats = {"samples": len(p), "thin_share": round(float(low.mean()), 3), "footprint_m": round(float(2 * r), 1),
             "views": len(views), "too_close": too_close, "seconds": round(time.time() - t0, 1)}
    on_line(f"suggest views: {len(views)} views (each about {2 * r:.0f} m across; {too_close} positions closer to "
            f"the surface than the photos there turned down), {stats['seconds']:.0f} s")
    return views, stats


def can_suggest(project):
    """None if suggest_views can run on project, else why not."""
    for f, why in ((os.path.join("sfmmesh", "mesh.ply"), "needs the SfM mesh (Sparse Mesh tab)"),
                   (os.path.join("sfmmesh", "stats.json"), "needs the SfM mesh (Sparse Mesh tab)"),
                   ("transforms.json", "needs an SfM export"),
                   (os.path.relpath(splat_mesh.splat_path(project), project), "needs a trained splat")):
        if not os.path.exists(os.path.join(project, f)):
            return why
    return None


def add_suggested(project, views):
    """Replace the earlier suggested targets in PROJECT's Difix list with `views` (suggest_views'); the views saved
    by hand stay."""
    from modelforge.gui import views as views_mod
    keep = [v for v in views_mod.load(project, VIEWS_FILE) if "suggested" not in v]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for v in views:
        keep.append({**v, "name": views_mod._unique(keep, v["name"]), "saved": now})
    views_mod._save(project, keep, VIEWS_FILE)
    return len(views)


def clear_views(project, suggested_only=True):
    """Remove the suggested targets (suggested_only) or every target from PROJECT's Difix list; the number removed."""
    from modelforge.gui import views as views_mod
    old = views_mod.load(project, VIEWS_FILE)
    keep = [v for v in old if "suggested" not in v] if suggested_only else []
    views_mod._save(project, keep, VIEWS_FILE)
    return len(old) - len(keep)
