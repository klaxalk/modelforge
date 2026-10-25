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

from modelforge import jobs, semantic, splat, splat_mesh

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKGS = next((p for p in (os.environ.get("MODELFORGE_DIFIX_PKGS"), "/opt/difix-pkgs",
                         os.path.join(_REPO, "build", "difix_pkgs")) if p and os.path.isdir(p)), None)
CODE = next((p for p in (os.environ.get("MODELFORGE_DIFIX_REPO"), "/opt/difix3d",
                         os.path.join(_REPO, "build", "difix", "Difix3D"))
             if p and os.path.exists(os.path.join(p, "src", "pipeline_difix.py"))), None)
MODEL = "nvidia/difix_ref"
VIEWS_FILE = "difix_views.json"
BACKUP = os.path.join("splat", "splat_enu.pre_difix.ply")
WORK = os.path.join("splat", "difix")
STEPS, ROUNDS, SHIFT_M = 3000, 4, 3.0
WIDTHS = {"training size": 0, "1024 px": 1024}  # Difix's input width (it runs at its input size; 15 GB at 1920)
_LINE = re.compile(r"\[difix\] (?:round (\d+) / (\d+): view (\d+) / (\d+)|step (\d+) / (\d+))")


def available(project=None):
    """None if a Difix refinement can run here (and on project, when given), else the reason."""
    why = splat.available()
    if why:
        return why
    if PKGS is None or CODE is None:
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


def refined(project):
    """{"backup": bool, "info": the last refinement's record} for the tab's status line."""
    rep = os.path.join(project, "report.json")
    info = (json.load(open(rep)).get("difix") if os.path.exists(rep) else None) or None
    return {"backup": os.path.exists(os.path.join(project, BACKUP)), "info": info}


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


def run(project, views, steps=STEPS, rounds=ROUNDS, shift_m=SHIFT_M, difix_width=0, on_line=print, job=None,
        on_progress=None):
    """Refine PROJECT's splat towards `views` (gui/views.py records). on_progress(phase, done, total): "difix" views
    cleaned in a round, "train" fine-tune steps. Returns the stats (also report.json["difix"])."""
    from modelforge.pipeline import Report
    why = available(project)
    if why:
        raise RuntimeError(why)
    if not views:
        raise RuntimeError("no target views: save some in the Difix tab first")
    job = job or jobs.Token()
    t0 = time.time()
    sp = splat_mesh.splat_path(project)
    backup = os.path.join(project, BACKUP)
    if not os.path.exists(backup):  # the splat before the first refinement (Undo)
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
            "transforms": os.path.join(project, "transforms.json"), "downscale": ds,
            "images_dirs": [os.path.join(project, f"images_{ds}"), os.path.join(project, "images")],
            "sky_dir": sky_dir if os.path.isdir(sky_dir) else None, "steps": int(steps), "rounds": int(rounds),
            "shift_m": float(shift_m), "rot_m": 10.0, "novel_prob": 0.3, "novel_weight": 0.3,
            "difix_width": int(difix_width), "targets": [c2w(v).tolist() for v in views], "workdir": work,
            "cache_gb": 4.0}
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
    env["PYTHONPATH"] = os.pathsep.join(p for p in (PKGS, env.get("PYTHONPATH")) if p)
    env["HF_HOME"] = os.path.join(semantic.MODEL_CACHE, "hf_cache")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "difix_refine.py")
    job.run([os.path.join(splat.NS_BIN, "python"), script, spec_path], line, env=env)
    on_line("difix: writing the refined splat")
    write_splat(sp, sp, {k: v for k, v in np.load(npz_out).items()})
    for f in (npz_in, npz_out):
        os.remove(f)
    st = json.load(open(os.path.join(work, "stats.json")))
    stats = {**st, "views": [v["name"] for v in views], "difix_width": int(difix_width) or "training size",
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
    rep = Report(project)
    if "difix" in rep.data:
        rep.set("difix", None)
