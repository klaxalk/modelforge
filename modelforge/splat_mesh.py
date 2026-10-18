"""Mesh from the Gaussian splat (an alternative to COLMAP's dense MVS): TSDF fusion of depth rendered
from splat/splat_enu.ply at every camera, then marching cubes. Output PROJECT/splatmesh/:
mesh.ply (vertex colours, ENU), preview.ply (simplified, for the View tab), stats.json; stats also go
to report.json["splat_mesh"].

The splat and the cameras are in the project's ENU frame, so the mesh lines up with them and with the
COLMAP mesh. Rendering and fusion run in nerfstudio's venv (export/splat_tsdf.py: gsplat + Open3D) on
the GPU; this module prepares its input from the splat PLY and transforms.json.

Quality sets the voxel as pixels at the median rendered depth (coarse 8, medium 4, fine 2), at the
render width (default 960 px). Depth beyond depth_factor x the median depth (default 3) is not fused:
that keeps the far background (and sky floaters) out.

Usage (CUDA container):
    python -m modelforge.splat_mesh data/site [--quality medium] [--width 960] [--voxel 0.1]
        [--max-depth 80]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from modelforge import jobs, splat

OUT = "splatmesh"
QUALITY = {"coarse": 8.0, "medium": 4.0, "fine": 2.0}  # voxel = this many pixels at the median depth
DEFAULTS = {"voxel": None, "voxel_px": 4.0, "max_depth": None, "depth_factor": 3.0, "min_alpha": 0.7,
            "edge_tol": 0.05, "min_weight": 2.0, "max_preview_faces": 300_000, "min_component": 0.002}
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "splat_tsdf.py")


def splat_path(project):
    return os.path.join(project, "splat", "splat_enu.ply")


def available(project=None):
    """None if a splat mesh can be built here (and for project, when given), else the reason."""
    why = splat.available()
    if why:
        return why
    if project is not None and not os.path.exists(splat_path(project)):
        return "no splat yet: train one in the Splat tab first (splat/splat_enu.ply)"
    return None


def load_splat(path):
    """splat_enu.ply -> means, quats (wxyz), log-scales, opacity logits, SH (N, K, 3)."""
    import plyfile
    v = plyfile.PlyData.read(path)["vertex"].data
    names = v.dtype.names
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    n = len(v)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    if rest:  # stored channel-major (all R coefficients, then G, then B), as nerfstudio / 3DGS write them
        r = np.stack([v[k] for k in rest], 1).reshape(n, 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, r], 1)
    else:
        sh = dc
    return {"means": np.stack([v["x"], v["y"], v["z"]], 1), "quats": np.stack([v[f"rot_{i}"] for i in range(4)], 1),
            "scales": np.stack([v[f"scale_{i}"] for i in range(3)], 1), "opacities": np.asarray(v["opacity"]),
            "sh": sh.astype(np.float32)}


def cameras(project, width=960):
    """World -> OpenCV camera matrices and pinhole intrinsics of the transforms.json frames, scaled to width."""
    t = json.load(open(os.path.join(project, "transforms.json")))
    s = min(1.0, width / float(t["w"]))
    W, H = int(round(t["w"] * s)), int(round(t["h"] * s))
    K = np.array([[t["fl_x"] * s, 0, t["cx"] * s], [0, t["fl_y"] * s, t["cy"] * s], [0, 0, 1]], np.float32)
    views = []
    for f in t["frames"]:
        c2w = np.array(f["transform_matrix"], np.float64)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes (the export did the reverse)
        views.append(np.linalg.inv(c2w))
    return np.array(views, np.float32), np.repeat(K[None], len(views), 0), W, H


def parse_progress(line):
    """(i, n) from the "[tsdf] view i / n" lines, else None."""
    if line.startswith("[tsdf] view "):
        i, _, n = line[len("[tsdf] view "):].partition(" / ")
        try:
            return int(i), int(n)
        except ValueError:
            return None
    return None


def run(project, quality="medium", width=960, voxel=None, max_depth=None, on_line=print, on_progress=None,
        job=None, **overrides):
    """Build PROJECT/splatmesh/ from the splat. Returns the stats (also in report.json["splat_mesh"])."""
    from modelforge.pipeline import Report
    why = available(project)
    if why:
        raise RuntimeError(why)
    if not os.path.exists(os.path.join(project, "transforms.json")):
        raise RuntimeError("transforms.json missing: run the SfM export stage first")
    job = job or jobs.Token()
    t0 = time.time()
    out = os.path.join(project, OUT)
    os.makedirs(out, exist_ok=True)
    g = load_splat(splat_path(project))
    views, Ks, W, H = cameras(project, width)
    inp = os.path.join(out, "input.npz")
    np.savez(inp, viewmats=views, Ks=Ks, width=W, height=H, **g)
    params = dict(DEFAULTS, voxel_px=QUALITY[quality], voxel=voxel, max_depth=max_depth, **overrides)
    pfile = os.path.join(out, "params.json")
    json.dump(params, open(pfile, "w"), indent=1)
    on_line(f"splat mesh: {len(g['means']):,} gaussians, {len(views)} views at {W}x{H}, quality {quality}")

    def line(text):
        on_line(text)
        p = parse_progress(text)
        if p and on_progress:
            on_progress(*p)
    try:
        job.run([os.path.join(splat.NS_BIN, "python"), SCRIPT, inp, out, pfile], line, env=splat._env())
    finally:
        if os.path.exists(inp):
            os.remove(inp)
    stats = json.load(open(os.path.join(out, "stats.json")))
    stats.update(quality=quality, mesh=os.path.join(out, "mesh.ply"), seconds=round(time.time() - t0, 1))
    Report(project).set("splat_mesh", stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="medium")
    ap.add_argument("--width", type=int, default=960, help="render width in px (default 960)")
    ap.add_argument("--voxel", type=float, help="voxel size in m (default: from the quality)")
    ap.add_argument("--max-depth", type=float, help="depth cut-off in m (default: 3 x the median depth)")
    args = ap.parse_args()
    try:
        print(run(os.path.abspath(args.project), args.quality, args.width, args.voxel, args.max_depth,
                  on_line=lambda m: print(m, flush=True)))
    except (RuntimeError, jobs.Cancelled) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
