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

Patch holes (`fill`): border loops up to a size, other than each piece's outer border, closed by a fan of
triangles from their mean point (mesh_unpatched.ply keeps the marching-cubes mesh).

Usage (CUDA container):
    python -m modelforge.splat_mesh data/site [--quality medium] [--width 960] [--voxel 0.1]
        [--max-depth 80]
    python -m modelforge.splat_mesh data/site --fill [MAX_M]
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
    for name in (UNPATCHED, "patched.npy"):  # a patch of the previous mesh
        if os.path.exists(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    stats = json.load(open(os.path.join(out, "stats.json")))
    stats.update(quality=quality, mesh=os.path.join(out, "mesh.ply"), seconds=round(time.time() - t0, 1))
    Report(project).set("splat_mesh", stats)
    return stats


UNPATCHED = "mesh_unpatched.ply"  # the marching-cubes mesh while mesh.ply is the patched one


def hole_loops(faces):
    """The mesh's border loops. Returns (edges (E, 2) directed as their faces have them, loop label per edge, piece
    label per edge, simple per edge: its loop is a clean cycle)."""
    from modelforge import sfm_mesh
    n_f = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    u = np.sort(e, axis=1)
    key = u[:, 0] * (int(faces.max()) + 1) + u[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    border = np.flatnonzero(cnt[inv] == 1)
    del u, key, inv, cnt
    b, face = e[border], border % n_f
    del e
    piece = sfm_mesh.components(faces, np.ones(n_f, bool))[face]
    # loops: each border edge continues with the border edge leaving its end vertex. Where two holes touch at a
    # vertex (common in marching-cubes meshes) it has two border edges in and two out: turning around the vertex
    # through the faces (edge u->w of face (u, w, q) -> w->q -> its twin q->w -> ...) reaches the other hole's
    # outgoing edge, so the incoming edge continues with the other one and each hole is closed on its own. Vertices
    # where more holes touch, or with more border edges in than out (or the reverse), form no clean cycle
    n_v = int(faces.max()) + 1
    out_deg, in_deg = np.bincount(b[:, 0], minlength=n_v), np.bincount(b[:, 1], minlength=n_v)
    by_src = np.argsort(b[:, 0], kind="stable")
    start_out = np.r_[0, np.cumsum(out_deg)[:-1]]
    succ = np.full(len(b), -1, np.int64)
    w = b[:, 1]
    plain = (out_deg[w] == 1) & (in_deg[w] == 1)
    succ[plain] = by_src[start_out[w[plain]]]
    pinch = (out_deg[w] == 2) & (in_deg[w] == 2)
    if pinch.any():
        bkey = b[:, 0] * n_v + b[:, 1]
        border_order = np.argsort(bkey)
        bkey_sorted = bkey[border_order]
        is_p = np.zeros(n_v, bool)
        is_p[w[pinch]] = True
        fp = faces[is_p[faces].any(1)]
        de = np.concatenate([fp[:, [0, 1, 2]], fp[:, [1, 2, 0]], fp[:, [2, 0, 1]]])  # edge (a, b) + third vertex
        dkey = de[:, 0] * n_v + de[:, 1]
        o = np.argsort(dkey)
        dkey, third = dkey[o], de[o, 2]

        def find(sorted_keys, k):
            i = np.minimum(np.searchsorted(sorted_keys, k), len(sorted_keys) - 1)
            return i, sorted_keys[i] == k
        todo = np.flatnonzero(pinch)
        u, ww = b[todo, 0], b[todo, 1]
        for _ in range(64):  # valence bound
            i, ok = find(dkey, u * n_v + ww)
            q = np.where(ok, third[i], -1)
            j, hit = find(bkey_sorted, ww * n_v + q)
            hit &= ok
            walked = border_order[j[hit]]
            first_out = by_src[start_out[ww[hit]]]  # the vertex's two outgoing border edges: take the other one
            succ[todo[hit]] = np.where(walked == first_out, by_src[start_out[ww[hit]] + 1], first_out)
            keep = ok & ~hit
            todo, u, ww = todo[keep], q[keep], ww[keep]  # continue from the twin q -> w
            if not len(todo):
                break
    has_pred = np.zeros(len(b), bool)
    has_pred[succ[succ >= 0]] = True
    linked = succ >= 0
    label = sfm_mesh.union_labels(len(b), np.flatnonzero(linked), succ[linked])
    bad_loop = np.zeros(len(b), bool)
    np.logical_or.at(bad_loop, label, ~linked | ~has_pred)
    simple = ~bad_loop[label]
    return b, label, piece, simple


def fill(project, max_hole=None, on_line=print, token=None):
    """Close the holes of PROJECT/splatmesh/mesh.ply up to max_hole across (the bounding box's longest side; None:
    40 voxels): each border loop that is not the outer border of its piece gets a fan of triangles from its mean
    point (colour and normal the loop's mean). Interpolated, not measured. The marching-cubes mesh is kept as
    mesh_unpatched.ply, so patching again (another size) starts from it; Build mesh replaces both."""
    from modelforge import sfm_mesh
    from modelforge.pipeline import Report
    token = token or jobs.Token()
    t0 = time.time()
    out = os.path.join(project, OUT)
    mesh, raw = os.path.join(out, "mesh.ply"), os.path.join(out, UNPATCHED)
    if not os.path.exists(mesh) and not os.path.exists(raw):
        raise RuntimeError("no splat mesh to patch: Build mesh from splat first")
    stats_path = os.path.join(out, "stats.json")
    stats = json.load(open(stats_path))
    if max_hole is None or max_hole <= 0:
        max_hole = 40 * stats["voxel_m"]
    if not os.path.exists(raw):
        os.replace(mesh, raw)
    on_line("splat mesh: reading the mesh")
    v, faces = sfm_mesh.read_mesh(raw)
    token.check()
    on_line(f"splat mesh: finding holes among {len(faces):,} faces")
    b, label, piece, simple = hole_loops(faces)
    token.check()
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    n_l = int(label.max()) + 1 if len(label) else 0
    lo, hi = np.full((n_l, 3), np.inf), np.full((n_l, 3), -np.inf)
    p = xyz[b[:, 0]]
    np.minimum.at(lo, label, p)
    np.maximum.at(hi, label, p)
    size = (hi - lo).max(1) if n_l else np.zeros(0)
    # each piece's widest loop is its outer border (for the ground: the scene's edge), never a hole
    loop_piece = np.zeros(n_l, np.int64)
    loop_piece[label] = piece
    order = np.lexsort((-size, loop_piece))
    outer = np.zeros(n_l, bool)
    first = np.r_[True, loop_piece[order][1:] != loop_piece[order][:-1]] if n_l else np.zeros(0, bool)
    outer[order[first]] = True
    ok_loop = np.zeros(n_l, bool)
    ok_loop[label[simple]] = True
    ok_loop[label[~simple]] = False
    exists = np.zeros(n_l, bool)
    exists[label] = True
    hole = exists & ~outer & (size <= max_hole) & ok_loop
    skipped = int((exists & ~outer & (size <= max_hole) & ~ok_loop).sum())
    sel = hole[label]
    holes = np.flatnonzero(hole)
    new_id = np.full(n_l, -1, np.int64)
    new_id[holes] = len(v) + np.arange(len(holes))
    cnt = np.bincount(label[sel], minlength=n_l)[holes]
    centre = np.zeros((n_l, 3))
    np.add.at(centre, label[sel], p[sel])
    nv = np.zeros(len(holes), v.dtype)
    for i, ax in enumerate("xyz"):
        nv[ax] = centre[holes, i] / cnt
    names = v.dtype.names
    for group in (("red", "green", "blue"), ("nx", "ny", "nz")):
        if all(g in names for g in group):
            acc = np.zeros((n_l, 3))
            np.add.at(acc, label[sel], np.c_[tuple(v[g][b[sel, 0]].astype(np.float64) for g in group)])
            m = acc[holes] / cnt[:, None]
            if group[0] == "nx":
                m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
            for i, g in enumerate(group):
                nv[g] = np.round(m[:, i]) if v.dtype[g].kind == "u" else m[:, i]
    new_faces = np.c_[b[sel, 1], b[sel, 0], new_id[label[sel]]]  # reversed: the hole's side of the border edge
    token.check()
    on_line(f"splat mesh: writing the patched mesh ({len(holes):,} holes)")
    all_faces = np.concatenate([faces, new_faces])
    n_v = sfm_mesh.write_mesh(mesh, np.concatenate([v, nv]), all_faces)
    np.save(os.path.join(out, "patched.npy"), np.r_[np.zeros(len(faces), bool), np.ones(len(new_faces), bool)])
    on_line("splat mesh: simplifying the preview")
    sfm_mesh.write_preview(out, mesh, len(all_faces), token)
    rep = {"holes": len(holes), "faces_added": len(new_faces), "loops": int(exists.sum()), "skipped_not_closed": skipped,
           "max_hole": round(float(max_hole), 3),
           "largest_patched": round(float(size[holes].max()), 3) if len(holes) else 0.0,
           "faces": len(all_faces), "vertices": n_v, "seconds": round(time.time() - t0, 1)}
    stats["fill"] = rep
    json.dump(stats, open(stats_path, "w"), indent=1)
    Report(project).set("splat_mesh", stats)
    on_line(f"splat mesh: {len(holes):,} holes up to {max_hole:.2f} across closed with {len(new_faces):,} faces "
            f"({int(exists.sum()):,} border loops, {skipped:,} of them small but not closed; the rest are pieces' outer borders, larger, or not closed)")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--quality", choices=tuple(QUALITY), default="medium")
    ap.add_argument("--width", type=int, default=960, help="render width in px (default 960)")
    ap.add_argument("--voxel", type=float, help="voxel size in m (default: from the quality)")
    ap.add_argument("--max-depth", type=float, help="depth cut-off in m (default: 3 x the median depth)")
    ap.add_argument("--fill", nargs="?", type=float, const=0.0, metavar="MAX",
                    help="only patch the existing mesh's holes up to MAX m across (default 40 voxels)")
    args = ap.parse_args()
    if args.fill is not None:
        print(fill(os.path.abspath(args.project), args.fill or None, on_line=lambda m: print(m, flush=True)))
        return
    try:
        print(run(os.path.abspath(args.project), args.quality, args.width, args.voxel, args.max_depth,
                  on_line=lambda m: print(m, flush=True)))
    except (RuntimeError, jobs.Cancelled) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
