"""Mesh from the SfM points (stage `sfmmesh`): a surface the splat can later be held to (dense depth / normals where
the SfM points are sparse), and a quick look at what the sparse model says the ground is.

The points of the export's model (sparse/export: triangulated without the photos the export left out; else
sparse/refined) get normals from their 16 nearest neighbours (PCA), turned towards a camera that sees them; COLMAP's
Poisson mesher (`poisson_mesher`, killable) closes them into a surface. Poisson closes everything, also across gaps
and far beyond the scene's edge, so faces with a vertex farther than `trim` from every SfM point are cut away (its
own density trim deletes nearly everything on sparse clouds). Octree depth from the extent and the point spacing
(cells of ~2 point spacings).

Output PROJECT/sfmmesh/: points.ply (the oriented points), poisson.ply (untrimmed) + kept.npy (its faces kept by
the trim), mesh.ply (vertex colours, the project's world frame), preview.ply (<= 300k faces, for the View tab),
stats in report.json["sfm_mesh"].

`fill` (stage `sfmmesh-fill`, run by hand) patches holes: the faces the trim cut form connected pieces; the pieces
enclosed by the kept mesh are Poisson's own smooth surface across a hole (everything open to the outside joins one
large piece around the scene), and those up to `max_hole` across are put back. mesh.ply then holds the patched
mesh (patched.npy marks its patched faces), mesh_trimmed.ply the trimmed one. Patched faces are interpolated, not
measured.

`colour` (stage `sfmmesh-colour`, run by hand): the vertices coloured from the photos (export/mesh_color.py in
nerfstudio's venv: projected with the cameras' distortion, occlusion by raycasting against the mesh, weighted mean
favouring near, head-on views) -> mesh_coloured.ply + preview_coloured.ply (<= 1M faces) for the View tab. A separate
file, so mesh.ply (the splat's depth prior, its cached depth maps) stays as it is.
"""
import argparse
import json
import os
import re
import time

import numpy as np

from modelforge import dense, jobs

OUT = "sfmmesh"
NEIGHBOURS = 16
PREVIEW_FACES = 300_000
COLOURED_PREVIEW_FACES = 1_000_000  # the coloured mesh is for looking at: more detail in the View tab
COLOURED = ("mesh_coloured.ply", "mesh_coloured.ply.json", "preview_coloured.ply", "colour_live.npy")


def depth_cloud(project):
    """The SfM mesh (its vertices are the splat's depth targets with the SfM mesh prior), or None."""
    p = os.path.join(project, OUT, "mesh.ply")
    return os.path.abspath(p) if os.path.exists(p) else None


def model_dir(project):
    """The export's model (sparse/export: triangulated without the photos it left out) when it belongs to the
    current refine (not older than sparse/refined), else sparse/refined."""
    exp, ref = (os.path.join(project, "sparse", n, "points3D.bin") for n in ("export", "refined"))
    if os.path.exists(exp) and (not os.path.exists(ref) or os.path.getmtime(exp) >= os.path.getmtime(ref)):
        return os.path.dirname(exp)
    if os.path.exists(ref):
        return os.path.dirname(ref)
    raise RuntimeError("no SfM model yet (sparse/refined): run refine first")


def oriented_points(rec):
    """(xyz, rgb, normals, median point spacing) of the model's points without strays; normals by PCA over the
    nearest neighbours, pointing towards the first camera that sees each point.

    Strays matter: Poisson's octree spans the bounding box of all its points, and a few SfM points kilometres out
    (fullrun_test: a 1.45 km box for a ~300 m scene) make its cells 5x coarser. Kept: points inside the 0.5-99.5 %
    box widened by 10 % per side whose NEIGHBOURS-th neighbour is within 10x the median of that distance."""
    import cv2
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts])
    rgb = np.array([p.color for p in pts], np.uint8)
    centre = {i: rec.image(i).projection_center() for i in rec.reg_image_ids()}
    cam = np.array([centre[p.track.elements[0].image_id] for p in pts])
    lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0)
    box = np.all((xyz >= lo - 0.1 * (hi - lo)) & (xyz <= hi + 0.1 * (hi - lo)), axis=1)
    xyz, rgb, cam = xyz[box], rgb[box], cam[box]
    q = (xyz - xyz.mean(0)).astype(np.float32)
    index = cv2.flann_Index(q, dict(algorithm=1, trees=4))
    nrm, near, far = np.zeros_like(xyz), np.zeros(len(q)), np.zeros(len(q))
    for s in range(0, len(q), 200_000):
        k, d2 = index.knnSearch(q[s:s + 200_000], NEIGHBOURS, params={"checks": 64})
        nb = q[k] - q[k].mean(1, keepdims=True)
        _, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
        nrm[s:s + 200_000] = vec[:, :, 0]  # smallest eigenvalue: the surface normal
        near[s:s + 200_000], far[s:s + 200_000] = np.sqrt(d2[:, 1]), np.sqrt(d2[:, -1])
    keep = far <= 10 * np.median(far)
    flip = np.einsum("ij,ij->i", nrm, cam - xyz) < 0
    nrm[flip] *= -1
    return xyz[keep], rgb[keep], nrm[keep], float(np.median(near[keep]))


def write_points(path, xyz, rgb, nrm):
    import plyfile
    v = np.empty(len(xyz), [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for k, a in enumerate("xyz"):
        v[a], v["n" + a] = xyz[:, k], nrm[:, k]
    v["red"], v["green"], v["blue"] = rgb.T
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(path)


def read_mesh(path):
    """(vertex records, faces (F, 3))."""
    import plyfile
    try:  # triangles: read as one (F, 3) block instead of a Python object per face (23M faces in seconds)
        ply = plyfile.PlyData.read(path, known_list_len={"face": {"vertex_indices": 3, "vertex_index": 3}})
    except Exception:  # noqa: BLE001 - other polygons: the general reader
        ply = plyfile.PlyData.read(path)
    f = ply["face"].data
    key = "vertex_indices" if "vertex_indices" in f.dtype.names else "vertex_index"  # COLMAP writes the latter
    if not len(f):
        return ply["vertex"].data, np.zeros((0, 3), int)
    idx = f[key]
    return ply["vertex"].data, (idx if idx.dtype != object else np.stack(idx)).astype(np.int64)


def write_mesh(path, v, faces):
    """The faces with only the vertices they use (binary PLY, written as one block). Returns the vertex count."""
    used = np.unique(faces)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    vert = np.asarray(v[used])
    names = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "u2": "ushort", "i2": "short",
             "u4": "uint", "i4": "int"}
    vdt = np.dtype([(n, vert.dtype[n].newbyteorder("<")) for n in vert.dtype.names])
    fe = np.empty(len(faces), [("n", "u1"), ("vertex_indices", "<i4", (3,))])
    fe["n"], fe["vertex_indices"] = 3, remap[faces]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vert)}"]
    header += [f"property {names[vdt[n].str[1:]]} {n}" for n in vdt.names]
    header += [f"element face {len(fe)}", "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(vert.astype(vdt).tobytes())
        fh.write(fe.tobytes())
    return len(used)


def transfer_colours(src, dst):
    """dst's vertices get the colour of the nearest vertex of src (if src has colours)."""
    import cv2
    v_src, _ = read_mesh(src)
    if "red" not in v_src.dtype.names:
        return
    v_dst, f_dst = read_mesh(dst)
    a = np.c_[v_src["x"], v_src["y"], v_src["z"]]
    c = a.mean(0)
    index = cv2.flann_Index((a - c).astype(np.float32), dict(algorithm=1, trees=4))
    b = np.c_[v_dst["x"], v_dst["y"], v_dst["z"]]
    k, _ = index.knnSearch((b - c).astype(np.float32), 1, params={"checks": 32})
    v = np.empty(len(v_dst), [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = b.T
    for ch in ("red", "green", "blue"):
        v[ch] = v_src[ch][k[:, 0]]
    write_mesh(dst, v, f_dst)


def trim(v, faces, xyz, max_dist):
    """Faces whose vertices all lie within max_dist of an SfM point (bool per face)."""
    import cv2
    vx = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    c = xyz.mean(0)
    index = cv2.flann_Index((xyz - c).astype(np.float32), dict(algorithm=1, trees=4))
    _, d2 = index.knnSearch((vx - c).astype(np.float32), 1, params={"checks": 32})
    near = np.sqrt(d2[:, 0]) <= max_dist
    return near[faces].all(1)


def write_preview(out, mesh, n_faces, token, name="preview.ply", faces=PREVIEW_FACES):
    """A simplified copy (<= faces) for the View tab, with the mesh's vertex colours: COLMAP's mesh_simplifier
    writes positions only, so every preview vertex takes the colour of the nearest vertex of the mesh."""
    preview = os.path.join(out, name)
    if n_faces > faces:
        dense._colmap("mesh_simplifier", "--input_path", mesh, "--output_path", preview,
                      "--MeshSimplification.target_face_ratio", f"{faces / n_faces:.6f}", token=token)
        transfer_colours(mesh, preview)
    elif os.path.exists(preview):
        os.remove(preview)


def run(project, depth=None, trim_spacings=6.0, on_line=print, token=None):
    """Build PROJECT/sfmmesh/. depth: Poisson octree depth (None: from extent / (2 x point spacing), 8-12);
    trim_spacings: faces farther than this many median point spacings from every SfM point are cut."""
    import pycolmap

    from modelforge.pipeline import Report
    token = token or jobs.Token()
    t0 = time.time()
    src = model_dir(project)
    out = os.path.join(project, OUT)
    os.makedirs(out, exist_ok=True)
    on_line(f"sfm mesh: reading {os.path.relpath(src, project)}")
    rec = pycolmap.Reconstruction(src)
    if rec.num_points3D() < 100:
        raise RuntimeError(f"{os.path.relpath(src, project)} has only {rec.num_points3D()} points")
    on_line(f"sfm mesh: normals of {rec.num_points3D():,} points")
    xyz, rgb, nrm, spacing = oriented_points(rec)
    del rec
    token.check()
    extent = float(np.ptp(xyz, axis=0).max())  # Poisson's octree spans the whole box
    if depth is None:
        depth = int(np.clip(np.ceil(np.log2(extent / (2 * spacing))), 8, 12))
    pts = os.path.join(out, "points.ply")
    write_points(pts, xyz, rgb, nrm)
    raw = os.path.join(out, "poisson.ply")
    on_line(f"sfm mesh: Poisson over {len(xyz):,} points, octree depth {depth} "
            f"(cells ~{extent / 2 ** depth:.2f}, point spacing {spacing:.3f})")
    dense._colmap("poisson_mesher", "--input_path", pts, "--output_path", raw, "--PoissonMeshing.depth", str(depth),
                  "--PoissonMeshing.trim", "0", "--PoissonMeshing.color", "1", token=token)
    token.check()
    mesh = os.path.join(out, "mesh.ply")
    max_dist = trim_spacings * spacing
    on_line(f"sfm mesh: cutting faces farther than {max_dist:.2f} from the points")
    v, faces = read_mesh(raw)
    kept = trim(v, faces, xyz, max_dist)
    np.save(os.path.join(out, "kept.npy"), kept)
    n_v, n_f, n_raw = write_mesh(mesh, v, faces[kept]), int(kept.sum()), len(faces)
    for name in ("mesh_trimmed.ply", "patched.npy") + COLOURED:  # earlier patch / colours: the earlier mesh
        if os.path.exists(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    on_line(f"sfm mesh: {n_f:,} of {n_raw:,} faces within {max_dist:.2f} of an SfM point; writing the preview")
    write_preview(out, mesh, n_f, token)
    stats = {"model": os.path.relpath(src, project), "points": len(xyz), "point_spacing": round(spacing, 4),
             "depth": depth, "cell": round(extent / 2 ** depth, 4), "trim_dist": round(max_dist, 4),
             "faces": n_f, "vertices": n_v, "faces_untrimmed": n_raw, "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    Report(project).set("sfm_mesh", stats)
    on_line(f"sfm mesh: {stats}")
    return stats


def components(faces, mask):
    """Connected pieces of the masked faces (sharing an edge): a label per face (-1 outside the mask)."""
    idx = np.flatnonzero(mask)
    label = np.full(len(faces), -1, np.int64)
    if not len(idx):
        return label
    f = faces[idx]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(idx)), 3)
    key = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    o = np.argsort(key, kind="stable")
    key, owner = key[o], owner[o]
    same = key[1:] == key[:-1]
    label[idx] = union_labels(len(idx), owner[:-1][same], owner[1:][same])  # faces sharing an edge
    return label


def union_labels(n, a, b):
    """Connected components of n nodes joined by the pairs (a, b): the smallest node of its component per node."""
    parent = np.arange(n)
    while True:  # hooking to the smaller root, then pointer jumping: a few dozen rounds
        pa, pb = parent[a], parent[b]
        if np.array_equal(pa, pb):
            break
        m = np.minimum(pa, pb)
        np.minimum.at(parent, pa, m)
        np.minimum.at(parent, pb, m)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
    return parent


def can_fill(project):
    """The mesh has what `fill` needs (meshes from before patching existed lack the untrimmed surface)."""
    return all(os.path.exists(os.path.join(project, OUT, n)) for n in ("poisson.ply", "kept.npy", "stats.json"))


def fill(project, max_hole=None, on_line=print, token=None):
    """Patch holes in PROJECT/sfmmesh/mesh.ply with Poisson's surface across them (see the module doc).
    max_hole: the largest piece put back, across (its bounding box's longest side; None: 40 point spacings)."""
    from modelforge.pipeline import Report
    token = token or jobs.Token()
    t0 = time.time()
    out = os.path.join(project, OUT)
    raw, kept_path = os.path.join(out, "poisson.ply"), os.path.join(out, "kept.npy")
    if not can_fill(project):
        raise RuntimeError("no SfM mesh to patch (none yet, or one built before hole patching existed): "
                           "Run SfM mesh first")
    stats = json.load(open(os.path.join(out, "stats.json")))
    if max_hole is None or max_hole <= 0:
        max_hole = 40 * stats["point_spacing"]
    on_line("sfm mesh: reading the Poisson surface")
    v, faces = read_mesh(raw)
    kept = np.load(kept_path)
    if len(kept) != len(faces):
        raise RuntimeError("sfmmesh/kept.npy does not match poisson.ply: run SfM mesh again")
    on_line(f"sfm mesh: finding holes among {int((~kept).sum()):,} cut faces")
    label = components(faces, ~kept)
    token.check()
    vx = np.c_[v["x"], v["y"], v["z"]]
    cut = np.flatnonzero(~kept)
    o = np.argsort(label[cut], kind="stable")
    cut = cut[o]
    starts = np.r_[0, np.flatnonzero(np.diff(label[cut])) + 1]
    ends = np.r_[starts[1:], len(cut)]
    patched, holes, sizes = np.zeros(len(faces), bool), 0, []
    for s, e in zip(starts, ends):
        piece = cut[s:e]
        size = float(np.ptp(vx[faces[piece].ravel()], axis=0).max())
        if size <= max_hole:
            patched[piece] = True
            holes += 1
            sizes.append(size)
    mesh = os.path.join(out, "mesh.ply")
    trimmed = os.path.join(out, "mesh_trimmed.ply")
    if not os.path.exists(trimmed):
        os.replace(mesh, trimmed)
    keep = kept | patched
    on_line(f"sfm mesh: writing the patched mesh ({holes} holes)")
    n_v = write_mesh(mesh, v, faces[keep])
    for name in COLOURED:  # coloured before the patch: the patch has no photo colours
        if os.path.exists(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    np.save(os.path.join(out, "patched.npy"), patched[keep])  # per face of mesh.ply
    write_preview(out, mesh, int(keep.sum()), token)
    rep = {"holes": holes, "faces_added": int(patched.sum()), "pieces_cut": len(starts), "max_hole": round(max_hole, 3),
           "largest_patched": round(max(sizes), 3) if sizes else 0.0, "faces": int(keep.sum()), "vertices": n_v,
           "seconds": round(time.time() - t0, 1)}
    stats["fill"] = rep
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    Report(project).set("sfm_mesh", stats)
    on_line(f"sfm mesh: {holes} holes up to {max_hole:.2f} across patched with {rep['faces_added']:,} faces "
            f"({len(starts)} cut pieces; the rest is open to the outside or larger)")
    return rep


LIVE = "colour_live.npy"  # the colours so far while colouring runs (uint8 RGB per vertex of mesh.ply)


def write_coloured(mesh_path, rgb, out_path):
    """mesh_path with its vertex colours replaced by rgb (N, 3 uint8)."""
    import plyfile
    ply = plyfile.PlyData.read(mesh_path)
    v = ply["vertex"].data
    if "red" not in v.dtype.names:
        from numpy.lib import recfunctions
        v = recfunctions.append_fields(v, ("red", "green", "blue"), [rgb[:, 0], rgb[:, 1], rgb[:, 2]],
                                       dtypes=["u1"] * 3, usemask=False)
    else:
        v = v.copy()
        v["red"], v["green"], v["blue"] = rgb.T
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex"), ply["face"]]).write(out_path + ".part.ply")
    os.replace(out_path + ".part.ply", out_path)


def colour(project, width=1920, on_line=print, token=None, on_live=None, every=100):
    """Colour PROJECT/sfmmesh/mesh.ply from the photos of transforms.json (see the module doc).
    on_live(path, k, n): the colours after k of n photos were saved to path (sfmmesh/colour_live.npy), every
    `every` photos. Stopped (jobs.Cancelled), the colours saved last become mesh_coloured.ply (partial) before the
    Cancelled goes on."""
    from modelforge import splat
    from modelforge.pipeline import Report
    token = token or jobs.Token()
    out = os.path.join(project, OUT)
    mesh = os.path.join(out, "mesh.ply")
    if not os.path.exists(mesh):
        raise RuntimeError("no SfM mesh yet: Run SfM mesh first")
    if not os.path.exists(os.path.join(project, "transforms.json")):
        raise RuntimeError("transforms.json missing: run export first")
    why = splat.available()
    if why:
        raise RuntimeError(f"colouring runs in nerfstudio's environment: {why}")
    coloured = os.path.join(out, "mesh_coloured.ply")
    live = os.path.join(out, LIVE)
    if os.path.exists(live):
        os.remove(live)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "mesh_color.py")
    done = [0, 0]

    def line(text):
        on_line(text)
        m = re.match(r"mesh colour: live (\d+) / (\d+)", text)
        if m:
            done[:] = int(m.group(1)), int(m.group(2))
            if on_live:
                on_live(live, *done)

    def finish(rep, tok):
        import plyfile
        on_line("mesh colour: writing the preview")
        n_faces = plyfile.PlyData.read(coloured)["face"].count
        write_preview(out, coloured, n_faces, tok, name="preview_coloured.ply", faces=COLOURED_PREVIEW_FACES)
        stats = json.load(open(os.path.join(out, "stats.json")))
        stats["colour"] = rep
        json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
        Report(project).set("sfm_mesh", stats)
    try:
        splat._stream([os.path.join(splat.NS_BIN, "python"), script, project, mesh, coloured, "--width", str(int(width)),
                       "--live", live, "--every", str(int(every))], line, token)
    except jobs.Cancelled:
        if os.path.exists(live):  # keep what is done: the colours saved last
            on_line(f"mesh colour: stopped after {done[0]} of {done[1]} photos: keeping the colours so far")
            write_coloured(mesh, np.load(live), coloured)
            rep = {"photos": done[1], "photos_done": done[0], "width": width, "partial": True}
            json.dump(rep, open(coloured + ".json", "w"))
            finish(rep, jobs.Token())  # the job's token is cancelled
        raise
    finally:
        if os.path.exists(live):
            os.remove(live)
    rep = json.load(open(coloured + ".json"))
    finish(rep, token)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project")
    ap.add_argument("--depth", type=int, help="Poisson octree depth (default: from extent and point spacing)")
    ap.add_argument("--trim", type=float, default=6.0, help="cut faces farther than this many point spacings")
    ap.add_argument("--fill", type=float, metavar="MAX_HOLE", nargs="?", const=0.0,
                    help="patch holes up to MAX_HOLE across in the existing mesh (default 40 point spacings)")
    a = ap.parse_args()
    if a.fill is not None:
        fill(a.project, max_hole=a.fill or None)
    else:
        run(a.project, depth=a.depth, trim_spacings=a.trim)


if __name__ == "__main__":
    main()
