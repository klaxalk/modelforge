"""Mesh from the SfM points (stage `sfmmesh`): a surface the splat can later be held to (dense depth / normals where
the SfM points are sparse), and a quick look at what the sparse model says the ground is.

The points of the export's model (sparse/export: triangulated without the photos the export left out; else
sparse/refined) get normals from their 16 nearest neighbours (PCA), turned towards a camera that sees them; COLMAP's
Poisson mesher (`poisson_mesher`, killable) closes them into a surface. Poisson closes everything, also across gaps
and far beyond the scene's edge, so faces with a vertex farther than `trim` from every SfM point are cut away (its
own density trim deletes nearly everything on sparse clouds). Octree depth from the extent and the point spacing
(cells of ~2 point spacings).

Output PROJECT/sfmmesh/: points.ply (the oriented points), mesh.ply (vertex colours, the project's world frame),
preview.ply (<= 300k faces, for the View tab), stats in report.json["sfm_mesh"].
"""
import argparse
import json
import os
import time

import numpy as np

from modelforge import dense, jobs

OUT = "sfmmesh"
NEIGHBOURS = 16
PREVIEW_FACES = 300_000


def model_dir(project):
    for name in ("export", "refined"):
        d = os.path.join(project, "sparse", name)
        if os.path.exists(os.path.join(d, "points3D.bin")):
            return d
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


def trim(mesh_path, out_path, xyz, max_dist):
    """Keep the faces whose vertices all lie within max_dist of an SfM point; drop unused vertices."""
    import cv2
    import plyfile
    ply = plyfile.PlyData.read(mesh_path)
    v, f = ply["vertex"].data, ply["face"].data
    vx = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    c = xyz.mean(0)
    index = cv2.flann_Index((xyz - c).astype(np.float32), dict(algorithm=1, trees=4))
    _, d2 = index.knnSearch((vx - c).astype(np.float32), 1, params={"checks": 32})
    near = np.sqrt(d2[:, 0]) <= max_dist
    faces = np.stack(f["vertex_indices"]) if len(f) else np.zeros((0, 3), int)
    faces = faces[near[faces].all(1)]
    used = np.unique(faces)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    fe = np.empty(len(faces), [("vertex_indices", "i4", (3,))])
    fe["vertex_indices"] = remap[faces]
    plyfile.PlyData([plyfile.PlyElement.describe(v[used], "vertex"),
                     plyfile.PlyElement.describe(fe, "face")]).write(out_path)
    return len(used), len(faces), int(len(f))


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
    n_v, n_f, n_raw = trim(raw, mesh, xyz, max_dist)
    os.remove(raw)
    on_line(f"sfm mesh: {n_f:,} of {n_raw:,} faces within {max_dist:.2f} of an SfM point")
    preview = os.path.join(out, "preview.ply")
    if n_f > PREVIEW_FACES:
        dense._colmap("mesh_simplifier", "--input_path", mesh, "--output_path", preview,
                      "--MeshSimplification.target_face_ratio", f"{PREVIEW_FACES / n_f:.6f}", token=token)
    elif os.path.exists(preview):
        os.remove(preview)
    stats = {"model": os.path.relpath(src, project), "points": len(xyz), "point_spacing": round(spacing, 4),
             "depth": depth, "cell": round(extent / 2 ** depth, 4), "trim_dist": round(max_dist, 4),
             "faces": n_f, "vertices": n_v, "faces_untrimmed": n_raw, "seconds": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    Report(project).set("sfm_mesh", stats)
    on_line(f"sfm mesh: {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project")
    ap.add_argument("--depth", type=int, help="Poisson octree depth (default: from extent and point spacing)")
    ap.add_argument("--trim", type=float, default=6.0, help="cut faces farther than this many point spacings")
    a = ap.parse_args()
    run(a.project, depth=a.depth, trim_spacings=a.trim)


if __name__ == "__main__":
    main()
