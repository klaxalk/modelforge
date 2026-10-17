"""Export the end products of a project into PROJECT/export/, ready for other tools.

  mesh   glb  textured (or vertex-coloured) glTF binary, one file      Y-up (glTF convention)
         obj  .obj + .mtl + texture .png                               Y-up (Blender's OBJ default)
         ply  COLMAP's files as they are (textured: mesh.ply with per-face UVs + texture.png)  ENU
  splat  ply    standard 3DGS PLY (SuperSplat, Postshot, nerfstudio, gsplat)    ENU
         splat  32 bytes per Gaussian (antimatter15 web viewer format)        ENU

ENU = the project's local frame: x east, y north, z up, metres, origin at enu_origin_lla.
Y-up = (x, z, -y) of ENU, so importers that expect Y-up show the scene upright.
export/georef.json records the origin and the axes of every exported file.

Usage:
    python -m modelforge.export.products PROJECT [--mesh delaunay] [--mesh-format glb|obj|ply]
                                                 [--splat-format ply|splat] [--no-mesh] [--no-splat]
"""

import argparse
import json
import os
import re
import shutil

import numpy as np

from modelforge import dense

MESH_FORMATS = ("glb", "obj", "ply")
SPLAT_FORMATS = ("ply", "splat")
ENU_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # (x, y, z) -> (x, z, -y)


def out_dir(project):
    return os.path.join(project, "export")


def available(project):
    """{"meshes": {method: {"textured": bool, "plain": bool}}, "splat": bool}"""
    ws = dense.workspace(project)
    meshes = {}
    for m in dense.MESH_METHODS:
        textured = os.path.exists(os.path.join(ws, f"textured-{m}", "mesh.ply"))
        plain = os.path.exists(os.path.join(ws, f"meshed-{m}.ply"))
        if textured or plain:
            meshes[m] = {"textured": textured, "plain": plain}
    return {"meshes": meshes, "splat": os.path.exists(os.path.join(project, "splat", "splat_enu.ply"))}


def _stem(project):
    return os.path.basename(os.path.abspath(project).rstrip("/"))


def load_textured(path):
    """COLMAP's textured PLY (per-face texcoord) -> trimesh with per-vertex UVs (corners unwelded,
    then identical position + UV pairs merged again)."""
    import plyfile
    import trimesh
    from PIL import Image
    ply = plyfile.PlyData.read(path)
    v = ply["vertex"]
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    faces = np.stack(ply["face"]["vertex_indices"])
    uv = np.stack(ply["face"]["texcoord"]).reshape(-1, 2)
    tex = next(c.split(None, 1)[1] for c in ply.comments if c.startswith("TextureFile"))
    image = Image.open(os.path.join(os.path.dirname(path), tex))
    image.load()
    mesh = trimesh.Trimesh(xyz[faces.reshape(-1)], np.arange(faces.size).reshape(-1, 3),
                           visual=trimesh.visual.TextureVisuals(uv=uv, image=image), process=False)
    mesh.merge_vertices()  # keeps vertices whose UVs differ apart
    return mesh


def load_mesh(project, method="delaunay", textured=True):
    import trimesh
    ws = dense.workspace(project)
    t = os.path.join(ws, f"textured-{method}", "mesh.ply")
    if textured and os.path.exists(t):
        return load_textured(t), True
    p = os.path.join(ws, f"meshed-{method}.ply")
    if not os.path.exists(p):
        raise RuntimeError(f"no {method} mesh in {ws}: run the dense stages first")
    return trimesh.load(p, process=False), False


def export_mesh(project, method="delaunay", fmt="glb", textured=True):
    """Write the mesh in `fmt`; returns the written paths (main file first)."""
    if fmt not in MESH_FORMATS:
        raise ValueError(f"format must be one of {MESH_FORMATS}")
    out = out_dir(project)
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"{_stem(project)}_mesh-{method}")
    ws = dense.workspace(project)
    if fmt == "ply":  # COLMAP's own files, ENU
        tdir = os.path.join(ws, f"textured-{method}")
        if textured and os.path.exists(os.path.join(tdir, "mesh.ply")):
            dst = base + "-textured"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(tdir, dst)
            paths = [os.path.join(dst, f) for f in sorted(os.listdir(dst), key=lambda f: not f.endswith(".ply"))]
        else:
            src = os.path.join(ws, f"meshed-{method}.ply")
            if not os.path.exists(src):
                raise RuntimeError(f"no {method} mesh in {ws}: run the dense stages first")
            shutil.copyfile(src, base + ".ply")
            paths = [base + ".ply"]
        _georef(project, paths[0], "ENU")
        return paths
    mesh, has_tex = load_mesh(project, method, textured)
    mesh.vertices = mesh.vertices @ ENU_TO_YUP.T
    if fmt == "glb":
        path = base + ".glb"
        mesh.export(path, file_type="glb")
        paths = [path]
    else:
        from trimesh.exchange.obj import export_obj
        name = os.path.basename(base)
        text, files = export_obj(mesh, include_texture=True, return_texture=True, mtl_name=f"{name}.mtl")
        path = base + ".obj"
        with open(path, "w") as f:
            f.write(text)
        paths = [path]
        # trimesh names the texture material_0.png: rename it per mesh, so exports don't collide
        renames = {f: name + os.path.splitext(f)[1] for f in files if not f.endswith(".mtl")}
        for fname, data in files.items():
            if fname.endswith(".mtl"):
                for a, b in renames.items():
                    data = data.replace(a.encode(), b.encode())
                # viewers multiply the texture by Kd; trimesh's 0.4 would darken it
                data = re.sub(rb"(?m)^Kd .*$", b"Kd 1.00000000 1.00000000 1.00000000", data)
            dst = os.path.join(out, renames.get(fname, fname))
            with open(dst, "wb") as f:
                f.write(data)
            paths.append(dst)
    _georef(project, paths[0], "Y-up", textured=has_tex)
    return paths


def to_splat_bytes(v, min_opacity=0.0):
    """antimatter15 .splat: per Gaussian position f32x3, scale f32x3 (linear), RGBA u8 (SH band 0
    + sigmoid opacity), rotation u8x4 (wxyz, q * 128 + 128). Sorted by opacity x volume, largest first."""
    sh0 = 0.28209479177387814
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float32)
    scale = np.exp(np.c_[v["scale_0"], v["scale_1"], v["scale_2"]]).astype(np.float32)
    rgb = np.clip((0.5 + sh0 * np.c_[v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]) * 255, 0, 255)
    alpha = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    q = np.c_[v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    keep = alpha >= min_opacity
    order = np.argsort(-(alpha * scale.prod(1))[keep])
    rec = np.zeros(int(keep.sum()), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = xyz[keep][order], scale[keep][order]
    rec["c"] = np.c_[rgb, alpha * 255][keep][order].astype(np.uint8)
    rec["r"] = np.clip(q[keep][order] * 128 + 128, 0, 255).astype(np.uint8)
    return rec.tobytes()


def export_splat(project, fmt="ply", min_opacity=0.0):
    """Write the ENU splat in `fmt`; returns [path]."""
    import plyfile
    if fmt not in SPLAT_FORMATS:
        raise ValueError(f"format must be one of {SPLAT_FORMATS}")
    src = os.path.join(project, "splat", "splat_enu.ply")
    if not os.path.exists(src):
        raise RuntimeError("splat/splat_enu.ply missing: train the splat first")
    out = out_dir(project)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{_stem(project)}_splat.{fmt}")
    if fmt == "ply":
        shutil.copyfile(src, path)
    else:
        with open(path, "wb") as f:
            f.write(to_splat_bytes(plyfile.PlyData.read(src)["vertex"], min_opacity))
    _georef(project, path, "ENU")
    return [path]


def _georef(project, path, axes, **extra):
    """export/georef.json: the ENU origin plus the axes of every exported file."""
    gpath = os.path.join(out_dir(project), "georef.json")
    g = json.load(open(gpath)) if os.path.exists(gpath) else {}
    poses = os.path.join(project, "poses.json")
    if os.path.exists(poses):
        meta = json.load(open(poses))
        g["enu_origin_lla"] = meta.get("enu_origin_lla")
    g["frames"] = {"ENU": "x east, y north, z up, metres from enu_origin_lla (lat, lon, alt)",
                   "Y-up": "ENU rotated for Y-up tools: (x, y, z) = (east, up, -north)"}
    g.setdefault("files", {})[os.path.relpath(path, out_dir(project))] = {"axes": axes, **extra}
    with open(gpath, "w") as f:
        json.dump(g, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--mesh", default="delaunay", choices=dense.MESH_METHODS)
    ap.add_argument("--mesh-format", default="glb", choices=MESH_FORMATS)
    ap.add_argument("--untextured", action="store_true", help="export the vertex-coloured mesh, not the textured one")
    ap.add_argument("--splat-format", default="ply", choices=SPLAT_FORMATS)
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--no-splat", action="store_true")
    args = ap.parse_args()
    have = available(args.project)
    if not args.no_mesh and args.mesh in have["meshes"]:
        for p in export_mesh(args.project, args.mesh, args.mesh_format, not args.untextured):
            print(p)
    if not args.no_splat and have["splat"]:
        for p in export_splat(args.project, args.splat_format):
            print(p)


if __name__ == "__main__":
    main()
