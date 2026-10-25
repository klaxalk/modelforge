"""Direct sunlight per pixel of every training view, from the SfM mesh (docs/todo_lighting.md, step 3).

Runs in nerfstudio's venv (Open3D, OpenCV; CPU raycasting). For each frame of DATASET/transforms.json, at the
training size, undistorted and cropped exactly as nerfstudio's datamanager does (export/mesh_depth.py,
undistorted_camera): a ray through every pixel centre to the mesh; from the surface point it hits (moved off the
surface along the normal) a ray towards that photo's sun (SUN.json: {image: {"azimuth", "elevation"}}, ° in the
transforms' world, ENU). Per pixel S = V x max(0, cos(normal, sun)): V = 1 when the sun ray leaves the mesh, 0 when
it is blocked (shadow) or the sun is below the horizon; -1 where the first ray hits no mesh (sky, beyond the mesh:
unknown). -> OUT/<image name>.npy (float16). Kept per mesh, downscale, camera and sun (OUT/manifest.json).

python export/sun_vis.py DATASET MESH SUN.json OUT --downscale N
"""
import argparse
import json
import os
import sys
import time

import numpy as np



def _load(name):
    """A module of this folder by its path: the folder on sys.path would shadow nerfstudio (export/nerfstudio.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"mf_{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                              f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_md = _load("mesh_depth")
frame_camera, undistorted_camera = _md.frame_camera, _md.undistorted_camera


def sun_vector(az_deg, el_deg):
    """Unit vector towards the sun in ENU."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("mesh")
    ap.add_argument("sun")
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=1)
    a = ap.parse_args()
    t = json.load(open(os.path.join(a.dataset, "transforms.json")))
    suns = json.load(open(a.sun))
    cams, index = frame_camera.cameras(t)
    und = [undistorted_camera(c, a.downscale, t) for c in cams]
    st = os.stat(a.mesh)
    key = {"mesh": os.path.realpath(a.mesh), "size": st.st_size, "mtime": st.st_mtime, "downscale": a.downscale,
           "cameras": [{"K": np.round(K, 6).tolist(), "w": w, "h": h} for K, w, h in und], "version": 2}
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.json")
    old = json.load(open(man)) if os.path.exists(man) else None
    if old is not None and old != key:
        for f in os.listdir(a.out):
            if f.endswith(".npy"):
                os.remove(os.path.join(a.out, f))
    if old != key:
        json.dump(key, open(man, "w"), indent=1)
    cam_of = {os.path.basename(f["file_path"]): i for f, i in zip(t["frames"], index)}
    names = [os.path.basename(f["file_path"]) for f in t["frames"]]
    todo = [f for f, n in zip(t["frames"], names)
            if n in suns and not os.path.exists(os.path.join(a.out, n + ".npy"))]
    print(f"sun visibility: {len(t['frames'])} photos, {sum(n in suns for n in names)} with a sun, {len(todo)} to "
          f"compute", flush=True)
    if not todo:
        return
    t0 = time.time()
    mesh = o3d.t.io.read_triangle_mesh(a.mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(mesh.vertex.positions.to(o3d.core.float32),
                                                    mesh.triangle.indices.to(o3d.core.int32)))
    rays_cam = {}
    lit_share = []
    for k, f in enumerate(todo):
        name = os.path.basename(f["file_path"])
        ci = cam_of[name]
        K, w, h = und[ci]
        if ci not in rays_cam:
            u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
            rays_cam[ci] = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)],
                                    -1).reshape(-1, 3)
        c2w = np.array(f["transform_matrix"], float)
        R = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])  # OpenGL -> OpenCV camera axes
        d = rays_cam[ci] @ R.T
        o = np.broadcast_to(c2w[:3, 3], d.shape)
        r = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
        t_hit = r["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        S = np.full(len(d), -1.0, np.float32)
        s = sun_vector(suns[name]["azimuth"], suns[name]["elevation"])
        if hit.any():
            n = r["primitive_normals"].numpy()[hit].astype(np.float64)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            n *= -np.sign(np.sum(n * d[hit], axis=1, keepdims=True) + 1e-12)  # towards the camera
            p = o[hit] + t_hit[hit, None] * d[hit]
            # off the surface, and a little towards the sun: the mesh's own bumps (Poisson over SfM points) would
            # shade grazing sunlight everywhere (fullrun_test at 6.7°: speckled self-shadow over the grass)
            p = p + n * np.maximum(0.05, 2e-3 * t_hit[hit])[:, None] + s * np.maximum(0.3, 0.01 * t_hit[hit])[:, None]
            cos = np.clip(n @ s, 0.0, 1.0)
            if suns[name]["elevation"] <= 0:
                vis = np.zeros(len(p), bool)
            else:
                rs = scene.cast_rays(o3d.core.Tensor(np.concatenate(
                    [p, np.broadcast_to(s, p.shape)], 1).astype(np.float32)))
                vis = ~np.isfinite(rs["t_hit"].numpy())
            S[hit] = vis * cos
            lit_share.append(float(np.mean(vis & (cos > 0))))
        out = os.path.join(a.out, name + ".npy")
        np.save(out + ".part.npy", S.reshape(h, w).astype(np.float16))
        os.replace(out + ".part.npy", out)
        if (k + 1) % 20 == 0 or k + 1 == len(todo):
            print(f"sun visibility: {k + 1} / {len(todo)} ({(time.time() - t0) / (k + 1):.2f} s per map; "
                  f"sunlit share median {100 * np.median(lit_share):.0f} %)", flush=True)


if __name__ == "__main__":
    main()
