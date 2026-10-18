"""SfM for videos without position metadata (project kind "video").

COLMAP recovers the poses from the images alone with one of three mappers (MAPPERS, unmodified):
incremental (default; pycolmap, Stop via its cancellation token), global (GLOMAP in COLMAP) or
hierarchical (both through the colmap CLI, so Stop kills them). The world frame is arbitrary, so
the largest model is then
  oriented  up from the plane through the camera centres (a walking or flying path), else the
            dominant plane of the points (the ground, if it agrees with the cameras' up within 60 deg),
            else the cameras' mean "up" (-y of the OpenCV camera); see up_alignment
  centred   at the median camera centre
  scaled    so the median scene depth seen by the cameras is DEPTH_UNITS (no metric scale exists)
and written to sparse/refined, which the export, dense and splat stages read as for DJI projects.
"""

import os
import shutil
import threading
import time

import numpy as np
import pycolmap

from modelforge import jobs
from modelforge.dense import COLMAP_BIN

DEPTH_UNITS = 10.0


def _cancel_on(token, ct, done):
    """Forward a modelforge Stop (jobs.Token) to the mapper's cancellation token."""
    while not done.wait(0.2):
        if token.cancelled:
            ct.cancel()
            return


def dominant_plane(points, iters=300, seed=0):
    """Normal of the plane most 3D points lie on (RANSAC), or None; usually the ground."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    c = np.median(points, axis=0)
    thr = 0.02 * float(np.median(np.linalg.norm(points - c, axis=1)))
    best, best_n = 0, None
    for _ in range(iters):
        a, b, d = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, d - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((points - a) @ n) < thr).sum())
        if k > best:
            best, best_n = k, n
    return best_n if best > 0.2 * len(points) else None


def up_alignment(rec):
    """(R, method) rotating the model so that "up" becomes +z. Cues, in order:
    the plane of the camera path (walking / flying, if it spans a plane), the dominant plane of the
    points (the ground; only if within 60 deg of the cameras' up, so a facade is not taken for it),
    the cameras' mean up (-y of the OpenCV camera: a phone held upright)."""
    ids = sorted(rec.reg_image_ids())
    centers = np.array([rec.image(i).projection_center() for i in ids])
    cam_up = np.array([-rec.image(i).cam_from_world().rotation.matrix()[1] for i in ids])  # -y_cam in world
    mean_up = cam_up.sum(0)
    mean_up /= np.linalg.norm(mean_up) or 1.0
    up, method = mean_up, "mean camera up"
    planar = False
    if len(centers) >= 3:
        _, sv, vt = np.linalg.svd(centers - centers.mean(0))
        if sv[1] > 0.2 * sv[0]:  # the path spans a plane, not just a line
            n = vt[2]
            up, method, planar = (n if n @ mean_up >= 0 else -n), "camera-path plane", True
    if not planar:
        n = dominant_plane(np.array([p.xyz for p in rec.points3D.values()]))
        if n is not None:
            n = n if n @ mean_up >= 0 else -n
            if n @ mean_up > np.cos(np.radians(60)):
                up, method = n, "ground plane (points)"
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-9:
        return (np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])), method
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c)), method


def normalize(rec):
    """Orient up, centre and scale the model in place; returns what was done."""
    R, method = up_alignment(rec)
    rec.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(R), np.zeros(3)))
    centers = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    depths = []
    for i in rec.reg_image_ids():
        img = rec.image(i)
        T = img.cam_from_world().matrix()
        pts = [rec.points3D[p.point3D_id].xyz for p in img.points2D if p.has_point3D()]
        if pts:
            depths.append(np.median((np.array(pts) @ T[:, :3].T + T[:, 3])[:, 2]))
    s = DEPTH_UNITS / float(np.median(depths)) if depths and np.median(depths) > 0 else 1.0
    t = -s * np.median(centers, axis=0)
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(np.eye(3)), t))
    return {"up": method, "scale": round(s, 6), "median_depth_units": DEPTH_UNITS}


def stats(rec):
    rec.update_point_3d_errors()
    err = np.array([p.error for p in rec.points3D.values()])
    cam = next(iter(rec.cameras.values()))
    return {"reg_images": rec.num_reg_images(), "points3D": rec.num_points3D(),
            "observations": rec.compute_num_observations(),
            "mean_track_length": round(rec.compute_mean_track_length(), 2),
            "point_error_px": {"median": round(float(np.median(err)), 3),
                               "p90": round(float(np.percentile(err, 90)), 3)} if len(err) else {},
            "camera": {"model": cam.model.name, "params": [round(float(x), 4) for x in cam.params]}}


MAPPERS = {
    "incremental": "registers images one by one with repeated BA: the most robust, best up to ~1000 images",
    "global": "all rotations, then all positions, then one BA (GLOMAP in COLMAP): much faster on large sets, "
              "needs well-connected images",
    "hierarchical": "clusters mapped incrementally in parallel, then merged: for very large scenes (thousands)",
}
_CLI = {"global": "global_mapper", "hierarchical": "hierarchical_mapper"}
_PROGRESS_WORDS = ("Registering", "Rotation", "Global", "positioning", "Bundle", "Retriangulat", "Merging",
                   "Track", "Filter", "Cluster", "Reconstructing")


def _incremental_models(db, images, out, token, progress):
    """pycolmap incremental mapping: Stop through its cancellation token, a callback per registered image."""
    opts = pycolmap.IncrementalPipelineOptions()
    opts.extract_colors = True
    ct, done, count = pycolmap.CancellationToken(), threading.Event(), [0]
    if token is not None:
        threading.Thread(target=_cancel_on, args=(token, ct, done), daemon=True).start()

    def next_image():
        count[0] += 1
        if progress:
            progress({"phase": "register", "registered": count[0]})
    try:
        return pycolmap.incremental_mapping(db, images, out, options=opts, next_image_callback=next_image,
                                            cancellation_token=ct)
    except InterruptedError:  # pycolmap's way of reporting the cancellation
        raise jobs.Cancelled("stopped")
    finally:
        done.set()


def _cli_models(mapper, db, images, out, token, progress):
    """Global / hierarchical mapping through the colmap CLI (pycolmap offers them without a Stop hook):
    killable by Stop, and its log lines are the progress."""
    def line(text):
        msg = text.split("] ", 1)[-1].strip()
        if progress and any(w in msg for w in _PROGRESS_WORDS):
            progress({"phase": "log", "line": msg[:120]})
    (token or jobs.Token()).run([COLMAP_BIN, _CLI[mapper], "--database_path", db, "--image_path", images,
                                 "--output_path", out], line)
    models = {}
    for d in sorted(os.listdir(out)):
        if os.path.exists(os.path.join(out, d, "images.bin")):
            models[int(d) if d.isdigit() else len(models)] = pycolmap.Reconstruction(os.path.join(out, d))
    return models


def run(project, mapper="incremental", token=None, progress=None):
    """SfM without priors on PROJECT/database.db with the chosen mapper (MAPPERS) -> sparse/refined
    (largest model, normalized). progress(event): {"phase": "register", "registered": k} (incremental)
    or {"phase": "log", "line": ...}. Returns stats for report.json."""
    if mapper not in MAPPERS:
        raise ValueError(f"mapper must be one of {tuple(MAPPERS)}")
    db, images = os.path.join(project, "database.db"), os.path.join(project, "images")
    out = os.path.join(project, "sparse", "models")
    shutil.rmtree(out, ignore_errors=True)  # no stale models from an earlier run
    os.makedirs(out)
    t0 = time.time()
    if mapper == "incremental":
        models = _incremental_models(db, images, out, token, progress)
    else:
        models = _cli_models(mapper, db, images, out, token, progress)
    jobs.check(token)
    if not models:
        raise RuntimeError("SfM found no model: too few matches between the images (more overlap, sharper or "
                           "more textured frames help)")
    sizes = sorted((m.num_reg_images() for m in models.values()), reverse=True)
    rec = max(models.values(), key=lambda m: m.num_reg_images())
    if mapper != "incremental":
        rec.extract_colors_for_all_images(images)
    frame = normalize(rec)
    refined = os.path.join(project, "sparse", "refined")
    os.makedirs(refined, exist_ok=True)
    rec.write(refined)
    n_images = pycolmap.Database.open(db).num_images()
    return {**stats(rec), "mapper": mapper, "images": n_images, "models": sizes, "frame": frame,
            "seconds": round(time.time() - t0, 1)}
