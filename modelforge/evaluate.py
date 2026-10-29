"""Splat quality on held-out images: images the splat (and its SfM) never saw, rendered from the splat
and compared with the real photos (PSNR / SSIM / LPIPS). Used by the subset calibration sweep
(modelforge/subset_sweep.py).

1. pose: each held-out image is located against the project's refined model without changing it:
   its SIFT features are matched (ratio test) to the `neighbors` model images nearest in capture
   time, the matched keypoints that have a 3D point give 2D-3D pairs, and pycolmap's LO-RANSAC +
   refinement (estimate_and_refine_absolute_pose) gives the pose with the model's camera. So the pose
   is in the splat's frame (ENU) and the held-out images do not help the subset's SfM.
2. render + score: export/splat_eval.py in nerfstudio's venv renders splat/splat_enu.ply at each pose
   (pinhole with the model's intrinsics, at `width`), undistorts the photo to the same pinhole, and
   scores it.

    python -m modelforge.evaluate PROJECT HOLDOUT.json [--width 1920]
HOLDOUT.json: {"images_dir": ..., "images": [{"image": name, "time_s": t, "source": s}, ...]}.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

MIN_INLIERS = 30


def _sift(db_path, images_dir, names, upright, camera_model, camera_params):
    import pycolmap
    if os.path.exists(db_path):
        os.remove(db_path)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    reader.camera_params = ",".join(f"{x:.10g}" for x in camera_params)
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 8192
    ext.sift.upright = upright
    pycolmap.extract_features(db_path, images_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)


def _features(db, name):
    im = db.read_image_with_name(name)
    return db.read_keypoints(im.image_id)[:, :2], np.asarray(db.read_descriptors(im.image_id).to_float().data, np.float32)


def pose_holdout(project, holdout, neighbors=6, ratio=0.8, log=print):
    """{name: {"viewmat": 4x4 world -> OpenCV camera, "inliers": n, "camera_id": COLMAP camera}} for the held-out
    images that could be located (others are reported and skipped)."""
    return locate(project, holdout["images_dir"], holdout["images"], os.path.join(project, "evaluation"),
                  neighbors=neighbors, ratio=ratio, log=log)


def locate(project, images_dir, images, ws, neighbors=6, ratio=0.8, min_inliers=MIN_INLIERS, log=print,
           on_progress=None, token=None, keep_matches=False):
    """Place photos the model does not hold into the refined model, which stays as it is (held-out evaluation,
    sfm/register.py): each photo takes the camera of its source's model images nearest in time (a source is one
    camera), its SIFT features (extracted with that camera into WS) are matched (ratio test) to those `neighbors`
    images' features, the matches with a 3D point give 2D-3D pairs, PnP + refinement gives the pose.
    With learned features (sfm/learned.py) the photo gets the same features and LightGlue matches instead.
    images: [{"image", "time_s", "source"}]. Returns {name: {"viewmat", "inliers", "camera_id", "matches",
    "error_px" (median reprojection error of the inliers)}} (keep_matches: also the inlier "p2d", "p3d");
    on_progress(done, total)."""
    import cv2
    import pycolmap
    holdout = {"images_dir": images_dir, "images": images}
    rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    report = json.load(open(os.path.join(project, "report.json")))
    upright = bool(report.get("features", {}).get("upright", True))
    os.makedirs(ws, exist_ok=True)
    mdb = pycolmap.Database.open(os.path.join(project, "features.db"))
    poses = json.load(open(os.path.join(project, "poses.json")))["images"]
    model_by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    cands = [(p.get("source", 0), p.get("time_s", 0.0), p["image"]) for p in poses if p["image"] in model_by_name]

    def nearest(h):
        return sorted((c for c in cands if c[0] == h.get("source", 0)),
                      key=lambda c: abs(c[1] - h.get("time_s", 0.0)))[:neighbors]
    cam_of = {h["image"]: model_by_name[near[0][2]].camera_id for h in holdout["images"] for near in [nearest(h)]
              if near}
    from modelforge.sfm import learned
    method = learned.method_of(project)
    lq = {}  # learned features: per photo (its keypoints, [(model name, matches)]) from one worker run
    if method:
        todo = [h for h in holdout["images"] if h["image"] in cam_of]
        qs = [{"image": h["image"], "model": [c[2] for c in nearest(h)]} for h in todo]
        if qs:
            res = learned.query(project, method, holdout["images_dir"], qs, ws, token=token,
                                progress=on_progress and (lambda k, n: on_progress(k, 2 * n)))
            lq = {q["image"]: r for q, r in zip(qs, res)}
    hdbs = {}  # camera id -> held-out features (one database per camera: SINGLE mode, its intrinsics)
    for cid in sorted(set(cam_of.values()) if not method else ()):
        cam = rec.cameras[cid]
        path = os.path.join(ws, "holdout_features.db" if len(rec.cameras) == 1 else f"holdout_features_c{cid}.db")
        _sift(path, holdout["images_dir"], [n for n, c in cam_of.items() if c == cid], upright, cam.model.name,
              list(cam.params))
        hdbs[cid] = pycolmap.Database.open(path)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out = {}
    try:
        for k_h, h in enumerate(holdout["images"]):
            if token is not None:
                token.check()
            if on_progress:
                n_h = len(holdout["images"])
                on_progress(n_h + k_h, 2 * n_h) if method else on_progress(k_h, n_h)  # learned: matching was half
            if h["image"] not in cam_of:
                log(f"evaluate: {h['image']} not located (no model image of its source)")
                continue
            cid = cam_of[h["image"]]
            cam = rec.cameras[cid]
            p2d, p3d, seen = [], [], set()
            if method:  # LightGlue matches of the project's learned features
                kp, per_model = lq[h["image"]]
                for mname, m in per_model:
                    pid = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in model_by_name[mname].points2D])
                    for q, t in m:
                        if t < len(pid) and pid[t] >= 0 and q not in seen:
                            seen.add(q)
                            p2d.append(kp[q])
                            p3d.append(rec.points3D[int(pid[t])].xyz)
            else:
                hdb = hdbs[cid]
                kp, desc = _features(hdb, h["image"])
            for _, _, mname in (nearest(h) if not method else ()):
                img = model_by_name[mname]
                mkp, mdesc = _features(mdb, mname)
                pid = np.array([p.point3D_id if p.has_point3D() else -1 for p in img.points2D])
                for pair in bf.knnMatch(desc, mdesc, k=2):
                    if len(pair) < 2 or pair[0].distance > ratio * pair[1].distance:
                        continue
                    q, t = pair[0].queryIdx, pair[0].trainIdx
                    if t < len(pid) and pid[t] >= 0 and q not in seen:
                        seen.add(q)
                        p2d.append(kp[q])
                        p3d.append(rec.points3D[int(pid[t])].xyz)
            res = None
            if len(p2d) >= MIN_INLIERS:
                res = pycolmap.estimate_and_refine_absolute_pose(np.array(p2d, float), np.array(p3d, float), cam)
            if res is None or res["num_inliers"] < min_inliers:
                log(f"evaluate: {h['image']} not located ({len(p2d)} 2D-3D matches)")
                continue
            m = np.eye(4)
            m[:3] = res["cam_from_world"].matrix()
            mask = np.asarray(res.get("inlier_mask", np.ones(len(p2d), bool)), bool)
            P2, P3 = np.array(p2d, float)[mask], np.array(p3d, float)[mask]
            q = (P3 @ m[:3, :3].T + m[:3, 3])
            proj = np.array([cam.img_from_cam(v) for v in q]).reshape(-1, 2)
            out[h["image"]] = {"viewmat": m.tolist(), "inliers": int(res["num_inliers"]), "camera_id": int(cid),
                               "matches": len(p2d),
                               "error_px": round(float(np.median(np.linalg.norm(proj - P2, axis=1))), 3)}
            if keep_matches:
                out[h["image"]]["p2d"], out[h["image"]]["p3d"] = P2, P3
    finally:
        for hdb in hdbs.values():
            hdb.close()
        mdb.close()
    return out


def run(project, holdout, width=1920, log=print, splat_path=None, located=None, report=True, gains=None,
        rasterize_mode="classic", sky=None):
    """Pose, render and score the held-out images; stats to report.json["evaluation"] (report=True) and
    returned. splat_path: another ENU splat than splat/splat_enu.ply (e.g. a training snapshot); located:
    pose_holdout's result, to locate the images once for several splats. gains: {image name: relative exposure}
    (modelforge.exposure): each render scaled by it (a splat trained with use_exposure). sky: a folder of sky masks
    (<name>.png, 0 = sky; modelforge/sky.py): the sky left out of the scores (a splat trained with sky masks renders it
    empty)."""
    import pycolmap
    from modelforge import splat, splat_mesh
    from modelforge.pipeline import Report
    t0 = time.time()
    located = located or pose_holdout(project, holdout, log=log)
    if not located:
        raise RuntimeError("no held-out image could be located in the model")
    rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    ws = os.path.join(project, "evaluation")
    names = list(located)
    first = next(iter(rec.cameras))  # older located records carry no camera: a one-camera project
    ids = sorted({int(located[n].get("camera_id", first)) for n in names})
    Ks, dists = [], []
    for cid in ids:
        cam = rec.cameras[cid]
        p = [float(x) for x in cam.params]
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if cam.model.name in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE") \
            else p[:4]
        Ks.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dists.append(p[4:8] if cam.model.name == "OPENCV" else ([p[3], p[4] if len(p) > 4 else 0.0, 0.0, 0.0]
                     if cam.model.name in ("SIMPLE_RADIAL", "RADIAL") else [0.0] * 4))
    inp = os.path.join(ws, "eval_input.npz")
    g = splat_mesh.load_splat(splat_path or os.path.join(project, "splat", "splat_enu.ply"))
    np.savez(inp, viewmats=np.array([located[n]["viewmat"] for n in names], np.float32),
             K=np.array(Ks, np.float64), dist=np.array(dists, np.float64),
             width=np.array([rec.cameras[c].width for c in ids]), height=np.array([rec.cameras[c].height for c in ids]),
             camera=np.array([ids.index(int(located[n].get("camera_id", first))) for n in names]),
             render_width=int(width), rasterize_mode=rasterize_mode,
             paths=np.array([os.path.join(holdout["images_dir"], n) for n in names]),
             **({"gains": np.array([float(gains.get(n, 1.0)) for n in names])} if gains else {}),
             **({"masks": np.array([os.path.join(sky, n + ".png") if os.path.exists(os.path.join(sky, n + ".png"))
                                    else "" for n in names])} if sky else {}), **g)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "splat_eval.py")
    out_json = os.path.join(ws, "metrics.json")
    lines = []
    r = subprocess.run([os.path.join(splat.NS_BIN, "python"), script, inp, out_json], env=splat._env(),
                       capture_output=True, text=True)
    lines = (r.stdout + r.stderr).strip().splitlines()
    if r.returncode:
        raise RuntimeError("splat_eval failed: " + (lines[-1] if lines else str(r.returncode)))
    os.remove(inp)
    metrics = json.load(open(out_json))
    per = metrics.pop("per_image")
    for n, m in zip(names, per):
        m["inliers"] = located[n]["inliers"]
        m["image"] = n
    stats = {**metrics, "held_out": len(holdout["images"]), "located": len(names), "width": int(width),
             "seconds": round(time.time() - t0, 1)}
    json.dump({**stats, "per_image": per}, open(out_json, "w"), indent=1)
    if report:
        Report(project).set("evaluation", stats)
    log(f"evaluate: {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("holdout")
    ap.add_argument("--width", type=int, default=1920)
    a = ap.parse_args()
    print(run(os.path.abspath(a.project), json.load(open(a.holdout)), a.width))


if __name__ == "__main__":
    main()
