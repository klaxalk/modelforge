"""Register more photos into the refined model (the user asked, 2026-10-25; docs/todo_register_more.md).

SfM runs on a subset (keyframes, every N-th); the other extracted frames (PROJECT/extracted/, the pool) can then be
placed into that model without running SfM again, so the splat can train on a denser set. Interpolated poses alone are
not enough (a splat needs ~1 px; 5-10 px off between keyframes), so each frame is located in the model, which stays as
it is: SIFT features with its camera, matched to the model photos of its flight nearest in time, 2D-3D from their
tracks, PnP + refinement (evaluate.locate, as for the held-out evaluation). Photos with fewer than MIN_INLIERS inliers,
or whose pose lies more than MAX_SHIFT_M / MAX_TURN_DEG from the trajectory's (interpolated) are left out.

The trajectory from the metadata is smooth, and so is its error (the user's point): the corrections the refine made at
the model photos (refined pose vs GPS / telemetry prior), interpolated in time per flight, give every other frame a
pose without any matching (`interpolated`). kladno_dopravnik, 24 frames between 56 model photos (4K; reprojection of
the model's points they see): raw metadata 54 px median, interpolated corrections 2.0 px (p90 7.7, max 33), registered
0.67 px (max 0.97). So the interpolation is the check (a wrong registration lands far from it), registration the pose.

Output: PROJECT/registered/ (images/ hard links to the pool, registered.json: the model it belongs to (sparse/refined's
images.bin size + mtime) and per photo its cam_from_world, camera, inliers). The export appends these photos to
transforms.json (frames marked "mf_registered") while the model is that one (add_frames); the splat then trains on
them like the others. A new refine makes them stale: register again.
"""
import json
import os
import shutil
import time

import numpy as np

DIR = "registered"
MIN_INLIERS = 60
MAX_SHIFT_M = 1.0  # registered vs interpolated pose: further = a wrong registration
MAX_TURN_DEG = 2.0


def _model_key(project):
    f = os.path.join(project, "sparse", "refined", "images.bin")
    st = os.stat(f)
    return {"size": st.st_size, "mtime": st.st_mtime}


def candidates(project, every=1, selection=None):
    """Pool records (extracted/poses.json images) not in use, every `every`-th per source in capture order. With
    selection, a choice over the whole pool as Sparse tab → Images used makes it (the user asked, 2026-10-25):
    {"names": [...]} (keyframes at a density, overlap.select_keyframes) or {"every", "offset", "start", "end"}
    (projects.subsample_selection); its photos not in use."""
    pool = os.path.join(project, "extracted", "poses.json")
    if not os.path.exists(pool):
        return []
    used = set(os.listdir(os.path.join(project, "images")))
    allr = json.load(open(pool))["images"]
    if selection is not None:
        from modelforge import projects
        if "names" in selection:
            pick = set(selection["names"])
            return [r for r in allr if r["image"] in pick and r["image"] not in used]
        keep = projects.subsample_selection(allr, **selection)
        return [allr[k] for k in keep if allr[k]["image"] not in used]
    recs = [r for r in allr if r["image"] not in used]
    out = []
    for s in sorted({r.get("source", 0) for r in recs}):
        rs = sorted((r for r in recs if r.get("source", 0) == s), key=lambda r: r.get("time_s", 0.0))
        out += rs[::max(1, int(every))]
    return out


def load(project):
    """registered.json, or None when missing or made for another model (stale)."""
    p = os.path.join(project, DIR, "registered.json")
    try:
        d = json.load(open(p))
        return d if d.get("model") == _model_key(project) else None
    except (OSError, ValueError):
        return None


def run(project, every=1, on_line=print, on_progress=None, token=None, selection=None, describe=None):
    """Register every `every`-th unused pool photo (or those of `selection`, see candidates) into PROJECT's refined
    model. Returns the stats."""
    from modelforge import evaluate
    t0 = time.time()
    cands = candidates(project, every, selection)
    if not cands:
        raise RuntimeError("no extracted photos to register: the selection holds only photos in use")
    on_line(f"register: {len(cands)} photos of the extracted pool into the model")
    out_dir = os.path.join(project, DIR)
    located = evaluate.locate(project, os.path.join(project, "extracted", "images"), cands,
                              os.path.join(out_dir, "work"), min_inliers=MIN_INLIERS, log=lambda m: None,
                              on_progress=on_progress, token=token)
    from modelforge.sfm import robust
    interp = interpolated(project, [r["image"] for r in cands])  # the check: the smooth trajectory's pose
    imgs, far, shifts = {}, [], []
    for r in cands:
        loc = located.get(r["image"])
        if loc is None:
            continue
        vm = np.array(loc["viewmat"], float)
        R, C = vm[:3, :3], -vm[:3, :3].T @ vm[:3, 3]
        rec_ = {"cam_from_world": vm[:3].tolist(), "camera_id": loc["camera_id"], "inliers": loc["inliers"],
                "error_px": loc.get("error_px"), "source": int(r.get("source", 0)), "frame": r.get("frame"),
                "time_s": r.get("time_s")}
        if r["image"] in interp:
            Ri, Ci = interp[r["image"]]
            d, a = float(np.linalg.norm(C - Ci)), float(np.degrees(np.linalg.norm(robust.log_so3(R @ Ri.T))))
            rec_["from_trajectory"] = {"m": round(d, 3), "deg": round(a, 3)}
            if d > MAX_SHIFT_M or a > MAX_TURN_DEG:
                far.append((r["image"], round(d, 2), round(a, 2)))
                continue
            shifts.append((d, a))
        imgs[r["image"]] = rec_
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    for f in os.listdir(os.path.join(out_dir, "images")):
        if f not in imgs:
            os.remove(os.path.join(out_dir, "images", f))
    for n in imgs:
        dst = os.path.join(out_dir, "images", n)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(project, "extracted", "images", n), dst)
            except OSError:
                shutil.copy2(os.path.join(project, "extracted", "images", n), dst)
    inl = [v["inliers"] for v in imgs.values()]
    stats = {"tried": len(cands), "registered": len(imgs), "every": int(every), "selection": describe,
             "not_located": len(cands) - len(located), "off_trajectory": len(far),
             "inliers_median": int(np.median(inl)) if inl else 0,
             "error_px_median": round(float(np.median([v["error_px"] for v in imgs.values()
                                                       if v.get("error_px") is not None] or [0])), 3),
             "from_trajectory_median": {"m": round(float(np.median([d for d, _ in shifts])), 3),
                                        "deg": round(float(np.median([a for _, a in shifts])), 3)} if shifts else None,
             "seconds": round(time.time() - t0, 1)}
    json.dump({"model": _model_key(project), "stats": stats, "images": imgs},
              open(os.path.join(out_dir, "registered.json"), "w"), indent=1)
    shutil.rmtree(os.path.join(out_dir, "work"), ignore_errors=True)
    n_tf = add_frames(project)
    on_line(f"register: {len(imgs)} of {len(cands)} registered (median {stats['inliers_median']} inliers, "
            f"{stats['error_px_median']} px; {stats['not_located']} not located, {len(far)} off the trajectory), "
            f"{n_tf} added to transforms.json, {stats['seconds']:.0f} s")
    from modelforge.pipeline import Report
    Report(project).set("register", stats)
    return stats


def interpolated(project, names):
    """{name: (R_cw, centre)} for pool photos: their metadata pose (extracted/poses.json) + the refine's corrections at
    the model photos (refined vs prior, rotation and position) interpolated in time within each flight. Photos
    outside a flight's model photos in time get the nearest one's correction."""
    import pycolmap
    from modelforge.sfm import priors as priors_mod, robust
    rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    pool, _ = priors_mod.load(os.path.join(project, "extracted"))
    by_src = {}
    for i in rec.reg_image_ids():
        img = rec.image(i)
        p = pool.get(img.name)
        if p is None or p.time is None or not np.all(np.isfinite(p.center)):
            continue
        m = img.cam_from_world().matrix()
        R, C = m[:, :3], -m[:, :3].T @ m[:, 3]
        by_src.setdefault(p.source, []).append((p.time, C - p.center, robust.log_so3(R @ p.R_cw.T)))
    out = {}
    for n in names:
        p = pool.get(n)
        k = by_src.get(getattr(p, "source", None))
        if p is None or not k or p.time is None or not np.all(np.isfinite(p.center)):
            continue
        k = sorted(k, key=lambda x: x[0])
        t = np.array([x[0] for x in k])
        dC = np.array([np.interp(p.time, t, [x[1][j] for x in k]) for j in range(3)])
        dR = np.array([np.interp(p.time, t, [x[2][j] for x in k]) for j in range(3)])
        out[n] = (robust.exp_so3(dR) @ p.R_cw, p.center + dC)
    return out


def add_frames(project):
    """Append the registered photos to PROJECT/transforms.json (once; frames "mf_registered"), when registered.json
    belongs to the current model. Returns how many it holds."""
    import pycolmap
    from modelforge.export import nerfstudio as ns_export
    path = os.path.join(project, "transforms.json")
    if not os.path.exists(path):
        return 0
    t = json.load(open(path))
    t["frames"] = [f for f in t["frames"] if not f.get("mf_registered")]
    reg = load(project)
    if reg and reg["images"]:
        rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
        per_frame = bool(t.get("mf_cameras"))
        from modelforge import exposure
        try:
            gains = exposure.gains_for(project, [{"image": n, "source": v["source"], "frame": v["frame"]}
                                                 for n, v in reg["images"].items()])
        except Exception:  # noqa: BLE001 - metadata unreadable: no gains
            gains = {}
        for n, v in sorted(reg["images"].items()):
            w2c = np.eye(4)
            w2c[:3] = np.array(v["cam_from_world"])
            c2w = np.linalg.inv(w2c)
            c2w[0:3, 1:3] *= -1  # OpenCV -> OpenGL camera axes
            c = rec.cameras[v["camera_id"]]
            f = {"file_path": f"{DIR}/images/{n}", "transform_matrix": c2w.tolist(), "mf_registered": True}
            if per_frame:
                f.update(ns_export.intrinsics(c), w=int(c.width), h=int(c.height))
            if n in gains:
                f["mf_exposure"] = gains[n]
            t["frames"].append(f)
    tmp = path + ".part"
    json.dump(t, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    return sum(1 for f in t["frames"] if f.get("mf_registered"))


def remove(project):
    """Back to the model's own photos: registered/ deleted, the frames taken out of transforms.json."""
    shutil.rmtree(os.path.join(project, DIR), ignore_errors=True)
    add_frames(project)
