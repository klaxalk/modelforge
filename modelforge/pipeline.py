"""Headless modelforge pipeline on an ingested project (output of modelforge.dji.extract).

Stages (artifacts in PROJECT/):
  features  features.db                    SIFT only; matching works on a copy
  pairs     pairs.txt                      prior-based pair selection
  match     database.db                    copy of features.db + matches + two-view geometry
  refine    sparse/round_N, sparse/refined prior-initialized triangulation + prior BA rounds
  export    transforms.json, sparse_pc.ply nerfstudio dataset
Dense stages (modelforge.dense, COLMAP MVS on the refined poses; artifacts in PROJECT/dense/):
  undistort images/, sparse/               PINHOLE workspace
  stereo    stereo/{depth,normal}_maps     PatchMatch, needs CUDA (MODELFORGE_GPU=1)
  fusion    fused.ply (+ .vis)
  mesh      meshed-<method>.ply, preview-<method>.ply (simplified for the GUI)
  texture   textured-<method>/mesh.ply + texture.png
Every stage records its stats in PROJECT/report.json.

Usage:
    python -m modelforge.pipeline data/clip_frames                  # sfm stages (default)
    python -m modelforge.pipeline data/clip_frames --stages dense   # undistort .. texture
    python -m modelforge.pipeline data/clip_frames --stages refine,export,dense --mesh poisson
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import pycolmap

from modelforge import dense, jobs
from modelforge.export import lidar as lidar_mod
from modelforge.export import nerfstudio
from modelforge.dji import ingest
from modelforge.sfm import checkpoint, chunked, features, isolated, pairs, priors as priors_mod, reconstruct, refine, robust, tie_points

SFM_STAGES = ("features", "pairs", "match", "refine", "export")  # DJI: refine from the pose priors
VIDEO_SFM_STAGES = ("features", "pairs", "match", "reconstruct", "export")  # no metadata: incremental SfM
DENSE_STAGES = ("undistort", "stereo", "fusion", "mesh", "texture")
STAGES = ("features", "pairs", "match", "refine", "reconstruct", "export") + DENSE_STAGES  # canonical order
GROUPS = {"sfm": SFM_STAGES, "dense": DENSE_STAGES, "all": SFM_STAGES + DENSE_STAGES}


def sparse_kind(project):
    """The kind whose sparse method a project uses: ingest.json's kind, except that a ROS bag without
    camera poses (pose source none: images only) reconstructs like a video without metadata."""
    kind = ingest.load_kind(project)
    if kind == "rosbag":
        try:
            images = json.load(open(os.path.join(project, "poses.json")))["images"]
        except (OSError, ValueError, KeyError):
            return kind
        if images and not any("colmap_qvec" in r for r in images):
            return "video"
    return kind


def sfm_stages(kind):
    """The sparse stages of a project kind (ingest.json "kind")."""
    return VIDEO_SFM_STAGES if kind == "video" else SFM_STAGES  # dji, rosbag: priors -> refine


LISTENERS = []  # callables(msg), e.g. the GUI log panel


def log(msg):
    print(msg, file=sys.stderr, flush=True)
    for fn in LISTENERS:
        fn(msg)


class Report:
    def __init__(self, project):
        self.path = os.path.join(project, "report.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def set(self, key, value):
        self.data[key] = value
        json.dump(self.data, open(self.path, "w"), indent=1)


def known_camera(project):
    """PROJECT/camera.json (from a ROS camera_info), or None: then the intrinsics are estimated."""
    path = os.path.join(project, "camera.json")
    return json.load(open(path)) if os.path.exists(path) else None


def stage_features(project, report, kind="dji", extractor="sift", max_features=8192, upright=None, progress=None):
    """DJI: upright SIFT (the gimbal keeps the horizon level) and an OPENCV camera refined against the
    priors. Videos without metadata: oriented SIFT (handheld footage rolls) and a RADIAL camera (one
    focal length: without priors, separate fx / fy drift apart on short sequences). upright None = by kind.
    progress(done, total, image name): features.extract's progress."""
    video = kind == "video"
    upright = (kind == "dji") if upright is None else upright
    cam = known_camera(project)  # ROS camera_info: fixed intrinsics
    model = cam["model"] if cam else ("RADIAL" if video else "OPENCV")
    stats = features.extract(project, max_num_features=max_features, upright=upright, extractor=extractor,
                             camera_model=model, camera_params=cam["params"] if cam else None, progress=progress)
    stats["camera"] = {"model": model, "known": bool(cam)}
    log(f"features: {stats}")
    report.set("features", stats)


PAIR_MODES = {
    "auto": "DJI: time neighbours + view overlap from the priors; no metadata: exhaustive up to 150 images, "
            "else sequential",
    "sequential": "time neighbours (Neighbours) + quadratic jumps within each source, a coarse grid between sources",
    "exhaustive": "every pair: the most thorough, cost grows with the square of the image count",
}


def stage_pairs(project, report, pri, meta, mode="auto", neighbors=10):
    """pairs.txt for the match stage; mode: PAIR_MODES."""
    if mode not in PAIR_MODES:
        raise ValueError(f"pairs mode must be one of {tuple(PAIR_MODES)}")
    names = priors_mod.by_order(pri)
    seqs = [[n for n in names if pri[n].source == s] for s in sorted({pri[n].source for n in names})]
    rep = {"mode": mode}
    if mode == "auto" and priors_mod.has_poses(pri):  # DJI: look-at overlap from the priors
        rep["ground_z"] = meta["ground_z"] if "ground_z" in meta else -meta["enu_origin_lla"][2]
        pp = pairs.select(pri, rep["ground_z"])
        terrain = _terrain(project)
        if terrain is not None:  # a re-run: look-ats on the surface of the earlier model, added to the flat ones
            flat = len(pp)
            pp = sorted(set(pp) | set(pairs.select(pri, rep["ground_z"], terrain=terrain)))
            rep.update(terrain_points=terrain.points, terrain_cell_m=round(terrain.cell, 2), flat_pairs=flat)
            log(f"pairs: terrain from {terrain.points:,} sparse points ({terrain.cell:.2f} m cells): "
                f"{len(pp) - flat:,} pairs added to {flat:,} on flat ground")
    elif mode == "exhaustive":
        pp = pairs.sequential(seqs, exhaustive_below=len(names))
    else:  # sequential, or auto without priors (exhaustive when small)
        pp = pairs.sequential(seqs, overlap=neighbors, exhaustive_below=150 if mode == "auto" else 0)
        rep["neighbors"] = neighbors
    pairs.write(project, pp)
    report.set("pairs", {"num_pairs": len(pp), **rep})
    log(f"pairs: {len(pp)} ({mode})")


def _terrain(project, min_points=1000):
    """pairs.Terrain from the project's sparse points (sparse_pc.ply of an earlier SfM of these images), or None."""
    path = os.path.join(project, "sparse_pc.ply")
    if not os.path.exists(path):
        return None
    import plyfile
    v = plyfile.PlyData.read(path)["vertex"].data
    if len(v) < min_points:
        return None
    t = pairs.Terrain(np.stack([v["x"], v["y"], v["z"]], 1))
    t.points = len(v)
    return t


def copy_db(src, dst):
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dst + ext):
            os.remove(dst + ext)
    shutil.copyfile(src, dst)


def match_stats(db_path, seconds):
    db = features.open_database(db_path)
    s = {"matched_pairs": db.num_matched_image_pairs(), "verified_pairs": db.num_verified_image_pairs(),
         "inlier_matches": db.num_inlier_matches(), "seconds": round(seconds, 1)}
    db.close()
    return s


def _watch_matches(db_path, total, progress, done, every=0.25):
    """progress(pairs done, total) while COLMAP matches: match_image_pairs has no callback, but each block
    of pairs is committed to the database as it finishes."""
    import sqlite3
    last, wait = -1, every
    while not done.wait(wait):
        try:
            t0 = time.time()
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.1)
            try:
                n = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            finally:
                con.close()
        except sqlite3.Error:
            continue
        wait = max(every, 4 * (time.time() - t0))  # kladno_dira's 10 GB database: a count takes seconds
        if n != last:
            last = n
            progress(min(n, total), total)


def stage_match(project, report, max_ratio=0.8, guided=False, min_inliers=15, token=None, progress=None):
    """Match the pairs in pairs.txt and verify them geometrically (two-view geometry, RANSAC).
    max_ratio: Lowe's ratio test; guided: re-match along the estimated geometry (more inliers, slower);
    min_inliers: a verified pair needs this many. token: Stop through pycolmap's cancellation token.
    progress(pairs done, total): polled from the database a few times a second."""
    db_path = os.path.join(project, "database.db")
    copy_db(os.path.join(project, "features.db"), db_path)
    opts = pycolmap.ImportedPairingOptions()
    opts.match_list_path = os.path.join(project, "pairs.txt")
    m = pycolmap.FeatureMatchingOptions()
    m.sift.max_ratio, m.guided_matching = float(max_ratio), bool(guided)
    v = pycolmap.TwoViewGeometryOptions()
    v.min_num_inliers = int(min_inliers)
    ct, done, watcher = pycolmap.CancellationToken(), threading.Event(), None
    if token is not None:
        threading.Thread(target=reconstruct._cancel_on, args=(token, ct, done), daemon=True).start()
    if progress:
        total = sum(1 for line in open(opts.match_list_path) if line.strip())
        watcher = threading.Thread(target=_watch_matches, args=(db_path, total, progress, done), daemon=True)
        watcher.start()
    t0 = time.time()
    try:
        pycolmap.match_image_pairs(db_path, matching_options=m, pairing_options=opts, verification_options=v,
                                   cancellation_token=ct)
    except InterruptedError:  # pycolmap's way of reporting the cancellation
        raise jobs.Cancelled("stopped")
    finally:
        done.set()
        if watcher is not None:  # its last count must end before the database is opened again (kladno_dira: the
            watcher.join(timeout=120)  # stats' open failed with "database is locked" on the 10 GB database)
    jobs.check(token)
    stats = {**match_stats(db_path, time.time() - t0), "max_ratio": float(max_ratio), "guided": bool(guided),
             "min_inliers": int(min_inliers)}
    log(f"match: {stats}")
    report.set("match", stats)


BA_MODES = ("auto", "global", "chunked")
# auto (the user's choice, 2026-10-23): global whenever it fits in RAM (chunks leave revisits apart: kladno_dira,
# dira2k), chunked otherwise (kladno_dira, 5681 images: the global refine was OOM-killed at 30 GB in round 1)
DEFAULT_BA_MODE = "auto"
# a global refine's peak RSS per million verified (inlier) matches: dira2k 0.49 (2000 images), berlin 0.67 (2535),
# kladno_dira 0.65 (2883), berlin in the GUI 0.72 (3142); kladno_dira at 5681 images (82.7M) did not fit in 31 GB
GLOBAL_GB_PER_M_MATCHES = 0.7
GLOBAL_GB_BASE = 1.0
RAM_SHARE = 0.8  # of the memory available when refine starts


def ram_available_gb():
    """MemAvailable from /proc/meminfo (host-wide, also inside the container), None if unknown."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return None


def global_ba_need_gb(inlier_matches):
    return GLOBAL_GB_BASE + GLOBAL_GB_PER_M_MATCHES * inlier_matches / 1e6


def auto_ba_mode(report_data, n_images=None, available_gb=None, min_images=None):
    """BA mode "auto": (mode, why) - global if its estimated peak (global_ba_need_gb of the match stage's inlier
    matches) fits in RAM_SHARE of the available memory, else chunked. Without match stats or a memory reading: chunked
    from min_images (chunked.Chunked.min_images) images on."""
    inl = (report_data.get("match") or {}).get("inlier_matches")
    avail = ram_available_gb() if available_gb is None else available_gb
    if inl is None or avail is None:
        min_images = min_images or chunked.Chunked.min_images
        mode = "chunked" if n_images is not None and n_images >= min_images else "global"
        return mode, f"no match statistics or memory reading: {mode} ({'≥' if mode == 'chunked' else '<'} {min_images} images)"
    need = global_ba_need_gb(inl)
    mode = "global" if need <= RAM_SHARE * avail else "chunked"
    return mode, (f"global needs ~{need:.0f} GB ({inl / 1e6:.1f}M inlier matches), {avail:.0f} GB available "
                  f"({RAM_SHARE:.0%} usable): {mode}")


def _rounds(project, rec, pri, model, rounds, prefix, emit, token, max_iterations, skip, refine_intrinsics,
            write=True, ctx_extra=None, skip_round=None, ck=None):
    """Refinement rounds [(i, Round)]: triangulate (poses fixed) -> global prior BA -> source offsets ->
    drift (not in a chunk: its model is a throwaway copy) -> prior-model calibration. Writes sparse/<prefix>_i[_tri] (write=False: a scratch triangulation
    folder, no snapshots, and no "round_done" events: a chunk). Returns (rec, the last Round that
    ran, {"<prefix>_i": stats}). ck (checkpoint.Checkpoint): finished rounds are skipped, each new one saved."""
    db_path = os.path.join(project, "database.db")
    image_path = os.path.join(project, "images")
    results, last = {}, None
    n = len(rounds)
    for k, (i, rnd) in enumerate(rounds):
        if max_iterations == 0 or (skip is not None and skip.is_set()):
            break
        last = rnd
        if ck is not None and ck.done(f"{prefix}_{i}"):
            results[f"{prefix}_{i}"] = ck.get(f"round:{prefix}_{i}")
            log(f"{prefix}_{i}: done before (resumed)")
            continue
        if ck is not None:
            ck.restore(rec, model)
        t0 = time.time()
        ctx = {"round": k, "rounds": n, "stage": prefix, "name": f"{prefix}_{i}", **(ctx_extra or {})}
        emit({"phase": "triangulate", **ctx, "rec": rec})
        jobs.check(token)
        tri_dir = os.path.join(project, "sparse", f"{prefix}_{i}_tri" if write else "_chunk_tri")
        rec = refine.triangulate(rec, db_path, image_path, tri_dir, rnd)
        jobs.check(token)
        tri = refine.evaluate(rec, pri)
        corrected = robust.corrected_priors(pri, model)
        live = rec
        ba = refine.prior_ba(rec, corrected, model, refine_intrinsics=refine_intrinsics, loss_scale_px=rnd.loss_scale_px,
                             relative=i >= model.start_round, token=token, max_iterations=max_iterations, skip=skip,
                             skip_round=skip_round,
                             on_iteration=lambda s: emit({"phase": "ba", **ctx, "rec": live, "iteration": s.iteration,
                                                          "cost": s.cost}))
        jobs.check(token)  # an aborted BA leaves no round_N
        if write:
            out = os.path.join(project, "sparse", f"{prefix}_{i}")
            os.makedirs(out, exist_ok=True)
            rec.write(out)
        ev = refine.evaluate(rec, pri)  # before the offset / drift shifts move cameras off their points
        offsets = robust.update_source_offsets(model, rec, pri, db_path)
        # not in round 0 (focal length still ~2x off: its split points measure nonsense) and not on the
        # calibration subset (separate blocks: the gaps between them are unconstrained; a chunked refine
        # estimates the drift over all images after the pose graph)
        drift = (robust.update_drift(model, rec, pri, db_path)
                 if write and prefix != "calib" and i >= model.start_round else None)
        calib = robust.update(model, rec, pri) if i >= model.start_round else {"model": model.summary(), "skipped": True}
        results[f"{prefix}_{i}"] = {"thresholds": vars(rnd), "after_triangulation": tri, "ba": ba, "after_ba": ev,
                                    "prior_model": calib, "source_offsets": offsets, "drift": drift,
                                    "seconds": round(time.time() - t0, 1)}
        log(f"{ctx['name']} ({rnd.max_reproj_px:g}px): tri {tri['points3D']} pts, median "
            f"{tri['point_error_px'].get('median')}px -> BA {ba['iterations']}/{max_iterations} it ({ba['termination']}, "
            f"{ba['seconds']} s), "
            f"median {ev['point_error_px'].get('median')}px, prior dev pos {ev['prior_deviation']['position_m']} "
            f"rot {ev['prior_deviation']['rotation_deg']}, focal {ev['camera']['params'][0]}")
        if offsets:
            log(f"   source offsets ({offsets['method']}): {offsets['offsets_m']}")
        if drift:
            log(f"   drift: {_drift_line(drift)}")
        m = calib["model"]
        if not calib.get("skipped") and write:
            log(f"   prior model: scales {calib.get('scales')} | abs {m['abs_sigmas']} rel {m['rel_sigmas']} | "
                f"B {m['boresight_norm_deg']} deg, dt {m['dt_ms']} ms | rejected abs {calib['abs_rejected']} "
                f"rel {calib['rel_rejected']} | d2 median abs {calib['abs_d2_median']} rel {calib['rel_d2_median']}")
        if ck is not None:
            ck.save(f"{prefix}_{i}", rec, model, **{f"round:{prefix}_{i}": results[f"{prefix}_{i}"]})
        if write:
            emit({"phase": "round_done", **ctx, "rec": rec, "eval": ev, "ba": ba, "max_iterations": max_iterations})
    return rec, last, results


def _drift_line(drift):
    return "; ".join(f"source {s}: {d['pairs']} pairs" + (
        f", far split median {d['far_split_median_m']} m, increment ≤ {d['increment_max_m']} m, "
        f"drift ≤ {d['drift_max_m']} m (z range {d['drift_range_z_m']} m)" if d["pairs"] else "")
        for s, d in drift["sources"].items())


DRIFT_PASSES = 3  # chunked: triangulate all + drift update, until the increments are below DRIFT_DONE_M
DRIFT_DONE_M = 0.05


def _pose(rec, image_id):
    return np.array(rec.frame(rec.image(image_id).frame_id).rig_from_world.params)


def _refine_chunks(project, rec, pri, model, rounds, cfg, emit, token, max_iterations, skip, skip_round=None,
                   ck=None):
    """Chunked refinement (modelforge.sfm.chunked): every chunk through `rounds` on its own (its images
    and matches only, intrinsics fixed), a pose graph over all cameras, then a triangulation of all
    images and a polish sweep of windows. rec (all images registered) gets the poses. Returns
    (rec, the last Round, stats). ck (checkpoint.Checkpoint): finished steps are skipped, each new one saved."""
    db_path = os.path.join(project, "database.db")
    image_path = os.path.join(project, "images")
    ck = ck or _NoCheckpoint()
    ck.restore(rec, model)  # the calibrated intrinsics and prior model, or the poses of the last finished step
    ids = {rec.image(i).name: i for i in rec.reg_image_ids()}
    order = priors_mod.by_order(pri, list(ids))
    parts = chunked.chunks(order, cfg.size, cfg.overlap)
    cams = {cid: np.array(cam.params) for cid, cam in rec.cameras.items()}
    log(f"refine: chunked, {len(order)} images in {len(parts)} chunks of {cfg.size} (overlap {cfg.overlap})")
    t0 = time.time()
    init, centrality, edges, rejected, per_chunk = ck.get("chunks", ({}, {}, [], set(), []))
    for k, names in enumerate(parts):
        if skip is not None and skip.is_set():
            break
        if ck.done(f"chunk_{k}"):
            continue
        jobs.check(token)
        tc = time.time()
        crec = refine.init_from_database(db_path, pri, names=names)
        for cid, p in cams.items():
            crec.camera(cid).params = p
        m = robust.copy_model(model)
        crec, _, cres = _rounds(project, crec, pri, m, rounds, "chunk", emit, token, max_iterations, skip, False,
                                write=False, ctx_extra={"chunk": k, "chunks": len(parts)}, skip_round=skip_round)
        seen = {crec.image(i).name: _pose(crec, i) for i in crec.reg_image_ids()
                if crec.image(i).num_points3D >= cfg.min_points}  # a camera without points sits at its prior
        edges += chunked.chunk_edges(cfg, [n for n in names if n in seen], seen)
        rejected |= m.abs_rejected
        for j, name in enumerate(names):
            c = -abs(j - (len(names) - 1) / 2)  # the chunk where an image is most central sets its start
            if name in seen and c > centrality.get(name, -np.inf):
                init[name], centrality[name] = seen[name], c
        last_round = cres[max(cres)] if cres else {}
        err = last_round.get("after_ba", {}).get("point_error_px", {})
        per_chunk.append({"images": len(names), "points": crec.num_points3D(), "error_median_px": err.get("median"),
                          "seconds": round(time.time() - tc, 1)})
        log(f"refine: chunk {k + 1}/{len(parts)}: {len(names)} images, {crec.num_points3D():,} points, "
            f"error median {err.get('median')} px, {time.time() - tc:.0f} s")
        emit({"phase": "chunk_done", "chunk": k, "chunks": len(parts)})
        del crec
        ck.save(f"chunk_{k}", chunks=(init, centrality, edges, rejected, per_chunk))
    shutil.rmtree(os.path.join(project, "sparse", "_chunk_tri"), ignore_errors=True)
    jobs.check(token)
    corrected = robust.corrected_priors(pri, model)
    for name in order:  # cameras no chunk could place start (and stay, but for their prior) at the prior
        if name not in init:
            init[name] = np.array(corrected[name].params)
    if ck.done("posegraph"):
        ck.restore(rec, model)
        stats = ck.get("stats")
    else:
        emit({"phase": "posegraph", "cameras": len(order), "edges": len(edges)})
        pg = chunked.pose_graph(init, edges, corrected, model, rejected, token=token,
                                solver=chunked.solver_options(cfg.posegraph_solver),
                                on_iteration=lambda s: emit({"phase": "posegraph_iteration", "iteration": s.iteration,
                                                             "max_iterations": 200}))
        jobs.check(token)
        log(f"refine: pose graph, {pg['cameras']} cameras, {pg['edges']} edges, {pg['abs_priors']} priors: "
            f"{pg['iterations']} it ({pg['termination']}), {pg['seconds']} s")
        for name, i in ids.items():
            frame = rec.frame(rec.image(i).frame_id)
            frame.rig_from_world = chunked._rigid(init[name])
        for cid, p in cams.items():
            rec.camera(cid).params = p
        stats = {"chunks": per_chunk, "pose_graph": pg, "chunk_size": cfg.size, "overlap": cfg.overlap,
                 "rejected_priors": len(rejected)}
        ck.save("posegraph", rec, model, stats=stats)
    rnd = rounds[-1][1]
    # revisits: the chunks and the pose graph never relate passes over the same ground at different times,
    # so with a drifting prior they triangulate as offset copies: loop-closure edges from the copies' offsets,
    # pose graph again, until they merge
    if cfg.revisit_passes and not (skip is not None and skip.is_set()) and not ck.done("revisits"):
        stats.setdefault("revisits", [])
        for k in range(cfg.revisit_passes):
            if ck.done(f"revisit_{k}"):
                continue
            ck.restore(rec, model)
            emit({"phase": "revisits", "pass": k, "passes": cfg.revisit_passes})
            rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "posegraph_tri"), rnd)
            jobs.check(token)
            loops, rs = chunked.revisit_edges(rec, pri, db_path, cfg)
            log(f"refine: revisit pass {k + 1}: {rs['pairs']} far image pairs split, median {rs['split_median_m']} m "
                f"(p90 {rs['split_p90_m']} m)")
            if not loops or rs["split_median_m"] < cfg.revisit_done_m:
                stats["revisits"].append(rs)
                ck.save("revisits", rec, model, stats=stats)
                break
            cur = {name: _pose(rec, i) for name, i in ids.items()}
            emit({"phase": "posegraph", "cameras": len(cur), "edges": len(edges), "loops": len(loops), "pass": k})
            if os.environ.get("MODELFORGE_DUMP_POSEGRAPH"):  # offline solver experiments (build/pg_bench.py)
                import pickle
                with open(os.path.join(os.environ["MODELFORGE_DUMP_POSEGRAPH"], f"posegraph_{k}.pkl"), "wb") as f:
                    pickle.dump({"init": cur, "edges": edges, "priors": corrected, "model": model,
                                 "rejected": rejected, "loops": loops, "loop_loss": cfg.revisit_loss}, f)
            rs["pose_graph"] = pg = chunked.pose_graph(
                cur, edges, corrected, model, rejected, loops=loops, loop_loss=cfg.revisit_loss, token=token,
                solver=chunked.solver_options(cfg.posegraph_solver),
                on_iteration=lambda s: emit({"phase": "posegraph_iteration", "iteration": s.iteration,
                                             "max_iterations": 200}))
            jobs.check(token)
            log(f"refine: revisit pass {k + 1}: pose graph with {len(loops)} loop edges: {pg['iterations']} it "
                f"({pg['termination']}), {pg['seconds']} s")
            stats["revisits"].append(rs)
            for name, i in ids.items():
                rec.frame(rec.image(i).frame_id).rig_from_world = chunked._rigid(cur[name])
            ck.save(f"revisit_{k}", rec, model, stats=stats)
    # what is left: measure and remove the drift globally
    if model.estimate_drift and not (skip is not None and skip.is_set()) and not ck.done("drift"):
        stats.setdefault("drift", [])
        for k in range(DRIFT_PASSES):
            if ck.done(f"drift_{k}"):
                continue
            ck.restore(rec, model)
            emit({"phase": "drift", "pass": k, "passes": DRIFT_PASSES})
            rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "posegraph_tri"), rnd)
            jobs.check(token)
            dr = robust.update_drift(model, rec, pri, db_path)
            stats["drift"].append(dr)
            log(f"refine: drift pass {k + 1}: {_drift_line(dr)}")
            inc = max((d.get("increment_max_m", 0.0) for d in dr["sources"].values()), default=0.0)
            ck.save(f"drift_{k}", rec, model, stats=stats)
            if inc < DRIFT_DONE_M:
                break
        ck.save("drift", rec, model, stats=stats)
    ck.restore(rec, model)
    corrected = robust.corrected_priors(pri, model)
    # a global BA pulls passes together that the pose graph leaves apart (each chunk keeps its own pass's drift):
    # over all cameras, on thinned points, each after a fresh triangulation (which merges what came closer)
    if cfg.global_passes and not (skip is not None and skip.is_set()) and not ck.done("global"):
        stats.setdefault("global", [])
        model.abs_rejected = rejected
        for k in range(cfg.global_passes):
            if ck.done(f"global_{k}"):
                continue
            ck.restore(rec, model)
            ev = {"round": k, "rounds": cfg.global_passes, "stage": "global"}
            emit({"phase": "triangulate", **ev})
            rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "posegraph_tri"), rnd)
            jobs.check(token)
            _, rs = chunked.revisit_edges(rec, pri, db_path, cfg)
            if k and rs["split_median_m"] < cfg.revisit_done_m:
                ck.save("global", rec, model, stats=stats)
                break
            live = rec
            g = chunked.global_ba(rec, corrected, model, cfg, token=token, skip=skip, loss_scale_px=rnd.loss_scale_px,
                                  max_iterations=max_iterations, skip_round=skip_round,
                                  on_iteration=lambda s: emit({"phase": "ba", **ev, "rec": live,
                                                               "iteration": s.iteration, "cost": s.cost}))
            jobs.check(token)
            stats["global"].append({**rs, "ba": g})
            log(f"refine: global pass {k + 1}: revisits split {rs['split_median_m']} m before (p90 {rs['split_p90_m']} m); "
                f"BA on {g['points']:,} of {g['of_points']:,} points, {g['iterations']} it ({g['termination']})")
            ck.save(f"global_{k}", rec, model, stats=stats)
    ck.restore(rec, model)
    if cfg.polish_sweeps and not (skip is not None and skip.is_set()) and not ck.done("polish"):
        emit({"phase": "triangulate", "round": 0, "rounds": 1, "stage": "polish", "name": "posegraph", "rec": rec})
        rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "posegraph_tri"), rnd)
        jobs.check(token)
        if stats.get("global"):  # what the global passes left
            _, stats["global_split"] = chunked.revisit_edges(rec, pri, db_path, cfg)
            log(f"refine: after the global passes, revisits split {stats['global_split']['split_median_m']} m "
                f"(p90 {stats['global_split']['split_p90_m']} m)")
        win = {}

        def on_window(w, nw, sweep, r):
            win.update(window=w, windows=nw, sweep=sweep)
        local = refine.Local(window=cfg.polish_window, sweeps=cfg.polish_sweeps)
        live = rec
        model.abs_rejected = rejected
        ba = refine.local_ba(rec, corrected, model, local, on_window=on_window, global_pass=False, token=token,
                             skip=skip, loss_scale_px=rnd.loss_scale_px, max_iterations=max_iterations,
                             skip_round=skip_round,
                             on_iteration=lambda s: emit({"phase": "ba", "round": 0, "rounds": 1, "stage": "polish",
                                                          "rec": live, **win, "iteration": s.iteration, "cost": s.cost}))
        jobs.check(token)
        stats["polish"] = ba
        log(f"refine: polish, {ba['windows']} windows, {ba['iterations']} it, {ba['seconds']} s")
        out = os.path.join(project, "sparse", "posegraph")
        os.makedirs(out, exist_ok=True)
        rec.write(out)
        emit({"phase": "round_done", "round": 0, "rounds": 1, "stage": "polish", "name": "posegraph", "rec": rec,
              "eval": refine.evaluate(rec, pri)})
        ck.save("polish", rec, model, stats=stats)
    ck.restore(rec, model)
    stats["seconds"] = round(time.time() - t0, 1)
    return rec, rnd, stats


class _NoCheckpoint:
    """checkpoint.Checkpoint without a file: nothing done, nothing restored or saved."""

    def done(self, step):
        return False

    def get(self, name, default=None):
        return default

    def restore(self, rec, model=None):
        pass

    def save(self, step, rec=None, model=None, **extra):
        pass


def refine_key(project, rounds, use_chunks, cfg, max_iterations, model):
    """The refine's checkpoint key: inputs (database.db, poses.json) and the settings that shape the result."""
    import dataclasses
    return checkpoint.key(project, {"rounds": [vars(r) for r in rounds], "chunked": bool(use_chunks),
                                    "cfg": dataclasses.asdict(cfg) if use_chunks else None,
                                    "max_iterations": max_iterations, "model": model.summary(),
                                    "tie_points": tie_points.load(project)})


def stage_refine(project, report, pri, *args, **kwargs):
    """_stage_refine with the project's manual tie points (sfm/tie_points.py) in every bundle adjustment."""
    with tie_points.active(project) as ties:
        if ties:
            log(f"refine: {len(ties)} tie points ({sum(len(t['obs']) for t in ties)} clicks)")
        return _stage_refine(project, report, pri, *args, **kwargs)


def _stage_refine(project, report, pri, rounds=refine.DEFAULT_ROUNDS, progress=None, model=None, token=None,
                  max_iterations=500, skip=None, ba_mode=DEFAULT_BA_MODE, chunks=None, skip_round=None, resume=False):
    """progress(event: dict) is called with phase "triangulate" | "ba" | "round_done" | "done", and in a
    chunked refine also "chunk_done" | "posegraph"; "ba" events carry the live Reconstruction (poses
    are updated every iteration). Events carry "stage" ("calib" | "round" | "chunk" | "polish"),
    "name" (the snapshot written) and, in a chunked refine, "chunk" / "chunks" or "window" / "windows".
    token (jobs.Token): stop between steps; BA aborts within an iteration.
    max_iterations: BA iteration cap per round (per chunk / window in a chunked refine); 0 = no
    bundle adjustment (poses stay at the priors, intrinsics at their initial values; points are only
    triangulated).
    skip (threading.Event): ends the running BA where it is and skips the remaining rounds; the
    final triangulation and outputs still follow, so the pipeline continues.
    skip_round (threading.Event): ends the running BA where it is (termination SKIPPED_ROUND) and is
    cleared; the next round (chunk, polish window) runs as usual.

    ba_mode: "global" = one BA over all images per round; "chunked" = modelforge.sfm.chunked (chunks
    on their own, a pose graph, a polish sweep); "auto" = chunked from chunks.min_images images on.
    A chunked refine first calibrates the intrinsics and the prior model on a subset
    (refine.calibration_subset: rounds 0-1, global BA, sparse/calib_i[_tri]); the chunks then run the
    rounds after round 0 with them fixed: round 0's loose thresholds exist only for the
    uncalibrated focal length.

    model (robust.PriorModel) holds prior sigmas, systematic corrections and rejected
    priors; it is self-calibrated after every round's BA.

    Every finished step is checkpointed (modelforge.sfm.checkpoint, sparse/refine_resume.pkl); resume=True skips the
    steps a checkpoint of the same inputs and settings holds and continues from its state.
    """
    emit = progress or (lambda event: None)
    model = model or robust.PriorModel()
    cfg = chunks or chunked.Chunked()
    db_path = os.path.join(project, "database.db")
    rec = refine.init_from_database(db_path, pri)
    n_images = rec.num_reg_images()
    auto_why = None
    if ba_mode == "auto":
        auto, auto_why = auto_ba_mode(report.data, n_images, min_images=cfg.min_images)
        log(f"refine: BA mode auto: {auto_why}")
    use_chunks = ba_mode == "chunked" or (ba_mode == "auto" and auto == "chunked")
    ck = checkpoint.Checkpoint(project, refine_key(project, rounds, use_chunks, cfg, max_iterations, model), resume)
    if ck.resumed:
        log(f"refine: resuming after {', '.join(ck.state['done'])}")
    results = {"init": refine.evaluate(rec, pri)}
    t_start = time.time()
    free_intrinsics = known_camera(project) is None
    main = list(enumerate(rounds))
    subset, last = None, rounds[0]  # last: threshold of the last round that ran (the final triangulation uses it)
    if use_chunks and max_iterations and len(rounds) > 1:
        subset = refine.calibration_subset(pri, [rec.image(i).name for i in rec.reg_image_ids()],
                                           cameras={rec.image(i).name: rec.image(i).camera_id
                                                    for i in rec.reg_image_ids()})
        log(f"refine: chunked ({n_images} images); calibrating on {len(subset)} images first")
        crec = refine.init_from_database(db_path, pri, names=subset)
        # rounds 0-1: round 0 converges the focal length, round 1 the prior model at a tight threshold
        crec, clast, cres = _rounds(project, crec, pri, model, main[:2], "calib", emit, token, max_iterations, skip,
                                    free_intrinsics, skip_round=skip_round, ck=ck)
        results.update(cres)
        if clast is not None:
            last = clast
            for cid, cam in crec.cameras.items():
                rec.camera(cid).params = cam.params
        del crec
        if not (skip is not None and skip.is_set()):
            rec, last, results["chunked"] = _refine_chunks(project, rec, pri, model, main[1:], cfg, emit, token,
                                                           max_iterations, skip, skip_round, ck=ck)
    else:
        rec, mlast, rres = _rounds(project, rec, pri, model, main, "round", emit, token, max_iterations, skip,
                                   free_intrinsics, skip_round=skip_round, ck=ck)
        results.update(rres)
        last = mlast or last
    if max_iterations == 0:
        log("refine: no bundle adjustment (0 iterations): poses at the priors, intrinsics at their start values")
    elif skip is not None and skip.is_set():
        log("refine: skipped by the user")
    results["ba"] = {"max_iterations": max_iterations, "skipped": bool(skip is not None and skip.is_set()),
                     "mode": "chunked" if use_chunks else "global", **({"auto": auto_why} if auto_why else {}),
                     **({"chunk_size": cfg.size, "overlap": cfg.overlap, "calibration_images": len(subset)}
                        if subset else {})}
    # final triangulation at the last round's threshold (the tightest after a full run) so the
    # exported cloud matches the refined poses
    jobs.check(token)
    ck.restore(rec, model)  # resumed after the last step: its poses
    image_path = os.path.join(project, "images")
    rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "refined"), last, refine=True)
    jobs.check(token)  # stopped: sparse/refined is not (over)written
    rec.extract_colors_for_all_images(image_path)
    rec.write(os.path.join(project, "sparse", "refined"))
    results["final"] = refine.evaluate(rec, pri)
    check = lidar_mod.sfm_check(rec, project)  # rosbag projects: SfM points vs the LiDAR cloud
    if check:
        results["lidar_check"] = check
        if "median_m" in check:
            log(f"refine: SfM points vs LiDAR ({check['covered'] * 100:.0f}% covered): median {check['median_m']} m, "
                f"p90 {check['p90_m']} m, {check['within_10cm'] * 100:.0f}% within 10 cm")
    results["prior_model"] = model.summary()
    if tie_points._ACTIVE:  # how well the refined cameras agree with the user's clicks
        results["tie_points"] = tie_points.errors(rec)
        log("refine: tie points " + ", ".join(f"{k} {v['error_px']} px ({v['clicks']} clicks)"
                                             for k, v in results["tie_points"].items()))
    results["seconds"] = round(time.time() - t_start, 1)
    import resource
    results["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)  # of this process
    if ck.resumed_from:
        results["resumed_after"] = ck.resumed_from
    report.set("refine", results)
    ck.remove()
    log(f"refine final: {results['final']}")
    emit({"phase": "done", "rec": rec, "eval": results["final"]})
    return rec


def stage_reconstruct(project, report, progress=None, token=None, mapper="incremental"):
    """SfM without priors (videos without metadata) -> sparse/refined; see modelforge.sfm.reconstruct."""
    stats = reconstruct.run(project, mapper=mapper, token=token, progress=progress)
    report.set("reconstruct", stats)
    log(f"reconstruct ({mapper}): {stats['reg_images']}/{stats['images']} images registered, "
        f"{stats['points3D']} points, "
        f"error median {stats['point_error_px'].get('median')} px, models {stats['models']}, {stats['seconds']} s")
    if len(stats["models"]) > 1:
        log(f"reconstruct: the images split into {len(stats['models'])} models; kept the largest")
    return stats


def stage_export(project, report, rec=None, lidar=True, drop_isolated=True, progress=None):
    """transforms.json + sparse_pc.ply; lidar: LiDAR points (kind rosbag) join the splat's initial cloud.
    drop_isolated: leave out photos no other pass ties in whose ground lies off the other passes' (sfm.isolated); the
    points are then triangulated again from the kept photos alone (poses fixed, as refine's final triangulation,
    into sparse/export), so the left-out photos neither add points nor pull any into place.
    progress(text, k, n): finding the isolated photos, triangulating, refining the points, writing."""
    def step(text, k=0, n=1):
        if progress:
            progress(text, k, n)
    if rec is None:
        step("reading sparse/refined")
        rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    kind = sparse_kind(project)  # a ROS bag of images only: an up-oriented local frame, like a video
    world = {"video": "local: up-oriented, arbitrary scale", "rosbag": "the bag's world frame"}.get(kind, "ENU")
    iso = None
    shutil.rmtree(os.path.join(project, "sparse", "export"), ignore_errors=True)  # only this export's model counts
    if drop_isolated and kind != "video":  # metres (the cover distance); a video's frame has no scale
        t0 = time.time()
        times = {r["image"]: r.get("time_s") for r in json.load(open(os.path.join(project, "poses.json")))["images"]}
        iso = isolated.find(rec, times, progress=progress)
        iso["points_before"] = rec.num_points3D()
        db_path = os.path.join(project, "database.db")
        if iso["drop"] and os.path.exists(db_path):
            log(f"export: triangulating again without the {len(iso['drop'])} photos left out")
            for name in iso["drop"]:
                rec.deregister_frame(rec.find_image_with_name(name).frame_id)
            image_path = os.path.join(project, "images")
            rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "export"),
                                     refine.DEFAULT_ROUNDS[-1], refine=True,  # refine's final threshold (4 px)
                                     progress=progress)
            step("colours of the points from the photos")
            rec.extract_colors_for_all_images(image_path)
            rec.write(os.path.join(project, "sparse", "export"))  # View: snapshot "export"
            iso["triangulated"] = True
        elif iso["drop"]:  # no database (copied model): drop the points only they see
            isolated.remove(rec, iso["drop"])
        iso["points"] = rec.num_points3D()
        iso["points_removed"] = iso["points_before"] - iso["points"]
        iso["seconds"] = round(time.time() - t0, 1)
        log(f"export: {len(iso['drop'])} of {iso['photos']} photos left out: weakly tied to other passes and over "
            f"{isolated.MAX_OFF:.0%} of their ground > {isolated.MAX_DZ} m off the others', also on open flat ground "
            f"({iso['isolated']} weakly tied, {iso.get('candidates', 0)} with their ground off); points {iso['points_before']:,} → {iso['points']:,}; stretches (t0, t1, photos, share of cells off) "
            f"{iso['stretches'][:10]}{' …' if len(iso['stretches']) > 10 else ''}")
    step("writing transforms.json + sparse_pc.ply")
    path = nerfstudio.export(rec, project, extra={"world": world, "source": "modelforge.pipeline"}, lidar=lidar,
                             exclude=set(iso["drop"]) if iso else ())
    n_lidar = json.load(open(path)).get("modelforge", {}).get("lidar_points_in_ply", 0)
    rep = {"transforms": path, "frames": len(json.load(open(path))["frames"]), "lidar_points": n_lidar}
    if iso:
        rep["isolated"] = {k: v for k, v in iso.items() if k != "drop"} | {"dropped": len(iso["drop"]),
                                                                          "names": iso["drop"]}
    report.set("export", rep)
    log(f"export: {path}")


def stage_dense(project, report, stage, mesh_methods=("delaunay",), quality=dense.DEFAULT_QUALITY,
                max_image_size=None, token=None, on_progress=None, lidar=True):
    """One dense stage. `quality` is a dense.QUALITY preset; `max_image_size` overrides its image size."""
    q = dict(dense.QUALITY[quality])
    size = q.pop("max_image_size") if max_image_size is None else max_image_size
    rep = report.data.get("dense", {})
    if stage == "undistort":
        rep = {"undistort": {**dense.undistort(project, max_image_size=size), "quality": quality}}  # new workspace
    elif stage == "stereo":
        rep["stereo"] = {**dense.stereo(project, max_image_size=size, token=token, on_progress=on_progress, **q),
                         "quality": quality}
    elif stage == "fusion":
        rep["fusion"] = dense.fusion(project)
        if lidar and lidar_mod.load(project) is not None:  # rosbag projects: LiDAR joins the fused cloud
            rep["fusion"]["lidar"] = dense.add_lidar(project)
            log(f"fusion: + LiDAR {rep['fusion']['lidar']}")
    elif stage == "mesh":
        rep["mesh"] = {m: {**dense.mesh(project, m, token=token), "preview": dense.preview(project, m, token=token)}
                       for m in mesh_methods}
    elif stage == "texture":
        rep["texture"] = {m: dense.texture(project, m, token=token) for m in mesh_methods}
    report.set("dense", rep)
    log(f"{stage}: {rep.get(stage)}")


def expand_stages(spec, groups=GROUPS):
    out = []
    for s in spec.split(","):
        for stage in groups.get(s, (s,)):
            if stage not in out:
                out.append(stage)
    return [s for s in STAGES if s in out]  # canonical order


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--stages", default="sfm",
                    help=f"comma-separated stages {STAGES} or groups {tuple(GROUPS)} (default: sfm)")
    ap.add_argument("--mesh", default="delaunay", choices=dense.MESH_METHODS + ("both",),
                    help="dense meshing method (default delaunay: keeps open surfaces)")
    ap.add_argument("--dense-quality", default=dense.DEFAULT_QUALITY, choices=tuple(dense.QUALITY),
                    help=f"dense speed/detail preset (default {dense.DEFAULT_QUALITY}: "
                         f"{dense.QUALITY[dense.DEFAULT_QUALITY]['max_image_size']} px, fast PatchMatch); "
                         "max = COLMAP defaults at full resolution")
    ap.add_argument("--dense-max-image-size", type=int,
                    help="override the preset's image size for PatchMatch/fusion (-1 = full resolution)")
    ap.add_argument("--ba-iterations", type=int, default=500,
                    help="bundle-adjustment iteration cap per refine round (default 500; it stops earlier once the "
                         "cameras settle); 0 = no BA (poses at the priors, intrinsics at their start values)")
    ap.add_argument("--resume-refine", action="store_true",
                    help="refine: continue after the last finished step of a refine that stopped or died "
                         "(sparse/refine_resume.pkl; same database, poses and settings)")
    ap.add_argument("--ba-mode", default=DEFAULT_BA_MODE, choices=BA_MODES,
                    help="refine BA: global (all images at once), chunked (chunks on their own + a pose graph + "
                         "a polish sweep, after calibrating the intrinsics on a subset), auto (default: global if its "
                         "estimated memory fits in the RAM available, else chunked)")
    ap.add_argument("--chunk-size", type=int, default=chunked.Chunked.size, help="images per chunk (default 500)")
    ap.add_argument("--chunk-overlap", type=int, default=chunked.Chunked.overlap,
                    help="images shared by consecutive chunks (default 100)")
    ap.add_argument("--polish-sweeps", type=int, default=chunked.Chunked.polish_sweeps,
                    help="window sweeps after the pose graph (default 1; 0 = none)")
    ap.add_argument("--features", default="sift", choices=tuple(features.EXTRACTORS),
                    help="feature extractor (default sift on the GPU; sift-robust: DSP + affine shape, CPU)")
    ap.add_argument("--max-features", type=int, default=8192, help="max SIFT features per image (default 8192)")
    ap.add_argument("--upright", action=argparse.BooleanOptionalAction, default=None,
                    help="one keypoint orientation (default: on for DJI, off for videos without metadata)")
    ap.add_argument("--pairs", default="auto", choices=tuple(PAIR_MODES),
                    help="which images get matched (default auto: priors for DJI, exhaustive / sequential otherwise)")
    ap.add_argument("--neighbors", type=int, default=10, help="time neighbours per image for --pairs sequential")
    ap.add_argument("--ratio", type=float, default=0.8, help="Lowe's ratio test for matching (default 0.8)")
    ap.add_argument("--guided", action="store_true", help="guided matching: more inliers, slower")
    ap.add_argument("--min-inliers", type=int, default=15, help="inliers a verified pair needs (default 15)")
    ap.add_argument("--mapper", default="incremental", choices=tuple(reconstruct.MAPPERS),
                    help="SfM mapper for videos without metadata (default incremental; global: faster on large sets)")
    ap.add_argument("--no-relative", action="store_true", help="refine without relative (consecutive-frame) priors")
    ap.add_argument("--no-calibration", action="store_true", help="keep the initial prior sigmas")
    ap.add_argument("--no-systematic", action="store_true", help="do not estimate boresight / time offset")
    ap.add_argument("--no-drift", action="store_true",
                    help="do not estimate the position drift over time (revisits of one long flight)")
    ap.add_argument("--keep-isolated", action="store_true",
                    help="export every photo, also those no other pass ties in (sfm.isolated)")
    args = ap.parse_args()
    unknown = [s for s in args.stages.split(",") if s not in STAGES and s not in GROUPS]
    if unknown:
        ap.error(f"unknown stages {unknown}")
    kind = sparse_kind(args.project)
    sfm = sfm_stages(kind)
    stages = expand_stages(args.stages, {"sfm": sfm, "dense": DENSE_STAGES, "all": sfm + DENSE_STAGES})

    report = Report(args.project)
    pri, meta = priors_mod.load(args.project)
    rec = None
    if "features" in stages:
        stage_features(args.project, report, kind, args.features, args.max_features, args.upright)
    if "pairs" in stages:
        stage_pairs(args.project, report, pri, meta, args.pairs, args.neighbors)
    if "match" in stages:
        stage_match(args.project, report, args.ratio, args.guided, args.min_inliers)
    if "refine" in stages:
        model = robust.PriorModel(use_relative=not args.no_relative, calibrate=not args.no_calibration,
                                  estimate_systematic=not args.no_systematic, estimate_drift=not args.no_drift)
        rec = stage_refine(args.project, report, pri, model=model, max_iterations=args.ba_iterations,
                           resume=args.resume_refine, ba_mode=args.ba_mode, chunks=chunked.Chunked(size=args.chunk_size, overlap=args.chunk_overlap,
                                                                         polish_sweeps=args.polish_sweeps))
    if "reconstruct" in stages:
        stage_reconstruct(args.project, report, mapper=args.mapper)
    if "export" in stages:
        stage_export(args.project, report, rec, drop_isolated=not args.keep_isolated)
    methods = dense.MESH_METHODS if args.mesh == "both" else (args.mesh,)
    for stage in DENSE_STAGES:
        if stage in stages:
            stage_dense(args.project, report, stage, methods, args.dense_quality, args.dense_max_image_size)


if __name__ == "__main__":
    main()
