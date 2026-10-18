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
import time

import pycolmap

from modelforge import dense, jobs
from modelforge.export import nerfstudio
from modelforge.dji import ingest
from modelforge.sfm import features, pairs, priors as priors_mod, reconstruct, refine, robust

SFM_STAGES = ("features", "pairs", "match", "refine", "export")  # DJI: refine from the pose priors
VIDEO_SFM_STAGES = ("features", "pairs", "match", "reconstruct", "export")  # no metadata: incremental SfM
DENSE_STAGES = ("undistort", "stereo", "fusion", "mesh", "texture")
STAGES = ("features", "pairs", "match", "refine", "reconstruct", "export") + DENSE_STAGES  # canonical order
GROUPS = {"sfm": SFM_STAGES, "dense": DENSE_STAGES, "all": SFM_STAGES + DENSE_STAGES}


def sfm_stages(kind):
    """The sparse stages of a project kind (ingest.json "kind")."""
    return VIDEO_SFM_STAGES if kind == "video" else SFM_STAGES


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


def stage_features(project, report, kind="dji"):
    """DJI: upright SIFT (the gimbal keeps the horizon level) and an OPENCV camera refined against the
    priors. Videos without metadata: oriented SIFT (handheld footage rolls) and a RADIAL camera (one
    focal length: without priors, separate fx / fy drift apart on short sequences)."""
    video = kind == "video"
    stats = features.extract(project, upright=not video, camera_model="RADIAL" if video else "OPENCV")
    log(f"features: {stats}")
    report.set("features", stats)


def stage_pairs(project, report, pri, meta):
    if not priors_mod.has_poses(pri):  # video without metadata: sequential + quadratic (+ exhaustive if small)
        names = priors_mod.by_order(pri)
        seqs = [[n for n in names if pri[n].source == s] for s in sorted({pri[n].source for n in names})]
        pp = pairs.sequential(seqs)
        pairs.write(project, pp)
        report.set("pairs", {"num_pairs": len(pp), "method": "sequential"})
        log(f"pairs: {len(pp)} (sequential)")
        return
    ground_z = meta.get("ground_z", -meta["enu_origin_lla"][2])
    pp = pairs.select(pri, ground_z)
    pairs.write(project, pp)
    report.set("pairs", {"num_pairs": len(pp), "ground_z": ground_z})
    log(f"pairs: {len(pp)}")


def copy_db(src, dst):
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(dst + ext):
            os.remove(dst + ext)
    shutil.copyfile(src, dst)


def match_stats(db_path, seconds):
    db = pycolmap.Database.open(db_path)
    s = {"matched_pairs": db.num_matched_image_pairs(), "verified_pairs": db.num_verified_image_pairs(),
         "inlier_matches": db.num_inlier_matches(), "seconds": round(seconds, 1)}
    db.close()
    return s


def stage_match(project, report):
    db_path = os.path.join(project, "database.db")
    copy_db(os.path.join(project, "features.db"), db_path)
    opts = pycolmap.ImportedPairingOptions()
    opts.match_list_path = os.path.join(project, "pairs.txt")
    t0 = time.time()
    pycolmap.match_image_pairs(db_path, pairing_options=opts)
    stats = match_stats(db_path, time.time() - t0)
    log(f"match: {stats}")
    report.set("match", stats)


def stage_refine(project, report, pri, rounds=refine.DEFAULT_ROUNDS, progress=None, model=None, token=None,
                 max_iterations=500, skip=None):
    """progress(event: dict) is called with phase "triangulate" | "ba" | "round_done" | "done";
    "ba" events carry the live Reconstruction (poses are updated every iteration).
    token (jobs.Token): stop between steps; BA aborts within an iteration.
    max_iterations: BA iteration cap per round; 0 = no bundle adjustment (poses stay at the priors,
    intrinsics at their initial values; points are only triangulated).
    skip (threading.Event): ends the running BA where it is and skips the remaining rounds; the
    final triangulation and outputs still follow, so the pipeline continues.

    model (robust.PriorModel) holds prior sigmas, systematic corrections and rejected
    priors; it is self-calibrated after every round's BA.
    """
    emit = progress or (lambda event: None)
    model = model or robust.PriorModel()
    db_path = os.path.join(project, "database.db")
    image_path = os.path.join(project, "images")
    rec = refine.init_from_database(db_path, pri)
    results = {"init": refine.evaluate(rec, pri)}
    t_start = time.time()
    last = rounds[0]  # threshold of the last round that ran (the final triangulation uses it)
    for i, rnd in enumerate(rounds):
        if max_iterations == 0 or (skip is not None and skip.is_set()):
            break
        last = rnd
        t0 = time.time()
        emit({"phase": "triangulate", "round": i, "rounds": len(rounds), "rec": rec})
        jobs.check(token)
        rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", f"round_{i}_tri"), rnd)
        jobs.check(token)
        tri = refine.evaluate(rec, pri)
        live = rec
        corrected = robust.corrected_priors(pri, model)
        ba = refine.prior_ba(rec, corrected, model, loss_scale_px=rnd.loss_scale_px, relative=i >= model.start_round,
                             on_iteration=lambda s, i=i: emit({"phase": "ba", "round": i, "rounds": len(rounds),
                                                               "iteration": s.iteration, "cost": s.cost, "rec": live}),
                             token=token, max_iterations=max_iterations, skip=skip)
        jobs.check(token)  # an aborted BA leaves no round_N
        out = os.path.join(project, "sparse", f"round_{i}")
        os.makedirs(out, exist_ok=True)
        rec.write(out)
        offsets = robust.update_source_offsets(model, rec, pri, db_path)
        ev = refine.evaluate(rec, pri)
        calib = robust.update(model, rec, pri) if i >= model.start_round else {"model": model.summary(), "skipped": True}
        results[f"round_{i}"] = {"thresholds": vars(rnd), "after_triangulation": tri, "ba": ba, "after_ba": ev,
                                 "prior_model": calib, "source_offsets": offsets, "seconds": round(time.time() - t0, 1)}
        log(f"round {i} ({rnd.max_reproj_px:g}px): tri {tri['points3D']} pts, median {tri['point_error_px']['median']}px "
            f"-> BA {ba['iterations']} it ({ba['termination']}), median {ev['point_error_px']['median']}px, "
            f"prior dev pos {ev['prior_deviation']['position_m']} rot {ev['prior_deviation']['rotation_deg']}")
        if offsets:
            log(f"   source offsets ({offsets['method']}): {offsets['offsets_m']}")
        m = calib["model"]
        if not calib.get("skipped"):
            log(f"   prior model: scales {calib.get('scales')} | abs {m['abs_sigmas']} rel {m['rel_sigmas']} | "
                f"B {m['boresight_norm_deg']} deg, dt {m['dt_ms']} ms | rejected abs {calib['abs_rejected']} "
                f"rel {calib['rel_rejected']} | d2 median abs {calib['abs_d2_median']} rel {calib['rel_d2_median']}")
        emit({"phase": "round_done", "round": i, "rounds": len(rounds), "rec": rec, "eval": ev})
    if max_iterations == 0:
        log("refine: no bundle adjustment (0 iterations): poses at the priors, intrinsics at their start values")
    elif skip is not None and skip.is_set():
        log(f"refine: skipped by the user after round {i}")
    results["ba"] = {"max_iterations": max_iterations, "skipped": bool(skip is not None and skip.is_set())}
    # final triangulation at the last round's threshold (the tightest after a full run) so the
    # exported cloud matches the refined poses
    jobs.check(token)
    rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "refined"), last)
    jobs.check(token)  # stopped: sparse/refined is not (over)written
    rec.extract_colors_for_all_images(image_path)
    rec.write(os.path.join(project, "sparse", "refined"))
    results["final"] = refine.evaluate(rec, pri)
    results["prior_model"] = model.summary()
    results["seconds"] = round(time.time() - t_start, 1)
    report.set("refine", results)
    log(f"refine final: {results['final']}")
    emit({"phase": "done", "rec": rec, "eval": results["final"]})
    return rec


def stage_reconstruct(project, report, progress=None, token=None):
    """Incremental SfM (videos without metadata) -> sparse/refined; see modelforge.sfm.reconstruct."""
    stats = reconstruct.incremental(project, token=token, progress=progress)
    report.set("reconstruct", stats)
    log(f"reconstruct: {stats['reg_images']}/{stats['images']} images registered, {stats['points3D']} points, "
        f"error median {stats['point_error_px'].get('median')} px, models {stats['models']}, {stats['seconds']} s")
    if len(stats["models"]) > 1:
        log(f"reconstruct: the images split into {len(stats['models'])} models; kept the largest")
    return stats


def stage_export(project, report, rec=None):
    if rec is None:
        rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    world = "ENU" if ingest.load_kind(project) != "video" else "local: up-oriented, arbitrary scale"
    path = nerfstudio.export(rec, project, extra={"world": world, "source": "modelforge.pipeline"})
    report.set("export", {"transforms": path, "frames": rec.num_reg_images()})
    log(f"export: {path}")


def stage_dense(project, report, stage, mesh_methods=("delaunay",), quality=dense.DEFAULT_QUALITY,
                max_image_size=None, token=None, on_progress=None):
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
    ap.add_argument("--no-relative", action="store_true", help="refine without relative (consecutive-frame) priors")
    ap.add_argument("--no-calibration", action="store_true", help="keep the initial prior sigmas")
    ap.add_argument("--no-systematic", action="store_true", help="do not estimate boresight / time offset")
    args = ap.parse_args()
    unknown = [s for s in args.stages.split(",") if s not in STAGES and s not in GROUPS]
    if unknown:
        ap.error(f"unknown stages {unknown}")
    kind = ingest.load_kind(args.project)
    sfm = sfm_stages(kind)
    stages = expand_stages(args.stages, {"sfm": sfm, "dense": DENSE_STAGES, "all": sfm + DENSE_STAGES})

    report = Report(args.project)
    pri, meta = priors_mod.load(args.project)
    rec = None
    if "features" in stages:
        stage_features(args.project, report, kind)
    if "pairs" in stages:
        stage_pairs(args.project, report, pri, meta)
    if "match" in stages:
        stage_match(args.project, report)
    if "refine" in stages:
        model = robust.PriorModel(use_relative=not args.no_relative, calibrate=not args.no_calibration,
                                  estimate_systematic=not args.no_systematic)
        rec = stage_refine(args.project, report, pri, model=model, max_iterations=args.ba_iterations)
    if "reconstruct" in stages:
        stage_reconstruct(args.project, report)
    if "export" in stages:
        stage_export(args.project, report, rec)
    methods = dense.MESH_METHODS if args.mesh == "both" else (args.mesh,)
    for stage in DENSE_STAGES:
        if stage in stages:
            stage_dense(args.project, report, stage, methods, args.dense_quality, args.dense_max_image_size)


if __name__ == "__main__":
    main()
