"""Headless modelforge pipeline on an ingested project (output of modelforge.dji.extract).

Stages (artifacts in PROJECT/):
  features  features.db                    SIFT only; matching works on a copy
  pairs     pairs.txt                      prior-based pair selection
  match     database.db                    copy of features.db + matches + two-view geometry
  refine    sparse/round_N, sparse/refined prior-initialized triangulation + prior BA rounds
  export    transforms.json, sparse_pc.ply nerfstudio dataset
Every stage records its stats in PROJECT/report.json.

Usage:
    python -m modelforge.pipeline data/clip_frames [--stages pairs,match,refine,export]
"""

import argparse
import json
import os
import shutil
import sys
import time

import pycolmap

from modelforge.export import nerfstudio
from modelforge.sfm import features, pairs, priors as priors_mod, refine

STAGES = ("features", "pairs", "match", "refine", "export")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Report:
    def __init__(self, project):
        self.path = os.path.join(project, "report.json")
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def set(self, key, value):
        self.data[key] = value
        json.dump(self.data, open(self.path, "w"), indent=1)


def stage_features(project, report):
    stats = features.extract(project)
    log(f"features: {stats}")
    report.set("features", stats)


def stage_pairs(project, report, pri, meta):
    ground_z = -meta["enu_origin_lla"][2]
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


def stage_refine(project, report, pri, rounds=refine.DEFAULT_ROUNDS):
    db_path = os.path.join(project, "database.db")
    image_path = os.path.join(project, "images")
    rec = refine.init_from_database(db_path, pri)
    results = {"init": refine.evaluate(rec, pri)}
    t_start = time.time()
    for i, rnd in enumerate(rounds):
        t0 = time.time()
        rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", f"round_{i}_tri"), rnd)
        tri = refine.evaluate(rec, pri)
        ba = refine.prior_ba(rec, pri, loss_scale_px=rnd.loss_scale_px)
        out = os.path.join(project, "sparse", f"round_{i}")
        os.makedirs(out, exist_ok=True)
        rec.write(out)
        ev = refine.evaluate(rec, pri)
        results[f"round_{i}"] = {"thresholds": vars(rnd), "after_triangulation": tri, "ba": ba, "after_ba": ev,
                                 "seconds": round(time.time() - t0, 1)}
        log(f"round {i} ({rnd.max_reproj_px:g}px): tri {tri['points3D']} pts, {tri['mean_reproj_error_px']}px "
            f"-> BA {ba['iterations']} it, {ev['points3D']} pts, {ev['mean_reproj_error_px']}px, "
            f"prior dev pos {ev['prior_deviation']['position_m']} rot {ev['prior_deviation']['rotation_deg']}")
    # final triangulation at the tightest threshold so the exported cloud matches the refined poses
    rec = refine.triangulate(rec, db_path, image_path, os.path.join(project, "sparse", "refined"), rounds[-1])
    rec.extract_colors_for_all_images(image_path)
    rec.write(os.path.join(project, "sparse", "refined"))
    results["final"] = refine.evaluate(rec, pri)
    results["seconds"] = round(time.time() - t_start, 1)
    report.set("refine", results)
    log(f"refine final: {results['final']}")
    return rec


def stage_export(project, report, rec=None):
    if rec is None:
        rec = pycolmap.Reconstruction(os.path.join(project, "sparse", "refined"))
    path = nerfstudio.export(rec, project, extra={"world": "ENU", "source": "modelforge.pipeline"})
    report.set("export", {"transforms": path, "frames": rec.num_reg_images()})
    log(f"export: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    args = ap.parse_args()
    stages = args.stages.split(",")
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages {unknown}")

    report = Report(args.project)
    pri, meta = priors_mod.load(args.project)
    rec = None
    if "features" in stages:
        stage_features(args.project, report)
    if "pairs" in stages:
        stage_pairs(args.project, report, pri, meta)
    if "match" in stages:
        stage_match(args.project, report)
    if "refine" in stages:
        rec = stage_refine(args.project, report, pri)
    if "export" in stages:
        stage_export(args.project, report, rec)


if __name__ == "__main__":
    main()
