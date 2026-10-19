"""Calibration sweep for the image subset (Images used): how few images still give a good splat.

A throwaway copy of a project (images hard-linked) holds back a fixed set of images (every
`holdout_every`-th of each source, from the middle of the first block) that no run trains or
reconstructs on. Every configuration (every N-th image, keyframes at each density level) then runs
SfM (features, pairs, match, refine, export), a splat with equal settings (fixed steps, downscale,
no camera optimizer) and the held-out evaluation (modelforge/evaluate.py). Results per run go to
SWEEP/runs/<name>/ and SWEEP/sweep.json (resumable: finished runs are skipped), a table to sweep.md.

    python -m modelforge.subset_sweep SRC_PROJECT SWEEP_DIR [--every 1,2,3,4,6,8]
        [--levels dense,normal,sparse] [--iterations 30000] [--downscale 2] [--eval-width 1920]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from modelforge import evaluate, projects, splat, splat_plan
from modelforge.dji import ingest
from modelforge.sfm import overlap


def log(msg):
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(src, ws, holdout_every=10):
    """SWEEP_DIR with the source's extracted images minus the held-out ones; holdout.json."""
    if os.path.exists(os.path.join(ws, "holdout.json")):
        return json.load(open(os.path.join(ws, "holdout.json")))
    img_dir, poses_path = overlap.pool(src)
    poses = json.load(open(poses_path))
    held = set()
    for names in overlap.ordered(poses["images"]).values():
        held.update(names[holdout_every // 2::holdout_every])
    os.makedirs(os.path.join(ws, "images"))
    keep = [im for im in poses["images"] if im["image"] not in held]
    for im in keep:
        os.link(os.path.join(img_dir, im["image"]), os.path.join(ws, "images", im["image"]))
    json.dump(dict(poses, images=keep), open(os.path.join(ws, "poses.json"), "w"), indent=1)
    for f in (ingest.PLAN_FILE, "camera.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(ws, f))
    holdout = {"images_dir": os.path.abspath(img_dir),
               "images": [im for im in poses["images"] if im["image"] in held]}
    json.dump(holdout, open(os.path.join(ws, "holdout.json"), "w"), indent=1)
    log(f"prepared {ws}: {len(keep)} images, {len(holdout['images'])} held out")
    return holdout


def one_run(ws, name, args, holdout, iterations, downscale, eval_width, analysis, res):
    """Fills res (kept as far as it got when a step fails)."""
    t0 = time.time()
    pool_images = projects.extracted_images(ws)
    sel = args.get("names") or [pool_images[k]["image"] for k in
                                projects.subsample_selection(pool_images, **{k: v for k, v in args.items()
                                                                            if k != "setting"})]
    n, total, archived = projects.apply_subsample(ws, **args)
    if archived:
        shutil.rmtree(archived)  # a throwaway workspace: the previous run's results are in runs/
    res.update(images=n, of=total, overlap=overlap.overlap_stats(pool_images, analysis, sel))
    log(f"{name}: {n} images, SfM")
    r = subprocess.run([sys.executable, "-m", "modelforge.pipeline", ws, "--stages",
                        "features,pairs,match,refine,export"], capture_output=True, text=True)
    open(os.path.join(ws, "sfm.log"), "w").write(r.stdout + r.stderr)
    rep = json.load(open(os.path.join(ws, "report.json"))) if os.path.exists(os.path.join(ws, "report.json")) else {}
    fin = rep.get("refine", {}).get("final", {})
    res["sfm"] = {"ok": r.returncode == 0, "registered": fin.get("reg_images"), "points": fin.get("points3D"),
                  "error_median_px": fin.get("point_error_px", {}).get("median"),
                  "seconds": round(sum(rep.get(s, {}).get("seconds", 0) or 0 for s in ("features", "match"))
                                   + (rep.get("refine", {}).get("seconds") or 0), 1)}
    if r.returncode or not fin.get("points3D") or fin["points3D"] < 100:
        res["failed"] = "SfM: " + (r.stderr.strip().splitlines()[-1] if r.returncode and r.stderr.strip()
                                   else f"{fin.get('points3D')} points")
        return res
    log(f"{name}: splat")
    plan = splat_plan.suggest(splat_plan.survey(ws), iterations=iterations, downscale=downscale, camera_opt="off")
    st = splat.run(ws, "splatfacto", iterations, downscale=downscale, schedule=plan, on_line=lambda m: None)
    res["splat"] = {"gaussians": st["num_gaussians"], "train_seconds": st["train_seconds"],
                    "strategy": plan["strategy"]}
    log(f"{name}: evaluation")
    ev = evaluate.run(ws, holdout, width=eval_width, log=lambda m: None)
    res["eval"] = {k: ev.get(k) for k in ("psnr", "ssim", "lpips", "psnr_p10", "empty", "located", "held_out")}
    res["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(ws, "runs", name)
    os.makedirs(out, exist_ok=True)
    for f in ("report.json", "evaluation/metrics.json", "sfm.log"):
        if os.path.exists(os.path.join(ws, f)):
            shutil.copy2(os.path.join(ws, f), os.path.join(out, os.path.basename(f)))
    return res


def table(results):
    rows = ["| run | images | SfM points | err px | SfM s | train s | gaussians | PSNR | PSNR p10 | SSIM | LPIPS "
            "| overlap min / p5 / median |", "|" + "---|" * 12]
    for r in results:
        o = (r.get("overlap") or {}).get("consecutive") or {}
        e, s, sp = r.get("eval", {}), r.get("sfm", {}), r.get("splat", {})
        rows.append(f"| {r['name']} | {r['images']} | {s.get('points')} | {s.get('error_median_px')} | "
                    f"{s.get('seconds')} | {sp.get('train_seconds', '')} | {sp.get('gaussians', '')} | "
                    f"{e.get('psnr', r.get('failed', ''))} | {e.get('psnr_p10', '')} | {e.get('ssim', '')} | "
                    f"{e.get('lpips', '')} | {o.get('min')} / {o.get('p5')} / {o.get('median')} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("ws")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--every", default="1,2,3,4,6,8")
    ap.add_argument("--levels", default=",".join(overlap.LEVELS))
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--eval-width", type=int, default=1920)
    a = ap.parse_args()
    ws = os.path.abspath(a.ws)
    holdout = prepare(os.path.abspath(a.src), ws, a.holdout_every)
    analysis = overlap.load(ws)
    if analysis is None:
        log("overlap analysis")
        analysis = overlap.analyze(ws, kind=ingest.load_kind(ws), log=log)
    done_path = os.path.join(ws, "sweep.json")
    results = json.load(open(done_path)) if os.path.exists(done_path) else []
    done = {r["name"] for r in results}
    pool = projects.extracted_images(ws)
    configs = [(f"every{n}", {"every": n}) for n in map(int, a.every.split(",")) if n]
    for lv in filter(None, a.levels.split(",")):
        names = overlap.select_keyframes(pool, analysis, overlap.LEVELS[lv])
        configs.append((f"keyframes-{lv}", {"names": names, "setting": {"method": "keyframes", "level": lv,
                                                                         "min_overlap": overlap.LEVELS[lv]}}))
    for name, args in configs:
        if name in done:
            continue
        res = {"name": name, "images": len(args.get("names") or [])}
        try:
            one_run(ws, name, args, holdout, a.iterations, a.downscale, a.eval_width, analysis, res)
        except Exception as e:  # noqa: BLE001 - record and go on with the next configuration
            import traceback
            traceback.print_exc()
            res["failed"] = f"{type(e).__name__}: {e}"
        log(f"{name}: {res.get('eval') or res.get('failed')}")
        results.append(res)
        json.dump(results, open(done_path, "w"), indent=1)
        open(os.path.join(ws, "sweep.md"), "w").write(table(results) + "\n")
    log("done")


if __name__ == "__main__":
    main()
