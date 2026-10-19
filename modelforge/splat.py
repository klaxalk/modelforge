"""Gaussian splat of a project with nerfstudio splatfacto, inside the CUDA dev image.

nerfstudio lives in its own venv (/opt/nerfstudio, docker/Dockerfile NERFSTUDIO=ON) because
its pins (viser 1.0.0, opencv 4.10, numpy<2) clash with ours; this module runs its CLIs as
subprocesses:
  train    ns-train <method> --data PROJECT --output-dir PROJECT/nerfstudio
           (loads our sparse_pc.ply as initial points; viewer on :7007 while training).
           Images are cached on disk by default (cache="disk": nothing preloaded into RAM or
           GPU memory, for large datasets); downscale N trains on PROJECT/images_N/, made here.
  export   ns-export gaussian-splat -> PROJECT/splat/splat.ply (nerfstudio's normalized frame)
  convert  modelforge.export.splat_to_enu -> PROJECT/splat/splat_enu.ply (ENU, aligned with
           the refined cameras and the COLMAP mesh)
Live preview: with preview_every=N, ns-train saves a checkpoint every N steps; a watcher dumps
each new one (export/ckpt_dump.py, in nerfstudio's venv) to PROJECT/splat/live.npz and calls
on_snapshot(step, npz, dataparser_transforms.json). The GUI shows it in the View tab.
Stats go to report.json["splat"]. The GUI Splat tab and scripts/splat both call `run`.
LiDAR depth supervision (lidar_depth=weight, projects with lidar/cloud.ply): trains splatfacto-lidar /
splatfacto-big-lidar from modelforge/nsplugin (registered through nerfstudio's environment variables,
nothing installed) with the lidar-data dataparser; its "[lidar] ..." depth error lines go to the report.

Training plan (modelforge/splat_plan.py): iterations, densification strategy (default with a reset
period scaled to the image count, or MCMC with a Gaussian budget), learning-rate decay length and camera
pose refinement come from the project's cameras and sparse points (count, overlap, GSD) and the GPU
memory, unless given; full resolution unless --downscale. splatfacto's fixed schedule never densifies
from ~2900 training images on.

Usage (CUDA container):
    python -m modelforge.splat data/site [--method splatfacto-big] [--iterations 15000]
        [--cache disk|cpu|gpu] [--downscale 2] [--strategy default|mcmc] [--max-gaussians N]
        [--camera-opt off|SO3xR3|SE3] [--no-plan] [--lidar-depth 0.2] [extra ns-train args]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from modelforge import jobs

NS_BIN = os.environ.get("NERFSTUDIO_BIN", "/opt/nerfstudio/bin")
METHODS = ("splatfacto", "splatfacto-big")
PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsplugin")
DEFAULT_LIDAR_DEPTH = 0.2  # weight of the LiDAR depth loss (relative depth error, see the plugin)
_LIDAR = re.compile(r"\[lidar\] step (\d+): depth error median ([\d.]+)% p90 ([\d.]+)% within 5% ([\d.]+)%")
DEFAULT_ITERATIONS = 30000  # splatfacto's default
CACHE_MODES = ("disk", "cpu", "gpu")  # nerfstudio's cache_images; disk keeps memory use flat
VIEWER_PORT = 7007
# nerfstudio's progress table rows: "29910 (99.70%)   12.345 ms   1 s, 230.21 ms ..."
_PROGRESS = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # the table is redrawn with cursor-up escapes


Stopped = jobs.Cancelled  # stopping is shared with the other stages (modelforge.jobs)
Job = jobs.Token


def available():
    """None if splat training can run here, else the reason."""
    if not os.path.exists(os.path.join(NS_BIN, "ns-train")):
        return ("nerfstudio is not installed in this image: use the CUDA image "
                "(compose.gpu.yaml builds it with NERFSTUDIO=ON) and MODELFORGE_GPU=1")
    if not glob.glob("/dev/nvidia[0-9]*") and not shutil.which("nvidia-smi"):
        return "no NVIDIA GPU visible in this container"
    return None


def _env():
    env = dict(os.environ, PYTHONUNBUFFERED="1", TERM="dumb", COLUMNS="160")
    env["PATH"] = NS_BIN + os.pathsep + env.get("PATH", "")  # nerfstudio's own tools first
    # torch >= 2.6 loads weights_only by default, which rejects nerfstudio 1.1.5 checkpoints
    # (numpy scalars inside); they are our own, written by ns-train in this project
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for k in ("VIRTUAL_ENV", "PYTHONPATH"):  # keep /opt/venv and the repo out of its imports
        env.pop(k, None)
    env["PYTHONPATH"] = PLUGIN_DIR  # only the nerfstudio plugin (splatfacto-lidar)
    env["NERFSTUDIO_METHOD_CONFIGS"] = ("splatfacto-lidar=mf_splatfacto_lidar:splatfacto_lidar,"
                                        "splatfacto-big-lidar=mf_splatfacto_lidar:splatfacto_big_lidar")
    env["NERFSTUDIO_DATAPARSER_CONFIGS"] = "lidar-data=mf_lidar_dataparser:lidar_dataparser"
    return env


def _stream(cmd, on_line, job):
    def line(text):
        text = _ANSI.sub("", text).rstrip()
        if text:
            on_line(text)
    job.run(cmd, line, env=_env())  # Cancelled when stopped, RuntimeError on failure


def ns_method(method, lidar_depth=None):
    """nerfstudio method name: the -lidar variant when LiDAR depth is on (a weight, also 0)."""
    return method if lidar_depth is None else method + "-lidar"


def has_lidar(project):
    import json
    path = os.path.join(project, "transforms.json")
    return os.path.exists(path) and "lidar_ply_file_path" in json.load(open(path))


DOWNSCALES = (1, 2, 4, 8)  # image downsample factors offered (GUI, CLI)


STRATEGIES = ("default", "mcmc")  # gsplat densification: split/clone/prune, or MCMC with a fixed budget
CAMERA_OPT = ("off", "SO3xR3", "SE3")  # splatfacto's camera optimizer (pose refinement while training)


def schedule_args(iterations=None, strategy=None, max_gaussians=None, stop_split_at=None, reset_alpha_every=None,
                  camera_opt=None):
    """ns-train flags for the densification and learning-rate schedule (see modelforge/splat_plan.py)."""
    args = []
    if iterations:  # nerfstudio's method config decays these over a fixed 30000 steps
        for group in ("means", "camera-opt", "bilateral-grid"):
            args += [f"--optimizers.{group}.scheduler.max-steps", str(int(iterations))]
    if strategy:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        args += ["--pipeline.model.strategy", strategy]
        if strategy == "mcmc":  # as nerfstudio's splatfacto-mcmc
            args += ["--pipeline.model.cull-alpha-thresh", "0.005"]
            if max_gaussians:
                args += ["--pipeline.model.max-gs-num", str(int(max_gaussians))]
    if stop_split_at:
        args += ["--pipeline.model.stop-split-at", str(int(stop_split_at))]
    if reset_alpha_every:
        args += ["--pipeline.model.reset-alpha-every", str(int(reset_alpha_every))]
    if camera_opt:
        if camera_opt not in CAMERA_OPT:
            raise ValueError(f"camera_opt must be one of {CAMERA_OPT}")
        args += ["--pipeline.model.camera-optimizer.mode", camera_opt]
    return args


def train_command(project, method="splatfacto", iterations=None, extra=(), cache="disk", downscale=1,
                  steps_per_save=None, lidar_depth=None, schedule=None):
    """lidar_depth: None = plain splatfacto; a weight (0 = only report the depth error) = the -lidar
    method of modelforge/nsplugin with the lidar-data dataparser. schedule: splat_plan settings
    (strategy, max_gaussians, stop_split_at, reset_alpha_every, camera_opt)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if cache not in CACHE_MODES:
        raise ValueError(f"cache must be one of {CACHE_MODES}")
    cmd = [os.path.join(NS_BIN, "ns-train"), ns_method(method, lidar_depth), "--data", project,
           "--output-dir", os.path.join(project, "nerfstudio"),
           "--viewer.quit-on-train-completion", "True", "--viewer.websocket-port", str(VIEWER_PORT),
           "--pipeline.datamanager.cache-images", cache,
           # splatfacto-big (and -mcmc) keep method_name "splatfacto", so their runs would land in
           # nerfstudio/*/splatfacto/ where the preview watcher and latest_config don't look
           "--method-name", ns_method(method, lidar_depth),
           # rendering every eval image every 1000 steps: minutes per pass with hundreds of eval images
           "--steps-per-eval-all-images", "0"]
    if iterations:
        cmd += ["--max-num-iterations", str(int(iterations))]
    if schedule:
        cmd += schedule_args(iterations, **{k: schedule.get(k) for k in
                                            ("strategy", "max_gaussians", "stop_split_at", "reset_alpha_every",
                                             "camera_opt")})
    if steps_per_save:
        cmd += ["--steps-per-save", str(int(steps_per_save))]
    if lidar_depth is not None:
        cmd += ["--pipeline.model.lidar-depth-lambda", str(float(lidar_depth))]
    # the dataparser subcommand goes last; an explicit factor (also 1) stops nerfstudio from
    # auto-picking a leftover images_2/ for images wider than 1600 px
    parser = "nerfstudio-data" if lidar_depth is None else "lidar-data"
    return cmd + list(extra) + [parser, "--downscale-factor", str(int(downscale))]


def prepare_downscaled(project, factor, on_line=print, job=None, on_count=None, workers=8):
    """PROJECT/images_<factor>/<name> for every transforms.json frame (nerfstudio's layout for
    --downscale-factor). Sizes are floor(w / factor), as nerfstudio rescales the cameras.

    JPEGs are decoded at reduced size (PIL draft: the DCT scales by 1/2, 1/4, 1/8 directly) in
    `workers` threads; on_count(done, total) reports progress and `job` (jobs.Token) is checked
    between images. Before, 8000 4K images took tens of minutes, silently and without Stop."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    if factor <= 1:
        return None
    out = os.path.join(project, f"images_{factor}")
    os.makedirs(out, exist_ok=True)
    frames = json.load(open(os.path.join(project, "transforms.json")))["frames"]

    def one(f):
        jobs.check(job)
        src = os.path.join(project, f["file_path"])
        dst = os.path.join(out, os.path.basename(f["file_path"]))
        with Image.open(src) as im:
            size = (im.width // factor, im.height // factor)
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                with Image.open(dst) as old:
                    if old.size == size:
                        return 0
            im.draft("RGB", size)  # JPEG: decode at >= size, a fraction of the full decode
            im.convert("RGB").resize(size, Image.BICUBIC).save(dst + ".part.jpg", quality=95)
        os.replace(dst + ".part.jpg", dst)  # a stopped run leaves no truncated image
        return 1

    made, step = 0, max(1, len(frames) // 50)
    with ThreadPoolExecutor(workers) as pool:
        for k, m in enumerate(pool.map(one, frames), 1):
            made += m
            if on_count and (k % step == 0 or k == len(frames)):
                on_count(k, len(frames))
    on_line(f"splat: images_{factor}/ ready ({made} of {len(frames)} images written)")
    return out


def parse_lidar(line):
    """{"step", "median_pct", "p90_pct", "within_5pct"} from the plugin's "[lidar] step ..." line."""
    m = _LIDAR.search(line)
    if not m:
        return None
    return {"step": int(m.group(1)), "median_pct": float(m.group(2)), "p90_pct": float(m.group(3)),
            "within_5pct": float(m.group(4))}


def parse_progress(line):
    """(step, percent) from a nerfstudio progress row, else None."""
    m = _PROGRESS.match(line)
    return (int(m.group(1)), float(m.group(2))) if m else None


def latest_config(project, method):
    configs = glob.glob(os.path.join(project, "nerfstudio", "*", method, "*", "config.yml"))
    if not configs:
        raise RuntimeError(f"no {method} training run found in {project}/nerfstudio")
    return max(configs, key=os.path.getmtime)


def _watch_checkpoints(project, method, t0, on_snapshot, stop, on_line):
    """Dump every new checkpoint of the run started at t0 and pass it to on_snapshot. Reports what
    it does through on_line (the GUI log), so a preview that never appears leaves a trace."""
    dump = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "ckpt_dump.py")
    out = os.path.join(project, "splat", "live.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seen, run_dir, failures = None, None, 0
    on_line("splat: live preview waiting for the first checkpoint")
    while not stop.wait(2.0):
        runs = [d for d in glob.glob(os.path.join(project, "nerfstudio", "*", method, "*"))
                if os.path.isdir(d) and os.path.getmtime(d) >= t0 - 2]
        if not runs:
            continue
        if max(runs, key=os.path.getmtime) != run_dir:
            run_dir = max(runs, key=os.path.getmtime)
            on_line(f"splat: live preview watching {run_dir}")
        ckpts = sorted(glob.glob(os.path.join(run_dir, "nerfstudio_models", "step-*.ckpt")))
        transforms = os.path.join(run_dir, "dataparser_transforms.json")
        if not ckpts or ckpts[-1] == seen or not os.path.exists(transforms):
            continue
        try:
            r = subprocess.run([os.path.join(NS_BIN, "python"), dump, ckpts[-1], out], env=_env(),
                               capture_output=True, text=True, timeout=300)
            rc, err = r.returncode, (r.stderr or "").strip()
        except subprocess.TimeoutExpired:
            rc, err = -1, "timed out"
        if rc:  # usually still being written: retry on the next poll, but say so when it persists
            failures += 1
            if failures in (2, 10) or failures % 50 == 0:
                on_line(f"splat: live preview cannot read {os.path.basename(ckpts[-1])} "
                        f"(attempt {failures}): {err.splitlines()[-1] if err else rc}")
            continue
        seen, failures = ckpts[-1], 0
        step = int(re.search(r"step-(\d+)", seen).group(1))
        try:
            on_snapshot(step, out, transforms)
        except Exception as e:  # a preview problem must never stop training
            on_line(f"splat: live preview failed at step {step}: {e}")


def run(project, method="splatfacto", iterations=None, extra=(), on_line=print, on_progress=None, job=None,
        cache="disk", downscale=1, preview_every=None, on_snapshot=None, lidar_depth=None, on_prepare=None,
        schedule=None):
    """Train, export and convert to ENU. Returns stats (also written to report.json["splat"]).
    schedule: densification / learning-rate settings from splat_plan.suggest (or None: splatfacto's).
    lidar_depth: weight of the LiDAR depth loss (None: off; needs lidar_ply_file_path in transforms.json).
    on_prepare(done, total): progress of writing the downscaled images (downscale > 1)."""
    from modelforge.export import splat_to_enu
    from modelforge.pipeline import Report
    reason = available()
    if reason:
        raise RuntimeError(reason)
    if not os.path.exists(os.path.join(project, "transforms.json")):
        raise RuntimeError("transforms.json missing: run the SfM export stage first")
    if lidar_depth is not None and not has_lidar(project):
        raise RuntimeError("LiDAR depth needs LiDAR points (a ROS 2 bag project with point cloud topics); "
                           "re-run the export stage after extraction")
    job = job or Job()
    t0 = time.time()
    lidar_log = []
    method_dir = ns_method(method, lidar_depth)

    def line(text):
        on_line(text)
        p = parse_progress(text)
        if p and on_progress:
            on_progress(*p)
        q = parse_lidar(text)
        if q:
            lidar_log.append(q)

    prepare_downscaled(project, int(downscale), on_line, job=job, on_count=on_prepare)
    on_line(f"splat: training {method_dir}, cache {cache}, downscale {downscale}"
            + (f", LiDAR depth weight {lidar_depth}" if lidar_depth is not None else "")
            + (f", {iterations or DEFAULT_ITERATIONS:,} steps" if schedule else "")
            + (f", {schedule['strategy']} densification" if schedule and schedule.get("strategy") else "")
            + (f" (≤{schedule['max_gaussians']:,} gaussians)" if schedule and schedule.get("max_gaussians") else "")
            + (f", camera optimizer {schedule['camera_opt']}" if schedule and schedule.get("camera_opt") else "")
            + " "
            f"(viewer on http://localhost:{VIEWER_PORT})")
    stop = threading.Event()
    if on_snapshot:
        threading.Thread(target=_watch_checkpoints, args=(project, method_dir, t0, on_snapshot, stop, on_line),
                         daemon=True).start()
    try:
        _stream(train_command(project, method, iterations, extra, cache, downscale,
                              preview_every if on_snapshot else None, lidar_depth, schedule), line, job)
    finally:
        stop.set()
    t_train = time.time() - t0
    config = latest_config(project, method_dir)
    out_dir = os.path.join(project, "splat")
    on_line(f"splat: exporting {config}")
    _stream([os.path.join(NS_BIN, "ns-export"), "gaussian-splat", "--load-config", config, "--output-dir", out_dir],
            on_line, job)
    on_line("splat: converting to the project's ENU frame")
    enu = os.path.join(out_dir, "splat_enu.ply")
    splat_to_enu.convert(os.path.join(out_dir, "splat.ply"),
                         os.path.join(os.path.dirname(config), "dataparser_transforms.json"), enu)
    stats = {"method": method_dir, "iterations": iterations or DEFAULT_ITERATIONS, "cache": cache,
             "downscale": int(downscale), "config": config, "splat": enu,
             "num_gaussians": _num_vertices(enu), "train_seconds": round(t_train, 1),
             "seconds": round(time.time() - t0, 1)}
    if schedule:
        stats["schedule"] = {k: schedule.get(k) for k in ("strategy", "max_gaussians", "stop_split_at",
                                                          "reset_alpha_every", "camera_opt") if schedule.get(k)}
    if lidar_depth is not None:
        stats["lidar_depth"] = {"weight": float(lidar_depth), "error": lidar_log[-1] if lidar_log else None,
                                "history": lidar_log}
    Report(project).set("splat", stats)
    on_line(f"splat: {dict(stats, lidar_depth={k: v for k, v in stats['lidar_depth'].items() if k != 'history'})}"
            if "lidar_depth" in stats else f"splat: {stats}")
    return stats


def _num_vertices(ply):
    import plyfile
    return len(plyfile.PlyData.read(ply)["vertex"].data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--method", choices=METHODS, default="splatfacto")
    ap.add_argument("--iterations", type=int, help="training steps (default: from the project's size)")
    ap.add_argument("--cache", choices=CACHE_MODES, default="disk",
                    help="where training images are kept (default disk: low memory, for large datasets)")
    ap.add_argument("--downscale", type=int, choices=DOWNSCALES,
                    help="train on images downscaled by this factor (default 1: full resolution)")
    ap.add_argument("--strategy", choices=STRATEGIES, help="densification (default: mcmc when the scene needs "
                    "more Gaussians than fit, else default with a reset period scaled to the image count)")
    ap.add_argument("--max-gaussians", type=int, help="MCMC budget (default: from the scene and GPU memory)")
    ap.add_argument("--camera-opt", choices=CAMERA_OPT, help="pose refinement while training (default SO3xR3)")
    ap.add_argument("--no-plan", action="store_true",
                    help="splatfacto's own defaults (30000 steps, no schedule) instead of the flight-based plan")
    ap.add_argument("--lidar-depth", type=float, metavar="W",
                    help=f"LiDAR depth loss weight (ROS 2 bag projects; e.g. {DEFAULT_LIDAR_DEPTH}; 0 = report only)")
    args, extra = ap.parse_known_args()
    project = os.path.abspath(args.project)
    try:
        if args.no_plan:
            iterations, downscale, schedule = args.iterations, args.downscale or 1, None
        else:
            from modelforge import splat_plan
            schedule = splat_plan.suggest(splat_plan.survey(project), iterations=args.iterations,
                                          downscale=args.downscale, strategy=args.strategy,
                                          max_gaussians=args.max_gaussians, camera_opt=args.camera_opt,
                                          method=args.method)
            print("splat plan: " + "; ".join(schedule["notes"]), flush=True)
            iterations, downscale = schedule["iterations"], schedule["downscale"]
        run(project, args.method, iterations, extra, on_line=lambda m: print(m, flush=True), cache=args.cache,
            downscale=downscale, lidar_depth=args.lidar_depth, schedule=schedule)
    except (RuntimeError, ValueError, Stopped) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
