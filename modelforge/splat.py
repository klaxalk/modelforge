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

Usage (CUDA container):
    python -m modelforge.splat data/site [--method splatfacto-big] [--iterations 15000]
        [--cache disk|cpu|gpu] [--downscale 2] [extra ns-train args]
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

NS_BIN = os.environ.get("NERFSTUDIO_BIN", "/opt/nerfstudio/bin")
METHODS = ("splatfacto", "splatfacto-big")
DEFAULT_ITERATIONS = 30000  # splatfacto's default
CACHE_MODES = ("disk", "cpu", "gpu")  # nerfstudio's cache_images; disk keeps memory use flat
VIEWER_PORT = 7007
# nerfstudio's progress table rows: "29910 (99.70%)   12.345 ms   1 s, 230.21 ms ..."
_PROGRESS = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # the table is redrawn with cursor-up escapes


class Stopped(Exception):
    pass


class Job:
    """Handle to stop a running `run` from another thread."""

    def __init__(self):
        self.proc, self.stopped = None, False

    def stop(self):
        self.stopped = True
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


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
    return env


def _stream(cmd, on_line, job):
    job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=_env())
    for line in job.proc.stdout:
        line = _ANSI.sub("", line).rstrip()
        if line:
            on_line(line)
    rc = job.proc.wait()
    if job.stopped:
        raise Stopped(f"{os.path.basename(cmd[0])} stopped")
    if rc:
        raise RuntimeError(f"{os.path.basename(cmd[0])} exited with code {rc} (see the log)")


def train_command(project, method="splatfacto", iterations=None, extra=(), cache="disk", downscale=1,
                  steps_per_save=None):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if cache not in CACHE_MODES:
        raise ValueError(f"cache must be one of {CACHE_MODES}")
    cmd = [os.path.join(NS_BIN, "ns-train"), method, "--data", project,
           "--output-dir", os.path.join(project, "nerfstudio"),
           "--viewer.quit-on-train-completion", "True", "--viewer.websocket-port", str(VIEWER_PORT),
           "--pipeline.datamanager.cache-images", cache]
    if iterations:
        cmd += ["--max-num-iterations", str(int(iterations))]
    if steps_per_save:
        cmd += ["--steps-per-save", str(int(steps_per_save))]
    # the dataparser subcommand goes last; an explicit factor (also 1) stops nerfstudio from
    # auto-picking a leftover images_2/ for images wider than 1600 px
    return cmd + list(extra) + ["nerfstudio-data", "--downscale-factor", str(int(downscale))]


def prepare_downscaled(project, factor, on_line=print):
    """PROJECT/images_<factor>/<name> for every transforms.json frame (nerfstudio's layout for
    --downscale-factor). Sizes are floor(w / factor), as nerfstudio rescales the cameras."""
    import json

    from PIL import Image
    if factor <= 1:
        return None
    out = os.path.join(project, f"images_{factor}")
    os.makedirs(out, exist_ok=True)
    frames = json.load(open(os.path.join(project, "transforms.json")))["frames"]
    made = 0
    for f in frames:
        src = os.path.join(project, f["file_path"])
        dst = os.path.join(out, os.path.basename(f["file_path"]))
        with Image.open(src) as im:
            size = (im.width // factor, im.height // factor)
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                with Image.open(dst) as old:
                    if old.size == size:
                        continue
            im.convert("RGB").resize(size, Image.BICUBIC).save(dst, quality=95)
            made += 1
    on_line(f"splat: images_{factor}/ ready ({made} of {len(frames)} images written)")
    return out


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
        cache="disk", downscale=1, preview_every=None, on_snapshot=None):
    """Train, export and convert to ENU. Returns stats (also written to report.json["splat"])."""
    from modelforge.export import splat_to_enu
    from modelforge.pipeline import Report
    reason = available()
    if reason:
        raise RuntimeError(reason)
    if not os.path.exists(os.path.join(project, "transforms.json")):
        raise RuntimeError("transforms.json missing: run the SfM export stage first")
    job = job or Job()
    t0 = time.time()

    def line(text):
        on_line(text)
        p = parse_progress(text)
        if p and on_progress:
            on_progress(*p)

    prepare_downscaled(project, int(downscale), on_line)
    on_line(f"splat: training {method}, cache {cache}, downscale {downscale} "
            f"(viewer on http://localhost:{VIEWER_PORT})")
    stop = threading.Event()
    if on_snapshot:
        threading.Thread(target=_watch_checkpoints, args=(project, method, t0, on_snapshot, stop, on_line),
                         daemon=True).start()
    try:
        _stream(train_command(project, method, iterations, extra, cache, downscale,
                              preview_every if on_snapshot else None), line, job)
    finally:
        stop.set()
    t_train = time.time() - t0
    config = latest_config(project, method)
    out_dir = os.path.join(project, "splat")
    on_line(f"splat: exporting {config}")
    _stream([os.path.join(NS_BIN, "ns-export"), "gaussian-splat", "--load-config", config, "--output-dir", out_dir],
            on_line, job)
    on_line("splat: converting to the project's ENU frame")
    enu = os.path.join(out_dir, "splat_enu.ply")
    splat_to_enu.convert(os.path.join(out_dir, "splat.ply"),
                         os.path.join(os.path.dirname(config), "dataparser_transforms.json"), enu)
    stats = {"method": method, "iterations": iterations or DEFAULT_ITERATIONS, "cache": cache,
             "downscale": int(downscale), "config": config, "splat": enu,
             "num_gaussians": _num_vertices(enu), "train_seconds": round(t_train, 1),
             "seconds": round(time.time() - t0, 1)}
    Report(project).set("splat", stats)
    on_line(f"splat: {stats}")
    return stats


def _num_vertices(ply):
    import plyfile
    return len(plyfile.PlyData.read(ply)["vertex"].data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--method", choices=METHODS, default="splatfacto")
    ap.add_argument("--iterations", type=int, help=f"training steps (default {DEFAULT_ITERATIONS})")
    ap.add_argument("--cache", choices=CACHE_MODES, default="disk",
                    help="where training images are kept (default disk: low memory, for large datasets)")
    ap.add_argument("--downscale", type=int, default=1, help="train on images downscaled by this factor (default 1)")
    args, extra = ap.parse_known_args()
    try:
        run(os.path.abspath(args.project), args.method, args.iterations, extra,
            on_line=lambda m: print(m, flush=True), cache=args.cache, downscale=args.downscale)
    except (RuntimeError, ValueError, Stopped) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
