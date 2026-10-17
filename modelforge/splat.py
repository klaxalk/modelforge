"""Gaussian splat of a project with nerfstudio splatfacto, inside the CUDA dev image.

nerfstudio lives in its own venv (/opt/nerfstudio, docker/Dockerfile NERFSTUDIO=ON) because
its pins (viser 1.0.0, opencv 4.10, numpy<2) clash with ours; this module runs its CLIs as
subprocesses:
  train    ns-train <method> --data PROJECT --output-dir PROJECT/nerfstudio
           (loads our sparse_pc.ply as initial points; viewer on :7007 while training)
  export   ns-export gaussian-splat -> PROJECT/splat/splat.ply (nerfstudio's normalized frame)
  convert  modelforge.export.splat_to_enu -> PROJECT/splat/splat_enu.ply (ENU, aligned with
           the refined cameras and the COLMAP mesh)
Stats go to report.json["splat"]. The GUI Run tab and scripts/splat both call `run`.

Usage (CUDA container):
    python -m modelforge.splat data/site [--method splatfacto-big] [--iterations 15000] [extra ns-train args]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

NS_BIN = os.environ.get("NERFSTUDIO_BIN", "/opt/nerfstudio/bin")
METHODS = ("splatfacto", "splatfacto-big")
DEFAULT_ITERATIONS = 30000  # splatfacto's default
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


def train_command(project, method="splatfacto", iterations=None, extra=()):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cmd = [os.path.join(NS_BIN, "ns-train"), method, "--data", project,
           "--output-dir", os.path.join(project, "nerfstudio"),
           "--viewer.quit-on-train-completion", "True", "--viewer.websocket-port", str(VIEWER_PORT)]
    if iterations:
        cmd += ["--max-num-iterations", str(int(iterations))]
    return cmd + list(extra)


def parse_progress(line):
    """(step, percent) from a nerfstudio progress row, else None."""
    m = _PROGRESS.match(line)
    return (int(m.group(1)), float(m.group(2))) if m else None


def latest_config(project, method):
    configs = glob.glob(os.path.join(project, "nerfstudio", "*", method, "*", "config.yml"))
    if not configs:
        raise RuntimeError(f"no {method} training run found in {project}/nerfstudio")
    return max(configs, key=os.path.getmtime)


def run(project, method="splatfacto", iterations=None, extra=(), on_line=print, on_progress=None, job=None):
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

    on_line(f"splat: training {method} (viewer on http://localhost:{VIEWER_PORT})")
    _stream(train_command(project, method, iterations, extra), line, job)
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
    stats = {"method": method, "iterations": iterations or DEFAULT_ITERATIONS, "config": config, "splat": enu,
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
    args, extra = ap.parse_known_args()
    try:
        run(os.path.abspath(args.project), args.method, args.iterations, extra,
            on_line=lambda m: print(m, flush=True))
    except (RuntimeError, ValueError, Stopped) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
