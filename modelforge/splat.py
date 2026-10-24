"""Gaussian splat of a project with nerfstudio splatfacto, inside the CUDA dev image.

nerfstudio lives in its own venv (/opt/nerfstudio, docker/Dockerfile NERFSTUDIO=ON) because
its pins (viser 1.0.0, opencv 4.10, numpy<2) clash with ours; this module runs its CLIs as
subprocesses:
  train    ns-train <method> --data PROJECT --output-dir PROJECT/nerfstudio
           (loads our sparse_pc.ply as initial points; viewer on :7007 while training).
           Images are cached by auto_cache: GPU memory when small, else decoded in RAM, else disk;
           downscale N trains on PROJECT/images_N/, made here.
  export   ns-export gaussian-splat -> PROJECT/splat/splat.ply (nerfstudio's normalized frame)
  convert  modelforge.export.splat_to_enu -> PROJECT/splat/splat_enu.ply (ENU, aligned with
           the refined cameras and the COLMAP mesh)
Live preview: with preview_every=N, ns-train saves a checkpoint every N steps; a watcher dumps
each new one (export/ckpt_dump.py, in nerfstudio's venv) to PROJECT/splat/live.npz and calls
on_snapshot(step, npz, dataparser_transforms.json). The GUI shows it in the View tab.
Stats go to report.json["splat"]. The GUI Splat tab and scripts/splat both call `run`.
Depth supervision (lidar_depth=weight): trains splatfacto-lidar / splatfacto-big-lidar from modelforge/nsplugin
(registered through nerfstudio's environment variables, nothing installed) with the lidar-data dataparser, against
the LiDAR points (projects with lidar/cloud.ply) or else the SfM points (sparse_pc.ply); its "[lidar] ..." depth
error lines go to the report.

Training plan (modelforge/splat_plan.py): iterations, densification strategy (default with a reset
period scaled to the image count, or MCMC with a Gaussian budget), learning-rate decay length and camera
pose refinement come from the project's cameras and sparse points (count, overlap, GSD) and the GPU
memory, unless given; full resolution unless --downscale. splatfacto's fixed schedule never densifies
from ~2900 training images on.

Usage (CUDA container):
    python -m modelforge.splat data/site [--method splatfacto-big] [--iterations 15000]
        [--cache disk|cpu|gpu] [--downscale 2] [--strategy default|mcmc] [--max-gaussians N]
        [--camera-opt off|SO3xR3|SE3] [--no-plan] [--lidar-depth 0.2] [--resume] [extra ns-train args]
Stop (GUI) / Pause: the trainer saves a checkpoint at the current step (nsplugin/mf_train.py); Stop then
exports the partial splat, and `resume` (--resume) continues with the run's own settings.
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from modelforge import jobs
from modelforge.splat_plan import DOWNSCALES  # image downsample factors offered (GUI, CLI)

NS_BIN = os.environ.get("NERFSTUDIO_BIN", "/opt/nerfstudio/bin")
METHODS = ("splatfacto", "splatfacto-big")
PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsplugin")
DEFAULT_LIDAR_DEPTH = 0.2  # weight of the LiDAR depth loss (relative depth error, see the plugin)
_LIDAR = re.compile(r"\[lidar\] step (\d+): depth error median ([\d.]+)% p90 ([\d.]+)% within 5% ([\d.]+)%(?: gated ([\d.]+)%)?")
DEFAULT_ITERATIONS = 30000  # splatfacto's default
CACHE_MODES = ("disk", "cpu", "gpu")  # nerfstudio's cache_images; disk keeps memory use flat


def auto_cache(project, downscale=1):
    """Image cache for training: gpu when the decoded photos take < 1/4 of the GPU memory, else cpu when twice
    their size fits in half the available RAM, else disk. gpu only up to 500 training images: nerfstudio
    switches to cpu above that (a 740-photo tile was planned gpu and cached in RAM). gpu took 11.7 GB of the 24 GB on karlak (471 4K photos),
    crowding the Gaussians out (training ran out of memory). cpu holds the photos twice (the trainer's copy + the
    data workers' shared memory): 2000 photos at 1920 px (12.4 GB decoded) took 11.7 + 12.1 GB and the kernel
    killed the training on a 31 GB machine. The speed differs little (karlak, 6000 steps: disk 405 s, gpu 381 s:
    nerfstudio decodes in parallel workers). Returns (mode, decoded GB)."""
    import json
    from modelforge import splat_plan
    from modelforge.export import frame_camera
    t = json.load(open(os.path.join(project, "transforms.json")))
    gb = sum((c["w"] // downscale) * (c["h"] // downscale) * 3 for c in map(lambda f: frame_camera.camera(t, f),
                                                                          t["frames"])) / 1e9
    vram = splat_plan.vram_gb() or splat_plan.DEFAULT_VRAM_GB
    try:
        avail = next(int(line.split()[1]) for line in open("/proc/meminfo")
                     if line.startswith("MemAvailable")) / 1e6
    except (OSError, StopIteration, ValueError):
        avail = 0.0
    # nerfstudio turns gpu into cpu above 500 training images (full_images_datamanager)
    if gb < 0.25 * vram and math.ceil(0.9 * len(t["frames"])) <= 500:
        return "gpu", gb
    if 2 * gb < 0.5 * avail:
        return "cpu", gb
    return "disk", gb
VIEWER_PORT = 7007
# nerfstudio's progress table rows: "29910 (99.70%)   12.345 ms   1 s, 230.21 ms ..."
_PROGRESS = re.compile(r"^\s*(\d+)\s+\((\d+(?:\.\d+)?)%\)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # the table is redrawn with cursor-up escapes


Stopped = jobs.Cancelled  # stopping is shared with the other stages (modelforge.jobs)
Job = jobs.Token
RESUME = os.path.join("splat", "resume.json")  # a stopped / paused run that can continue
CONTROL = os.path.join("splat", "control")  # asks the trainer for a checkpoint and exit (nsplugin/mf_train.py)


class Paused(jobs.Cancelled):
    """Training was paused: checkpoint saved, resume.json written, nothing exported."""


class StoppedPartial(jobs.Cancelled):
    """Training was stopped: checkpoint saved, the partial splat exported to splat/splat_enu.ply."""


def available():
    """None if splat training can run here, else the reason."""
    if not os.path.exists(os.path.join(NS_BIN, "ns-train")):
        return ("nerfstudio is not installed in this image: use the CUDA image "
                "(compose.gpu.yaml builds it with NERFSTUDIO=ON) and MODELFORGE_GPU=1")
    if not glob.glob("/dev/nvidia[0-9]*") and not shutil.which("nvidia-smi"):
        return "no NVIDIA GPU visible in this container"
    return None


def _env(extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TERM="dumb", COLUMNS="160", **(extra or {}))
    env["PATH"] = NS_BIN + os.pathsep + env.get("PATH", "")  # nerfstudio's own tools first
    # torch >= 2.6 loads weights_only by default, which rejects nerfstudio 1.1.5 checkpoints
    # (numpy scalars inside); they are our own, written by ns-train in this project
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    # karlak at 4K: 5.8 GB of 22 GB reserved but unusable (fragmentation) when training ran out of memory
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k in ("VIRTUAL_ENV", "PYTHONPATH"):  # keep /opt/venv and the repo out of its imports
        env.pop(k, None)
    env["PYTHONPATH"] = PLUGIN_DIR  # only the nerfstudio plugin (splatfacto-lidar)
    env["NERFSTUDIO_METHOD_CONFIGS"] = ("splatfacto-lidar=mf_splatfacto_lidar:splatfacto_lidar,"
                                        "splatfacto-big-lidar=mf_splatfacto_lidar:splatfacto_big_lidar")
    env["NERFSTUDIO_DATAPARSER_CONFIGS"] = "lidar-data=mf_lidar_dataparser:lidar_dataparser"
    return env


def _stream(cmd, on_line, job, env=None, soft_stop=None):
    def line(text):
        text = _ANSI.sub("", text).rstrip()
        if text:
            on_line(text)
    job.run(cmd, line, env=_env(env), soft_stop=soft_stop)  # Cancelled when stopped, RuntimeError on failure


def ns_method(method, lidar_depth=None):
    """nerfstudio method name: the -lidar variant when LiDAR depth is on (a weight, also 0)."""
    return method if lidar_depth is None else method + "-lidar"


def has_lidar(project):
    import json
    path = os.path.join(project, "transforms.json")
    return os.path.exists(path) and "lidar_ply_file_path" in json.load(open(path))


def depth_source(project):
    """What the depth loss (lidar_depth) measures against: "LiDAR" (lidar_ply_file_path), "SfM points" (the
    dataset's sparse_pc.ply; the lidar-data dataparser falls back to it) or None."""
    import json
    path = os.path.join(project, "transforms.json")
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    if "lidar_ply_file_path" in t:
        return "LiDAR"
    return "SfM points" if os.path.exists(os.path.join(project, t.get("ply_file_path", "sparse_pc.ply"))) else None


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
                  steps_per_save=None, lidar_depth=None, schedule=None, load_dir=None, depth_cloud=None,
                  depth_maps=None, regularize=None, seed_ply=None, loss_masks=None):
    """lidar_depth: None = plain splatfacto; a weight (0 = only report the depth error) = the -lidar
    method of modelforge/nsplugin with the lidar-data dataparser. schedule: splat_plan settings
    (strategy, max_gaussians, stop_split_at, reset_alpha_every, camera_opt). load_dir: continue from the
    latest checkpoint there (a paused / stopped run; `run` trains through nsplugin/mf_train.py, which
    makes that work for splatfacto). depth_cloud: the depth loss's points from this ply instead (absolute path;
    the SfM mesh, sfm_mesh.depth_cloud). depth_maps: dense depth targets instead, a folder of per-image maps
    (mesh_depth_maps). regularize: {"flatten": w, "exposure": bool, "bilateral": bool} (see regularizing; exposure:
    each render scaled by its photo's relative exposure, transforms.json "mf_exposure"; bilateral: splatfacto's
    bilateral grid per photo, plain splatfacto too); flatten and exposure need the -lidar
    method (lidar_depth 0 = depth loss off). seed_ply: extra starting points (sfm_mesh.seed_points), started flat;
    lidar-data only. loss_masks: a folder of per-image masks at the undistorted training camera (splat_tiles.loss_masks:
    a tile's pixels outside it are left out of the losses); lidar-data only."""
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
    if load_dir:
        cmd += ["--load-dir", load_dir]
    if lidar_depth is not None:
        cmd += ["--pipeline.model.lidar-depth-lambda", str(float(lidar_depth))]
        reg = regularize or {}
        if reg.get("flatten"):
            cmd += ["--pipeline.model.flatten-lambda", f"{float(reg['flatten']):g}"]
        if reg.get("exposure"):
            cmd += ["--pipeline.model.use-exposure", "True"]
    if (regularize or {}).get("bilateral"):
        cmd += ["--pipeline.model.use-bilateral-grid", "True"]
    if (regularize or {}).get("antialiased"):  # Mip-Splatting's 2D filter (gsplat)
        cmd += ["--pipeline.model.rasterize-mode", "antialiased"]
    # the dataparser subcommand goes last; an explicit factor (also 1) stops nerfstudio from
    # auto-picking a leftover images_2/ for images wider than 1600 px
    parser = "nerfstudio-data" if lidar_depth is None else "lidar-data"
    scale = (schedule or {}).get("scale_factor")  # world units -> nerfstudio's (else 1 / max |camera position|)
    scale_args = ["--auto-scale-poses", "False", "--scale-factor", f"{float(scale):.8g}"] if scale else []
    depth_args = [] if lidar_depth is None else (["--depth-maps", str(depth_maps)] if depth_maps else
                                                 ["--depth-ply", str(depth_cloud)] if depth_cloud else [])
    if seed_ply and lidar_depth is not None:
        depth_args += ["--seed-ply", str(seed_ply)]
    if loss_masks and lidar_depth is not None:
        depth_args += ["--loss-masks", str(loss_masks)]
    return cmd + list(extra) + [parser, "--downscale-factor", str(int(downscale))] + scale_args + depth_args


FLATTEN = 0.02  # default flattening weight: 0.1 flattened every Gaussian completely (fullrun_test)
EXPOSURE = True  # default: renders scaled by the photos' exposure from the metadata (karlak A/B 2026-10-24: held-out
#                  +0.24 dB rendered with the photo's gain, +0.23 dB colour-corrected; docs/todo_splat_quality.md)


def exposure_known(project):
    """True when transforms.json carries the photos' exposure (mf_exposure: DJI metadata, written by the export)."""
    try:
        t = json.load(open(os.path.join(project, "transforms.json")))
    except (OSError, ValueError):
        return False
    return any("mf_exposure" in f for f in t.get("frames", []))


def regularizing(regularize):
    """True when flattening is on (modelforge/nsplugin/mf_splatfacto_lidar.py)."""
    return bool(regularize) and (float(regularize.get("flatten") or 0) > 0 or bool(regularize.get("exposure")))
    # older records may name far_mask: gone


def mesh_depth_maps(project, mesh, downscale, on_line=print, job=None, on_count=None):
    """Depth maps of the mesh at every camera of PROJECT/transforms.json (a tile's dataset works too), as the
    training images are undistorted and cropped (export/mesh_depth.py in nerfstudio's venv), in
    <mesh folder>/depth_x<downscale>/ (shared by the tiles; kept while mesh, downscale and camera stay).
    Returns the folder."""
    out = os.path.join(os.path.dirname(mesh), f"depth_x{int(downscale)}")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "mesh_depth.py")

    def line(text):
        on_line(text)
        m = re.search(r"mesh depth: (\d+) / (\d+)", text)
        if m and on_count:
            on_count(int(m.group(1)), int(m.group(2)))
    _stream([os.path.join(NS_BIN, "python"), script, project, mesh, out, "--downscale", str(int(downscale))],
            line, job or Job())
    return out


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
            "within_5pct": float(m.group(4)), **({"gated_pct": float(m.group(5))} if m.group(5) else {})}


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
        schedule=None, pause=None, load_dir=None, stop_at=None, depth_cloud=None, depth_dense=False, regularize=None,
        seed_mesh=None, loss_masks=None):
    """Train, export and convert to ENU. Returns stats (also written to report.json["splat"]).
    schedule: densification / learning-rate settings from splat_plan.suggest (or None: splatfacto's).
    Stopping the job (job.cancel) asks the trainer for a checkpoint at the current step first; the run's
    settings go to splat/resume.json (see `resume`). With `pause` (threading.Event) set: Paused, nothing
    exported. Otherwise the partial splat is exported and converted as usual, then StoppedPartial.
    load_dir: continue from that checkpoint folder (used by `resume`).
    stop_at: end at this step with a checkpoint (splat/resume.json, as a Stop) and export the splat so far, but
    return normally with stats["stopped_at_step"] (splat tiles: the end of a round).
    lidar_depth: weight of the depth loss (None: off): against the LiDAR points (lidar_ply_file_path), else the
    SfM points (sparse_pc.ply), or `depth_cloud` (a ply path: the SfM mesh's vertices) when given; with
    depth_dense the mesh rendered into every training camera instead (mesh_depth_maps, before training).
    on_prepare(done, total): progress of writing the downscaled images (downscale > 1).
    regularize: {"flatten": w}; on without a depth weight trains the -lidar method with the depth loss off (lidar_depth 0).
    seed_mesh: the SfM mesh (sfmmesh/mesh.ply): extra starting Gaussians on it where the SfM points are thin, flat in
    its surface (sfm_mesh.seed_points -> splat/mesh_seeds.ply; the -lidar method, as above).
    loss_masks: per-image loss masks (a folder, see train_command), or a function of the depth maps' folder that writes
    them and returns theirs (splat_tiles: the tile mask, from the dense depth maps)."""
    from modelforge.export import splat_to_enu
    from modelforge.pipeline import Report
    reason = available()
    if reason:
        raise RuntimeError(reason)
    if not os.path.exists(os.path.join(project, "transforms.json")):
        raise RuntimeError("transforms.json missing: run the SfM export stage first")
    from modelforge.splat_plan import ply_count
    if ply_count(os.path.join(project, "sparse_pc.ply")) == 0 and not any("random-init" in a for a in extra):
        raise RuntimeError("sparse_pc.ply has no points: SfM found no 3D points to start the splat from (too little "
                           "overlap between the images? see the refine History line; Images used: keep more)")
    if depth_cloud and not os.path.exists(depth_cloud):
        raise RuntimeError(f"depth prior {depth_cloud} missing: run SfM mesh again (Sparse tab)")
    if lidar_depth is not None and not depth_cloud and depth_source(project) is None:
        raise RuntimeError("the depth loss needs LiDAR or SfM points (sparse_pc.ply): run the export stage first")
    if (regularizing(regularize) or seed_mesh) and lidar_depth is None:
        lidar_depth = 0.0  # the plugin's model and dataparser: depth loss off, the regularization / seeds on
    if loss_masks is not None and lidar_depth is None:
        lidar_depth = 0.0
    if seed_mesh and not os.path.exists(seed_mesh):
        raise RuntimeError(f"SfM mesh {seed_mesh} missing: run SfM mesh again (Sparse Mesh tab)")
    job = job or Job()
    t0 = time.time()
    lidar_log = []
    method_dir = ns_method(method, lidar_depth)

    def line(text):
        on_line(text)
        p = parse_progress(text)
        if p and on_progress:
            on_progress(*p)
        if p and stop_at and p[0] >= stop_at and not os.path.exists(control):
            open(control, "w").write("checkpoint")  # the trainer saves and ends by itself
        q = parse_lidar(text)
        if q:
            lidar_log.append(q)

    settings = {"method": method, "iterations": iterations, "extra": list(extra), "cache": cache,
                "downscale": int(downscale), "lidar_depth": lidar_depth, "schedule": schedule,
                "depth_cloud": depth_cloud, "depth_dense": bool(depth_dense), "regularize": regularize,
                "seed_mesh": seed_mesh}
    if load_dir is None:
        _remove(os.path.join(project, RESUME))  # a new run replaces the paused one
    prepare_downscaled(project, int(downscale), on_line, job=job, on_count=on_prepare)
    on_line(("splat: resuming " if load_dir else "splat: training ")
            + f"{method_dir}, cache {cache}, downscale {downscale}"
            + (f", depth weight {lidar_depth} ({('SfM mesh, dense' if depth_dense else 'SfM mesh vertices') if depth_cloud else depth_source(project)})"
               if lidar_depth is not None else "")
            + (f", {iterations or DEFAULT_ITERATIONS:,} steps" if schedule else "")
            + (f", {schedule['strategy']} densification" if schedule and schedule.get("strategy") else "")
            + (f" (≤{schedule['max_gaussians']:,} gaussians)" if schedule and schedule.get("max_gaussians") else "")
            + (f", camera optimizer {schedule['camera_opt']}" if schedule and schedule.get("camera_opt") else "")
            + " "
            f"(viewer on http://localhost:{VIEWER_PORT})")
    if regularizing(regularize):
        reg = regularize
        on_line(f"splat: flattening weight {float(reg.get('flatten') or 0):g}"
                + (", exposure from the metadata" if reg.get("exposure") else ""))
    stop = threading.Event()
    if on_snapshot:
        threading.Thread(target=_watch_checkpoints, args=(project, method_dir, t0, on_snapshot, stop, on_line),
                         daemon=True).start()
    control = os.path.join(project, CONTROL)
    os.makedirs(os.path.dirname(control), exist_ok=True)  # a fresh project has no splat/ yet
    for f in (control, control + ".done"):
        _remove(f)
    maps = None
    if lidar_depth is not None and depth_cloud and depth_dense:
        maps = mesh_depth_maps(project, depth_cloud, downscale, on_line, job=job, on_count=on_prepare)
    if callable(loss_masks):
        loss_masks = loss_masks(maps) if maps else None
    settings["loss_masks"] = loss_masks
    seeds = None
    if seed_mesh:
        from modelforge import sfm_mesh
        t = json.load(open(os.path.join(project, "transforms.json")))
        seeds = os.path.join(project, "splat", "mesh_seeds.ply")
        sfm_mesh.seed_points(seed_mesh, os.path.join(project, t.get("ply_file_path", "sparse_pc.ply")), seeds,
                             on_line=on_line)
        seeds = os.path.abspath(seeds)
    cmd = train_command(project, method, iterations, extra, cache, downscale, preview_every if on_snapshot else None,
                        lidar_depth, schedule, load_dir, depth_cloud, maps, regularize, seeds, loss_masks)
    cmd = [os.path.join(NS_BIN, "python"), "-m", "mf_train"] + cmd[1:]  # ns-train + checkpoint on request
    stopped_at, reached = None, False

    def checkpointed():  # the trainer's checkpoint on request -> resume.json; its step (None: none saved)
        try:
            done = json.load(open(control + ".done"))
        except (OSError, ValueError):
            return None
        json.dump(dict(settings, load_dir=done["dir"], step=int(done["step"]),
                       transforms_mtime=os.path.getmtime(os.path.join(project, "transforms.json"))),
                  open(os.path.join(project, RESUME), "w"), indent=1)
        return int(done["step"])
    try:
        _stream(cmd, line, job, env={"MF_SPLAT_CONTROL": control},
                soft_stop=lambda: open(control, "w").write("checkpoint"))
        if stop_at:
            stopped_at = checkpointed()
            reached = stopped_at is not None
            if reached:
                on_line(f"splat: reached step {stopped_at:,}; exporting the splat so far")
    except jobs.Cancelled:
        stopped_at = checkpointed()
        if stopped_at is None:
            raise jobs.Cancelled("stopped before a checkpoint was saved") from None
        if pause is not None and pause.is_set():
            on_line(f"splat: paused at step {stopped_at:,}; Resume continues from there")
            raise Paused(f"paused at step {stopped_at:,} of {iterations or DEFAULT_ITERATIONS:,}") from None
        on_line(f"splat: stopped at step {stopped_at:,}; exporting the partial splat")
        job = Job()  # the job is cancelled; the export of the partial result still has to run
    finally:
        stop.set()
        for f in (control, control + ".done"):
            _remove(f)
    if stopped_at is None:
        _remove(os.path.join(project, RESUME))  # trained to the end
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
    if stopped_at is not None:
        stats["stopped_at_step"] = stopped_at  # a partial result; splat/resume.json continues it
    if load_dir:
        stats["resumed_from"] = load_dir
    if regularizing(regularize):
        stats["regularize"] = regularize
    if seed_mesh:
        stats["seed_mesh"] = seed_mesh
    if schedule:
        stats["schedule"] = {k: schedule.get(k) for k in ("strategy", "max_gaussians", "stop_split_at",
                                                          "reset_alpha_every", "camera_opt", "scale_factor")
                             if schedule.get(k)}
    if lidar_depth is not None:
        stats["lidar_depth"] = {"weight": float(lidar_depth), "error": lidar_log[-1] if lidar_log else None,
                                "history": lidar_log}
    Report(project).set("splat", stats)
    on_line(f"splat: {dict(stats, lidar_depth={k: v for k, v in stats['lidar_depth'].items() if k != 'history'})}"
            if "lidar_depth" in stats else f"splat: {stats}")
    if stopped_at is not None and not reached:
        raise StoppedPartial(f"stopped at step {stopped_at:,} of {iterations or DEFAULT_ITERATIONS:,}; the partial "
                             "splat is exported (splat/splat_enu.ply) and Resume continues the training")
    return stats


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def resume_info(project):
    """The paused / stopped run's settings (splat/resume.json) if it can continue, else None: its
    checkpoint must still exist and transforms.json must be unchanged (same cameras and scene frame)."""
    try:
        info = json.load(open(os.path.join(project, RESUME)))
    except (OSError, ValueError):
        return None
    if info.get("tiles"):  # a tiled run (modelforge/splat_tiles.py)
        from modelforge import splat_tiles
        return splat_tiles.resume_info(project, info)
    if not glob.glob(os.path.join(info.get("load_dir", ""), "step-*.ckpt")):
        return None
    t = os.path.join(project, "transforms.json")
    if not os.path.exists(t) or abs(os.path.getmtime(t) - info.get("transforms_mtime", 0)) > 1e-3:
        return None
    return info


def resume(project, **kwargs):
    """Continue the paused / stopped run with its own settings (method, steps, schedule, ...) from its
    last checkpoint: optimizer state, learning-rate schedules and the step count carry on. kwargs as
    for `run` (callbacks, job, pause, preview_every)."""
    info = resume_info(project)
    if info is None:
        raise RuntimeError("nothing to resume: no paused or stopped splat run with a checkpoint "
                           "(or transforms.json changed since)")
    if info.get("tiles"):
        from modelforge import splat_tiles
        return splat_tiles.run(project, resume=True, **kwargs)
    kwargs.pop("on_tile", None)  # GUI callers pass it for tiled runs
    return run(project, info["method"], info["iterations"], tuple(info["extra"]), cache=info["cache"],
               downscale=info["downscale"], lidar_depth=info["lidar_depth"], schedule=info["schedule"],
               load_dir=info["load_dir"], depth_cloud=info.get("depth_cloud"),
               depth_dense=info.get("depth_dense", False), regularize=info.get("regularize"),
               seed_mesh=info.get("seed_mesh"), loss_masks=info.get("loss_masks"), **kwargs)


def _num_vertices(ply):
    import plyfile
    return len(plyfile.PlyData.read(ply)["vertex"].data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project")
    ap.add_argument("--method", choices=METHODS, default="splatfacto")
    ap.add_argument("--iterations", type=int, help="training steps, per tile in a tiled run (default 50000, "
                    "splat_plan.DEFAULT_ITERATIONS; 0 = from the project's size; --no-plan: splatfacto's 30000)")
    ap.add_argument("--cache", choices=("auto",) + CACHE_MODES, default="auto",
                    help="where training images are kept (default auto: gpu if small, else cpu RAM, else disk)")
    ap.add_argument("--downscale", type=int, choices=DOWNSCALES,
                    help="train on images downscaled by this factor (default: to ~2K, i.e. 2 for 4K photos)")
    ap.add_argument("--strategy", choices=STRATEGIES, help="densification (default: default, with a reset period "
                    "scaled to the image count; mcmc built up near-camera fog on large scenes)")
    ap.add_argument("--max-gaussians", type=int, help="MCMC budget (default: from the scene and GPU memory)")
    ap.add_argument("--camera-opt", choices=CAMERA_OPT, help="pose refinement while training (default SO3xR3)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the paused / stopped run (splat/resume.json) with its own settings")
    ap.add_argument("--no-plan", action="store_true",
                    help="splatfacto's own defaults (30000 steps, no schedule) instead of the flight-based plan")
    ap.add_argument("--tiles", choices=("auto", "on", "off"), default="auto",
                    help="train in tiles (modelforge/splat_tiles.py); auto: above 800 training images")
    ap.add_argument("--tile-photos", type=int, default=None, help="about this many photos look into each tile (default 500)")
    ap.add_argument("--tile-layout", choices=("rectangles", "covisibility"), default="covisibility",  # DEFAULT_LAYOUT
                    help="tiles cut by straight lines, or regions from what the photos see together (seams on flat "
                         "ground; modelforge/tile_regions.py)")
    ap.add_argument("--skip-tiles", metavar="K,K", default=None,
                    help="tiles (1-based, as in the GUI) to leave out of training and the merge, e.g. 1,4")
    ap.add_argument("--round-steps", type=int, default=None,
                    help="tiled runs: every tile advances this many steps per round, merged after each "
                         "(default 10000; 0 = one tile after the other)")
    ap.add_argument("--tile", type=int, metavar="K", default=None,
                    help="train only tile K (1-based) of the set-up tiles from scratch; the others keep their splats "
                         "and all are merged again")
    ap.add_argument("--setup-tiles", action="store_true",
                    help="only divide the scene into tiles (tiles/tiles.json + datasets), without training")
    ap.add_argument("--lidar-depth", type=float, metavar="W",
                    help=f"depth loss weight (e.g. {DEFAULT_LIDAR_DEPTH}; 0 = report only): against the LiDAR points "
                         "(ROS bags), else the SfM points (sparse_pc.ply)")
    ap.add_argument("--flatten", type=float, default=FLATTEN, metavar="W",
                    help="weight of the flattening loss (smallest / middle scale: flat Gaussians, no needles; "
                         f"default {FLATTEN}; 0 = off)")
    ap.add_argument("--seed-mesh", action="store_true",
                    help="extra starting Gaussians on the SfM mesh (sfmmesh/mesh.ply) where the SfM points are thin, "
                         "flat in its surface")
    ap.add_argument("--mesh-prior", choices=("off", "vertices", "dense"), default="off",
                    help="depth targets from the SfM mesh (sfmmesh/mesh.ply): its vertices, or rendered into every "
                         "training camera (dense); needs --lidar-depth W")
    ap.add_argument("--no-tile-mask", dest="tile_mask", action="store_false", default=True,
                    help="tiled runs: train every photo's whole image (default: its pixels showing other tiles are left "
                         "out of the losses; needs --mesh-prior dense)")
    ap.add_argument("--no-exposure", dest="exposure", action="store_false", default=EXPOSURE,
                    help="do not scale each training render by its photo's relative exposure from the DJI metadata "
                         "(transforms.json mf_exposure, written by the export; on by default when present)")
    ap.add_argument("--bilateral-grid", action="store_true", help="splatfacto's bilateral grid per photo (colour / tone)")
    ap.add_argument("--antialiased", action="store_true", help="anti-aliased rasterization (Mip-Splatting's 2D filter)")
    args, extra = ap.parse_known_args()
    project = os.path.abspath(args.project)
    reg = {"flatten": args.flatten, "exposure": bool(args.exposure and exposure_known(project)),
           "bilateral": args.bilateral_grid, "antialiased": args.antialiased}
    reg = reg if regularizing(reg) or args.bilateral_grid or args.antialiased else None
    from modelforge import sfm_mesh as _sm
    seed = _sm.depth_cloud(project) if args.seed_mesh else None
    if args.seed_mesh and seed is None:
        sys.exit("error: --seed-mesh needs the SfM mesh (python -m modelforge.sfm_mesh PROJECT)")
    prior = _sm.depth_cloud(project) if args.mesh_prior != "off" else None
    if args.mesh_prior != "off" and (prior is None or args.lidar_depth is None):
        sys.exit("error: --mesh-prior needs the SfM mesh and --lidar-depth W")
    try:
        if args.resume:
            resume(project, on_line=lambda m: print(m, flush=True))
            return
        from modelforge import splat_plan, splat_tiles
        if args.iterations is None and not args.no_plan:
            args.iterations = splat_plan.DEFAULT_ITERATIONS
        if args.iterations == 0:  # auto: from the project's size
            args.iterations = None
        if args.setup_tiles:
            splat_tiles.setup(project, args.tile_photos or splat_tiles.PHOTOS_PER_TILE,
                              on_line=lambda m: print(m, flush=True), layout=args.tile_layout)
            return
        if args.tile is not None or (not args.no_plan and splat_tiles.wanted(splat_plan.survey(project)["train_images"],
                                                                             args.tiles)):
            splat_tiles.run(project, args.method, dict(iterations=args.iterations, downscale=args.downscale,
                                                       strategy=args.strategy, max_gaussians=args.max_gaussians,
                                                       camera_opt=args.camera_opt), extra,
                            on_line=lambda m: print(m, flush=True), cache=args.cache, lidar_depth=args.lidar_depth,
                            photos=args.tile_photos or splat_tiles.PHOTOS_PER_TILE, layout=args.tile_layout,
                            skip=None if args.skip_tiles is None else
                            [int(k) - 1 for k in args.skip_tiles.split(",") if k.strip()], regularize=reg,
                            seed_mesh=seed, depth_cloud=prior, depth_dense=args.mesh_prior == "dense",
                            tile_mask=args.tile_mask,
                            round_steps=splat_tiles.ROUND_STEPS if args.round_steps is None else args.round_steps,
                            only=None if args.tile is None else args.tile - 1)
            return
        if args.no_plan:
            iterations, downscale, schedule = args.iterations, args.downscale or 1, None
        else:
            schedule = splat_plan.suggest(splat_plan.survey(project), iterations=args.iterations,
                                          downscale=args.downscale, strategy=args.strategy,
                                          max_gaussians=args.max_gaussians, camera_opt=args.camera_opt,
                                          method=args.method)
            print("splat plan: " + "; ".join(schedule["notes"]), flush=True)
            iterations, downscale = schedule["iterations"], schedule["downscale"]
        cache = args.cache if args.cache != "auto" else auto_cache(project, downscale)[0]
        run(project, args.method, iterations, extra, on_line=lambda m: print(m, flush=True), cache=cache,
            downscale=downscale, lidar_depth=args.lidar_depth, schedule=schedule, regularize=reg, seed_mesh=seed,
            depth_cloud=prior, depth_dense=args.mesh_prior == "dense")
    except (RuntimeError, ValueError, Stopped) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
