"""Splat training settings from the flight: image count, ground area, overlap and ground sampling distance.

splatfacto's defaults are tuned for a few hundred images of an object-sized scene. For a drone survey:
  - densification: splatfacto pauses refinement for (training images + 100) steps after every opacity
    reset, and resets every 30 x 100 = 3000 steps, so with ~2900+ training images refinement never runs
    (gsplat DefaultStrategy: refine only while step % reset_every >= pause). The reset period is scaled
    to 2 x the pause here; or MCMC is used, which has no reset pause and keeps a hard Gaussian cap.
  - schedule: iterations grow with the image count (each image seen ~EPOCHS times), splitting stops
    halfway (MCMC: at 5/6, nerfstudio's ratio), and the position / camera learning-rate decay is
    stretched to the run (it is fixed at 30000 steps in nerfstudio's method config).
  - budget: the cameras' ground footprints give the area and the ground sampling distance (GSD), so the
    Gaussians needed at each downsample factor (one per PX_PER_GAUSSIAN^2 ground pixels) can be compared
    with what fits in GPU memory; the smallest factor that fits is suggested, and MCMC with a cap when
    the full resolution would not fit.
  - camera optimizer: SO3xR3 lets training correct small pose errors.

The cameras come from transforms.json (refined) or, before SfM, from poses.json (the flight priors, focal
from camera.json or an assumed field of view). The ground is poses.json's ground_z (DJI / ROS 2 bag),
else a low percentile of the sparse points. All heuristics; the Splat tab shows the numbers and every
value can be overridden.
"""

import json
import math
import os
import subprocess

import numpy as np

EPOCHS = 20  # each training image seen ~this many times
MIN_ITERATIONS, MAX_ITERATIONS = 30_000, 300_000
PX_PER_GAUSSIAN = 2.0  # one Gaussian per this many ground pixels (per side) at the training resolution
GAUSSIANS_PER_GB = 350_000  # training memory incl. optimizer state and rendering, rough
VRAM_RESERVE_GB = 4.0  # images, rasterization buffers, the CUDA context
DEFAULT_VRAM_GB = 12.0  # when nvidia-smi is not available
ASSUMED_HFOV_DEG = 80.0  # priors without a calibrated camera (DJI wide cameras: ~80-84 deg)
MAX_RANGE_X_HEIGHT = 5.0  # footprint rays end at this many camera heights (oblique views, horizon)
TRAIN_FRACTION = 0.9  # nerfstudio-data's train_split_fraction
DOWNSCALES = (1, 2, 4, 8)


def vram_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cameras(project):
    """(centres n x 3, R_wc n x 3 x 3 OpenCV, fx, w, h, source)."""
    path = os.path.join(project, "transforms.json")
    if os.path.exists(path):
        t = json.load(open(path))
        c2w = np.array([f["transform_matrix"] for f in t["frames"]], float)
        R = c2w[:, :3, :3] * np.array([1.0, -1.0, -1.0])  # OpenGL -> OpenCV camera axes
        return c2w[:, :3, 3], R, float(t["fl_x"]), int(t["w"]), int(t["h"]), "refined cameras"
    poses = json.load(open(os.path.join(project, "poses.json")))
    imgs = [i for i in poses["images"] if "q_world_from_cam" in i]
    if not imgs:
        raise ValueError("no camera poses (poses.json has no priors; run SfM first)")
    qw, qx, qy, qz = np.array([i["q_world_from_cam"] for i in imgs], float).T  # wxyz
    R = np.stack([np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
                  np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
                  np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1)], 1)
    C = np.array([[i["enu_x"], i["enu_y"], i["enu_z"]] for i in imgs], float)
    from PIL import Image
    with Image.open(os.path.join(project, "images", imgs[0]["image"])) as im:
        w, h = im.size
    cam = os.path.join(project, "camera.json")
    if os.path.exists(cam):
        fx = float(json.load(open(cam))["params"][0])
    else:
        fx = w / 2 / math.tan(math.radians(ASSUMED_HFOV_DEG / 2))
    return C, R, fx, w, h, "flight priors"


def _ground_z(project, C):
    path = os.path.join(project, "poses.json")
    if os.path.exists(path):
        g = json.load(open(path)).get("ground_z")
        if g is not None:
            return float(g)
    ply = os.path.join(project, "sparse_pc.ply")
    if os.path.exists(ply):
        import plyfile
        v = plyfile.PlyData.read(ply)["vertex"]
        return float(np.percentile(np.asarray(v["z"]), 5))
    return float(C[:, 2].min()) - 10.0


def ply_count(path):
    """Vertex count from a PLY header (0 if missing)."""
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        for _ in range(50):
            line = f.readline()
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    return 0


def survey(project):
    """Image count, ground area covered (m²), overlap (views per ground point), GSD (m / px at full
    resolution, median over cameras), initial points and where the cameras came from."""
    import cv2
    C, R, fx, w, h, source = _cameras(project)
    gz = _ground_z(project, C)
    u, v = np.meshgrid(np.linspace(0, w, 9), np.linspace(0, h, 9))
    rays_c = np.stack([(u.ravel() - w / 2) / fx, (v.ravel() - h / 2) / fx, np.ones(u.size)], 1)
    hulls, gsd = [], []
    for c, r in zip(C, R):
        height = c[2] - gz
        if height <= 0:
            continue
        d = rays_c @ r.T  # world directions
        t = np.where(d[:, 2] < -1e-9, -height / np.minimum(d[:, 2], -1e-9), np.inf)
        dist = t * np.linalg.norm(d, axis=1)
        keep = dist <= MAX_RANGE_X_HEIGHT * height
        if keep.sum() < 3:
            continue
        pts = c[:2] + d[keep, :2] * t[keep, None]
        hulls.append(cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2))
        gsd.append(np.median(dist[keep]) / fx)
    out = {"images": len(C), "train_images": math.ceil(TRAIN_FRACTION * len(C)), "source": source,
           "width": w, "height": h, "initial_points": ply_count(os.path.join(project, "sparse_pc.ply")),
           "area_m2": 0.0, "overlap": 0.0, "gsd_m": None}
    if not hulls:
        return out
    allp = np.concatenate(hulls)
    lo, extent = allp.min(0), float((allp.max(0) - allp.min(0)).max()) or 1.0
    cell = extent / 1024
    grid = np.zeros((1026, 1026), np.uint8)
    total = 0.0
    for hl in hulls:
        total += float(cv2.contourArea(hl))
        cv2.fillConvexPoly(grid, np.round((hl - lo) / cell).astype(np.int32), 1)
    area = float(grid.sum()) * cell * cell
    out.update(area_m2=area, overlap=total / area if area else 0.0, gsd_m=float(np.median(gsd)))
    return out


def needed_gaussians(s, downscale):
    if not s.get("gsd_m"):
        return None
    return int(s["area_m2"] / (PX_PER_GAUSSIAN * s["gsd_m"] * downscale) ** 2)


def suggest(s, vram=None, iterations=None, downscale=None, strategy=None, max_gaussians=None, camera_opt=None):
    """Settings for splat.train_command from a survey: iterations, downscale, strategy, max_gaussians,
    stop_split_at, reset_alpha_every, camera_opt, plus `notes` (why). Explicit arguments win over the
    suggestion (None, 0 or "auto" = suggest); the dependent values follow them."""
    auto = lambda v: v in (None, 0, "", "auto")  # noqa: E731
    vram = vram or vram_gb() or DEFAULT_VRAM_GB
    cap = max(1_000_000, int((vram - VRAM_RESERVE_GB) * GAUSSIANS_PER_GB))
    n = max(1, s["train_images"])
    iters = (int(min(MAX_ITERATIONS, max(MIN_ITERATIONS, math.ceil(EPOCHS * n / 1000) * 1000)))
             if auto(iterations) else int(iterations))
    need = {f: needed_gaussians(s, f) for f in DOWNSCALES}
    if not auto(downscale):
        down = int(downscale)
    elif need[1] is None:
        down = 1
    else:
        down = next((f for f in DOWNSCALES if need[f] <= cap), DOWNSCALES[-1])
        while down > 1 and s["width"] / down < 800:  # keep enough pixels to match against
            down //= 2
    if auto(strategy):
        strategy = "mcmc" if need[1] is not None and need[1] > cap else "default"
    auto_budget = auto(max_gaussians)
    if strategy == "mcmc" and auto_budget:
        # MCMC grows to its cap: what the area needs at that factor, within the GPU estimate
        max_gaussians = max(min(cap, need[down] or cap), s.get("initial_points", 0), 500_000)
    pause = n + 100  # splatfacto's pause after an opacity reset (steps)
    reset_steps = max(3000, 2 * pause)
    plan = {"iterations": iters, "downscale": down, "strategy": strategy,
            "max_gaussians": int(max_gaussians) if strategy == "mcmc" else None,
            "stop_split_at": iters // 2 if strategy == "default" else iters * 5 // 6,
            "reset_alpha_every": math.ceil(reset_steps / 100),
            "camera_opt": "SO3xR3" if auto(camera_opt) else camera_opt,
            "vram_gb": round(vram, 1), "cap": cap, "needed": need}
    notes = [f"{s['images']:,} images ({n:,} for training) from the {s['source']}"]
    if s.get("gsd_m"):
        notes.append(f"ground {s['area_m2'] / 1e4:,.1f} ha, overlap {s['overlap']:.0f}×, "
                     f"GSD {s['gsd_m'] * 100:.1f} cm/px at full resolution")
        notes.append(f"~{need[down] / 1e6:.1f}M Gaussians at ×{down} (full resolution ~{need[1] / 1e6:.1f}M); "
                     f"{vram:.0f} GB GPU fits ~{cap / 1e6:.1f}M")
    notes.append(f"{iters:,} steps (~{iters / n:.0f} views per image), splitting until {plan['stop_split_at']:,}")
    if strategy == "mcmc":
        notes.append(f"MCMC: a fixed budget of {plan['max_gaussians'] / 1e6:.1f}M Gaussians")
        if s.get("initial_points", 0) > min(cap, need[down] or cap) and auto_budget:
            notes.append(f"the sparse cloud ({s['initial_points'] / 1e6:.1f}M points) exceeds the estimate; "
                         "the budget is raised to it")
    elif pause > 3000 - 100:
        notes.append(f"opacity reset every {reset_steps:,} steps (splatfacto's 3,000 would block densification "
                     f"with {n:,} training images)")
    plan["notes"] = notes
    return plan
