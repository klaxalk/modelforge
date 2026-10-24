"""Splat training settings from the cameras and the sparse points: image count, overlap, scene surface
and ground sampling distance (GSD).

splatfacto's defaults are tuned for a few hundred images of an object-sized scene. For a drone survey:
  - densification: splatfacto pauses refinement for (training images + 100) steps after every opacity
    reset, and resets every 30 x 100 = 3000 steps, so with ~2900+ training images refinement never runs
    (gsplat DefaultStrategy: refine only while step % reset_every >= pause). The reset period is scaled
    to 2 x the pause here. MCMC (no reset pause, a hard Gaussian cap) only when asked: on karlak it built up
    semi-transparent "fog" near the camera path (a held-out view fell to 9.5 dB PSNR) and smudged detail,
    while default densification did not, with half the Gaussians in 1/3 to 1/7 of the time.
  - schedule: iterations grow with the image count (each image seen ~EPOCHS times), splitting stops
    halfway (MCMC: at 5/6, nerfstudio's ratio), and the position / camera learning-rate decay is
    stretched to the run (it is fixed at 30000 steps in nerfstudio's method config).
  - resolution: photos are trained at ~2K (MAX_TRAIN_WIDTH; 4K -> x2). On karlak (471 4K photos) full 4K
    with default densification grew no more Gaussians than x2 (2.0M vs 1.9M: its split threshold is
    resolution-independent) and was no better even scored at 4K, in 2.4x the time.
  - budget: the scene needs about (all image pixels) / (overlap x PX_PER_GAUSSIAN^2) Gaussians at the
    training resolution, a rough figure (karlak x1: 14M estimated, 2.0M grown) that sets MCMC's budget.
  - camera optimizer: SO3xR3 lets training correct small pose errors.
  - scale: one nerfstudio unit = WORLD_UNIT_M. nerfstudio scales the poses by 1 / max |camera position|,
    and splatfacto's position learning rate and its size thresholds (clone vs split below 0.01 units,
    cull above 0.5) are in those units, so a 380 m flight (unit ~200 m) trained very differently from a
    50 m one (unit ~28 m): held-out quality fell with more steps and the Gaussian count shrank at every
    opacity reset. 10 m: dira2k (2000 photos) 12000 steps 14.9 -> 17.2 dB (LPIPS 0.94 -> 0.74; 4 m and
    28 m worse), karlak 22.99 -> 23.24 dB (LPIPS 0.333 -> 0.301). Projects without metric scale (videos)
    have their median scene depth at 10 units.

No ground plane is assumed (the flight may start in the air, the terrain may drop anywhere): the scene is
the sparse cloud (sparse_pc.ply, a random subset). Each point counts the cameras it projects into
(in front, inside the frame, nearer than 4x that camera's median depth: a rough stand-in for occlusion),
which gives the overlap; each camera's median point depth / focal is its GSD. The surface seen is
sum(footprints) / overlap. Without sparse points (before SfM) only the image count is known.
All heuristics; the Splat tab shows the numbers and every value can be overridden.
"""

import json
import math
import os
import subprocess

import numpy as np

EPOCHS = 20  # each training image seen ~this many times
MIN_ITERATIONS, MAX_ITERATIONS = 30_000, 300_000
PX_PER_GAUSSIAN = 2.0  # one Gaussian per this many surface pixels (per side) at the training resolution
GAUSSIANS_PER_GB = 350_000  # training memory incl. optimizer state and rendering, rough
VRAM_RESERVE_GB = 4.0  # images, rasterization buffers, the CUDA context
DEFAULT_VRAM_GB = 12.0  # when nvidia-smi is not available
ASSUMED_HFOV_DEG = 80.0  # priors without a calibrated camera (DJI wide cameras: ~80-84 deg)
MAX_POINTS = 10_000  # sparse points sampled for the survey
FAR_X_MEDIAN = 4.0  # points beyond this many times a camera's median depth don't count as seen
TRAIN_FRACTION = 0.9  # nerfstudio-data's train_split_fraction
MAX_TRAIN_WIDTH = 2700  # auto downsample: the smallest factor that brings the photos to this width (4K -> x2)
DOWNSCALES = (1, 2, 4, 8)
WORLD_UNIT_M = 10.0  # one nerfstudio unit (fixed, instead of 1 / max |camera position|)


def vram_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cameras(project):
    """(centres n x 3, R_wc n x 3 x 3 OpenCV, fx, w, h, source); fx, w, h are numbers, or arrays (n,) when the frames
    have cameras of their own (multi-camera projects)."""
    path = os.path.join(project, "transforms.json")
    if os.path.exists(path):
        from modelforge.export import frame_camera
        t = json.load(open(path))
        c2w = np.array([f["transform_matrix"] for f in t["frames"]], float)
        R = c2w[:, :3, :3] * np.array([1.0, -1.0, -1.0])  # OpenGL -> OpenCV camera axes
        cams, index = frame_camera.cameras(t)
        if len(cams) > 1:
            fx, w, h = (np.array([float(cams[i][k]) for i in index]) for k in ("fl_x", "w", "h"))
            return c2w[:, :3, 3], R, fx, w.astype(int), h.astype(int), "refined cameras"
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


def _points(project, max_points, seed=0):
    """A random subset of sparse_pc.ply (n x 3), or None."""
    path = os.path.join(project, "sparse_pc.ply")
    if not os.path.exists(path):
        return None
    import plyfile
    v = plyfile.PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(v[k], float) for k in ("x", "y", "z")], 1)
    xyz = xyz[np.isfinite(xyz).all(1)]
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)]
    return xyz


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


def survey(project, max_points=MAX_POINTS):
    """Image count, overlap (cameras per sparse point), surface seen (m², sum of footprints / overlap),
    GSD (m / px at full resolution, median over cameras), initial points and the camera source."""
    C, R, fx, w, h, source = _cameras(project)
    fxs, ws, hs = (np.broadcast_to(np.asarray(x), (len(C),)) for x in (fx, w, h))  # per camera
    if np.ndim(w):  # several cameras: the widest decides the resolution rules
        k = int(np.argmax(ws))
        w, h = int(ws[k]), int(hs[k])
    out = {"images": len(C), "train_images": math.ceil(TRAIN_FRACTION * len(C)), "source": source,
           "width": w, "height": h, "initial_points": ply_count(os.path.join(project, "sparse_pc.ply")),
           "area_m2": 0.0, "overlap": 0.0, "gsd_m": None}
    xyz = _points(project, max_points)
    if xyz is None or len(xyz) < 10:
        return out
    seen = np.zeros(len(xyz), np.int64)
    depths, used = [], []
    for i, (c, r) in enumerate(zip(C, R)):
        p = (xyz - c) @ r  # camera coordinates (OpenCV)
        z = p[:, 2]
        front = np.flatnonzero(z > 1e-9)
        u = fxs[i] * p[front, 0] / z[front] + ws[i] / 2
        v = fxs[i] * p[front, 1] / z[front] + hs[i] / 2
        idx = front[(u >= 0) & (u < ws[i]) & (v >= 0) & (v < hs[i])]
        if len(idx) < 5:
            continue
        d = float(np.median(z[idx]))
        depths.append(d)
        used.append(i)
        seen[idx[z[idx] < FAR_X_MEDIAN * d]] += 1
    if not depths or not seen.any():
        return out
    overlap = float(seen[seen > 0].mean())
    depths, used = np.asarray(depths), np.asarray(used)
    footprints = float(np.sum(ws[used] * hs[used] * (depths / fxs[used]) ** 2))
    out.update(area_m2=footprints / overlap, overlap=overlap, gsd_m=float(np.median(depths / fxs[used])),
               cameras_seeing=len(depths))
    return out


def needed_gaussians(s, downscale=1):
    """Gaussians the scene needs at a downsample factor: unique surface pixels / PX_PER_GAUSSIAN²."""
    if not s.get("overlap"):
        return None
    return int(s["images"] * s["width"] * s["height"] / (s["overlap"] * (PX_PER_GAUSSIAN * downscale) ** 2))


DEFAULT_ITERATIONS = 50000  # the Splat tab's and the CLI's Iterations (the user's choice, 2026-10-23; 0 = auto below)


def auto_downscale(width):
    """The smallest downsample factor that trains the photos at <= MAX_TRAIN_WIDTH."""
    return next((f for f in DOWNSCALES if width / f <= MAX_TRAIN_WIDTH), DOWNSCALES[-1])


def suggest(s, vram=None, iterations=None, downscale=None, strategy=None, max_gaussians=None, camera_opt=None,
            method="splatfacto", tiles=None):
    """Settings for splat.train_command from a survey: iterations, downscale, strategy, max_gaussians,
    stop_split_at, reset_alpha_every, camera_opt, scale_factor, tiles (train in tiles: modelforge/splat_tiles.py;
    tiles = auto / on / off), plus `notes` (why). Explicit arguments win over the
    suggestion (None, 0 or "auto" = suggest); the dependent values follow them. method: unused (kept for
    callers; all splatfacto variants take the same flags)."""
    auto = lambda v: v in (None, 0, "", "auto")  # noqa: E731
    vram = vram or vram_gb() or DEFAULT_VRAM_GB
    cap = max(1_000_000, int((vram - VRAM_RESERVE_GB) * GAUSSIANS_PER_GB))
    n = max(1, s["train_images"])
    iters = (int(min(MAX_ITERATIONS, max(MIN_ITERATIONS, math.ceil(EPOCHS * n / 1000) * 1000)))
             if auto(iterations) else int(iterations))
    down = auto_downscale(s["width"]) if auto(downscale) else int(downscale)
    need = needed_gaussians(s, down)
    if auto(strategy):
        strategy = "default"
    auto_budget = auto(max_gaussians)
    if strategy == "mcmc" and auto_budget:
        # MCMC grows to its cap: what the scene needs at that factor, within the GPU estimate
        max_gaussians = max(min(cap, need or cap), s.get("initial_points", 0), 500_000)
    pause = n + 100  # splatfacto's pause after an opacity reset (steps)
    reset_steps = max(3000, 2 * pause)
    plan = {"iterations": iters, "downscale": down, "strategy": strategy,
            "max_gaussians": int(max_gaussians) if strategy == "mcmc" else None,
            "stop_split_at": iters // 2 if strategy == "default" else iters * 5 // 6,
            "reset_alpha_every": math.ceil(reset_steps / 100),
            "camera_opt": "SO3xR3" if auto(camera_opt) else camera_opt,
            "scale_factor": 1.0 / WORLD_UNIT_M,
            "vram_gb": round(vram, 1), "cap": cap, "needed": need}
    notes = [f"{s['images']:,} images ({n:,} for training) from the {s['source']}"]
    if s.get("gsd_m"):
        notes.append(f"surface ~{s['area_m2'] / 1e4:,.1f} ha seen {s['overlap']:.0f}× on average (sparse points), "
                     f"GSD {s['gsd_m'] * 100:.1f} cm/px at full resolution (median)")
        notes.append(f"~{need / 1e6:.1f}M Gaussians at ×{down} (rough); {vram:.0f} GB GPU fits ~{cap / 1e6:.1f}M")
    else:
        notes.append("no sparse points yet: the scene size is unknown (run SfM + export for the estimate)")
    if auto(downscale) and down > 1:
        notes.append(f"×{down}: {s['width']} px photos trained at {s['width'] // down} px (pick 1 for full "
                     "resolution: on 4K photos no better, 2.4× slower)")
    notes.append(f"{iters:,} steps (~{iters / n:.0f} views per image), splitting until {plan['stop_split_at']:,}")
    notes.append(f"scene unit {WORLD_UNIT_M:g} m (learning rates and size thresholds in metres, not in "
                 "fractions of the flight)")
    if strategy == "mcmc":
        notes.append(f"MCMC: a fixed budget of {plan['max_gaussians'] / 1e6:.1f}M Gaussians")
        if s.get("initial_points", 0) > min(cap, need or cap) and auto_budget:
            notes.append(f"the sparse cloud ({s['initial_points'] / 1e6:.1f}M points) exceeds the estimate; "
                         "the budget is raised to it")
    elif pause > 3000 - 100:
        notes.append(f"opacity reset every {reset_steps:,} steps (splatfacto's 3,000 would block densification "
                     f"with {n:,} training images)")
    from modelforge import splat_tiles
    plan["tiles"] = splat_tiles.wanted(n, tiles)
    if plan["tiles"]:
        notes.append(f"**tiles** of ≤ ~{splat_tiles.PHOTOS_PER_TILE} photos (+ oblique views from the neighbours), "
                     f"~{max(1, math.ceil(s['images'] / splat_tiles.PHOTOS_PER_TILE))} of them, each trained with its own "
                     "plan (steps from its photos), then cropped to its cell and merged"
                     + (f": one splat over {n:,} training images stays blurry" if n > splat_tiles.TILE_ABOVE else ""))
    plan["notes"] = notes
    return plan
