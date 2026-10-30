"""Mesh from the LiDAR scans of a ROS bag project, placed with the finished SfM poses (right panel → LiDAR Mesh).

The ingest put every scan into the world with the bag's own poses (TF / odometry: the robot's SLAM). After the
sparse reconstruction the photos have better poses. Per photo the correction C = T_world_cam(SfM) ·
T_world_cam(bag)⁻¹ says how far the SfM moved that camera; a scan at time t gets C interpolated between the photos
around t (translation linear, rotation slerp; before the first / after the last photo: the nearest), so
T_world_sensor = C(t) · T_world_sensor(bag, t). Poses "bag" skips the correction (the ingest's placement), for a
comparison.

Then the scans are fused: export/lidar_tsdf.py (nerfstudio's venv, Open3D on the GPU) turns each scan into six
90° depth images around the sensor (a cube map, z-depth per pixel, the nearest return) and integrates them into a
TSDF (Open3D VoxelBlockGrid, truncation 4 voxels), so the free space between the sensor and a surface carves
what moving people or noise left; marching cubes over voxels seen by at least `min_weight` scans, small pieces
dropped. Vertex colours are the LiDAR intensity (grey); "Colour from the photos" colours it like the SfM mesh.

PROJECT/lidarmesh/: scans.npz (the scans in their sensor frames + world poses; the worker's input), cloud.ply
(the corrected cloud, voxel-downsampled), mesh.ply, preview.ply (<= 300k faces), stats.json; report "lidar_mesh".

python -m modelforge.lidar_mesh PROJECT [--poses sfm|bag] [--voxel M] [--max-range M] [--every N]
"""
import argparse
import json
import os
import time

import numpy as np

from modelforge import jobs

OUT = "lidarmesh"
POSES = ("sfm", "bag")
MAX_SCANS = 2000  # every N-th scan auto: at most this many per cloud topic
FACE_RES = 256  # cube-map face size [px]: 90° / 256 = 0.35° per pixel
MIN_RANGE = 0.3  # returns nearer than this [m] (the robot itself) are dropped


def available(project):
    """None when the project can have a LiDAR mesh, else why not."""
    try:
        meta = json.load(open(os.path.join(project, "poses.json")))
    except (OSError, ValueError):
        return "no extracted images yet (poses.json)"
    if meta.get("kind") != "rosbag":
        return "only ROS bag projects carry LiDAR scans"
    ros = meta.get("ros") or {}
    if not ros.get("cloud_topics"):
        return "the project's Ingest settings pick no point cloud topic"
    if ros.get("pose_source", "tf") == "none":
        return "the bag gives no poses (pose source none): the scans cannot be placed"
    return None


def shown_cloud(project):
    """(path, what) of the LiDAR cloud to show: the aggregated one (lidarmesh/cloud.ply, with its poses) when built,
    else the ingest's (lidar/cloud.ply, the bag's poses, <= 400 scans per topic), else (None, None)."""
    p = os.path.join(project, OUT, "cloud.ply")
    if os.path.exists(p):
        try:
            info = json.load(open(os.path.join(project, OUT, "cloud.json")))
            what = f"aggregated, poses {info['poses']}, {info['scans']:,} scans, {info['voxel'] * 100:.1f} cm voxel"
        except (OSError, ValueError, KeyError):
            what = "aggregated"
        return p, what
    p = os.path.join(project, "lidar", "cloud.ply")
    return (p, "the ingest's (bag poses)") if os.path.exists(p) else (None, None)


def depth_cloud(project):
    """The LiDAR mesh (the splat's depth prior with Prior mesh lidar), or None."""
    p = os.path.join(project, OUT, "mesh.ply")
    return os.path.abspath(p) if os.path.exists(p) else None


PRIOR_MESHES = ("auto", "sfm", "lidar")


def prior_mesh(project, choice="auto"):
    """The splat's depth-prior mesh: auto = the LiDAR mesh when the project has one (measured), else the SfM mesh;
    sfm / lidar = that one (None when it does not exist)."""
    from modelforge import sfm_mesh
    if choice == "lidar" or (choice == "auto" and depth_cloud(project)):
        return depth_cloud(project)
    return sfm_mesh.depth_cloud(project)


def _model_dir(project):
    from modelforge import sfm_mesh
    return sfm_mesh.model_dir(project)


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def corrections(project, meta=None, info=None):
    """Per source index: (stamps (n,), C (n, 4, 4)) sorted by stamp: C = T_world_cam(SfM) · T_world_cam(bag)⁻¹ of
    every photo the SfM model holds, without the camera mount's rotation: the rotation the SfM gives every photo
    alike in the camera frame (kn_odin: 1.0°, 0.19° left per photo) is the camera's calibration against the robot in
    the bag's TF, not a pose error of the robot, so the scans do not turn with it. info (dict) gets "mount_deg"."""
    import pycolmap
    from modelforge.rosbag.tf import quat_to_rot
    meta = meta or json.load(open(os.path.join(project, "poses.json")))
    rec = pycolmap.Reconstruction(_model_dir(project))
    sfm = {}
    for im in rec.images.values():
        c = im.cam_from_world
        c = c() if callable(c) else c
        M = np.eye(4)
        M[:3] = c.matrix()
        sfm[im.name] = np.linalg.inv(M)
    pairs = []
    for r in meta["images"]:
        T = sfm.get(r["image"])
        if T is None or "enu_x" not in r or "stamp" not in r:
            continue
        P = np.eye(4)
        P[:3, :3] = quat_to_rot(np.roll(np.asarray(r["q_world_from_cam"], float), -1))  # (w,x,y,z) -> (x,y,z,w)
        P[:3, 3] = r["enu_x"], r["enu_y"], r["enu_z"]
        pairs.append((r["source"], r["stamp"], P, T))
    if not pairs:
        return {}
    U, _, Vt = np.linalg.svd(sum(P[:3, :3].T @ T[:3, :3] for _, _, P, T in pairs))  # the mean camera-frame rotation
    B = U @ Vt
    if np.linalg.det(B) < 0:
        B = U @ np.diag([1, 1, -1]) @ Vt
    if info is not None:
        info["mount_deg"] = _angle(B)
        info["mount_spread_deg"] = float(np.median([_angle(B.T @ P[:3, :3].T @ T[:3, :3]) for _, _, P, T in pairs]))
    out = {}
    for k, stamp, P, T in pairs:
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ B.T  # the camera as the robot carries it (the mount's rotation taken out)
        out.setdefault(k, []).append((stamp, T @ np.linalg.inv(P)))
    res = {}
    for k, v in out.items():
        v.sort(key=lambda x: x[0])
        res[k] = (np.array([s for s, _ in v]), np.stack([C for _, C in v]))
    return res


def interpolate(stamps, Cs, t):
    """C at time t: linear translation, slerped rotation between the photos around t; the nearest outside."""
    from modelforge.rosbag.tf import quat_to_rot, rot_to_quat, slerp
    if t <= stamps[0]:
        return Cs[0]
    if t >= stamps[-1]:
        return Cs[-1]
    i = int(np.searchsorted(stamps, t))
    a = (t - stamps[i - 1]) / max(stamps[i] - stamps[i - 1], 1e-9)
    q = slerp(rot_to_quat(Cs[i - 1][:3, :3]), rot_to_quat(Cs[i][:3, :3]), a)
    C = np.eye(4)
    C[:3, :3] = quat_to_rot(q)
    C[:3, 3] = (1 - a) * Cs[i - 1][:3, 3] + a * Cs[i][:3, 3]
    return C


def _correction_stats(corr):
    Cs = np.concatenate([c for _, c in corr.values()]) if corr else np.zeros((0, 4, 4))
    if not len(Cs):
        return {}
    ang = np.array([_angle(C[:3, :3]) for C in Cs])  # the robot's rotation by the SfM, per photo
    return {"photos": int(len(Cs)), "rotation_deg_median": float(np.median(ang)), "rotation_deg_max": float(ang.max())}


def collect(project, poses="sfm", every=0, max_range=None, on_line=print, token=None, meta=None):
    """The scans of every source's range: (points in the sensor frame (N, 3) float32, intensity (N,), offsets
    (scans + 1,), T_world_sensor (scans, 4, 4), stats)."""
    from modelforge.rosbag import bag
    from modelforge.rosbag.ingest import Poser
    token = token or jobs.Token()
    meta = meta or json.load(open(os.path.join(project, "poses.json")))
    ros = meta["ros"]
    max_range = float(max_range or ros.get("max_range", 60.0))
    info = {}
    corr = corrections(project, meta, info) if poses == "sfm" else {}
    if poses == "sfm" and not corr:
        raise RuntimeError("the SfM model holds none of the photos with a bag pose")
    pts, its, offs, Ts = [], [], [0], []
    stats = {"topics": {}}
    for k, src in enumerate(meta["sources"]):
        path = src["video"]
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(project)), "..", path)
        if not os.path.exists(path):
            raise RuntimeError(f"the bag {src['video']} is gone (Videos tab)")
        poser = Poser(path, ros)
        t_first = float(bag.image_stamps(path, ros["image_topic"])[0])
        t0 = t_first + float(src.get("start") or 0.0)
        t1 = t_first + float(src["end"]) if src.get("end") is not None else np.inf
        inv = bag.inventory(path)
        for topic in ros["cloud_topics"]:
            n_msgs = inv["topics"].get(topic, {}).get("count", 0)
            step = int(every) if every else max(1, int(np.ceil(n_msgs / MAX_SCANS)))
            used = skipped = 0
            on_line(f"lidar mesh: reading {topic} of {os.path.basename(path)} (every {step}. of {n_msgs} scans)")
            for t, frame, xyz, inten in bag.iter_clouds(path, topic, t0, t1, step):
                token.check()
                T = poser.world_from(frame, t)
                if T is None:
                    skipped += 1
                    continue
                if k in corr:
                    C = interpolate(*corr[k], t)
                    T = C @ T
                r = np.linalg.norm(xyz, axis=1)
                keep = np.isfinite(r) & (r >= MIN_RANGE) & (r <= max_range)
                pts.append(xyz[keep].astype(np.float32))
                its.append((inten[keep] if inten is not None else np.zeros(int(keep.sum()))).astype(np.float32))
                offs.append(offs[-1] + int(keep.sum()))
                Ts.append(T)
                used += 1
                if used % 50 == 0:
                    on_line(f"lidar mesh: scan {used}")
            stats["topics"][topic] = {"scans": used, "skipped_no_pose": skipped, "every": step}
            on_line(f"lidar mesh: {topic}: {used} scans" + (f", {skipped} without a pose" if skipped else ""))
    if not Ts:
        raise RuntimeError("no LiDAR scan with a pose in the sources' ranges")
    stats["correction"] = dict(_correction_stats(corr), **info)
    if info:
        on_line(f"lidar mesh: the camera's mount is {info['mount_deg']:.2f}° off the bag's TF (left out: it is the "
                f"camera's, not the robot's); the SfM turned the robot by {stats['correction']['rotation_deg_median']:.2f}° "
                "median")
    return np.concatenate(pts), np.concatenate(its), np.asarray(offs, np.int64), np.stack(Ts), stats


def shift_stats(project, meta=None):
    """How far the SfM moved the bag's camera poses: centre shift median / p90 / max [m]."""
    meta = meta or json.load(open(os.path.join(project, "poses.json")))
    corr = corrections(project, meta)
    d = []
    for r in meta["images"]:
        if r["source"] in corr and "enu_x" in r and "stamp" in r:
            C = interpolate(*corr[r["source"]], r["stamp"])
            p = np.array([r["enu_x"], r["enu_y"], r["enu_z"], 1.0])
            d.append(np.linalg.norm((C @ p)[:3] - p[:3]))
    d = np.asarray(d)
    return {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)), "max_m": float(d.max())} \
        if len(d) else {}


def write_cloud(path, xyz, inten, voxel):
    from modelforge.rosbag.ingest import write_cloud as wc
    if voxel > 0:
        _, first = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
        xyz, inten = xyz[first], inten[first]
    wc(path, xyz, inten)
    return len(xyz)


def run(project, poses="sfm", voxel=0.0, max_range=None, every=0, min_weight=2, on_line=print, token=None,
        cloud_only=False):
    """Build PROJECT/lidarmesh/ (see the module doc). voxel 0 = auto (median return range / 100, 1-20 cm).
    cloud_only: only the aggregated cloud (cloud.ply + cloud.json, report "lidar_cloud"), no fusion."""
    from modelforge import splat
    from modelforge.pipeline import Report
    token = token or jobs.Token()
    why = available(project)
    if why:
        raise RuntimeError(why)
    if poses not in POSES:
        raise ValueError(f"poses: one of {POSES}")
    why = None if cloud_only else splat.available()
    if why:
        raise RuntimeError(f"the LiDAR mesh is fused in nerfstudio's environment (Open3D, GPU): {why}")
    t0 = time.time()
    meta = json.load(open(os.path.join(project, "poses.json")))
    out = os.path.join(project, OUT)
    os.makedirs(out, exist_ok=True)
    shift = shift_stats(project, meta) if poses == "sfm" else {}
    if shift:
        on_line(f"lidar mesh: the SfM moved the bag's camera poses by {shift['median_m'] * 100:.1f} cm median, "
                f"{shift['p90_m'] * 100:.1f} cm p90, {shift['max_m'] * 100:.1f} cm at most")
    pts, inten, offs, Ts, st = collect(project, poses, every, max_range, on_line, token, meta)
    rng = np.linalg.norm(pts[:: max(1, len(pts) // 200000)], axis=1)
    if not voxel:
        voxel = float(np.clip(np.median(rng) / 100, 0.01, 0.2))
    hi = float(np.percentile(inten, 99)) if len(inten) else 0.0
    grey = np.clip(inten / hi, 0, 1).astype(np.float32) if hi > 0 else np.full(len(inten), 0.7, np.float32)
    token.check()
    on_line(f"lidar mesh: writing the aggregated cloud ({len(pts):,} returns of {len(Ts)} scans, one point per "
            f"{voxel * 100:.1f} cm voxel)")
    world = np.concatenate([pts[offs[i]:offs[i + 1]] @ Ts[i][:3, :3].T + Ts[i][:3, 3] for i in range(len(Ts))])
    n_cloud = write_cloud(os.path.join(out, "cloud.ply"), world, inten, voxel)
    del world
    cloud = dict(poses=poses, voxel=voxel, scans=int(len(Ts)), returns=int(len(pts)), points=int(n_cloud),
                 shift=shift, seconds=round(time.time() - t0, 1), **st)
    json.dump(cloud, open(os.path.join(out, "cloud.json"), "w"), indent=1)
    if cloud_only:
        Report(project).set("lidar_cloud", cloud)
        on_line(f"lidar cloud: {n_cloud:,} points from {len(Ts)} scans in {cloud['seconds']:.0f} s")
        return cloud
    scans = os.path.join(out, "scans.npz")
    np.savez(scans, points=pts, grey=grey, offsets=offs, poses=Ts.astype(np.float64))
    params = {"voxel": voxel, "max_range": float(max_range or meta["ros"].get("max_range", 60.0)),
              "face_res": FACE_RES, "min_weight": int(min_weight)}
    pfile = os.path.join(out, "params.json")
    json.dump(params, open(pfile, "w"))
    mesh = os.path.join(out, "mesh.ply")
    for name in ("mesh.ply", "preview.ply", "mesh_coloured.ply", "preview_coloured.ply"):
        p = os.path.join(out, name)
        if os.path.exists(p):
            os.remove(p)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "lidar_tsdf.py")
    on_line(f"lidar mesh: fusing {len(Ts)} scans at a {voxel * 100:.1f} cm voxel")
    splat._stream([os.path.join(splat.NS_BIN, "python"), script, scans, mesh, pfile], on_line, token)
    info = json.load(open(mesh + ".json"))
    stats = dict(info, poses=poses, voxel=voxel, point_spacing=2 * voxel,  # the splat's mesh seeds' spacing scans=int(len(Ts)), returns=int(len(pts)), cloud_points=int(n_cloud),
                 shift=shift, model=os.path.relpath(_model_dir(project), project) if poses == "sfm" else None,
                 seconds=round(time.time() - t0, 1), **st)
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    Report(project).set("lidar_mesh", stats)
    os.remove(scans)  # the worker's input: ~12 bytes per return
    on_line(f"lidar mesh: {info['faces']:,} faces from {len(Ts)} scans in {stats['seconds']:.0f} s")
    return stats


def colour(project, on_line=print, token=None, on_live=None):
    """Colour lidarmesh/mesh.ply from the photos (as the SfM mesh: sfm_mesh.colour)."""
    from modelforge import sfm_mesh
    return sfm_mesh.colour(project, on_line=on_line, token=token, on_live=on_live, folder=OUT,
                           report_key="lidar_mesh")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("project")
    ap.add_argument("--poses", choices=POSES, default="sfm", help="sfm (default): the scans moved with the SfM "
                    "poses; bag: as the ingest placed them")
    ap.add_argument("--voxel", type=float, default=0.0, help="TSDF voxel [m] (0 = auto)")
    ap.add_argument("--max-range", type=float, help="drop returns farther [m] (default: the Ingest setting)")
    ap.add_argument("--every", type=int, default=0, help=f"every N-th scan (0 = auto, at most {MAX_SCANS})")
    ap.add_argument("--min-weight", type=int, default=2, help="voxels seen by fewer scans make no surface")
    ap.add_argument("--colour", action="store_true", help="colour the existing mesh from the photos")
    ap.add_argument("--cloud-only", action="store_true", help="only the aggregated cloud (lidarmesh/cloud.ply)")
    a = ap.parse_args()
    if a.colour:
        colour(a.project)
    else:
        run(a.project, a.poses, a.voxel, a.max_range, a.every, a.min_weight, cloud_only=a.cloud_only)


if __name__ == "__main__":
    main()
