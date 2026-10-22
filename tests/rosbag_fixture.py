"""A realistic synthetic ROS bag (ROS 2, or ROS 1 Noetic with ros1=True) from a DJI project (images + TF + odometry + two LiDARs), for the
rosbag project kind's tests. The DJI project gives the truth: its video, its aligned telemetry poses
(world = the project's ENU frame) and its fused dense cloud (as LiDAR returns).

    images      /camera/image_raw/compressed  (JPEG, half resolution, frame camera_optical)
    intrinsics  /camera/camera_info           (plumb_bob, from the project's refined camera)
    poses       /tf  world -> base_link (30 Hz, a FLU body frame)
                /tf_static base_link -> camera_optical, base_link -> lidar, base_link -> lidar2,
                world -> odom (offset + rotated, so odometry has to be chained into world)
                /odom (nav_msgs/Odometry of base_link in odom, 30 Hz)
    LiDAR       /lidar/points (2 Hz, 40 m), /lidar2/points (2 Hz, 25 m, mounted rotated)
truth.json holds the camera poses (world <- camera_optical) at every image stamp.
"""

import json
import os
import shutil

import cv2
import numpy as np

from modelforge.dji import extract as ex
from modelforge.dji import ingest
from modelforge.rosbag.tf import inv, make, rot_to_quat

EPOCH = 1_700_000_000.0
R_BASE_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # optical (x right, y down, z fwd) in FLU
T_BASE_CAM = make(R_BASE_CAM, (0.05, 0.0, -0.02))
T_BASE_LIDAR = make(np.eye(3), (0.0, 0.0, 0.1))
T_BASE_LIDAR2 = make(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float), (0.1, 0.0, 0.0))  # yawed 90 deg
T_WORLD_ODOM = make(np.array([[np.cos(0.5), -np.sin(0.5), 0], [np.sin(0.5), np.cos(0.5), 0], [0, 0, 1]]),
                    (10.0, -5.0, 0.0))


def camera_poses(project, video, frames):
    """T_world_cam (OpenCV optical camera) from the DJI clip's aligned telemetry, in the project's ENU."""
    meta = json.load(open(os.path.join(project, "poses.json")))
    origin = meta["enu_origin_lla"]
    c = ingest.load_clip(video)
    alt_offset = meta["sources"][0].get("alt_offset_m", 0.0)
    out = []
    for k in frames:
        a = dict(zip(ex.AIRCRAFT_COLS, c.aircraft(k - c.lead)[0]))
        g = dict(zip(ex.GIMBAL_COLS, c.gimbal(k - c.lead)[0]))
        p = ex.ecef_to_enu(ex.lla_to_ecef(a["lat"], a["lon"], a["rel_alt_m"] + alt_offset), origin)
        R = ex.R_ENU_NED @ ex.r_ned_body(a["ac_yaw_deg"], g["gimbal_pitch_deg"], g["gimbal_roll_deg"]) @ ex.R_BODY_CAM
        out.append(make(R, p))
    return c, out


def make_bag(out, project, video, t0, t1, fps=5.0, scale=0.5, quality=85, seed=0, ros1=False):
    """ros1: out is a .bag file (Noetic, frame names with a leading "/", CameraInfo K / D / R / P); its truth goes to
    <out>.truth.json (a ROS 2 bag folder holds truth.json)."""
    from pathlib import Path

    from plyfile import PlyData
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    if ros1 and "tf2_msgs/msg/TFMessage" not in ts.types:  # not in rosbags' Noetic store (bags carry their own)
        from rosbags.typesys import get_types_from_msg
        ts.register(get_types_from_msg("geometry_msgs/TransformStamped[] transforms", "tf2_msgs/msg/TFMessage"))
    T_ = ts.types
    slash = "/" if ros1 else ""
    Time, Header = T_["builtin_interfaces/msg/Time"], T_["std_msgs/msg/Header"]
    Vec3, Quat = T_["geometry_msgs/msg/Vector3"], T_["geometry_msgs/msg/Quaternion"]
    Tf, TfS = T_["geometry_msgs/msg/Transform"], T_["geometry_msgs/msg/TransformStamped"]
    TfM = T_["tf2_msgs/msg/TFMessage"]
    rng = np.random.default_rng(seed)

    def header(t, frame):
        sec = int(t)
        stamp_ = Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))
        return Header(seq=0, stamp=stamp_, frame_id=slash + frame) if ros1 else Header(stamp=stamp_, frame_id=frame)

    def tfs(t, parent, child, T):
        q = rot_to_quat(T[:3, :3])
        return TfS(header=header(t, parent), child_frame_id=slash + child,
                   transform=Tf(translation=Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                                rotation=Quat(x=q[0], y=q[1], z=q[2], w=q[3])))

    if os.path.isdir(out):
        shutil.rmtree(out)
    elif os.path.exists(out):
        os.remove(out)
    c0 = ingest.load_clip(video)
    k0, k1 = int(np.searchsorted(c0.pts, t0)), int(np.searchsorted(c0.pts, t1))
    tf_frames = list(range(k0, k1))  # 30 Hz poses
    clip, T_wc = camera_poses(project, video, tf_frames)
    T_wb = [Tw @ inv(T_BASE_CAM) for Tw in T_wc]  # world <- base_link
    stamp = {k: EPOCH + float(clip.pts[k]) for k in tf_frames}
    step = max(1, int(round(clip.fps / fps)))
    img_frames = tf_frames[::step]
    cam = json.load(open(os.path.join(project, "report.json")))["refine"]["final"]["camera"]
    fx, fy, cx, cy, k1_, k2_, p1, p2 = cam["params"]
    w, h = int(round(clip.width * scale)), int(round(clip.height * scale))
    fused = PlyData.read(os.path.join(project, "dense", "fused.ply"))["vertex"]
    world_pts = np.c_[fused["x"], fused["y"], fused["z"]].astype(np.float64)
    truth = []
    if ros1:
        from rosbags.rosbag1 import Writer
        writer = Writer(Path(out))
    else:
        from rosbags.rosbag2 import Writer
        writer = Writer(out, version=9)
    with writer as wr:
        c_img = wr.add_connection("/camera/image_raw/compressed", "sensor_msgs/msg/CompressedImage", typestore=ts)
        c_info = wr.add_connection("/camera/camera_info", "sensor_msgs/msg/CameraInfo", typestore=ts)
        c_tf = wr.add_connection("/tf", "tf2_msgs/msg/TFMessage", typestore=ts)
        c_tfs = wr.add_connection("/tf_static", "tf2_msgs/msg/TFMessage", typestore=ts)
        c_odom = wr.add_connection("/odom", "nav_msgs/msg/Odometry", typestore=ts)
        c_l1 = wr.add_connection("/lidar/points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        c_l2 = wr.add_connection("/lidar2/points", "sensor_msgs/msg/PointCloud2", typestore=ts)

        def write(conn, t, msg):
            wr.write(conn, int(t * 1e9), (ts.serialize_ros1 if ros1 else ts.serialize_cdr)(msg, conn.msgtype))

        t_first = stamp[tf_frames[0]]
        write(c_tfs, t_first, TfM(transforms=[tfs(t_first, "base_link", "camera_optical", T_BASE_CAM),
                                              tfs(t_first, "base_link", "lidar", T_BASE_LIDAR),
                                              tfs(t_first, "base_link", "lidar2", T_BASE_LIDAR2),
                                              tfs(t_first, "world", "odom", T_WORLD_ODOM)]))
        Odom, Pose, Point = T_["nav_msgs/msg/Odometry"], T_["geometry_msgs/msg/Pose"], T_["geometry_msgs/msg/Point"]
        PoseWC, Twist = T_["geometry_msgs/msg/PoseWithCovariance"], T_["geometry_msgs/msg/Twist"]
        TwistWC = T_["geometry_msgs/msg/TwistWithCovariance"]
        T_ow = inv(T_WORLD_ODOM)
        for k, Tb in zip(tf_frames, T_wb):
            t = stamp[k]
            write(c_tf, t, TfM(transforms=[tfs(t, "world", "base_link", Tb)]))
            To = T_ow @ Tb
            q = rot_to_quat(To[:3, :3])
            write(c_odom, t, Odom(header=header(t, "odom"), child_frame_id=slash + "base_link",
                                  pose=PoseWC(pose=Pose(position=Point(x=To[0, 3], y=To[1, 3], z=To[2, 3]),
                                                        orientation=Quat(x=q[0], y=q[1], z=q[2], w=q[3])),
                                              covariance=np.zeros(36)),
                                  twist=TwistWC(twist=Twist(linear=Vec3(x=0.0, y=0.0, z=0.0),
                                                            angular=Vec3(x=0.0, y=0.0, z=0.0)),
                                                covariance=np.zeros(36))))
        # images (+ camera_info alongside)
        CImg, Info = T_["sensor_msgs/msg/CompressedImage"], T_["sensor_msgs/msg/CameraInfo"]
        Roi = T_["sensor_msgs/msg/RegionOfInterest"]
        pose_of = dict(zip(tf_frames, T_wc))
        for k in img_frames:
            rgb = clip.frame_image(clip.pts[k], width=w)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
            t = stamp[k]
            write(c_img, t, CImg(header=header(t, "camera_optical"), format="jpeg", data=jpg.reshape(-1)))
            mats = dict(d=np.array([k1_, k2_, p1, p2, 0.0]),
                        k=np.array([fx * scale, 0, cx * scale, 0, fy * scale, cy * scale, 0, 0, 1.0]),
                        r=np.eye(3).reshape(-1), p=np.zeros(12))
            write(c_info, t, Info(header=header(t, "camera_optical"), height=h, width=w, distortion_model="plumb_bob",
                                  **{(k.upper() if ros1 else k): v for k, v in mats.items()},
                                  binning_x=0, binning_y=0,
                                  roi=Roi(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)))
            truth.append({"stamp": t, "T_world_cam": pose_of[k].tolist()})
        # LiDARs: returns from the fused cloud around the current position, in the sensor frame
        PC2, PF = T_["sensor_msgs/msg/PointCloud2"], T_["sensor_msgs/msg/PointField"]
        fields = [PF(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(("x", "y", "z", "intensity"))]
        lstep = max(1, int(round(clip.fps / 2.0)))
        for k, Tb in list(zip(tf_frames, T_wb))[::lstep]:
            t = stamp[k]
            for conn, T_bl, frame, rmax, n in ((c_l1, T_BASE_LIDAR, "lidar", 40.0, 20000),
                                                 (c_l2, T_BASE_LIDAR2, "lidar2", 25.0, 10000)):
                T_wl = Tb @ T_bl
                near = world_pts[np.linalg.norm(world_pts - T_wl[:3, 3], axis=1) < rmax]
                if len(near) > n:
                    near = near[rng.choice(len(near), n, replace=False)]
                local = (near - T_wl[:3, 3]) @ T_wl[:3, :3]  # into the sensor frame
                arr = np.c_[local, np.full(len(local), 100.0)].astype(np.float32)
                write(conn, t, PC2(header=header(t, frame), height=1, width=len(arr), fields=fields,
                                   is_bigendian=False, point_step=16, row_step=16 * len(arr),
                                   data=arr.reshape(-1).view(np.uint8), is_dense=True))
    json.dump({"images": truth, "world": "world", "camera_frame": "camera_optical",
               "camera": {"model": "OPENCV", "width": w, "height": h,
                          "params": [fx * scale, fy * scale, cx * scale, cy * scale, k1_, k2_, p1, p2]}},
              open(out + ".truth.json" if ros1 else os.path.join(out, "truth.json"), "w"), indent=1)
    return out
