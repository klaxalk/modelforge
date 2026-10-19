"""Project kind "rosbag": TF buffer, bag reading (images, camera_info, TF, odometry, PointCloud2),
poses from TF or odometry, extraction (priors, camera.json, LiDAR cloud), and the GUI flow on the
berlin-derived synthetic bag (skipped without the sample clip and data/fullrun_test)."""

import json
import os
import socket
import time

import cv2
import numpy as np
import pytest

from modelforge import pipeline, projects
from modelforge.dji import ingest as dji
from modelforge.rosbag import bag
from modelforge.rosbag import ingest as ri
from modelforge.rosbag.tf import TFBuffer, inv, make, quat_to_rot, rot_to_quat
from modelforge.sfm import priors
from tests.sample_clip import sample_video

FULLRUN = os.path.join(os.path.dirname(__file__), "..", "data", "fullrun_test")


def test_tf_buffer():
    b = TFBuffer()
    yaw90 = rot_to_quat(quat_to_rot((0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))))
    b.add("world", "base", 0.0, (0, 0, 0), (0, 0, 0, 1))
    b.add("world", "base", 1.0, (2, 0, 0), yaw90)
    b.add("base", "cam", 0.0, (0, 0, 1), (0, 0, 0, 1), static=True)
    T = b.lookup("world", "cam", 0.5)
    assert np.allclose(T[:3, 3], (1, 0, 1)) and np.isclose(np.degrees(np.arctan2(T[1, 0], T[0, 0])), 45)
    assert np.allclose(b.lookup("cam", "world", 0.5), inv(T))
    assert b.lookup("world", "cam", 5.0) is None and b.lookup("world", "nowhere", 0.5) is None
    assert b.frames == ["base", "cam", "world"]


def write_small_bag(path, n=30, w=160, h=120, mcap=False, uncalibrated=False):
    """A camera moving 0.1 m per image along x over a frozen test pattern, + TF, odometry, camera_info
    and one LiDAR (sqlite3, or MCAP storage); returns the true T_world_cam per image."""
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T_ = ts.types
    Time, Header = T_["builtin_interfaces/msg/Time"], T_["std_msgs/msg/Header"]
    Vec3, Quat, Point = T_["geometry_msgs/msg/Vector3"], T_["geometry_msgs/msg/Quaternion"], T_["geometry_msgs/msg/Point"]

    def header(t, frame):
        return Header(stamp=Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id=frame)

    def tfs(t, parent, child, T):
        q = rot_to_quat(T[:3, :3])
        return T_["geometry_msgs/msg/TransformStamped"](
            header=header(t, parent), child_frame_id=child,
            transform=T_["geometry_msgs/msg/Transform"](translation=Vec3(x=T[0, 3], y=T[1, 3], z=T[2, 3]),
                                                        rotation=Quat(x=q[0], y=q[1], z=q[2], w=q[3])))
    R_base_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T_base_cam = make(R_base_cam, (0.1, 0, 0))
    T_world_odom = make(np.eye(3), (5.0, 0, 0))
    tex = np.random.default_rng(0).integers(0, 255, (h, w * 4, 3), np.uint8)  # a texture to pan across
    truth = []
    storage = StoragePlugin.MCAP if mcap else StoragePlugin.SQLITE3
    with Writer(path, version=9, storage_plugin=storage) as wr:
        conns = {t: wr.add_connection(t, ty, typestore=ts) for t, ty in (
            ("/cam/image/compressed", "sensor_msgs/msg/CompressedImage"), ("/cam/info", "sensor_msgs/msg/CameraInfo"),
            ("/tf", "tf2_msgs/msg/TFMessage"), ("/tf_static", "tf2_msgs/msg/TFMessage"),
            ("/odom", "nav_msgs/msg/Odometry"), ("/scan", "sensor_msgs/msg/PointCloud2"))}

        def write(topic, t, msg):
            wr.write(conns[topic], int(t * 1e9), ts.serialize_cdr(msg, conns[topic].msgtype))
        write("/tf_static", 100.0, T_["tf2_msgs/msg/TFMessage"](transforms=[
            tfs(100.0, "base", "cam_optical", T_base_cam), tfs(100.0, "base", "lidar", make(np.eye(3), (0, 0, 0.2))),
            tfs(100.0, "map", "odom", T_world_odom)]))
        for i in range(n):
            t = 100.0 + 0.2 * i
            T_wb = make(np.eye(3), (0.1 * i, 0, 1.0))
            write("/tf", t, T_["tf2_msgs/msg/TFMessage"](transforms=[tfs(t, "map", "base", T_wb)]))
            To = inv(T_world_odom) @ T_wb
            q = rot_to_quat(To[:3, :3])
            write("/odom", t, T_["nav_msgs/msg/Odometry"](
                header=header(t, "odom"), child_frame_id="base",
                pose=T_["geometry_msgs/msg/PoseWithCovariance"](
                    pose=T_["geometry_msgs/msg/Pose"](position=Point(x=To[0, 3], y=To[1, 3], z=To[2, 3]),
                                                      orientation=Quat(x=q[0], y=q[1], z=q[2], w=q[3])),
                    covariance=np.zeros(36)),
                twist=T_["geometry_msgs/msg/TwistWithCovariance"](
                    twist=T_["geometry_msgs/msg/Twist"](linear=Vec3(x=0.0, y=0.0, z=0.0),
                                                        angular=Vec3(x=0.0, y=0.0, z=0.0)), covariance=np.zeros(36))))
            img = tex[:, 2 * i:2 * i + w]
            ok, jpg = cv2.imencode(".jpg", img)
            write("/cam/image/compressed", t, T_["sensor_msgs/msg/CompressedImage"](
                header=header(t, "cam_optical"), format="jpeg", data=jpg.reshape(-1)))
            write("/cam/info", t, T_["sensor_msgs/msg/CameraInfo"](  # uncalibrated drivers publish all zeros
                header=header(t, "cam_optical"), height=0 if uncalibrated else h, width=0 if uncalibrated else w,
                distortion_model="" if uncalibrated else "plumb_bob",
                d=np.zeros(0) if uncalibrated else np.array([0.01, 0.0, 0.0, 0.0, 0.0]),
                k=np.zeros(9) if uncalibrated else np.array([100.0, 0, 80, 0, 100.0, 60, 0, 0, 1]),
                r=np.eye(3).reshape(-1), p=np.zeros(12), binning_x=0, binning_y=0,
                roi=T_["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)))
            pts = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, -0.5], [np.nan, 0, 0]], np.float32)  # in the lidar frame
            fields = [T_["sensor_msgs/msg/PointField"](name=c, offset=4 * k, datatype=7, count=1)
                      for k, c in enumerate("xyz")]
            write("/scan", t, T_["sensor_msgs/msg/PointCloud2"](
                header=header(t, "lidar"), height=1, width=3, fields=fields, is_bigendian=False, point_step=12,
                row_step=36, data=pts.reshape(-1).view(np.uint8), is_dense=False))
            truth.append(T_wb @ T_base_cam)
    return truth


def test_sharpest_streams_per_window(monkeypatch):
    """Each window's sharpest frame comes out as soon as decoding has passed the window (progress while the
    bag is read, one window's frames in memory), and it is the sharpest of its window."""
    rng = np.random.default_rng(0)
    read = []

    def fake_iter(video, topic, indices):
        for k in indices:
            read.append(k)
            sharpness = 1 + (k % 3 == 1)  # the middle frame of every 3 is the sharpest
            img = (rng.random((40, 40, 3)) * 255 * sharpness / 2).astype(np.uint8) if sharpness > 1 else \
                np.full((40, 40, 3), 128, np.uint8)
            yield k, float(k), img
    monkeypatch.setattr(ri.bag, "iter_images", fake_iter)

    class Clip:
        video, topic = "bag", "/img"
    windows = [(0, 2), (3, 5), (6, 8), (9, 11)]
    got, read_at = [], []
    for k, rgb, sharp in ri._sharpest(Clip(), windows):
        got.append(k)
        read_at.append(len(read))
    assert got == [1, 4, 7, 10]
    assert read_at[:3] == [4, 7, 10]  # window 0 out after frame 3 was read, not after the whole bag


def test_extrinsic_rpy(tmp_path):
    """A camera frame missing from TF: poses of its mount frame + Extrinsic RPY. The test bag mounts the
    optical frame on base looking forward (0.1 m ahead), so base + RPY 0, 0, 0 gives the true
    orientations; pitch 90 looks straight down with the image's top towards the front."""
    R = ri.mount_rotation([0, 0, 0])
    assert np.allclose(R[:, 2], [1, 0, 0]) and np.allclose(R[:, 1], [0, 0, -1])  # forward, image down = down
    R = ri.mount_rotation([0, 90, 0])
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(-R[:, 1], [1, 0, 0])  # down, image top = front
    R = ri.mount_rotation([0, 30, 0])
    assert np.isclose(np.degrees(np.arcsin(-R[2, 2])), 30)  # forward, tilted 30 deg down
    path = str(tmp_path / "small_bag")
    truth = write_small_bag(path)
    d = ri.ros_defaults(path)
    c = ri.load_clip(path, ros=dict(d, camera_frame="base", camera_rpy=[0.0, 0.0, 0.0]))
    assert c.valid_pose.all() and c.align["camera_rpy"] == [0.0, 0.0, 0.0]
    for k in range(c.n):
        assert np.allclose(c.T[k][:3, :3], truth[k][:3, :3], atol=1e-9)
        assert np.isclose(np.linalg.norm(c.T[k][:3, 3] - truth[k][:3, 3]), 0.1)  # the mount's offset


def test_uncalibrated_camera_info(tmp_path):
    """A camera_info of zeros (uncalibrated driver) is no calibration: extraction works without camera.json
    (it divided by the zero size before), and SfM estimates the intrinsics."""
    path = str(tmp_path / "zero_info_bag")
    write_small_bag(path, uncalibrated=True)
    assert bag.camera_info(path, "/cam/info") is None
    d = ri.ros_defaults(path)
    proj = projects.create(str(tmp_path), "zeroproj", kind="rosbag")
    lines = []
    out = ri.run(proj, [dji.Source(path)], dji.Settings(sampling="distance", num_images=6, ros=d), log=lines.append)
    assert len(out["images"]) == 6 and not os.path.exists(os.path.join(proj, "camera.json"))
    assert any("no calibration" in m for m in lines)


def test_small_bag(tmp_path):
    path = str(tmp_path / "small_bag")
    truth = write_small_bag(path)
    assert bag.is_bag(path) and bag.find_bags(str(tmp_path)) == [path]
    d = ri.ros_defaults(path)
    assert d["image_topic"] == "/cam/image/compressed" and d["info_topic"] == "/cam/info"
    assert d["world_frame"] == "map" and d["pose_source"] == "tf" and d["cloud_topics"] == ["/scan"]
    for source in ("tf", "odometry"):  # odometry: odom -> base, chained map -> odom (static) and base -> camera
        c = ri.load_clip(path, ros=dict(d, pose_source=source))
        assert c.n == 30 and (c.width, c.height) == (160, 120) and c.valid_pose.all()
        assert max(np.abs(c.T[k] - truth[k]).max() for k in range(30)) < 1e-6
    cam = bag.camera_info(path, "/cam/info")
    assert cam["model"] == "OPENCV" and cam["params"][:4] == [100.0, 100.0, 80.0, 60.0]
    proj = projects.create(str(tmp_path), "bagproj", kind="rosbag")
    out = ri.run(proj, [dji.Source(path)], dji.Settings(sampling="distance", num_images=6, ros=d))
    assert len(out["images"]) == 6 and dji.load_kind(proj) == "rosbag"
    assert json.load(open(os.path.join(proj, "camera.json")))["params"][:2] == [100.0, 100.0]
    pri, meta = priors.load(proj)
    assert priors.has_poses(pri) and meta["kind"] == "rosbag"
    r = out["images"][0]
    assert np.allclose([r["enu_x"], r["enu_y"], r["enu_z"]], truth[r["frame"]][:3, 3])
    from modelforge.export import lidar
    cloud = lidar.load(proj)  # two returns per scan, 0.1 m apart per scan, in the world frame (voxel 5 cm)
    assert cloud is not None and len(cloud) > 30 and np.all(np.isfinite(cloud))
    assert np.allclose(sorted(set(np.round(cloud[:, 2], 3))), [0.7, 1.2])  # base z 1 + lidar 0.2 (- 0.5)
    assert pipeline.sfm_stages("rosbag") == pipeline.SFM_STAGES


@pytest.mark.skipif(not (os.path.exists(sample_video()) and os.path.exists(os.path.join(FULLRUN, "dense", "fused.ply"))),
                    reason="needs the sample clip and data/fullrun_test (refined, with dense/fused.ply)")
def test_rosbag_project_through_the_gui(tmp_path):
    """The berlin-derived bag: TF poses (telemetry), camera_info, two LiDARs; extract 20 images, SfM."""
    from modelforge.gui.app import App
    from tests.rosbag_fixture import make_bag
    root = str(tmp_path / "data")
    os.makedirs(root)
    bag_path = make_bag(str(tmp_path / "data" / "bag"), FULLRUN, sample_video(), 306.0, 316.0)
    path = projects.create(root, "robot", kind="rosbag")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    app = App(path, port=port, root=root)
    try:
        ing = app.ingest
        assert ing.kind == "rosbag" and ing.g_ros.visible and not ing.g_upload.visible
        assert app.g_buttons["refine"].visible and app.g_dense_lidar.visible and not app.g_upright.value
        ing.sources = [dji.Source(os.path.relpath(bag_path))]
        ing.settings = dji.Settings(sampling="distance", num_images=20)
        ing._rebuild_sources()
        ing.request_plan()
        t0 = time.time()
        while (ing.plan is None or ing.plan.num_images != 20) and time.time() - t0 < 300:
            time.sleep(0.5)
        assert ing.plan.num_images == 20
        assert ing.g_ros_image.value == "/camera/image_raw/compressed" and set(ing._cloud_boxes) == {
            "/lidar/points", "/lidar2/points"}
        ing.extract()
        while ing.running and time.time() - t0 < 600:
            time.sleep(0.5)
        assert app.project is not None and len(app.project.names) == 20 and app.project.has_priors
        app.run_stage("sfm")
        while app.running and time.time() - t0 < 900:
            time.sleep(0.5)
        assert "sfm finished" in app.g_state.content, app.g_state.content
        d = pipeline.Report(path).data
        assert d["features"]["camera"] == {"model": "OPENCV", "known": True}
        f = d["refine"]["final"]
        assert f["reg_images"] == 20 and f["point_error_px"]["median"] < 1.0
        known = json.load(open(os.path.join(path, "camera.json")))["params"]
        assert np.allclose(f["camera"]["params"][:4], known[:4], atol=1e-3)  # camera_info kept fixed
        assert d["refine"]["lidar_check"]["median_m"] < 0.3  # SfM points on the LiDAR surfaces
        assert d["export"]["lidar_points"] > 1000  # the splat starts from the LiDAR points too
    finally:
        app.server.stop()


def test_splat_lidar_command():
    from modelforge import splat
    cmd = splat.train_command("p", "splatfacto-big", 100, lidar_depth=0.3)
    assert cmd[1] == "splatfacto-big-lidar" and cmd[cmd.index("--method-name") + 1] == "splatfacto-big-lidar"
    assert cmd[cmd.index("--pipeline.model.lidar-depth-lambda") + 1] == "0.3" and cmd[-3] == "lidar-data"
    plain = splat.train_command("p", "splatfacto")
    assert plain[1] == "splatfacto" and "nerfstudio-data" in plain and "lidar-data" not in plain
    assert splat.parse_lidar("[lidar] step 1500: depth error median 0.64% p90 2.57% within 5% 95.3%") == {
        "step": 1500, "median_pct": 0.64, "p90_pct": 2.57, "within_5pct": 95.3}


@pytest.mark.skipif(not os.path.exists("/opt/nerfstudio/bin/python"), reason="needs nerfstudio's venv (CUDA image)")
def test_lidar_depth_image_in_nerfstudio_venv():
    """The plugin's per-view LiDAR depth: nearest return per pixel, returns seen through the gaps of a
    nearer surface dropped; and the plugin's methods register with ns-train."""
    import subprocess
    from modelforge import splat
    code = """
import torch
from mf_splatfacto_lidar import lidar_depth_image
g = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 21), torch.linspace(-1, 1, 21), indexing="ij"), -1).reshape(-1, 2)
front = torch.cat([g, torch.full((len(g), 1), 5.0)], 1)                    # sparse wall at z = 5
back = torch.cat([g * 6 + 0.05, torch.full((len(g), 1), 10.0)], 1)          # far wall, peeking through
K = torch.tensor([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
d = lidar_depth_image(torch.cat([front, back]), torch.eye(4), K, 100, 100, window=5, tol=0.05)
v = d[d > 0]
centre = d[42:58, 42:58][d[42:58, 42:58] > 0]  # inside the near wall (u, v in 40..60)
assert d.shape == (100, 100, 1) and torch.allclose(centre, torch.tensor(5.0)), centre.unique()
assert (v == 10.0).any()  # the far wall outside the near one stays
print("ok")
"""
    r = subprocess.run([os.path.join(splat.NS_BIN, "python"), "-c", code], env=splat._env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
    r = subprocess.run([os.path.join(splat.NS_BIN, "ns-train"), "splatfacto-lidar", "--help"], env=splat._env(),
                       capture_output=True, text=True, timeout=300)
    assert "lidar-depth-lambda" in r.stdout and "Error" not in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
