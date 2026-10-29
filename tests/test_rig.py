"""Articulated objects (modelforge/rig.py, gui/rig.py) on a synthetic splat: a box with an arm standing on flat
ground."""

import json
import os

import numpy as np
import plyfile

from modelforge import rig, semantic
from modelforge.export import splat_to_enu


def scene(seed=0):
    """Ground (z ~ 0, 30 x 30 m), a box body (2 x 2 x 1.5 m, from 0.3 m up) with faint Gaussians inside it and a
    weakly matching underside, and an arm from (1, 0, 1.2) to (6, 0, 1.2). Returns means, opacity, relevance and
    the slices (ground, body, faint, under, arm)."""
    rng = np.random.default_rng(seed)
    g = np.c_[rng.uniform(-15, 15, (40000, 2)), rng.normal(0, 0.02, 40000)]
    body = rng.uniform([-1, -1, 0.6], [1, 1, 1.8], (4000, 3))
    faint = rng.uniform([-0.8, -0.8, 0.7], [0.8, 0.8, 1.7], (500, 3))
    under = rng.uniform([-1, -1, 0.45], [1, 1, 0.6], (600, 3))  # the tracks: below the matches, weak relevance
    t = rng.uniform(1, 6, 1500)
    arm = np.c_[t, rng.normal(0, 0.08, 1500), 1.2 + rng.normal(0, 0.08, 1500)]
    means = np.concatenate([g, body, faint, under, arm])
    n = np.cumsum([0, len(g), len(body), len(faint), len(under), len(arm)])
    parts = {k: slice(n[i], n[i + 1]) for i, k in enumerate(("ground", "body", "faint", "under", "arm"))}
    op = np.full(len(means), 0.9)
    op[parts["faint"]] = 0.02
    rel = np.full(len(means), 0.3)
    rel[parts["body"]] = 0.75
    rel[parts["arm"]] = rng.uniform(0.52, 0.7, 1500)  # matched weakly: joins by growing, not as seeds
    rel[parts["faint"]] = 0.3
    rel[parts["under"]] = 0.4
    return means, op, rel, parts


def test_cut_out_takes_the_object_and_not_the_ground():
    means, op, rel, parts = scene()
    objs = semantic.objects(means, rel, 0.7)
    idx, ground = rig.cut_out(means, op, rel, 0.7, objs[0])
    m = np.zeros(len(means), bool)
    m[idx] = True
    for k in ("body", "faint", "under", "arm"):
        assert m[parts[k]].mean() > 0.95, k
    assert m[parts["ground"]].mean() < 0.001
    assert abs(float(ground(0.0, 0.0))) < 0.1  # under the body: inpainted from around it, not on the body
    rgb = np.full((len(means), 3), 0.4)
    f = rig.fill(means, rgb, op, m, ground)
    fx = f["means"]
    assert len(fx) and np.abs(fx[:, 2]).max() < 0.15 and np.all(np.abs(fx[:, 1]) < 2.0)
    assert np.allclose(0.5 + rig.SH_C0 * f["sh0"][:, 0], 0.4, atol=0.02)  # the ground's colour


def test_joint_cuts_the_arm_and_poses_it():
    means, op, rel, parts = scene()
    obj = np.concatenate([means[parts["body"]], means[parts["arm"]]])
    j = rig.new_joint([1.0, -1.0, 1.2], [1.0, 1.0, 1.2], [5.0, 0.0, 1.2])  # hinge across the arm at its root
    labels = rig.assign_parts(obj, [j])
    assert np.all(labels[:4000][obj[:4000, 0] < 1.0] == 0) and (labels[4000:] == 1).mean() > 0.99
    j["value"] = 90.0
    T = rig.transforms([j])[1]
    tip = T[:3, :3] @ np.array([6.0, 0, 1.2]) + T[:3, 3]
    assert np.allclose(tip, [1.0, 0, 1.2 + 5.0], atol=1e-9) or np.allclose(tip, [1.0, 0, 1.2 - 5.0], atol=1e-9)
    assert np.allclose(rig.transforms([dict(j, type="prismatic", value=2.0)])[1][:3, 3], 2 * np.array(j["axis"]))


def _write_splat(path, means, degree=1, seed=0):
    rng = np.random.default_rng(seed)
    K = (degree + 1) ** 2 - 1
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * K)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(len(means), [(k, "f4") for k in names])
    for i, c in enumerate("xyz"):
        v[c] = means[:, i]
    for k in names[3:]:
        v[k] = rng.normal(0, 0.3, len(means))
    v["rot_0"] = 1.0
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(path)
    return v


def test_posed_ply_turns_positions_and_view_dependent_colour(tmp_path):
    p = str(tmp_path)
    os.makedirs(os.path.join(p, "splat"))
    means = np.array([[0.0, 0, 0], [3.0, 0, 1], [3.1, 0.05, 1]])
    v = _write_splat(os.path.join(p, "splat", "splat_enu.ply"), means)
    f = {"means": np.zeros((1, 3), np.float32), "quats": np.array([[1.0, 0, 0, 0]], np.float32),
         "scales": np.zeros((1, 3), np.float32), "opacities": np.zeros(1, np.float32),
         "sh0": np.zeros((1, 1, 3), np.float32)}
    rig.create(p, "thing", np.array([1, 2]), f)
    r = rig.load(p, "thing")
    r["joints"].append(dict(rig.new_joint([2.0, 0, 1], [2.0, 1, 1], [3.5, 0, 1]), value=90.0))
    rig.save(p, "thing", r)
    labels = rig.relabel(p, "thing", means[[1, 2]])
    assert list(labels) == [1, 1]
    out = rig.posed_ply(p, "thing", os.path.join(p, "export", "posed.ply"))
    w = plyfile.PlyData.read(out)["vertex"].data
    assert len(w) == 4  # the rest (1) + the fill (1) + the parts (2)
    T = rig.transforms(r["joints"])[1]
    R, t = T[:3, :3], T[:3, 3]
    got = np.stack([w["x"], w["y"], w["z"]], 1)[-2:]
    assert np.allclose(got, means[[1, 2]] @ R.T + t, atol=1e-5)
    # colour seen from direction R d after the turn = colour seen from d before
    d = np.random.default_rng(1).normal(size=(20, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rest = sorted((n for n in v.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    F0 = np.stack([v[n][1] for n in rest]).reshape(3, -1)
    F1 = np.stack([w[n][-2] for n in rest]).reshape(3, -1)
    before = splat_to_enu.sh_basis(d, 1) @ F0.T
    after = splat_to_enu.sh_basis(d @ R.T, 1) @ F1.T
    assert np.allclose(before, after, atol=1e-4)
    assert json.load(open(os.path.join(p, "rig", "thing", "rig.json")))["joints"][0]["value"] == 90.0
    assert rig.list_rigs(p) == ["thing"] and rig.current(p, "thing")


def test_panel_hides_the_object_in_the_splat():
    from modelforge.gui.rig import RigPanel

    class V:
        value = True
    panel = RigPanel.__new__(RigPanel)
    panel.rig, panel.obj, panel.g_show = {"joints": []}, (np.zeros((1, 3)),), V()
    panel.hidden = np.zeros(10, bool)
    panel.hidden[[2, 3, 4]] = True
    c = np.arange(10, dtype=np.float32)[:, None].repeat(3, 1)
    out = panel.decorate(c, c, c, c[:, :1], np.arange(10))
    assert len(out[0]) == 7 and 3.0 not in out[0][:, 0]
    # merged (level of detail): shown Gaussian 0 from sources 0-2 (one hidden), 1 from 3-4 (both hidden)
    out = panel.decorate(c[:3], c[:3], c[:3], c[:3, :1], np.array([0, 1, 2, 3, 4]), np.array([0, 0, 0, 1, 1]))
    assert len(out[0]) == 2  # 0 stays (1 of 3 hidden), 1 goes, 2 has no sources: stays


def test_unirig_limbs_become_hinges_where_they_leave_the_body():
    """rig_unirig: a creature-like skeleton (spine through a body, an arm out to +x that overshoots its tip, a leg
    down) over Gaussians of a box body and an arm: one joint, hinged at the body's edge, across the arm; the leg and
    the spine are no joints. The similarity maps UniRig's frame back exactly; the consensus keeps limbs >= 2 runs
    share, opposite ones apart."""
    from modelforge import rig_unirig
    means, op, rel, parts = scene()
    obj = np.concatenate([means[parts["body"]], means[parts["arm"]]])
    heads = np.array([[-1.0, 0, 1.2], [0.0, 0, 1.2], [0.0, 0, 1.2], [1.5, 0, 1.2], [4.0, 0, 1.2], [0.5, 0.5, 1.0]])
    tails = np.array([[0.0, 0, 1.2], [1.0, 0, 1.2], [1.5, 0, 1.2], [4.0, 0, 1.2], [7.5, 0, 1.2], [0.5, 0.5, 0.0]])
    parents = np.array([-1, 0, 0, 2, 3, 0])
    L = rig_unirig.limbs(heads, tails, parents, obj)
    assert len(L) == 1
    o, d = L[0]["origin"], L[0]["dir"]
    assert 0.9 <= o[0] <= 1.8 and abs(o[1]) < 0.1  # where the arm leaves the body (its edge at x = 1)
    assert np.allclose(d, [1, 0, 0], atol=1e-6) and abs(L[0]["axis"] @ [0, 1, 0]) > 0.99
    assert 5.5 <= o[0] + L[0]["reach"] <= 6.5  # clipped at the arm's tip (6 m), not UniRig's 7.5
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3))
    R = rig.rotation([0.2, 1, 0.3], 0.7)
    s, Rf, a0, b0 = rig_unirig.similarity(A, 3.0 * A @ R.T + [1.0, 2, 3])
    assert abs(s - 3.0) < 1e-9 and np.allclose(Rf, R, atol=1e-9)
    left = dict(L[0], dir=-L[0]["dir"], origin=L[0]["origin"] - [2.5, 0, 0], seed=L[0]["seed"] - [6.0, 0, 0])
    runs = [[L[0], left], [dict(L[0], seed=L[0]["seed"] + [0.8, 0.1, 0])], [left], []]
    c = rig_unirig.consensus(runs)
    assert len(c) == 2 and sorted(x["runs"] for x in c) == [2, 2]
    assert len(rig_unirig.consensus(runs, min_runs=3)) == 0


def test_animation_tracks():
    """The demo moves each joint through its limits and back, then all together; keyframes loop back to the
    first; sampling eases between keyframes and loops or holds."""
    joints = [{"type": "revolute", "limits": [-30.0, 60.0]}, {"type": "prismatic", "limits": [0.0, 1.0]}]
    keys = rig.demo_track(joints)
    vals = np.array([v for _, v in keys])
    assert vals[:, 0].max() == 60.0 and vals[:, 0].min() == -30.0 and vals[:, 1].max() == 1.0
    assert np.all(np.diff([t for t, _ in keys]) >= 0) and keys[-1][1] == [0.0, 0.0]
    assert any(v[0] == 60.0 and v[1] == 1.0 for _, v in keys)  # all together at the end
    k2 = rig.keyframe_track([[0.0, 0.0], [40.0, 0.5]], 2.0)
    assert [t for t, _ in k2] == [0.0, 2.0, 4.0]
    assert rig.sample(k2, 1.0) == [20.0, 0.25]  # halfway, eased symmetric
    assert rig.sample(k2, 0.5)[0] < 10.0  # smoothstep: slow at the start
    assert rig.sample(k2, 5.0) == rig.sample(k2, 1.0)  # loops
    assert rig.sample(k2, 9.0, loop=False) == [0.0, 0.0]  # holds the last


def test_vision_model_answer_becomes_motions():
    """rig_vlm: "positive raises the far end" maps onto each hinge's own sign, angles are clipped to the limits,
    joints are named J<k>, a motion that moves nothing is dropped, and JSON is found after "JSON:" in prose."""
    from modelforge import rig_vlm
    up = rig.new_joint([0, 0, 1.0], [0, 1, 1.0], [3, 0, 1.0])  # axis +y: +angle turns x towards -z (down)
    down = rig.new_joint([0, 0, 1.0], [0, -1, 1.0], [3, 0, 1.0])  # axis -y: +angle raises
    slew = dict(rig.new_joint([0, 0, 0.0], [0, 0, -1.0], [2, 0, 0.5]), limits=[-180.0, 180.0])
    joints = [up, down, slew]
    assert rig_vlm.signs(joints) == [-1.0, 1.0, -1.0]
    text = 'Thinking about it... JSON:\n```json\n{"machine": "m", "motions": [' \
           '{"name": "fold", "steps": [{"seconds": 2, "angles": {"J1": 30, "J2": 30}}, ' \
           '{"seconds": 1, "angles": {"J3": 90, "J9": 5, "J1": "x"}}]}, ' \
           '{"name": "nothing", "steps": [{"seconds": 3, "angles": {"J1": 0}}]}]}\n```'
    ans = rig_vlm.parse(text)
    mos = rig_vlm.motions(ans, joints)
    assert [m["name"] for m in mos] == ["fold"]
    keys = mos[0]["keys"]
    assert [t for t, _ in keys] == [0.0, 2.0, 3.0]
    assert keys[1][1] == [-30.0, 30.0, 0.0] and keys[2][1] == [-30.0, 30.0, -90.0]
    T = rig.transforms(joints, keys[1][1])
    for k in (1, 2):  # both hinges raise their seeds
        s = np.asarray(joints[k - 1]["seed"])
        assert (T[k][:3, :3] @ s + T[k][:3, 3])[2] > s[2]
    clipped = rig_vlm.motions({"motions": [{"name": "far", "steps": [{"seconds": 1, "angles": {"J1": 200}}]}]},
                              joints)
    assert clipped[0]["keys"][1][1][0] == -90.0  # -200 clipped to the lower limit


def test_loops_and_starts_have_no_jumps():
    """A looping motion that ends elsewhere gets a hold and a smooth return; playback starts from the pose shown
    with a lead-in; consecutive samples never jump."""
    joints = [{"type": "revolute", "limits": [-90.0, 90.0]}, {"type": "revolute", "limits": [-90.0, 90.0]}]
    keys = [(0.0, [0.0, 0.0]), (3.0, [60.0, 30.0])]  # ends folded
    c = rig.closed(keys, joints)
    assert c[-1][1] == [0.0, 0.0] and c[-2][1] == [60.0, 30.0] and c[-1][0] - c[-2][0] == 3.0  # played back
    assert rig.closed(c, joints) == c  # already closed
    end = c[-1][0]
    ts = np.linspace(0, 2 * end, 2000)
    vals = np.array([rig.sample(c, t) for t in ts])
    assert np.abs(np.diff(vals, axis=0)).max() < 1.0  # no jump, also across the loop
    intro = rig.lead_in(c, joints, [10.0, -20.0])
    assert intro[0][1] == [10.0, -20.0] and intro[-1][1] == [0.0, 0.0] and rig.lead_in(c, joints, [0.0, 0.0]) == []


def test_partfield_parts_become_joints():
    """rig_partfield.joints_from_parts on a synthetic excavator: undercarriage (base), a house sitting on it (a
    turntable: vertical axis), an arm out of the house (a hinge across it, at the house), gravel by the tracks (stays
    with the base); KMeans separates well-apart clusters."""
    from modelforge import rig_partfield
    rng = np.random.default_rng(0)
    under = rng.uniform([-2, -1.5, 0.0], [2, 1.5, 1.0], (4000, 3))
    house = rng.uniform([-1.8, -1.4, 1.0], [1.8, 1.4, 2.5], (3000, 3))
    arm = np.c_[rng.uniform(1.8, 7.0, 1500), rng.normal(0, 0.15, 1500), 2.0 + rng.normal(0, 0.15, 1500)]
    gravel = rng.uniform([-6, -1, 0.0], [-2.2, 1, 0.4], (800, 3))
    means = np.concatenate([under, house, arm, gravel])
    parts = np.repeat([0, 1, 2, 3], [4000, 3000, 1500, 800])
    joints = rig_partfield.joints_from_parts(means, parts)
    kinds = {j["part"]: (j["name"].split()[0], j["parent"]) for j in joints}
    assert kinds[1] == ("slew", 0) and np.allclose(np.abs(joints[0]["axis"]), [0, 0, 1])
    assert kinds[2][0] == "hinge" and kinds[2][1] == 1  # the arm hangs on the house's joint
    arm_joint = next(j for j in joints if j["part"] == 2)
    assert 1.2 < arm_joint["origin"][0] < 2.6 and abs(arm_joint["axis"][0]) < 0.1  # at the house, across the arm
    assert 3 not in kinds  # the gravel stays with the base
    labels = rig.assign_parts(means, joints, parts=parts)
    assert (labels[4000:7000] == 1).all() and (labels[7000:8500] == 2).all() and (labels[8500:] == 0).all()
    X = np.concatenate([rng.normal(c, 0.1, (300, 8)) for c in (0.0, 3.0, -3.0)])
    lab = rig_partfield.kmeans(X, 3)
    assert len({tuple(np.unique(lab[i * 300:(i + 1) * 300])) for i in range(3)}) == 3
    assert all(len(np.unique(lab[i * 300:(i + 1) * 300])) == 1 for i in range(3))


def test_turntable_keeps_the_hanging_arm_and_template_joints():
    """A turntable cut (horizontal, at the ring) takes the house and the arm hanging down in front of it below the
    ring, not the tracks; rig_vlm.template_joints builds the chain from named pins and part points, arm hinges
    parallel."""
    from modelforge import rig_vlm
    rng = np.random.default_rng(1)
    under = rng.uniform([-2, -1.5, 0.0], [2, 1.5, 1.0], (4000, 3))
    house = rng.uniform([-1.8, -1.4, 1.05], [1.8, 1.4, 2.5], (3000, 3))
    t = rng.uniform(0, 1, 2000)
    boom = np.c_[1.8 + 3.0 * t, rng.normal(0, 0.1, 2000), 2.4 + 0.5 * t]  # up and out
    s = rng.uniform(0, 1, 2000)
    stick = np.c_[4.8 + 0.6 * s, rng.normal(0, 0.1, 2000), 2.9 - 2.8 * s]  # down to the ground, in front
    means = np.concatenate([under, house, boom, stick])
    tmpl = {"machine": "an excavator", "joints": [
        {"name": "swing", "type": "slew", "parent": "tracks", "child": "upper body", "pin": "ring"},
        {"name": "boom", "type": "hinge", "parent": "upper body", "child": "boom", "pin": "boom foot"},
        {"name": "stick", "type": "hinge", "parent": "boom", "child": "stick", "pin": "elbow"}]}
    pins = {"swing": np.array([0.0, 0.0, 1.02]), "boom": np.array([1.8, 0.0, 2.4]), "stick": np.array([4.8, 0.0, 2.9])}
    centres = {"upper body": np.array([0.0, 0.0, 1.8]), "boom": np.array([3.3, 0.0, 2.65]),
               "stick": np.array([5.1, 0.0, 1.5])}
    joints = rig_vlm.template_joints(tmpl, pins, centres)
    assert [j["parent"] for j in joints] == [0, 1, 2] and np.allclose(np.abs(joints[0]["axis"]), [0, 0, 1])
    assert np.allclose(joints[1]["axis"], joints[2]["axis"]) and abs(joints[1]["axis"][1]) > 0.99  # across the arm
    labels = rig.assign_parts(means, joints)
    assert (labels[:4000] == 0).mean() > 0.99  # the tracks stay
    assert (labels[4000:7000] == 1).mean() > 0.95
    assert (labels[7000:9000] == 2).mean() > 0.9 and (labels[9000:] == 3).mean() > 0.9  # boom and stick, all of it


def test_ring_from_a_side_click():
    """rig.ring: a click on the side at the gap (z 1.0, at the house's edge) -> the click's height, centred in the
    house above it."""
    rng = np.random.default_rng(2)
    tracks = rng.uniform([-2, -1.5, 0.0], [2, 1.5, 1.0], (6000, 3))
    house = rng.uniform([-1.8, -1.4, 1.0], [1.8, 1.4, 2.5], (5000, 3)) + [0.5, 0.2, 0]
    p = rig.ring(np.concatenate([tracks, house]), [2.3, 0.0, 1.0])
    assert abs(p[2] - 1.0) < 1e-9 and np.linalg.norm(p[:2] - [0.5, 0.2]) < 0.35
