"""Motions for a rig planned by a local vision-language model (the user asked, 2026-10-27; Qwen2.5-VL-7B-Instruct,
Apache-2.0, offline: nothing leaves the machine).

**Views** (`views`): the object alone, rendered from 4 sides (azimuths 45° apart from the axes, 35° above, framing
the whole object; export/rig_views.py in nerfstudio's venv) with the moving parts tinted, every hinge axis drawn
(white, black outline) and labelled J<k> on it, plus the project photo that shows the object largest (context). **Question** (`PROMPT`):
what the machine and each part are, whether each hinge axis is plausible, and 2-3 motions it really performs, as
JSON with angles per joint per step. **Signs:** a hinge's angle sign is arbitrary in the rig, so the model is told
"positive raises the part's far end" (vertical axes: counter-clockwise seen from above; slides: outwards), and
`signs` maps that onto each joint (its seed turned by +10°: up or down). **Result** (`plan`): rig.json `vlm` (the
answer: machine, joint roles) and `motions` ([{name, description, keys: [[t, values]]}]) for rig.sample; the Rig
tab plays them.

Runs in UniRig's Python (build/unirig/venv, scripts/install_unirig) with the weights in build/hf_cache
(`scripts/install_unirig --vlm` downloads them, ~16 GB); ~17 GB of GPU memory while it answers (~1 min).
"""

import glob
import json
import os
import re

import numpy as np

from modelforge import jobs, rig, rig_unirig, splat, splat_mesh

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
COLOUR_NAMES = ("red", "green", "yellow", "blue", "orange", "purple", "cyan", "magenta", "lime", "teal")
TINT = 0.6
CLOSE_TINT = 0.35


def model_path():
    snaps = sorted(glob.glob(os.path.join(rig_unirig.HF_HOME, "hub", "models--" + MODEL.replace("/", "--"),
                                          "snapshots", "*", "config.json")))
    return os.path.dirname(snaps[-1]) if snaps else None


def available():
    why = rig_unirig.available()
    if why:
        return why.replace("UniRig is", "UniRig's environment (the vision model runs in it) is")
    if model_path() is None:
        return f"the vision model ({MODEL}) is not downloaded (scripts/install_unirig --vlm)"
    return None


def signs(joints):
    """Per joint the factor that turns the model's convention (positive raises the far end; vertical axes:
    counter-clockwise from above; slides: outwards) into the rig's angle."""
    out = []
    for j in joints:
        a = np.asarray(j["axis"], float)
        a = a / max(np.linalg.norm(a), 1e-12)
        if j["type"] == "prismatic":
            out.append(1.0 if (np.asarray(j["seed"]) - np.asarray(j["origin"])) @ a >= 0 else -1.0)
        elif abs(a[2]) > 0.7:
            out.append(1.0 if a[2] > 0 else -1.0)
        else:
            T = rig.joint_matrix(j, 10.0)
            s = np.asarray(j["seed"], float)
            out.append(1.0 if (T[:3, :3] @ s + T[:3, 3])[2] >= s[2] else -1.0)
    return out


def _look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(f, right)
    R = np.stack([right, down, f], 1)  # world <- OpenCV camera
    V = np.eye(4)
    V[:3, :3], V[:3, 3] = R.T, -R.T @ eye
    return V


def views(project, name, out_dir, width=896, height=672, fov=50.0, on_line=print, token=None):
    """Render the annotated views (module doc) into out_dir; returns the image paths: 4 overviews (every moving part
    tinted), a close-up per joint (only its part tinted, from the overview side that shows it largest, labelled),
    then the photo."""
    import cv2
    import plyfile
    from modelforge.gui import data as gdata
    r = rig.load(project, name)
    joints = r["joints"]
    idx, labels, _ = rig.arrays(project, name)
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data[idx]
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    natural = gdata.splat_arrays(v, None)[2].astype(np.float64)

    def tinted(parts, amount=TINT):
        rgb = natural.copy()
        for k in parts:
            col = np.array(rig_colour(k)) / 255.0
            rgb[labels == k] = (1 - amount) * rgb[labels == k] + amount * col
        return rgb
    lo, hi = np.percentile(means, 1, axis=0), np.percentile(means, 99, axis=0)
    c = (lo + hi) / 2
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    f = 0.5 * height / np.tan(np.radians(fov) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    dist = radius / np.sin(np.radians(fov) / 2) * 1.05
    sample = means[np.random.default_rng(0).choice(len(means), min(5000, len(means)), replace=False)]

    def frame(target, pts, ray, fill_to=0.85):
        d = dist
        for _ in range(5):  # move in until the points fill fill_to of the frame
            V = _look_at(target + d * ray, target)
            x = pts @ V[:3, :3].T + V[:3, 3]
            u = (x @ K.T)[:, :2] / x[:, 2:]
            lo_u, hi_u = np.percentile(u, 1, axis=0), np.percentile(u, 99, axis=0)
            fill = max(abs(lo_u[0] - K[0, 2]) / (width / 2), abs(hi_u[0] - K[0, 2]) / (width / 2),
                       abs(lo_u[1] - K[1, 2]) / (height / 2), abs(hi_u[1] - K[1, 2]) / (height / 2))
            d *= fill / fill_to
        return _look_at(target + d * ray, target)
    rays = [np.array([np.cos(np.radians(35)) * np.cos(np.radians(az)), np.cos(np.radians(35)) * np.sin(np.radians(az)),
                      np.sin(np.radians(35))]) for az in (45.0, 135.0, 225.0, 315.0)]
    viewmats = [frame(c, sample, ray) for ray in rays]
    rgbs = [tinted(range(1, len(joints) + 1))] * len(viewmats)
    marks = [list(range(len(joints)))] * len(viewmats)
    for k in range(len(joints)):  # a close-up per joint: its part and its axis, from the side showing them largest
        part = means[labels == k + 1]
        if not len(part):
            continue
        pts = np.concatenate([part, np.asarray(joints[k]["origin"], float)[None].repeat(len(part) // 4 + 1, 0)])
        best = max(rays, key=lambda ray: np.ptp((pts @ _look_at(c + dist * ray, c)[:3, :3].T)[:, :2], 0).prod())
        target = (np.percentile(pts, 5, axis=0) + np.percentile(pts, 95, axis=0)) / 2
        viewmats.append(frame(target, pts, best, 0.6))
        rgbs.append(tinted([k + 1], CLOSE_TINT))  # lighter: what the part is must stay visible
        marks.append([k])
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, "input.npz")
    np.savez(inp, means=means, quats=np.stack([v[f"rot_{i}"] for i in range(4)], 1),
             scales=np.stack([v[f"scale_{i}"] for i in range(3)], 1), opacities=v["opacity"], rgbs=np.array(rgbs),
             viewmats=np.array(viewmats), Ks=np.repeat(K[None], len(viewmats), 0), width=width, height=height)
    try:
        (token or jobs.Token()).run([os.path.join(splat.NS_BIN, "python"), os.path.join(
            os.path.dirname(__file__), "export", "rig_views.py"), inp, out_dir], env=splat._env())
    finally:
        os.remove(inp)
    paths = []
    for k, (V, mk) in enumerate(zip(viewmats, marks)):
        p = os.path.join(out_dir, f"view_{k}.png")
        img = cv2.imread(p)

        def px(P):
            x = V[:3, :3] @ P + V[:3, 3]
            u = K @ x
            return int(u[0] / u[2]), int(u[1] / u[2])
        for jk in mk:
            j = joints[jk]
            o, ax = np.asarray(j["origin"], float), np.asarray(j["axis"], float)
            ax = ax / np.linalg.norm(ax)
            half = 0.12 * radius
            cv2.line(img, px(o - half * ax), px(o + half * ax), (0, 0, 0), 7)
            cv2.line(img, px(o - half * ax), px(o + half * ax), (255, 255, 255), 3)
            u, w = px(o + half * ax)
            for th, colr in ((5, (0, 0, 0)), (2, (255, 255, 255))):
                cv2.putText(img, f"J{jk + 1}", (u + 6, w - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colr, th)
        cv2.imwrite(p, img)
        paths.append(p)
    photo = _best_photo(project, means)
    if photo:
        out = os.path.join(out_dir, "photo.jpg")
        im = cv2.imread(photo)
        s = 1024 / max(im.shape[:2])
        cv2.imwrite(out, cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        paths.append(out)
    on_line(f"vision model: {len(paths)} images")
    return paths


def rig_colour(k):
    """RGB 0-255 of part k >= 1 (gui.rig.PART_COLOURS order)."""
    from modelforge.gui.rig import PART_COLOURS
    return tuple(int(round(x * 255)) for x in PART_COLOURS[k % len(PART_COLOURS)])


def _best_photo(project, means):
    """The transforms.json photo in which the object is whole and largest."""
    t = json.load(open(os.path.join(project, "transforms.json")))
    views_, Ks, W, H = splat_mesh.cameras(project, 640)
    pts = means[np.random.default_rng(0).choice(len(means), min(4000, len(means)), replace=False)]
    best, score = None, 0.0
    for f, V, K in zip(t["frames"], views_, Ks):
        x = pts @ V[:3, :3].T + V[:3, 3]
        if (x[:, 2] <= 0).any():
            continue
        u = (x @ K.T)[:, :2] / x[:, 2:]
        inside = ((u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)).mean()
        area = np.ptp(u[:, 0]) * np.ptp(u[:, 1]) / (W * H)
        if inside > 0.97 and area > score:
            best, score = os.path.join(project, f["file_path"]), area
    return best


PROMPT = """Images 1-4 are renders of one machine, photographed by a drone at a quarry, seen from four sides on a grey background. {closeups}The last image is a real photo of the machine in place.

The machine has been cut into rigid parts. The parts in their natural colours form the base, which does not move. Each tinted part is attached by a joint; a thick white line (black outline) marks the joint's axis and is labelled J1, J2, ...:
{joints}
Angles are in degrees; 0 is the pose shown in the images. For a hinge with a horizontal axis a positive angle raises the part's far end and a negative one lowers it; for a vertical axis a positive angle turns the part counter-clockwise seen from above; for a slide a positive value moves it outwards.

First think, in a few sentences each:
1. What kind of machine is this (be specific)?
2. For each joint: what is its tinted part (name it by what it is, not by its colour), and how does such a part move on this kind of machine, for example when it is set up for work or folded for transport? Is the marked axis where and how it really turns?
3. Which 2 or 3 motions does this machine really perform, and in which order do its parts move? Mind the sign convention above: folding a conveyor up is a positive angle.

Then write a line "JSON:" followed by JSON only:
{{"machine": "<specific name>",
 "joints": [{{"joint": "J1", "part": "<what the part is>", "function": "<what moving it does>", "axis_ok": <true or false>, "suggested_axis": "<if false: e.g. vertical, for slewing>"}}],
 "motions": [{{"name": "<short name>", "description": "<one sentence>", "steps": [{{"seconds": <duration of this step>, "angles": {{"J1": <angle at the end of the step>}}}}]}}]}}
Every motion starts from the pose shown (all angles 0); joints not named in a step keep their angle; stay within each joint's range; a motion may end in a new pose (e.g. folded) or return to 0."""


def _joint_lines(joints, labels=None):
    lines = []
    for k, j in enumerate(joints):
        if labels is not None and not (labels == k + 1).any():
            lines.append(f"- J{k + 1}: its part is empty (nothing moves): ignore it")
            continue
        kind = ("a slide" if j["type"] == "prismatic" else
                "a hinge with a vertical axis" if abs(np.asarray(j["axis"])[2]) > 0.7 else
                "a hinge with a horizontal axis")
        lo, hi = j["limits"]
        unit = "m" if j["type"] == "prismatic" else "degrees"
        lines.append(f"- J{k + 1}: {kind}; it moves the {COLOUR_NAMES[(k + 1) % len(COLOUR_NAMES)]} part; "
                     f"range {lo:g} to {hi:g} {unit}")
    return "\n".join(lines)


def parse(text):
    """The JSON object in the model's answer (code fences and stray text around it tolerated)."""
    text = text.split("JSON:")[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("the vision model gave no JSON")
    return json.loads(m.group(0))


def motions(answer, joints):
    """The answer's motions as keyframe tracks [{name, description, keys}] in the rig's angles (signs, limits)."""
    sg = signs(joints)
    out = []
    for mo in answer.get("motions", []):
        vals, t, keys = [0.0] * len(joints), 0.0, [(0.0, [0.0] * len(joints))]
        for st in mo.get("steps", []):
            for name, ang in (st.get("angles") or {}).items():
                m = re.fullmatch(r"\s*J(\d+)\s*", str(name))
                if not m or not 1 <= int(m.group(1)) <= len(joints):
                    continue
                k = int(m.group(1)) - 1
                lo, hi = joints[k]["limits"]
                try:
                    vals[k] = float(np.clip(sg[k] * float(ang), lo, hi))
                except (TypeError, ValueError):
                    continue
            t += max(float(st.get("seconds") or 1.0), 0.3)
            keys.append((t, list(vals)))
        if len(keys) > 1 and any(v != keys[0][1] for _, v in keys):  # a "motion" that moves nothing is left out
            out.append({"name": str(mo.get("name") or f"motion {len(out) + 1}")[:40],
                        "description": str(mo.get("description") or ""), "keys": keys})
    return out


def plan(project, name, on_line=print, token=None):
    """Ask the vision model about the rig (module doc); store and return (answer, motions)."""
    why = available()
    if why:
        raise RuntimeError(why)
    token = token or jobs.Token()
    r = rig.load(project, name)
    if not r["joints"]:
        raise RuntimeError("the rig has no joints yet")
    work = os.path.join(rig.rig_dir(project, name), "vlm")
    paths = views(project, name, work, on_line=on_line, token=token)
    labels = rig.arrays(project, name)[1]
    empty = [k for k in range(len(r["joints"])) if not (labels == k + 1).any()]
    close = "".join(f"Image {5 + i} is a close-up of joint J{k + 1} with only its part tinted. "
                    for i, k in enumerate(k for k in range(len(r["joints"])) if k not in empty))
    prompt = PROMPT.format(closeups=close, joints=_joint_lines(r["joints"], labels))
    req = os.path.join(work, "request.json")
    json.dump({"model": model_path(), "images": paths, "prompt": prompt, "max_new_tokens": 2500,
               "max_pixels": 640 * 28 * 28}, open(req, "w"))
    ans = os.path.join(work, "answer.txt")
    on_line("vision model: thinking (~1 min)")
    log = []
    try:
        token.run([os.path.join(rig_unirig.HOME, "venv", "bin", "python"), os.path.join(
            os.path.dirname(__file__), "export", "vlm_ask.py"), req, ans], log.append, env=rig_unirig._env())
    except RuntimeError as e:
        raise RuntimeError(f"the vision model failed: {e}") from None
    finally:
        open(os.path.join(work, "log.txt"), "w").write("\n".join(log))
    answer = parse(open(ans).read())
    mos = motions(answer, r["joints"])
    r = rig.load(project, name)  # the user may have moved sliders meanwhile
    r["vlm"] = {"model": MODEL, "machine": answer.get("machine"), "joints": answer.get("joints", [])}
    r["motions"] = [{**m, "keys": [[t, v] for t, v in m["keys"]]} for m in mos]
    rig.save(project, name, r)
    on_line(f"vision model: {answer.get('machine')}; {len(mos)} motion(s)")
    return answer, mos


# ---------------------------------------------------------------------------------------------- template + pointing
TEMPLATE_PROMPT = """These photos show one machine (cropped around it). Describe its kinematic structure: its rigid \
parts in a chain from the part standing on the ground, and the joint between each part and the part it is mounted \
on. Use these joint types: "slew" (turns about a vertical axis, e.g. an excavator's upper body on its tracks), \
"hinge" (turns about a horizontal pin, e.g. a boom), "slide" (telescopic). For each joint give "pin": a short \
phrase that names the spot where it is, so that someone can point at it in a photo (e.g. "the pin at the foot of \
the boom, where it joins the upper body").

Example, for a crawler crane (a different machine): parts ["crawler tracks", "upper body", "boom", "jib"]; joints:
"swing" slew, crawler tracks -> upper body, pin "the centre of the slewing ring between the tracks and the upper
body"; "boom" hinge, upper body -> boom, pin "the pin at the foot of the boom on the upper body"; "jib" hinge,
boom -> jib, pin "the pin at the tip of the boom where the jib starts". Every part that can move on its own gets
its own joint; parts that are bolted together (a cab on the upper body) are one part.

Answer with JSON only:
{"machine": "<name>", "parts": ["<part on the ground>", "<next part>", ...],
 "joints": [{"name": "<short name>", "type": "slew|hinge|slide", "parent": "<part>", "child": "<part>", \
"pin": "<where the pin is>"}]}"""

POINT_PROMPT = """This is {machine}. Point to {target}. Answer only with JSON: [{{"point_2d": [x, y], \
"label": "..."}}]"""


def photo_views(project, name, out_dir, n=6, width=1600, min_angle=40.0, token=None):
    """Up to n project photos that show the rig's object whole and large, from directions >= min_angle apart: each
    a crop around the object (written to out_dir) with its camera and the object's depth rendered from it (the object
    alone). Returns [{"path", "V", "K" (of the crop), "depth" (crop, 0 = not the object)}]."""
    import cv2
    import plyfile
    t = json.load(open(os.path.join(project, "transforms.json")))
    views, Ks, W, H = splat_mesh.cameras(project, width)
    idx = rig.arrays(project, name)[0]
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data[idx]
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    pts = means[np.random.default_rng(0).choice(len(means), min(3000, len(means)), replace=False)]
    centre = np.median(means[:, :2], 0)
    cands = []
    for i, (V, K) in enumerate(zip(views, Ks)):
        x = pts @ V[:3, :3].T + V[:3, 3]
        if (x[:, 2] <= 0).any():
            continue
        u = (x @ K.T)[:, :2] / x[:, 2:]
        if ((u[:, 0] < 0) | (u[:, 0] >= W) | (u[:, 1] < 0) | (u[:, 1] >= H)).mean() > 0.02:
            continue
        lo, hi = np.percentile(u, 1, axis=0), np.percentile(u, 99, axis=0)
        eye = -V[:3, :3].T @ V[:3, 3]
        cands.append(((hi - lo).prod(), i, lo, hi, np.arctan2(*(eye[:2] - centre)[::-1])))
    cands.sort(key=lambda c: -c[0])
    chosen = []
    for c in cands:
        if all(abs((c[4] - o[4] + np.pi) % (2 * np.pi) - np.pi) > np.radians(min_angle) for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    if not chosen:
        raise RuntimeError("no photo shows the whole object")
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, "depth_input.npz")
    np.savez(inp, means=means, quats=np.stack([v[f"rot_{i}"] for i in range(4)], 1),
             scales=np.stack([v[f"scale_{i}"] for i in range(3)], 1), opacities=v["opacity"],
             rgbs=np.ones((len(chosen), len(means), 3), np.float32), viewmats=np.array([views[c[1]] for c in chosen]),
             Ks=np.array([Ks[c[1]] for c in chosen]), width=W, height=H)
    try:
        (token or jobs.Token()).run([os.path.join(splat.NS_BIN, "python"), os.path.join(
            os.path.dirname(__file__), "export", "rig_views.py"), inp, os.path.join(out_dir, "depth")], env=splat._env())
    finally:
        os.remove(inp)
    out = []
    for k, (_, i, lo, hi, _) in enumerate(chosen):
        img = cv2.imread(os.path.join(project, t["frames"][i]["file_path"]))
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        m = 0.15 * (hi - lo)
        x0, y0 = np.maximum(lo - m, 0).astype(int)
        x1, y1 = np.minimum(hi + m, [W, H]).astype(int)
        path = os.path.join(out_dir, f"photo_{k}.jpg")
        cv2.imwrite(path, img[y0:y1, x0:x1])
        K = Ks[i].copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        depth = np.load(os.path.join(out_dir, "depth", f"view_{k}_depth.npy")).astype(np.float32)[y0:y1, x0:x1]
        out.append({"path": path, "V": views[i], "K": K, "depth": depth, "size": (x1 - x0, y1 - y0)})
    return out


def lift(view, x, y, search=15):
    """The 3D point (world) under pixel (x, y) of a photo view: the object's depth there (the nearest pixel with
    depth within `search` px), or None."""
    d = view["depth"]
    h, w = d.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    y0, y1, x0, x1 = max(0, yi - search), min(h, yi + search + 1), max(0, xi - search), min(w, xi + search + 1)
    win = d[y0:y1, x0:x1]
    ys, xs = np.nonzero(win > 0)
    if not len(ys):
        return None
    k = np.argmin((ys + y0 - yi) ** 2 + (xs + x0 - xi) ** 2)
    z = float(np.median(win[max(0, ys[k] - 2):ys[k] + 3, max(0, xs[k] - 2):xs[k] + 3][
        win[max(0, ys[k] - 2):ys[k] + 3, max(0, xs[k] - 2):xs[k] + 3] > 0]))
    ray = np.linalg.solve(view["K"], [xi, yi, 1.0])
    X = ray * z
    R, tt = view["V"][:3, :3], view["V"][:3, 3]
    return R.T @ (X - tt)


def consensus_point(points, radius=1.0):
    """The point most views agree on: the largest group within `radius` metres of one of them, averaged; (point,
    views agreeing) or (None, 0)."""
    P = np.array([p for p in points if p is not None])
    if not len(P):
        return None, 0
    near = np.linalg.norm(P[:, None] - P[None], axis=2) < radius
    best = int(near.sum(1).argmax())
    return P[near[best]].mean(0), int(near[best].sum())


def _point_of(answer, size, crop):
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", answer)
    if not m:
        return None
    sw, sh = size
    return float(m.group(1)) * crop[0] / sw, float(m.group(2)) * crop[1] / sh


def _ask(queries, work, token, on_line, name="points"):
    req = os.path.join(work, f"{name}_request.json")
    ans = os.path.join(work, f"{name}_answers.json")
    json.dump({"model": model_path(), "queries": queries, "max_pixels": 1024 * 28 * 28}, open(req, "w"))
    log = []
    try:
        token.run([os.path.join(rig_unirig.HOME, "venv", "bin", "python"), os.path.join(
            os.path.dirname(__file__), "export", "vlm_ask.py"), req, ans],
            lambda s: (log.append(s), on_line(f"vision model: {s.split('] ')[-1]}") if "query" in s else None),
            env=rig_unirig._env())
    finally:
        open(os.path.join(work, f"{name}_log.txt"), "w").write("\n".join(log))
    return json.load(open(ans))


def template_joints(template, pins, centres):
    """Joints (rig.new_joint dicts, chain order) from a template and its 3D pins / part centres (dicts by joint
    name / part name). A slew turns about the vertical; a hinge about a horizontal axis across the direction from its
    pin to its part's centre, snapped parallel to its parent hinge when within 30° (an arm's pins are parallel)."""
    joints, joint_of, axis_of = [], {}, {}
    for jt in template["joints"]:
        pin, seed = pins.get(jt["name"]), centres.get(jt["child"])
        if pin is None or seed is None:
            continue
        kind = jt.get("type", "hinge")
        if kind == "slew":
            axis = np.array([0.0, 0.0, 1.0])
        else:
            d = seed - pin
            d[2] = 0.0
            if np.linalg.norm(d) < 1e-3:
                d = np.array([1.0, 0.0, 0.0])
            axis = np.cross(d / np.linalg.norm(d), [0.0, 0.0, 1.0])
            prev = axis_of.get(jt["parent"])
            if prev is not None and abs(prev @ axis) > np.cos(np.radians(30)):
                axis = prev * np.sign(prev @ axis)
        lim = [-180.0, 180.0] if kind == "slew" else ([-2.0, 2.0] if kind == "slide" else [-60.0, 60.0])
        j = rig.new_joint(pin, pin + axis, seed, parent=joint_of.get(jt["parent"], 0),
                          kind="prismatic" if kind == "slide" else "revolute", name=str(jt["name"])[:30], limits=lim)
        if kind == "slew":
            j["slew"] = True  # its part is cut across the axis (rig.assign_parts)
        joints.append(j)
        joint_of[jt["child"]] = len(joints)
        if kind == "hinge":
            axis_of[jt["child"]] = axis
    return joints


def template(project, name, on_line=print, token=None):
    """The vision model's kinematic template of the rig's object, from up to 3 photos cropped around it
    (TEMPLATE_PROMPT): {"machine", "parts", "joints": [{"name", "type", "parent", "child", "pin"}]}; stored as
    rig/<name>/vlm/template.json."""
    why = available()
    if why:
        raise RuntimeError(why)
    token = token or jobs.Token()
    work = os.path.join(rig.rig_dir(project, name), "vlm")
    on_line("vision model: choosing photos")
    views_ = photo_views(project, name, os.path.join(work, "photos"), n=3, token=token)
    on_line("vision model: the machine's structure (~30 s)")
    tmpl = parse(_ask([{"images": [v["path"] for v in views_], "prompt": TEMPLATE_PROMPT, "max_new_tokens": 800}],
                      work, token, on_line, "template")[0]["answer"])
    joints = []
    for j in tmpl.get("joints", []):
        if isinstance(j, dict) and j.get("child") and j.get("parent"):
            joints.append({"name": str(j.get("name") or j["child"])[:30], "type": str(j.get("type", "hinge")),
                           "parent": str(j["parent"]), "child": str(j["child"]),
                           "pin": str(j.get("pin") or f"the joint between the {j['parent']} and the {j['child']}")})
    tmpl = {"machine": str(tmpl.get("machine") or "a machine"), "parts": tmpl.get("parts", []), "joints": joints}
    json.dump(tmpl, open(os.path.join(work, "template.json"), "w"), indent=1)
    return tmpl


def rig_from_template(project, name, on_line=print, token=None, n_views=6):
    """The vision model's template of the object, its pins and parts pointed in photos and lifted to 3D, as joints
    (module doc). Stores rig/<name>/vlm/template.json; returns (joints, info)."""
    why = available()
    if why:
        raise RuntimeError(why)
    token = token or jobs.Token()
    work = os.path.join(rig.rig_dir(project, name), "vlm")
    on_line("vision model: choosing photos")
    views_ = photo_views(project, name, os.path.join(work, "photos"), n=n_views, token=token)
    crops = [v["path"] for v in views_[:3]]
    on_line("vision model: the machine's structure")
    tmpl = parse(_ask([{"images": crops, "prompt": TEMPLATE_PROMPT, "max_new_tokens": 800}], work, token, on_line,
                      "template")[0]["answer"])
    machine = tmpl.get("machine") or "a machine"
    targets = [("pin", j["name"], j.get("pin") or f"the joint between the {j['parent']} and the {j['child']}")
               for j in tmpl.get("joints", [])]
    targets += [("part", p, f"the middle of the {p}") for p in {j["child"] for j in tmpl.get("joints", [])}]
    queries = [{"images": [v["path"]], "prompt": POINT_PROMPT.format(machine=machine, target=t[2]),
                "max_new_tokens": 60} for v in views_ for t in targets]
    on_line(f"vision model: pointing ({len(queries)} questions)")
    answers = _ask(queries, work, token, on_line)
    pins, centres, support = {}, {}, {}
    for ti, (kind, key, _) in enumerate(targets):
        pts = []
        for vi, v in enumerate(views_):
            a = answers[vi * len(targets) + ti]
            px = _point_of(a["answer"], a["sizes"][0], v["size"])
            pts.append(lift(v, *px) if px else None)
        p, n = consensus_point(pts)
        support[f"{kind}:{key}"] = n
        if p is not None:
            (pins if kind == "pin" else centres)[key] = p
    joints = template_joints(tmpl, pins, centres)
    info = {"template": tmpl, "support": support, "views": len(views_),
            "pins": {k: v.tolist() for k, v in pins.items()}, "centres": {k: v.tolist() for k, v in centres.items()}}
    json.dump(info, open(os.path.join(work, "template.json"), "w"), indent=1)
    on_line(f"vision model: {machine}; {len(joints)} of {len(tmpl.get('joints', []))} joints placed")
    return joints, info
