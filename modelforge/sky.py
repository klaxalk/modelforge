"""Sky masks for splat training (the user asked, 2026-10-24: Gaussians that represent the sky; docs/todo_sky.md).

SegFormer-B5 (ADE20K, class sky) segments every training photo (export/sky_masks.py in nerfstudio's venv with the
RADIO packages' transformers); the masks (0 = sky, 255 = the rest) are written at the training images' size (the
photo's floor(w / N) x floor(h / N), before undistortion: nerfstudio undistorts them with the photo) to
ROOT/sky/masks_x<N>/<image>.png, ROOT being the project whose images/ the dataset uses (tiles share it), kept per
photo. The lidar-data dataparser hands them to nerfstudio (--sky-masks), which leaves those pixels out of the
photometric loss; the plugin adds a loss on the rendered opacity there (sky_alpha_lambda), so nothing grows in the sky.
"""
import json
import os
import re

MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
LONG_SIDE = 1024  # px the photos are segmented at
ALPHA = 0.3  # weight of the opacity loss on the sky (its core: 1 % of the width from anything else, in the plugin);
#             berlin tile 0: 0.05 left 0.21 mean opacity over the sky, 0.3 0.16, the ground the same
DILATE = 0.004  # the sky widened by this share of the width (8 px at 1920): edge pixels learn no sky colour
_LINE = re.compile(r"sky masks: photo (\d+) / (\d+)")


DEFAULT_KINDS = ("dji",)  # project kinds with sky masks on by default (the user asked, 2026-10-24)


def default_on(project):
    """Sky masks on by default for this project: its kind (missing ingest.json = dji), and only where they can be
    made (nerfstudio's venv + transformers), so a default never fails a run."""
    from modelforge.dji import ingest
    try:
        return ingest.load_kind(project) in DEFAULT_KINDS and available() is None
    except Exception:  # noqa: BLE001 - no project yet
        return False


def available():
    """None if sky masks can be computed here, else the reason (nerfstudio's venv + transformers)."""
    from modelforge import semantic
    return semantic.available()


def root(project):
    """The project whose photos a dataset uses (a tile's images/ links to its project's)."""
    return os.path.dirname(os.path.realpath(os.path.join(project, "images")))


def folder(project, downscale):
    return os.path.join(root(project), "sky", f"masks_x{int(downscale)}")


def masks(project, downscale, on_line=print, job=None, on_count=None):
    """Sky masks of every photo of PROJECT/transforms.json at the training size (downscale N); returns the folder.
    Kept while the model and the size stay (manifest.json); only missing ones are computed."""
    from modelforge import jobs, semantic, splat
    from modelforge.export import frame_camera
    out = folder(project, downscale)
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, "manifest.json")
    key = {"model": MODEL, "long_side": LONG_SIDE, "downscale": int(downscale), "dilate": DILATE}
    if os.path.exists(man) and json.load(open(man)) != key:
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
    json.dump(key, open(man, "w"))
    t = json.load(open(os.path.join(project, "transforms.json")))
    todo = []
    for f in t["frames"]:
        cam = frame_camera.camera(t, f)
        name = os.path.basename(f["file_path"])
        dst = os.path.join(out, name + ".png")
        if not os.path.exists(dst):
            todo.append({"src": os.path.realpath(os.path.join(project, f["file_path"])), "dst": dst,
                         "w": int(cam["w"]) // int(downscale), "h": int(cam["h"]) // int(downscale)})
    n = len(t["frames"])
    if not todo:
        on_line(f"sky masks: {n} photos, all made already")
        return out
    why = available()
    if why:
        raise RuntimeError(f"sky masks: {why}")
    on_line(f"sky masks: {len(todo)} of {n} photos to segment ({MODEL.split('/')[-1]}, {LONG_SIDE} px)")
    spec = os.path.join(out, "jobs.json")
    json.dump({"model": MODEL, "long_side": LONG_SIDE, "dilate": DILATE, "jobs": todo}, open(spec, "w"))
    stats = os.path.join(out, "last_stats.json")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "sky_masks.py")

    def line(text):
        m = _LINE.search(text)
        if m and on_count:
            on_count(int(m.group(1)), int(m.group(2)))
        if not m or int(m.group(1)) % 50 == 0:
            on_line(text)
    (job or jobs.Token()).run([os.path.join(splat.NS_BIN, "python"), script, spec, stats], on_line=line,
                              env=semantic._env())
    st = json.load(open(stats))
    share = sorted(st["sky_share"].values())
    if share:
        on_line(f"sky masks: {len(share)} photos in {st['seconds']:.0f} s; sky median {100 * share[len(share) // 2]:.1f} %"
                f", max {100 * share[-1]:.0f} %, {sum(s > 0.01 for s in share)} photos with > 1 %")
    return out
