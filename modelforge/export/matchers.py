"""Learned features and matchers (modelforge/sfm/learned.py). Runs in build/matchers/venv (scripts/install_matchers):
LightGlue, GIM, RoMa on nerfstudio's torch.

    python matchers.py REQUEST.json

REQUEST.json {"op": ..., "method": "aliked-lightglue" | "gim-lightglue", ...}:
- extract: "images_dir", "names", "out_dir" -> out_dir/<name>.npz (keypoints float32 (N, 2) in pixels, descriptors
  float16 (N, D), size [w, h], stamp: the photo's size + mtime); a photo whose file has its current stamp is taken
  from it (the cache SfM and Register more photos share).
- match: "feat_dir", "pairs" (a pairs.txt), "db" (a COLMAP database holding the images) -> LightGlue matches written
  into its matches table, every pair (also those with none).
- roma: "images_dir", "pairs" [[a, b], ...], "out_dir", "num" -> out_dir/roma_<k>.npz per block of BLOCK pairs (a, b,
  offsets, ka, kb float32 pixels, certainty); blocks already written are skipped.
- query: "feat_dir", "images_dir" (the queries'), "queries" [{"image", "model": [names]}], "out" -> npz: per query i
  q<i>_kp, and per model image j q<i>_m<j> (query index, model keypoint index). The queries' features come from /
  go to the same cache.
Progress lines: "[matchers] <op> k / n".
"""
import json
import os
import sqlite3
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch

DEV = torch.device("cuda")
HOME = os.environ.get("MODELFORGE_MATCHERS") or os.path.join(  # build/matchers
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "build", "matchers")
GIM = os.path.join(HOME, "src", "gim")
PAIR_ID = 2147483647
BLOCK = 200
CACHE = 1500  # feature sets kept on the GPU while matching (~1 MB each)


def say(op, k, n, extra=""):
    print(f"[matchers] {op} {k} / {n}" + (f" {extra}" if extra else ""), flush=True)


# ------------------------------------------------------------------------------------------------- the two methods
class Aliked:
    dims = 128

    def __init__(self):
        import lightglue
        self.ext = lightglue.ALIKED(max_num_keypoints=4096).eval().to(DEV)
        self.lg = None

    @torch.no_grad()
    def extract(self, path):
        from lightglue.utils import load_image
        f = self.ext.extract(load_image(path).to(DEV), resize=None)
        return f["keypoints"][0].cpu().numpy(), f["descriptors"][0].cpu().numpy(), f["image_size"][0].cpu().numpy()

    @torch.no_grad()
    def match(self, a, b):
        import lightglue
        if self.lg is None:
            self.lg = lightglue.LightGlue(features="aliked").eval().to(DEV)
        out = self.lg({"image0": a, "image1": b})
        return out["matches"][0].cpu().numpy()

    @staticmethod
    def pack(kp, desc, size):
        return {"keypoints": kp[None], "descriptors": desc[None], "image_size": size[None]}


class Gim:
    dims = 256

    def __init__(self):
        sys.path.insert(0, GIM)
        from networks.lightglue.superpoint import SuperPoint
        self.det = SuperPoint({"max_num_keypoints": 2048, "force_num_keypoints": True, "detection_threshold": 0.0,
                               "nms_radius": 3, "trainable": False})
        sd = torch.load(os.path.join(GIM, "weights", "gim_lightglue_100h.ckpt"), map_location="cpu")
        self.sd = sd.get("state_dict", sd)
        self.det.load_state_dict({k[len("superpoint."):]: v for k, v in self.sd.items() if k.startswith("superpoint.")})
        self.det = self.det.eval().to(DEV)
        self.lg = None

    @torch.no_grad()
    def extract(self, path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        g = torch.from_numpy(im[: h // 8 * 8, : w // 8 * 8] / 255.0).float()[None, None].to(DEV)
        p = self.det({"image": g})
        size = np.array([g.shape[-1], g.shape[-2]], np.float32)  # what GIM's LightGlue normalizes by
        return p["keypoints"][0].cpu().numpy(), p["descriptors"][0].cpu().numpy(), size

    @torch.no_grad()
    def match(self, a, b):
        if self.lg is None:
            from networks.lightglue.models.matchers.lightglue import LightGlue
            self.lg = LightGlue({"filter_threshold": 0.1, "flash": False, "checkpointed": True})
            self.lg.load_state_dict({k[len("model."):]: v for k, v in self.sd.items() if k.startswith("model.")})
            self.lg = self.lg.eval().to(DEV)
        data = {"keypoints0": a["keypoints"], "descriptors0": a["descriptors"], "image_size0": a["image_size"],
                "keypoints1": b["keypoints"], "descriptors1": b["descriptors"], "image_size1": b["image_size"]}
        return self.lg(data)["matches"][0].cpu().numpy()

    @staticmethod
    def pack(kp, desc, size):
        return {"keypoints": kp[None], "descriptors": desc[None], "image_size": size[None]}


METHODS = {"aliked-lightglue": Aliked, "gim-lightglue": Gim}


def to_gpu(d, model):
    return model.pack(torch.from_numpy(d["keypoints"]).float().to(DEV),
                      torch.from_numpy(d["descriptors"]).float().to(DEV),
                      torch.from_numpy(np.asarray(d["size"], np.float32)).to(DEV))


class Features:
    """Feature sets from feat_dir, an LRU cache on the GPU."""

    def __init__(self, feat_dir, model):
        self.dir, self.model, self.cache = feat_dir, model, OrderedDict()

    def __getitem__(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        d = np.load(os.path.join(self.dir, name + ".npz"))
        f = to_gpu(d, self.model)
        self.cache[name] = f
        if len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return f


# ------------------------------------------------------------------------------------------------- operations
def stamp(path):
    """The photo's size and modification time: a cached feature set is valid while they are the same (the photos in
    use are hard links to the extracted pool's, so both share one)."""
    st = os.stat(path)
    return np.array([st.st_size, st.st_mtime], np.float64)


def cached(feat_dir, name, path):
    """The photo's cached features (keypoints, descriptors, size) if they belong to the photo as it is, else None."""
    f = os.path.join(feat_dir, name + ".npz")
    if not os.path.exists(f):
        return None
    try:
        d = np.load(f)
        if "stamp" not in d.files or not np.array_equal(d["stamp"], stamp(path)):
            return None
        return d["keypoints"], d["descriptors"], d["size"]
    except (OSError, ValueError, KeyError):  # a broken file: extract again
        return None


def extract_cached(model, feat_dir, name, path):
    """(keypoints, descriptors, size, from the cache?): the cache's, else extracted and saved there."""
    c = cached(feat_dir, name, path)
    if c is not None:
        return (*c, True)
    kp, desc, size = model.extract(path)
    kp, desc, size = kp.astype(np.float32), desc.astype(np.float16), np.asarray(size, np.float32)
    os.makedirs(os.path.dirname(os.path.join(feat_dir, name)) or feat_dir, exist_ok=True)
    dst = os.path.join(feat_dir, name + ".npz")
    tmp = dst + ".part.npz"
    np.savez(tmp, keypoints=kp, descriptors=desc, size=size, stamp=stamp(path))
    os.replace(tmp, dst)
    return kp, desc, size, False


def op_extract(req):
    model = METHODS[req["method"]]()
    os.makedirs(req["out_dir"], exist_ok=True)
    names, hits = req["names"], 0
    for k, n in enumerate(names):
        hits += extract_cached(model, req["out_dir"], n, os.path.join(req["images_dir"], n))[3]
        if k % 20 == 0 or k == len(names) - 1:
            say("extract", k + 1, len(names), n)
    print(f"[matchers] cached {hits} of {len(names)}", flush=True)


def read_pairs(path):
    out = []
    for line in open(path):
        s = line.split()
        if len(s) >= 2:
            out.append((s[0], s[1]))
    return out


def op_match(req):
    model = METHODS[req["method"]]()
    feats = Features(req["feat_dir"], model)
    pairs = sorted(read_pairs(req["pairs"]))  # grouped by the first photo: few cache misses
    con = sqlite3.connect(req["db"], timeout=60)
    ids = {n: i for i, n in con.execute("SELECT image_id, name FROM images")}
    rows = []

    def flush():
        con.executemany("INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)", rows)
        con.commit()
        rows.clear()
    for k, (a, b) in enumerate(pairs):
        m = model.match(feats[a], feats[b]).astype(np.uint32).reshape(-1, 2)
        i, j = ids[a], ids[b]
        if i > j:
            i, j, m = j, i, m[:, ::-1]
        rows.append((i * PAIR_ID + j, len(m), 2, np.ascontiguousarray(m).tobytes()))
        if len(rows) >= 500:
            flush()
        if k % 200 == 0 or k == len(pairs) - 1:
            say("match", k + 1, len(pairs))
    flush()
    con.close()


def op_roma(req):
    from romatch import roma_outdoor
    model = roma_outdoor(device=DEV, use_custom_corr=False)  # local_corr (a CUDA extension) is not installed
    pairs, num = req["pairs"], int(req.get("num", 2000))
    os.makedirs(req["out_dir"], exist_ok=True)
    size = {}

    def wh(p):  # (w, h) from the header
        if p not in size:
            from PIL import Image
            with Image.open(p) as im:
                size[p] = im.size
        return size[p]
    for blk in range(0, len(pairs), BLOCK):
        dst = os.path.join(req["out_dir"], f"roma_{blk // BLOCK:05d}.npz")
        if os.path.exists(dst):
            continue
        A, B, KA, KB, C, off = [], [], [], [], [], [0]
        for a, b in pairs[blk: blk + BLOCK]:
            pa, pb = os.path.join(req["images_dir"], a), os.path.join(req["images_dir"], b)
            (wa, ha), (wb, hb) = wh(pa), wh(pb)
            with torch.no_grad():
                warp, cert = model.match(pa, pb, device=DEV)
                m, c = model.sample(warp, cert, num=num)
                ka, kb = model.to_pixel_coordinates(m, ha, wa, hb, wb)
            A.append(a), B.append(b)
            KA.append(ka.cpu().numpy().astype(np.float32)), KB.append(kb.cpu().numpy().astype(np.float32))
            C.append(c.cpu().numpy().astype(np.float16))
            off.append(off[-1] + len(ka))
        tmp = dst + ".part.npz"
        np.savez(tmp, a=np.array(A), b=np.array(B), offsets=np.array(off, np.int64), ka=np.concatenate(KA),
                 kb=np.concatenate(KB), certainty=np.concatenate(C))
        os.replace(tmp, dst)
        say("roma", min(blk + BLOCK, len(pairs)), len(pairs))


def op_query(req):
    model = METHODS[req["method"]]()
    feats = Features(req["feat_dir"], model)
    out = {}
    hits = 0
    for i, q in enumerate(req["queries"]):
        kp, desc, size, hit = extract_cached(model, req["feat_dir"], q["image"],
                                             os.path.join(req["images_dir"], q["image"]))
        hits += hit
        f = model.pack(torch.from_numpy(kp).float().to(DEV), torch.from_numpy(desc).float().to(DEV),
                       torch.from_numpy(np.asarray(size, np.float32)).to(DEV))
        out[f"q{i}_kp"] = kp.astype(np.float32)
        for j, name in enumerate(q["model"]):
            out[f"q{i}_m{j}"] = model.match(f, feats[name]).astype(np.int64).reshape(-1, 2)
        say("query", i + 1, len(req["queries"]))
    np.savez(req["out"], **out)
    print(f"[matchers] cached {hits} of {len(req['queries'])}", flush=True)


def main():
    req = json.load(open(sys.argv[1]))
    {"extract": op_extract, "match": op_match, "roma": op_roma, "query": op_query}[req["op"]](req)


if __name__ == "__main__":
    main()
