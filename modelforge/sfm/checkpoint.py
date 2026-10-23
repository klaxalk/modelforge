"""Resumable refine: a checkpoint after every finished step (pipeline.stage_refine(resume=True), CLI --resume-refine,
Sparse tab "Resume refine").

Why (2026-10-23): the kernel OOM-killed the GUI with kladno_dira's refine inside it twice; round 0 alone had taken
52 min and everything restarted from the priors.

PROJECT/sparse/refine_resume.pkl holds, after the last finished step: the camera poses (by image name), the
intrinsics, the prior model, the stats so far and the chunked refine's own state (chunk edges, start poses, ...).
No points: every step starts with a fresh triangulation at the poses. A step is a round of a global refine, a
calibration round, a chunk, the pose graph, a revisit / drift / global pass or the polish of a chunked one.
It counts only for the same inputs and settings (`key`: database.db's content, poses.json, the refine settings); a
finished refine deletes it.
"""

import copy
import os
import pickle

import numpy as np

FILE = os.path.join("sparse", "refine_resume.pkl")


def _database(path):
    """database.db by its content (images, keypoints, verified matches): COLMAP touches its mtime when it opens it,
    even for reading."""
    import sqlite3
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
        return [q("SELECT COUNT(*) FROM images")[0], q("SELECT COUNT(*), TOTAL(rows) FROM keypoints"),
                q("SELECT COUNT(*), TOTAL(rows) FROM two_view_geometries WHERE rows > 0")]
    finally:
        con.close()


def key(project, settings):
    """What a checkpoint must match: database.db's content, poses.json's size + mtime and the settings."""
    p = os.path.join(project, "poses.json")
    return {"database": _database(os.path.join(project, "database.db")),
            "poses": [os.path.getsize(p), round(os.path.getmtime(p), 3)] if os.path.exists(p) else None,
            "settings": settings}


def info(project, k):
    """The checkpoint's finished steps if it matches `k`, else None."""
    st = _load(project)
    return list(st["done"]) if st and st.get("key") == k else None


def _load(project):
    try:
        with open(os.path.join(project, FILE), "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
        return None


class Checkpoint:
    """Use: done(step) skips a finished step; restore(rec, model) puts the last finished step's state into rec / model
    before the first step that runs; save(step, rec, model, **extra) after each step."""

    def __init__(self, project, k, resume):
        self.path = os.path.join(project, FILE)
        st = _load(project) if resume else None
        if st is None or st.get("key") != k:
            st = {"key": k, "done": [], "poses": {}, "cams": {}, "model": None, "extra": {}}
        self.state, self.live, self.model_restored = st, False, False
        self.resumed_from = list(st["done"])  # the steps done before this run

    @property
    def resumed(self):
        return bool(self.state["done"]) and not self.live

    def done(self, step):
        return step in self.state["done"]

    def get(self, name, default=None):
        return copy.deepcopy(self.state["extra"].get(name, default))

    def restore(self, rec, model=None):
        """The saved poses (images rec has) and intrinsics into rec, the prior model into model (once). Nothing
        after this run has saved a step of its own: its state is newer."""
        if self.live or not self.state["done"]:
            return
        import pycolmap
        for i in rec.reg_image_ids():
            im = rec.image(i)
            p = self.state["poses"].get(im.name)
            if p is not None:
                r = pycolmap.Rigid3d()
                r.params = np.asarray(p, float)
                rec.frame(im.frame_id).rig_from_world = r
        for cid, p in self.state["cams"].items():
            if cid in rec.cameras:
                rec.camera(cid).params = np.asarray(p, float)
        if model is not None and self.state["model"] is not None and not self.model_restored:
            model.__dict__.update(copy.deepcopy(self.state["model"]).__dict__)
            self.model_restored = True

    def save(self, step, rec=None, model=None, **extra):
        st = self.state
        if rec is not None:
            for i in rec.reg_image_ids():
                im = rec.image(i)
                st["poses"][im.name] = np.array(rec.frame(im.frame_id).rig_from_world.params)
            st["cams"] = {cid: np.array(c.params) for cid, c in rec.cameras.items()}
        if model is not None:
            st["model"] = copy.deepcopy(model)
        st["extra"].update(copy.deepcopy(extra))
        st["done"].append(step)
        self.live = True
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + ".part", "wb") as f:
            pickle.dump(st, f)
        os.replace(self.path + ".part", self.path)

    def remove(self):
        try:
            os.remove(self.path)
        except OSError:
            pass
