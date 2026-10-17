"""Files of one project folder, for the GUI file manager and browser uploads.

Every path is relative to the project and must stay inside it (no '..', no absolute paths).
Symlinks are listed as links; deleting one removes the link, never its target.
Uploaded videos go to PROJECT/videos/ (with their .SRT sidecar), where the Ingest tab finds them.
"""

import io
import os
import shutil
import time
import zipfile

from modelforge import projects
from modelforge.dji import ingest

VIDEO_DIR = "videos"
UPLOAD_EXTS = tuple(ingest.VIDEO_EXTS) + (".srt",)
ZIP_LIMIT = 4 * 1024 ** 3  # folder downloads are zipped in memory
_STORED = (".jpg", ".jpeg", ".png", ".mp4", ".mov", ".zip", ".glb", ".splat")  # already compressed


def resolve(project, rel=""):
    """Absolute path of `rel` inside the project; ValueError if it would leave it."""
    root = os.path.abspath(project)
    path = os.path.abspath(os.path.join(root, rel))
    if os.path.isabs(rel) or (path != root and not path.startswith(root + os.sep)):
        raise ValueError(f"path outside the project: {rel}")
    return path


def listdir(project, rel=""):
    """Entries of a project folder: folders first, then files, by name."""
    d = resolve(project, rel)
    out = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        link = os.path.islink(p)
        is_dir = os.path.isdir(p) and not link
        out.append({"name": name, "rel": os.path.relpath(p, os.path.abspath(project)), "dir": is_dir,
                    "link": link, "bytes": 0 if link else projects.size(p),
                    "mtime": os.lstat(p).st_mtime})
    return sorted(out, key=lambda e: (not e["dir"], e["name"].lower()))


def delete(project, rel):
    path = resolve(project, rel)
    if path == os.path.abspath(project):
        raise ValueError("refusing to delete the project folder itself")
    if not os.path.lexists(path):
        raise FileNotFoundError(rel)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def safe_name(filename):
    """Bare file name from a browser upload; ValueError for empty / hidden / path-like names."""
    name = os.path.basename(filename.replace("\\", "/")).strip()
    if not name or name.startswith(".") or name in ("..",):
        raise ValueError(f"invalid file name: {filename!r}")
    return name


def save_upload(project, rel_dir, filename, data, overwrite=False):
    """Write uploaded bytes to PROJECT/rel_dir/<name>; returns the path. Refuses to overwrite by default."""
    d = resolve(project, rel_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, safe_name(filename))
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(f"{os.path.relpath(path, project)} already exists")
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def save_video_upload(project, filename, data, overwrite=False):
    """A browser-uploaded video or .SRT sidecar -> PROJECT/videos/."""
    if not safe_name(filename).lower().endswith(UPLOAD_EXTS):
        raise ValueError(f"not a video or .SRT file: {filename} (allowed: {', '.join(UPLOAD_EXTS)})")
    return save_upload(project, VIDEO_DIR, filename, data, overwrite)


def download_payload(project, rel):
    """(filename, bytes) for the browser: a file as is, a folder as a .zip (up to ZIP_LIMIT)."""
    path = resolve(project, rel)
    if not os.path.isdir(path):
        with open(path, "rb") as f:
            return os.path.basename(path), f.read()
    total = projects.size(path)
    if total > ZIP_LIMIT:
        raise ValueError(f"folder is {projects.human(total)}, more than the {projects.human(ZIP_LIMIT)} zip limit; "
                         "download its files one by one")
    buf = io.BytesIO()
    base = os.path.basename(path.rstrip(os.sep)) or os.path.basename(os.path.abspath(project))
    with zipfile.ZipFile(buf, "w") as z:
        for d, _, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(d, f)
                comp = zipfile.ZIP_STORED if f.lower().endswith(_STORED) else zipfile.ZIP_DEFLATED
                z.write(p, os.path.join(base, os.path.relpath(p, path)), compress_type=comp)
    return base + ".zip", buf.getvalue()


def fmt_time(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
