"""The video library: DJI clips (+ .SRT sidecars) under the data root, shared by all projects.

Videos are found anywhere under the root except inside project folders (ingest.find_videos);
browser uploads go to ROOT/videos/. A clip is "used" by every project whose ingest.json lists
it as a source; renaming or deleting a used clip is refused, so no plan points at a missing file.
Renames and deletes move the .SRT sidecar along with the video.
"""

import json
import os
import subprocess

from modelforge import projects
from modelforge.dji import ingest

LIBRARY_DIR = "videos"
UPLOAD_EXTS = tuple(ingest.VIDEO_EXTS) + (".srt",)
_probe_cache = {}


def library_dir(root):
    return os.path.join(root, LIBRARY_DIR)


def _inside(root, path):
    root, path = os.path.realpath(root), os.path.realpath(path)
    return path.startswith(root + os.sep)


def check_video(root, path):
    """Absolute path of a library video; ValueError unless it is a video under root, outside projects."""
    p = os.path.abspath(path)
    if not p.lower().endswith(ingest.VIDEO_EXTS) or not os.path.isfile(p) or not _inside(root, p):
        raise ValueError(f"not a video in the library: {path}")
    d = os.path.dirname(p)
    while _inside(root, d):
        if projects.is_project(d):
            raise ValueError(f"{path} belongs to project folder {d}")
        d = os.path.dirname(d)
    return p


def probe(path):
    """{duration_s, width, height, fps, created} from ffprobe, cached by size + mtime."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime)
    if key not in _probe_cache:
        info = {"duration_s": None, "width": None, "height": None, "fps": None, "created": None}
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=width,height,avg_frame_rate:format=duration:format_tags=creation_time",
                                  "-of", "json", path], capture_output=True, text=True, timeout=30, check=True)
            d = json.loads(out.stdout)
            s = (d.get("streams") or [{}])[0]
            num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
            info.update(duration_s=float(d.get("format", {}).get("duration") or 0) or None,
                        width=s.get("width"), height=s.get("height"),
                        fps=round(float(num) / float(den or 1), 3) if float(den or 1) else None,
                        created=(d.get("format", {}).get("tags") or {}).get("creation_time"))
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
        _probe_cache[key] = info
    return _probe_cache[key]


def usage(root):
    """{realpath of video: [project names using it]} from every project's ingest.json."""
    out = {}
    for p in projects.list_projects(root):
        sources, _ = ingest.load_plan(p)
        for s in sources:
            out.setdefault(os.path.realpath(s.video), []).append(os.path.basename(p))
    return out


def list_videos(root, with_probe=True):
    """Library entries, newest file first: path, size, srt, used_by and the ffprobe info."""
    used = usage(root)
    out = []
    for v in ingest.find_videos(root):
        try:
            srt = ingest.find_srt(v)
            e = {"path": v, "name": os.path.basename(v), "rel": os.path.relpath(v, root),
                 "bytes": os.path.getsize(v), "mtime": os.path.getmtime(v), "srt": srt,
                 "used_by": sorted(set(used.get(os.path.realpath(v), [])))}
            if with_probe:
                e.update(probe(v))
        except FileNotFoundError:  # renamed or deleted while listing
            continue
        out.append(e)
    return sorted(out, key=lambda e: -e["mtime"])


def _refuse_if_used(root, path, action):
    users = usage(root).get(os.path.realpath(path))
    if users:
        raise ValueError(f"cannot {action} {os.path.basename(path)}: used by project(s) {', '.join(sorted(set(users)))}"
                         " (remove it from their sources first)")


def safe_name(filename):
    """Bare file name from a browser upload or rename; ValueError for empty / hidden names."""
    name = os.path.basename(filename.replace("\\", "/")).strip()
    if not name or name.startswith("."):
        raise ValueError(f"invalid file name: {filename!r}")
    return name


def save_upload(root, filename, data, overwrite=False):
    """A browser-uploaded video or .SRT -> ROOT/videos/<name> (via a temp file)."""
    name = safe_name(filename)
    if not name.lower().endswith(UPLOAD_EXTS):
        raise ValueError(f"not a video or .SRT file: {name} (allowed: {', '.join(UPLOAD_EXTS)})")
    d = library_dir(root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(f"{os.path.relpath(path, root)} already exists in the library")
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def rename(root, path, new_name):
    """Rename a video and its .SRT (same folder; the extension is kept). Returns the new path."""
    p = check_video(root, path)
    _refuse_if_used(root, p, "rename")
    stem = os.path.splitext(safe_name(new_name))[0]
    ext = os.path.splitext(p)[1]
    dst = os.path.join(os.path.dirname(p), stem + ext)
    srt = ingest.find_srt(p)
    srt_dst = os.path.join(os.path.dirname(p), stem + os.path.splitext(srt)[1]) if srt else None
    for t in (dst, srt_dst):
        if t and os.path.lexists(t) and os.path.abspath(t) != p:
            raise FileExistsError(f"{os.path.basename(t)} already exists")
    os.rename(p, dst)
    if srt:
        os.rename(srt, srt_dst)
    return dst


def delete(root, path):
    """Delete a video and its .SRT. Returns the deleted paths."""
    p = check_video(root, path)
    _refuse_if_used(root, p, "delete")
    gone = [p] + ([ingest.find_srt(p)] if ingest.find_srt(p) else [])
    for f in gone:
        os.unlink(f)
    return gone
