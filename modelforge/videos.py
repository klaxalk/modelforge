"""The video library: DJI clips (+ .SRT sidecars) under the data root, shared by all projects.

Videos are found anywhere under the root except inside project folders (ingest.find_videos);
browser uploads go to ROOT/videos/. A clip is "used" by every project whose ingest.json lists
it as a source; renaming or deleting a used clip is refused, so no plan points at a missing file.
Renames and deletes move the .SRT sidecar along with the video.

The GUI Videos tab is a file manager over the root: folders (create / delete), videos (upload,
rename, delete, download) and stray .SRT files. Project folders and hidden files are not part of
the library and are never listed, entered or deleted here (the Projects tab handles projects).
"""

import json
import os
import re
import shutil
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


def save_upload(root, filename, data, overwrite=False, rel_dir=LIBRARY_DIR):
    """A browser-uploaded video or .SRT -> ROOT/rel_dir/<name> (via a temp file)."""
    name = safe_name(filename)
    if not name.lower().endswith(UPLOAD_EXTS):
        raise ValueError(f"not a video or .SRT file: {name} (allowed: {', '.join(UPLOAD_EXTS)})")
    d = resolve_dir(root, rel_dir, create=True)
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


# --- file manager: folders of the library ------------------------------------------------------
FOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()-]{0,63}$")


def resolve_dir(root, rel="", create=False):
    """Absolute library folder ROOT/rel; ValueError if it leaves the root, is hidden or is (inside) a
    project folder. create: make it if missing (uploads into ROOT/videos)."""
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, rel))
    if path != base and not path.startswith(base + os.sep):
        raise ValueError(f"outside the library: {rel}")
    parts = os.path.relpath(path, base).split(os.sep) if path != base else []
    d = base
    for part in parts:
        if part.startswith("."):
            raise ValueError(f"hidden folder: {rel}")
        d = os.path.join(d, part)
        if projects.is_project(d):
            raise ValueError(f"{os.path.relpath(d, base)} is a project folder (Projects tab)")
    if create:
        os.makedirs(path, exist_ok=True)
    elif not os.path.isdir(path):
        raise ValueError(f"no such folder: {rel}")
    return path


def listdir(root, rel="", with_probe=True):
    """{"folders": [...], "videos": [...], "files": [stray .SRT]} of one library folder, by name.
    Folders carry their video count and size (recursive, projects excluded)."""
    d = resolve_dir(root, rel)
    used = usage(root)
    out = {"folders": [], "videos": [], "files": []}
    names = sorted(os.listdir(d), key=str.lower)
    videos = {os.path.splitext(n)[0].lower() for n in names if n.lower().endswith(ingest.VIDEO_EXTS)}
    for n in names:
        p = os.path.join(d, n)
        if n.startswith("."):
            continue
        if os.path.isdir(p):
            if projects.is_project(p):
                continue
            vids = ingest.find_videos(p)
            out["folders"].append({"name": n, "path": p, "rel": os.path.relpath(p, os.path.realpath(root)),
                                   "videos": len(vids), "bytes": sum(os.path.getsize(v) for v in vids)})
        elif n.lower().endswith(ingest.VIDEO_EXTS):
            e = {"path": p, "name": n, "rel": os.path.relpath(p, os.path.realpath(root)),
                 "bytes": os.path.getsize(p), "mtime": os.path.getmtime(p), "srt": ingest.find_srt(p),
                 "used_by": sorted(set(used.get(os.path.realpath(p), [])))}
            if with_probe:
                e.update(probe(p))
            out["videos"].append(e)
        elif n.lower().endswith(".srt") and os.path.splitext(n)[0].lower() not in videos:
            out["files"].append({"path": p, "name": n, "bytes": os.path.getsize(p)})
    return out


def make_folder(root, rel, name):
    """New folder ROOT/rel/name; returns its path."""
    name = name.strip()
    if not FOLDER_RE.match(name):
        raise ValueError("folder names use letters, digits, spaces and '_ . ( ) -' (max 64), starting with a "
                         "letter or digit")
    path = os.path.join(resolve_dir(root, rel), name)
    if os.path.lexists(path):
        raise FileExistsError(f"{name} already exists")
    os.makedirs(path)
    return path


def delete_folder(root, rel):
    """Delete a library folder with its contents; refused for the root, for folders holding a project
    and for folders holding a video a project uses. Returns (files, bytes)."""
    path = resolve_dir(root, rel)
    if path == os.path.realpath(root):
        raise ValueError("refusing to delete the library root")
    used = usage(root)
    files = nbytes = 0
    for d, dirs, names in os.walk(path):
        if projects.is_project(d):
            raise ValueError(f"cannot delete {rel}: it contains the project folder {os.path.relpath(d, path)}")
        for n in names:
            p = os.path.join(d, n)
            users = used.get(os.path.realpath(p))
            if users:
                raise ValueError(f"cannot delete {rel}: {n} is used by project(s) {', '.join(sorted(set(users)))}")
            files += 1
            nbytes += os.path.getsize(p)
    shutil.rmtree(path)
    return files, nbytes


def delete_file(root, path):
    """Delete a stray .SRT (no video next to it) from the library."""
    p = os.path.abspath(path)
    resolve_dir(root, os.path.relpath(os.path.dirname(p), os.path.realpath(root)))
    if not p.lower().endswith(".srt") or not os.path.isfile(p):
        raise ValueError(f"not an .SRT file in the library: {path}")
    os.unlink(p)
