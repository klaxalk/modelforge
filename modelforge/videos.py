"""The video library: DJI clips (+ .SRT sidecars) and ROS bags under the data root, shared by all
projects. A bag is a ROS 1 .bag file, a ROS 2 .mcap file or a ROS 2 folder with metadata.yaml (rosbag.bag.is_bag);
bag folders are listed as one entry, not opened. Bags arrive as a .bag / .mcap, as an archive of a bag folder (.zip / .tar /
.tar.gz, unpacked here) or as the loose files of one (metadata.yaml + .db3 / .mcap into a folder).

The library is ROOT/videos/ (projects live next to it in ROOT, so names never clash); the Ingest tab
still finds clips anywhere under ROOT outside project folders (ingest.find_videos). A clip is "used"
by every project whose ingest.json lists it as a source: renames and moves rewrite those projects'
paths (update_references), deleting a used clip is refused. The .SRT sidecar follows its video.

The GUI Videos tab is a file manager over ROOT/videos: folders (create / delete), videos (upload,
rename, delete, download) and stray .SRT files. Project folders and hidden files are not part of
the library and are never listed, entered or deleted here (the Projects tab handles projects).
"""

import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import zipfile

from modelforge import projects
from modelforge.dji import ingest

LIBRARY_DIR = projects.LIBRARY_DIR  # ROOT/videos: the library; projects live next to it in ROOT
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")
BAG_EXTS = (".mcap", ".bag")  # a single-file bag: ROS 2 MCAP, ROS 1
BAG_PART_EXTS = (".db3", ".mcap")  # files inside a bag folder, with metadata.yaml
UPLOAD_EXTS = tuple(ingest.VIDEO_EXTS) + (".srt",) + BAG_EXTS + ARCHIVE_EXTS + (".db3", ".yaml")
_probe_cache = {}
_bag_cache = {}


def is_bag(path):
    from modelforge.rosbag import bag
    return bag.is_bag(path)


def _tree_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, n)) for d, _, ns in os.walk(path) for n in ns)


def bag_info(path):
    """{duration_s, created, topics, images, clouds, tf, odometry, error} of a bag, cached by mtime."""
    key = (os.path.abspath(path), os.path.getmtime(path), _tree_size(path))
    if key not in _bag_cache:
        from modelforge.rosbag import bag
        info = {"duration_s": None, "created": None, "topics": 0, "images": [], "clouds": [], "tf": False,
                "odometry": [], "error": None}
        try:
            inv = bag.inventory(path)
            import datetime
            info.update(duration_s=inv["end"] - inv["start"], topics=len(inv["topics"]),
                        created=datetime.datetime.fromtimestamp(inv["start"], datetime.timezone.utc).isoformat(),
                        images=bag.topics_of(inv, bag.IMAGE_TYPES), clouds=bag.topics_of(inv, bag.CLOUD_TYPE),
                        odometry=bag.topics_of(inv, bag.ODOM_TYPE), tf=bool(bag.topics_of(inv, bag.TF_TYPE)))
        except Exception as e:  # unreadable / still being copied: list it anyway
            info["error"] = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        _bag_cache[key] = info
    return _bag_cache[key]


def library_dir(root):
    return os.path.join(root, LIBRARY_DIR)


def _inside(root, path):
    root, path = os.path.realpath(root), os.path.realpath(path)
    return path.startswith(root + os.sep)


def check_video(root, path):
    """Absolute path of a library video or ROS bag; ValueError unless it is one under root, outside
    projects."""
    p = os.path.abspath(path)
    video = p.lower().endswith(ingest.VIDEO_EXTS) and os.path.isfile(p)
    if not (video or is_bag(p)) or not _inside(root, p):
        raise ValueError(f"not a video or ROS bag in the library: {path}")
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


def save_upload(root, filename, data, overwrite=False, rel_dir=""):
    """A browser-uploaded video, .SRT or ROS bag -> ROOT/videos/rel_dir/<name> (via a temp file).
    Archives (.zip / .tar / .tar.gz) of a bag folder are unpacked (extract_bags); metadata.yaml and .db3
    are parts of a bag folder (upload them into one). Returns the saved path (an unpacked archive: its
    first bag)."""
    name = safe_name(filename)
    low = name.lower()
    if not low.endswith(UPLOAD_EXTS) or (low.endswith(".yaml") and low != "metadata.yaml"):
        raise ValueError(f"not a video, .SRT or ROS bag file: {name} (allowed: {', '.join(UPLOAD_EXTS)}; "
                         ".yaml only as a bag's metadata.yaml)")
    d = resolve_dir(root, rel_dir, create=True)
    if low.endswith(ARCHIVE_EXTS):
        return extract_bags(root, name, data, d)
    path = os.path.join(d, name)
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(f"{os.path.relpath(path, root)} already exists in the library")
    tmp = path + f".part{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def _archive_stem(name):
    low = name.lower()
    ext = next(e for e in sorted(ARCHIVE_EXTS, key=len, reverse=True) if low.endswith(e))
    return name[:-len(ext)]


def extract_bags(root, name, data, dest):
    """Unpack an uploaded archive holding ROS bag(s) into the library folder dest. Loose bag files
    (metadata.yaml at the archive's top) go into a folder named after the archive; otherwise its
    top-level entries are kept. Members leaving the folder, links and devices are refused. Returns the
    first bag's path; ValueError (nothing kept) without a bag inside."""
    tmp = os.path.join(dest, f".upload{os.getpid()}-{threading.get_ident()}")
    os.makedirs(tmp)
    try:
        low = name.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.infolist():
                    target = os.path.realpath(os.path.join(tmp, m.filename))
                    if not target.startswith(os.path.realpath(tmp) + os.sep):
                        raise ValueError(f"archive member leaves the folder: {m.filename}")
                z.extractall(tmp)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                t.extractall(tmp, filter="data")  # refuses absolute / .. paths, links out, devices
        tops = [n for n in os.listdir(tmp) if not n.startswith(".") and n != "__MACOSX"]
        if "metadata.yaml" in tops:  # the bag folder's files themselves
            moves = [(tmp, os.path.join(dest, _archive_stem(name)))]
        else:
            moves = [(os.path.join(tmp, n), os.path.join(dest, n)) for n in tops]
        from modelforge.rosbag import bag
        bags = [b for src, _ in moves for b in ([src] if is_bag(src) else bag.find_bags(src)
                                                  if os.path.isdir(src) else [])]
        if not bags:
            raise ValueError(f"no ROS bag in {name} (a ROS 1 .bag, an .mcap, or a folder with metadata.yaml)")
        for _, dst in moves:
            if os.path.lexists(dst):
                raise FileExistsError(f"{os.path.relpath(dst, root)} already exists in the library")
        first = None
        for src, dst in moves:
            os.rename(src, dst)
            for b in bags:  # where the first bag ended up
                if first is None and (b == src or b.startswith(src + os.sep)):
                    first = dst + b[len(src):]
        return first
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bag_zip(path):
    """A bag folder as .zip bytes (stored: bag data rarely compresses well), for the browser download."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for d, _, names in os.walk(path):
            for n in sorted(names):
                f = os.path.join(d, n)
                z.write(f, os.path.join(os.path.basename(path), os.path.relpath(f, path)))
    return buf.getvalue()


def _tree_moves(src, dst):
    """{old realpath: new path} for a file or a folder with everything in it (folders too: a bag
    folder is a project source itself)."""
    if not os.path.isdir(src):
        return {os.path.realpath(src): dst}
    moves = {}
    for d, _, names in os.walk(src):
        moves[os.path.realpath(d)] = os.path.join(dst, os.path.relpath(d, src)) if d != src else dst
        for n in names:
            old = os.path.join(d, n)
            moves[os.path.realpath(old)] = os.path.join(dst, os.path.relpath(old, src))
    return moves


def rename(root, path, new_name):
    """Rename a video and its .SRT (same folder; the extension is kept); projects using it are updated
    (update_references). Returns the new path."""
    p = check_video(root, path)
    if os.path.isdir(p):  # a bag folder: the whole name, the folder rules
        new_name = new_name.strip()
        if not FOLDER_RE.match(new_name):
            raise ValueError("bag folder names use letters, digits, spaces and '_ . ( ) -' (max 64), starting "
                             "with a letter or digit")
        dst = os.path.join(os.path.dirname(p), new_name)
        if os.path.lexists(dst):
            raise FileExistsError(f"{new_name} already exists")
        moves = _tree_moves(p, dst)
        os.rename(p, dst)
        update_references(root, moves)
        return dst
    stem = os.path.splitext(safe_name(new_name))[0]
    ext = os.path.splitext(p)[1]
    dst = os.path.join(os.path.dirname(p), stem + ext)
    srt = _srt(p)
    srt_dst = os.path.join(os.path.dirname(p), stem + os.path.splitext(srt)[1]) if srt else None
    for t in (dst, srt_dst):
        if t and os.path.lexists(t) and os.path.abspath(t) != p:
            raise FileExistsError(f"{os.path.basename(t)} already exists")
    moves = {os.path.realpath(p): dst, **({os.path.realpath(srt): srt_dst} if srt else {})}
    os.rename(p, dst)
    if srt:
        os.rename(srt, srt_dst)
    update_references(root, moves)
    return dst


def remove_from_projects(root, path):
    """Drop every project source that is path (a clip or bag) or lies inside it (a folder) from the
    projects' ingest.json. Extracted images and results stay. Returns the projects changed."""
    p = os.path.realpath(path)
    changed = []
    for proj in projects.list_projects(root):
        sources, settings = ingest.load_plan(proj)
        keep = [x for x in sources if os.path.realpath(x.video) != p
                and not os.path.realpath(x.video).startswith(p + os.sep)]
        if len(keep) != len(sources):
            ingest.save_plan(proj, keep, settings)
            changed.append(os.path.basename(proj))
    return changed


def _srt(p):
    """The .SRT sidecar of a video (bags have none)."""
    return ingest.find_srt(p) if p.lower().endswith(ingest.VIDEO_EXTS) else None


def delete(root, path):
    """Delete a video and its .SRT, or a bag (file or folder). Returns the deleted paths."""
    p = check_video(root, path)
    _refuse_if_used(root, p, "delete")
    if os.path.isdir(p):
        shutil.rmtree(p)
        return [p]
    gone = [p] + ([_srt(p)] if _srt(p) else [])
    for f in gone:
        os.unlink(f)
    return gone


# --- file manager: folders of the library ------------------------------------------------------
FOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()-]{0,63}$")


def _lib(root):
    """The library folder ROOT/videos (absolute, created if missing): the Videos tab's root."""
    d = os.path.realpath(library_dir(root))
    os.makedirs(d, exist_ok=True)
    return d


def resolve_dir(root, rel="", create=False):
    """Absolute library folder ROOT/videos/rel; ValueError if it leaves the library, is hidden or is
    (inside) a project folder. create: make it if missing."""
    base = _lib(root)
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
    """{"folders": [...], "videos": [...], "bags": [...], "files": [stray .SRT]} of one library folder,
    by name. Folders carry their video + bag count and size (recursive, projects excluded)."""
    from modelforge.rosbag import bag
    d = resolve_dir(root, rel)
    used = usage(root)
    out = {"folders": [], "videos": [], "bags": [], "files": [], "inside_bag": is_bag(d)}
    names = sorted(os.listdir(d), key=str.lower)
    videos = {os.path.splitext(n)[0].lower() for n in names if n.lower().endswith(ingest.VIDEO_EXTS)}
    for n in names:
        p = os.path.join(d, n)
        if n.startswith("."):
            continue
        if out["inside_bag"] and n.lower().endswith(BAG_PART_EXTS):  # the bag's own storage files
            continue
        if is_bag(p):
            e = {"path": p, "name": n, "rel": os.path.relpath(p, _lib(root)), "bytes": _tree_size(p),
                 "mtime": os.path.getmtime(p), "folder": os.path.isdir(p),
                 "used_by": sorted(set(used.get(os.path.realpath(p), [])))}
            if with_probe:
                e.update(bag_info(p))
            out["bags"].append(e)
        elif os.path.isdir(p):
            if projects.is_project(p):
                continue
            vids, bags = ingest.find_videos(p), bag.find_bags(p)
            out["folders"].append({"name": n, "path": p, "rel": os.path.relpath(p, _lib(root)),
                                   "videos": len(vids) + len(bags),
                                   "bytes": sum(os.path.getsize(v) for v in vids) + sum(_tree_size(b) for b in bags)})
        elif n.lower().endswith(ingest.VIDEO_EXTS):
            e = {"path": p, "name": n, "rel": os.path.relpath(p, _lib(root)),
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
    if path == _lib(root):
        raise ValueError("refusing to delete the library root")
    used = usage(root)
    files = nbytes = 0
    for d, dirs, names in os.walk(path):
        if projects.is_project(d):
            raise ValueError(f"cannot delete {rel}: it contains the project folder {os.path.relpath(d, path)}")
        users = used.get(os.path.realpath(d))  # a bag folder used as a source
        if users:
            raise ValueError(f"cannot delete {rel}: {os.path.basename(d)} is used by project(s) "
                             f"{', '.join(sorted(set(users)))}")
        for n in names:
            p = os.path.join(d, n)
            users = used.get(os.path.realpath(p))
            if users:
                raise ValueError(f"cannot delete {rel}: {n} is used by project(s) {', '.join(sorted(set(users)))}")
            files += 1
            nbytes += os.path.getsize(p)
    shutil.rmtree(path)
    return files, nbytes


def rename_folder(root, rel, new_name):
    """Rename a library folder in place; projects using clips inside are updated. Refused for folders
    holding a project. Returns the new path."""
    path = resolve_dir(root, rel)
    if path == _lib(root):
        raise ValueError("refusing to rename the library root")
    new_name = new_name.strip()
    if not FOLDER_RE.match(new_name):
        raise ValueError("folder names use letters, digits, spaces and '_ . ( ) -' (max 64), starting with a "
                         "letter or digit")
    return _move_folder(root, path, os.path.join(os.path.dirname(path), new_name), rel)


def delete_file(root, path):
    """Delete a stray .SRT (no video next to it) from the library."""
    p = os.path.abspath(path)
    resolve_dir(root, os.path.relpath(os.path.dirname(p), _lib(root)))
    if not p.lower().endswith(".srt") or not os.path.isfile(p):
        raise ValueError(f"not an .SRT file in the library: {path}")
    os.unlink(p)


# --- moving, and keeping projects pointed at their clips ----------------------------------------
def all_folders(root):
    """Every library folder, relative to the library ("" = ROOT/videos), projects and hidden ones left out."""
    base = _lib(root)
    out = [""]
    for d, dirs, _ in os.walk(base):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and not projects.is_project(os.path.join(d, x)))
        out += [os.path.relpath(os.path.join(d, x), base) for x in dirs]
    return out


def _stored_like(old, new_abs):
    """new_abs written the way the old path was stored: absolute, or relative to the working directory."""
    return new_abs if os.path.isabs(old) else os.path.relpath(new_abs)


def update_references(root, moves):
    """Point every project at moved files. moves: {old realpath: new absolute path}. Rewrites the
    sources in ingest.json and every matching path string in poses.json. Returns the projects changed."""
    changed = []
    exts = tuple(ingest.VIDEO_EXTS) + (".srt",) + BAG_EXTS

    def fix(v):  # paths only: a clip / bag file, or anything with a separator (a bag folder)
        if isinstance(v, str) and v and (v.lower().endswith(exts) or "/" in v) and os.path.realpath(v) in moves:
            return _stored_like(v, moves[os.path.realpath(v)]), True
        if isinstance(v, list):
            out = [fix(x) for x in v]
            return [x for x, _ in out], any(c for _, c in out)
        if isinstance(v, dict):
            out = {k: fix(x) for k, x in v.items()}
            return {k: x for k, (x, _) in out.items()}, any(c for _, c in out.values())
        return v, False

    for p in projects.list_projects(root):
        hit = False
        for name in (ingest.PLAN_FILE, "poses.json"):
            f = os.path.join(p, name)
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                data, c = fix(json.load(fh))
            if c:
                with open(f, "w") as fh:
                    json.dump(data, fh, indent=1)
                hit = True
        if hit:
            changed.append(os.path.basename(p))
    return changed


def _move_lead_cache(src_dir, dst_dir, video):
    """Carry the clip's cached optical-flow lead (keyed by size + mtime) to its new folder."""
    st = os.stat(video)
    key = f"{st.st_size}:{st.st_mtime_ns}"
    try:
        entry = json.load(open(os.path.join(src_dir, ingest.LEAD_CACHE)))[key]
    except (OSError, ValueError, KeyError):
        return
    path = os.path.join(dst_dir, ingest.LEAD_CACHE)
    try:
        cache = json.load(open(path)) if os.path.exists(path) else {}
        cache[key] = entry
        with open(path, "w") as f:
            json.dump(cache, f, indent=1)
    except (OSError, ValueError):
        pass


def _move_folder(root, path, dst, rel):
    for d, _, _ in os.walk(path):
        if projects.is_project(d):
            raise ValueError(f"cannot move {rel}: it contains the project folder {os.path.relpath(d, path)}")
    if os.path.lexists(dst):
        raise FileExistsError(f"{os.path.basename(dst)} already exists there")
    moves = _tree_moves(path, dst)
    os.rename(path, dst)
    update_references(root, moves)
    return dst


def move(root, path, dest_rel):
    """Move a clip (with its .SRT), a bag or a library folder into the library folder dest_rel; projects
    using the moved clips / bags are updated. Returns the new path."""
    dest = resolve_dir(root, dest_rel)
    p = os.path.abspath(path)
    if os.path.isdir(p):
        rel = os.path.relpath(os.path.realpath(p), _lib(root))
        src = resolve_dir(root, rel)
        if src == _lib(root):
            raise ValueError("refusing to move the library root")
        if dest == src or dest.startswith(src + os.sep):
            raise ValueError(f"cannot move {rel} into itself")
        if os.path.dirname(src) == dest:
            raise ValueError(f"{rel} is already there")
        return _move_folder(root, src, os.path.join(dest, os.path.basename(src)), rel)
    v = check_video(root, p)
    if os.path.dirname(v) == dest:
        raise ValueError(f"{os.path.basename(v)} is already there")
    srt = _srt(v)
    pairs = [(v, os.path.join(dest, os.path.basename(v)))] + ([(srt, os.path.join(dest, os.path.basename(srt)))]
                                                             if srt else [])
    for _, t in pairs:
        if os.path.lexists(t):
            raise FileExistsError(f"{os.path.basename(t)} already exists there")
    moves = {os.path.realpath(a): b for a, b in pairs}  # resolved before the files move
    if srt is not None or v.lower().endswith(ingest.VIDEO_EXTS):
        _move_lead_cache(os.path.dirname(v), dest, v)
    for a, b in pairs:
        os.rename(a, b)
    update_references(root, moves)
    return pairs[0][1]
