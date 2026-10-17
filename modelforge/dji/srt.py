"""Parser for the DJI .SRT telemetry sidecar (one entry per video frame)."""

import re

_KV = re.compile(r"\[(\w+)\s*:\s*([^\]]*)\]")
_ALT = re.compile(r"rel_alt:\s*([-\d.]+)\s+abs_alt:\s*([-\d.]+)")


def parse_srt(path):
    """Return a list of dicts, one per frame, with raw string values.

    `pp_current` / `pp_target` are parsed into (w, x, y, z) tuples; they are the
    camera attitude and, unlike the djmd track, in sync with the video.
    """
    entries = []
    for block in open(path, encoding="utf-8", errors="replace").read().strip().split("\n\n"):
        e = {k: v.strip() for k, v in _KV.findall(block)}
        m = _ALT.search(block)
        if m:
            e["rel_alt"], e["abs_alt"] = m.groups()
        for k in ("pp_current", "pp_target"):
            if k in e:
                e[k] = tuple(float(x) for x in e[k].split(","))
        entries.append(e)
    return entries
