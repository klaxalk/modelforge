"""Check flight alignment (Ingest tab, before extraction; the user asked, 2026-10-25): several DJI flights get their
altitude offset from the takeoff altitudes (abs_alt), which can be wrong (karlak3: 240 m low), and a wrong offset
put that flight far from the others. This measures the offsets from the images on a few photos per flight:

1. Where the flights look: every 3rd frame's look-at point, its optical axis meeting the ground at that flight's own
   takeoff height (rel_alt is reliable, the offset between flights is not), horizontally (GPS is good to metres).
   Cells of CELL_M that two or more flights look into are shared ground.
2. Per flight, runs of RUN frames STEP apart (a short baseline, so each flight triangulates its own copy of the
   spot well) in the shared cells, round robin over them, up to PER_FLIGHT photos. Single frames spread over the
   area (the first version) left each spot to one or two photos per flight: the copies were noise, and the
   horizontal offsets wandered by tens of metres between rounds on karlak (the altitude still came out).
3. A small project in PROJECT/flight_check/: those frames extracted (the sharpest of +-WINDOW frames), features,
   every pair, matching, the refine with per-flight offsets (robust.update_source_offsets: from the split points
   between the flights' copies of the scene; recovers a shift of 20x the flying height in its first round).
4. Per flight: the refine's offset added to the one used = the offset the images say, with how many points tie the
   flight to the others (merged) as the confidence. result.json; the Ingest tab offers them as Altitude offsets.
"""

import json
import math
import os
import shutil
import time

import numpy as np

from modelforge.dji import extract as ex
from modelforge.dji import ingest

DIR = "flight_check"
CELL_M = 25.0  # m: shared ground is counted in cells of this size
PER_FLIGHT = 42  # photos per flight
RUN, STEP = 3, 10  # frames per run, frames apart (1/3 s at 30 fps)
WINDOW = 4  # frames either side: the sharpest is taken
MIN_MERGED = 50  # points shared with the other flights after the refine: fewer = the offset is not trusted
EXTRA_ROUNDS = 3  # rounds at the final threshold after the usual three: karlak_2 still moved 1.6 m in the third
MAX_LAST_STEP_M = 1.0  # the altitude still moved more than this in the last round: not settled, not trusted


def look_cells(sp, origin, yaw_source="auto", step=3):
    """(frames, cells): every step-th valid frame of the source and the cell its look-at point falls in."""
    tel = sp.clip.telemetry(yaw_source)
    k = np.flatnonzero(sp.valid)[::step]
    if not len(k):
        return k, np.zeros((0, 2), int)
    a = tel["a"][k]
    C = ingest.enu(a[:, 0], a[:, 1], a[:, 2] + sp.alt_offset, origin)
    axis = ex.optical_axis_enu(tel["yaw"][k], tel["pitch"][k])
    depth = ingest.look_at_depth({"a": a, "pitch": tel["pitch"][k]})  # to the ground at its own takeoff height
    xy = C[:, :2] + depth[:, None] * axis[:, :2]
    return k, np.floor(xy / CELL_M).astype(int)


def choose(plan, yaw_source="auto", per_flight=PER_FLIGHT):
    """{plan source index: [frames]} from the shared cells, and {flight pair: shared cells}."""
    first = next(sp for sp in plan.sources if sp.valid.any())
    a0 = first.clip.telemetry(yaw_source)["a"][int(np.flatnonzero(first.valid)[0])]
    origin = (a0[0], a0[1], a0[2] + first.alt_offset)
    cells = {sp.index: look_cells(sp, origin, yaw_source) for sp in plan.sources}
    by_cell = {}  # cell -> {source: [frames]}
    for s, (k, c) in cells.items():
        for f, cc in zip(k, map(tuple, c)):
            by_cell.setdefault(cc, {}).setdefault(s, []).append(int(f))
    shared = {cc: d for cc, d in by_cell.items() if len(d) >= 2}
    pairs = {}
    for d in shared.values():
        ss = sorted(d)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                pairs[(ss[i], ss[j])] = pairs.get((ss[i], ss[j]), 0) + 1
    chosen = {}
    for s in cells:
        mine = sorted(((cc, d[s]) for cc, d in shared.items() if s in d), key=lambda x: (-len(shared[x[0]]), x[0]))
        valid = next(sp for sp in plan.sources if sp.index == s).valid
        picks, rnd = set(), 0
        while len(picks) < per_flight and any(len(fr) > rnd for _, fr in mine):
            for _, fr in mine:  # one run per cell per round; within a cell spread over its frames
                if len(fr) > rnd and len(picks) < per_flight:
                    f0 = fr[(rnd * 7919 + len(fr) // 2) % len(fr)] if rnd else fr[len(fr) // 2]
                    run_ = [f for f in range(f0, f0 + RUN * STEP, STEP) if f < len(valid) and valid[f]]
                    if all(abs(f - p) >= STEP for f in run_ for p in picks):  # no near-duplicate photos
                        picks |= set(run_)
            rnd += 1
        chosen[s] = sorted(picks)
    return chosen, pairs


def run(project, sources=None, settings=None, on_line=print, on_progress=None, token=None):
    """The check for PROJECT's Ingest plan (ingest.json unless sources / settings are given). Returns the result
    (also PROJECT/flight_check/result.json)."""
    from modelforge import pipeline
    from modelforge.sfm import priors as priors_mod
    from modelforge.sfm import robust
    t0 = time.time()
    if sources is None:
        sources, settings = ingest.load_plan(project)
    plan = ingest.make_plan(sources, settings)
    if len(plan.sources) < 2:
        raise ValueError("the flight check needs two or more enabled sources")
    chosen, shared = choose(plan, settings.yaw_source)
    names = {sp.index: os.path.basename(sp.source.video) for sp in plan.sources}
    for sp in plan.sources:
        on_line(f"flight check: {names[sp.index]}: {len(chosen[sp.index])} photos over shared ground "
                f"(altitude offset now {sp.alt_offset:+.2f} m)")
    if any(not v for v in chosen.values()):
        alone = [names[s] for s, v in chosen.items() if not v]
        raise ValueError(f"no ground shared with the other flights: {', '.join(alone)} (the check needs the flights "
                         "to look at the same place)")
    out = os.path.join(project, DIR)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    windows = {s: [(max(0, f - WINDOW), f + WINDOW) for f in fr] for s, fr in chosen.items()}  # STEP > 2 WINDOW
    total = sum(len(v) for v in windows.values())

    def progress(done, n, msg):
        if token is not None:
            token.check()
        if on_progress:
            on_progress(done, 2 * total)
    ingest.run(out, sources, settings, progress=progress, overwrite=True, log=lambda m: None, windows=windows)
    report = pipeline.Report(out)
    pri, meta = priors_mod.load(out)
    on_line("flight check: features, every pair, matching")
    pipeline.stage_features(out, report, "dji")
    if token is not None:
        token.check()
    pipeline.stage_pairs(out, report, pri, meta, "exhaustive")
    pipeline.stage_match(out, report, token=token)
    on_line("flight check: refine with per-flight offsets")
    # one constant offset per flight, vertical only (the altitude is what is in doubt; GPS is good to metres
    # horizontally, and these few photos measure it badly); the horizontal disagreement is reported
    model = robust.PriorModel(estimate_drift=False, source_offset_axes=(0.0, 0.0, 1.0))
    from modelforge.sfm import refine
    rounds = refine.DEFAULT_ROUNDS + (refine.DEFAULT_ROUNDS[-1],) * EXTRA_ROUNDS  # until the offsets settle
    rec = pipeline.stage_refine(out, report, pri, model=model, rounds=rounds, ba_mode="global", token=token,
                                progress=lambda ev: on_progress and on_progress(total + total // 2, 2 * total))
    edges = robust.split_point_edges(rec, pri, os.path.join(out, "database.db"))
    flights = []
    for sp in plan.sources:
        s = sp.index
        d = np.asarray(model.source_offsets.get(s, np.zeros(3)), float)
        merged = sum(e["merged"] for k, e in edges.items() if s in k)
        last = np.asarray(getattr(model, "_offset_inc", {}).get(s, np.zeros(3)), float)  # the last round's step
        settled = abs(float(last[2])) < MAX_LAST_STEP_M
        flights.append({"source": s, "video": sp.source.video, "name": names[s], "photos": len(chosen[s]),
                        "offset_used_m": round(sp.alt_offset, 3), "correction_m": np.round(d, 3).tolist(),
                        "alt_offset_m": round(sp.alt_offset + float(d[2]), 2), "merged_points": int(merged),
                        "last_step_m": np.round(last, 3).tolist(), "trusted": merged >= MIN_MERGED and settled})
    apart = {f"{names[a]} – {names[b]}": {"merged": int(e["merged"]), "split": int(e["n_split"]),
                                           "apart_m": np.round(e["median"], 2).tolist()}
             for (a, b), e in sorted(edges.items())}  # what is left between the flights' copies (x, y: GPS)
    res = {"flights": flights, "apart": apart, "shared_cells": {f"{a}-{b}": n for (a, b), n in sorted(shared.items())},
           "registered": int(rec.num_reg_images()) if rec is not None else 0, "photos": total,
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    for f in flights:
        on_line(f"flight check: {f['name']}: altitude offset {f['offset_used_m']:+.2f} → {f['alt_offset_m']:+.2f} m, "
                f"{f['merged_points']} "
                f"points shared with the other flights, last step {f['last_step_m'][2]:+.2f} m"
                + ("" if f["trusted"] else " — not trusted (too few shared points or still moving)"))
    for k, e in apart.items():
        on_line(f"flight check: {k}: copies of the scene still {e['apart_m']} m apart (x, y, z; median of "
                f"{e['split']} split points, {e['merged']} merged)")
    on_line(f"flight check: {res['registered']} of {total} photos registered, {res['seconds']:.0f} s")
    return res


def load(project):
    """result.json of the last check, or None."""
    try:
        return json.load(open(os.path.join(project, DIR, "result.json")))
    except (OSError, ValueError):
        return None
