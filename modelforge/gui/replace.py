"""The Replace tab (left panel): content-aware repair of the splat (modelforge/replace.py; the user asked, 2026-10-31).

Find and rate objects (a job), then either **automatic** (proposals: objects rendered much blurrier than their photos,
each to be removed with its ground filled, or replaced by a donor where one fits; tick them, Apply checked) or
**manual** (Pick, Ctrl + click an object; its donor candidates listed best first; remove + fill, or replace with the
chosen donor); **ground texture** (Map weak ground: a detail map of render vs photo; weak areas outlined in magenta,
ticked ones repaired; or a circle picked by Ctrl + click with a radius). Every applied edit is listed with Go and Undo. Objects are marked by upright cylinders: orange the
selected one, red the proposals, cyan the donor candidates."""
import os
import threading

import cv2
import numpy as np
import viser

from modelforge import replace
from modelforge.gui.semantic import best_view, cylinder_segments

NODE = "/replace"
COLOURS = {"selected": (255, 140, 0), "proposal": (230, 40, 40), "donor": (0, 200, 230), "area": (230, 0, 230),
           "circle": (255, 230, 0)}
MAX_ROWS = 40


class ReplacePanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self.lock = threading.RLock()
        self.objs, self.selected, self.donor_list, self.proposals = None, None, [], []
        self.picking = None  # "object" / "circle": what the next Ctrl + click picks
        self.areas, self.area_props, self.circle = [], [], None  # weak ground areas, their rows, the picked circle
        self.request = None  # what the next replace job does: ("find",) / ("apply", items) / ("undo", id)
        self._rows, self._markers = [], []
        gui.add_markdown(
            "Repair weak parts of the splat by **copying from better parts of it**, like content-aware fill: an "
            "object that renders much blurrier than its photos (a smeared tree or bush) is removed, the ground under "
            "it is filled with similar ground from elsewhere, and optionally a sharper object of the same kind is "
            "placed there instead. Every edit can be undone.")
        self.g_kind = gui.add_text(
            "Look for", initial_value=", ".join(replace.QUERIES),
            hint="What the objects are, as words the semantic search understands, separated by commas (default: "
                 "vegetation, trees). Needs the semantic features (Search tab).")
        b = gui.add_button("Find and rate objects", icon=viser.Icon.SEARCH,
                           hint="Finds every object of that kind standing on the ground (one per crown), then renders "
                                "each from the photos that see it best and compares: how much of the photos' detail "
                                "the splat keeps (sharpness 1 = all, a smear << 1). A few minutes.")
        b.on_click(lambda _: self._start(("find",)))
        self.g_status = gui.add_markdown("")
        with gui.add_folder("Automatic", expand_by_default=True):
            self.g_auto_action = gui.add_dropdown(
                "Action", options=("remove + fill ground", "replace by a donor where one fits"),
                initial_value="remove + fill ground",
                hint="What the proposals do. A donor must be as sharp as its photos, clean (nothing else standing "
                     "over it) and of a similar shape; where none fits, the object is removed and its ground filled.")
            b = gui.add_button("Suggest", icon=viser.Icon.SPARKLES,
                               hint="Lists the objects rendered much blurrier than their photos (sharpness < "
                                    f"{replace.TARGET_SHARP}) that are mostly vegetation. Tick the ones to change.")
            b.on_click(lambda _: self.suggest())
            self.g_auto_folder = gui.add_folder("Proposals", expand_by_default=True)
            b = gui.add_button("Apply checked", icon=viser.Icon.CHECK,
                               hint="Applies the ticked proposals (one job; ~3 s per edit).")
            b.on_click(lambda _: self.apply_checked())
        with gui.add_folder("Manual", expand_by_default=True):
            b = gui.add_button("Pick an object (Ctrl + click)", icon=viser.Icon.POINTER,
                               hint="Then Ctrl + click a tree or bush in the 3D view.")
            b.on_click(lambda _: self._arm())
            self.g_pick_md = gui.add_markdown("")
            self.g_manual_folder = gui.add_folder("Donor candidates", expand_by_default=True)
            row = gui.add_button_group("", options=("Remove + fill ground", "Replace by the donor"))
            row.on_click(lambda ev: self.manual(ev.target.value != "Remove + fill ground"))
            self.g_donor = gui.add_dropdown("Donor", options=("-",), initial_value="-",
                                            hint="The candidate placed instead (best first; Go flies to one).")
        with gui.add_folder("Ground texture", expand_by_default=True):
            gui.add_markdown("Ground the splat renders much blurrier than the photos show it (no object on it) gets "
                             "its ground layer (grass, low shrubs up to 1.5 m) replaced by similar ground from places "
                             "the splat renders sharply.")
            b = gui.add_button("Map weak ground", icon=viser.Icon.MAP,
                               hint="Renders 300 photos and compares each with the splat window by window: how much "
                                    "of the photos' detail the splat keeps, placed on a 2 m grid on the ground. Weak "
                                    f"areas: below {replace.WEAK_GROUND} of it, at least {replace.MIN_GROUND_M2:.0f} "
                                    "m2, no object on them. Needs Find and rate objects first. Seconds to a minute.")
            b.on_click(lambda _: self._start(("map",)))
            self.g_area_md = gui.add_markdown("")
            self.g_show_areas = gui.add_checkbox("Show weak areas", initial_value=True,
                                                 hint="Magenta outlines on the ground, labelled g<id>.")
            self.g_show_areas.on_update(lambda _: self._draw_markers())
            self.g_area_folder = gui.add_folder("Weak areas (weakest first)", expand_by_default=True)
            b = gui.add_button("Repair checked areas", icon=viser.Icon.CHECK,
                               hint="Replaces the ground layer of the ticked areas (one job).")
            b.on_click(lambda _: self.repair_checked())
            b = gui.add_button("Pick a spot (Ctrl + click)", icon=viser.Icon.POINTER,
                               hint="Then Ctrl + click the ground: a circle of the radius below is repaired there.")
            b.on_click(lambda _: self._arm("circle"))
            self.g_radius = gui.add_slider("Radius (m)", min=0.5, max=20.0, step=0.5, initial_value=3.0,
                                           hint="The size of the circle Repair circle replaces.")
            self.g_radius.on_update(lambda _: self._draw_markers())
            self.g_circle_md = gui.add_markdown("")
            b = gui.add_button("Repair circle", icon=viser.Icon.CIRCLE_CHECK,
                               hint="Replaces the ground layer inside the yellow circle.")
            b.on_click(lambda _: self.repair_circle())
        with gui.add_folder("Edits", expand_by_default=True):
            self.g_edits_folder = gui.add_folder("Applied", expand_by_default=True)
            b = gui.add_button("Undo all", icon=viser.Icon.ARROW_BACK_UP, hint="Back to the splat before any edit.")
            b.on_click(lambda _: self._start(("undo", "all")))
        self.refresh()

    # --------------------------------------------------------------------------------------------- state
    def refresh(self):
        """Re-read the objects and edits of the open project (after a project switch, a job, a new splat)."""
        p = self.app.path
        with self.lock:
            self.objs = replace.objects(p) if p else None
            if self.objs is None:
                self.selected, self.proposals, self.donor_list = None, [], []
            areas = replace.ground_areas(p) if p and self.objs is not None else []
            done = sorted(e.get("area") for e in replace.edits(p) if e.get("kind") == "ground" and "area" in e) \
                if p else []
            key = ([a["id"] for a in areas], done)  # the list changes with the areas and with what is repaired
            self.areas = areas
            if key != getattr(self, "_areas_key", None):
                self._areas_key = key
                self._draw_areas()
        self._area_status()
        self._status()
        self._draw_edits()
        self._draw_markers()

    def close(self):
        self.app.server.scene.remove_by_name(NODE)
        self._markers = []

    def _status(self):
        if self.objs is None:
            self.g_status.content = "_No objects found yet: Find and rate objects._"
            return
        rated = [o for o in self.objs if o.get("sharp") is not None]
        weak = replace.suggest_targets(self.objs)
        self.g_status.content = (f"{len(self.objs)} objects, {len(rated)} rated, {len(weak)} rendered much blurrier "
                                 f"than their photos; {len(replace.edits(self.app.path))} edits applied.")

    def _area_status(self):
        if self.objs is None:
            self.g_area_md.content = "_Find and rate objects first._"
        elif not self.areas:
            self.g_area_md.content = "_No weak ground mapped (Map weak ground)._"
        else:
            done = {e.get("area") for e in replace.edits(self.app.path) if e.get("kind") == "ground"}
            self.g_area_md.content = (f"{len(self.areas)} weak areas, {sum(a['area_m2'] for a in self.areas):,.0f} m2; "
                                      f"{len(done & {a['id'] for a in self.areas})} repaired.")

    def _draw_areas(self):
        for h in getattr(self, "_arows", []):
            h.remove()
        self._arows, self.area_props = [], []
        done = {e.get("area") for e in replace.edits(self.app.path) if e.get("kind") == "ground"}
        with self.g_area_folder:
            for a in [a for a in self.areas if a["id"] not in done][:MAX_ROWS]:
                cb = self.gui.add_checkbox(f"g{a['id']} · {a['area_m2']:.0f} m2 · detail {a['detail']:.2f}",
                                           initial_value=a["area_m2"] >= 4 * replace.MIN_GROUND_M2)
                b = self.gui.add_button(f"Go to g{a['id']}", icon=viser.Icon.FOCUS_2)
                b.on_click(lambda ev, a=a: self._go_ground(a["x"], a["y"], a["radius"], ev.client))
                self.area_props.append((a["id"], cb))
                self._arows += [cb, b]

    def repair_checked(self):
        items = [{"area": aid} for aid, cb in self.area_props if cb.value]
        if items:
            self._start(("apply", items))

    def repair_circle(self):
        if self.circle is None:
            self.g_circle_md.content = "_Pick a spot first._"
            return
        x, y = self.circle
        self.circle = None
        self.g_circle_md.content = ""
        self._start(("apply", [{"circle": [x, y, float(self.g_radius.value)]}]))

    def _ground_z(self, x, y):
        d = np.load(os.path.join(self.app.path, replace.DIR, "objects.npz"))
        return float(replace.Ground(d["ground_grid"], d["ground_origin"])(x, y))

    def _go_ground(self, x, y, r, client):
        if client is None:
            return
        target = np.array([x, y, self._ground_z(x, y)])
        pos = best_view(self.app.snap, target, self.app.fov, self.app.aspect, max(r, 2.0))
        if pos is None:
            pos = target + np.array([0.0, -1.0, 1.0]) / np.sqrt(2) * max(4 * r, 8.0)
        self.app.glide_to(client, pos, target)

    def _objs_by_id(self):
        return {o["id"]: o for o in (self.objs or [])}

    def _start(self, request):
        self.request = request
        self.app.run_stage("replace")

    # --------------------------------------------------------------------------------------------- the job
    def run_job(self, project, on_line, on_progress, token):
        """Run the pending request in the GUI's job thread; returns the History line."""
        req = self.request or ("find",)
        self.request = None
        if req[0] == "find":
            qs = {q.strip(): replace.QUERIES.get(q.strip(), 0.58) for q in self.g_kind.value.split(",") if q.strip()}
            objs = replace.find(project, qs, on_line)
            objs = replace.rate(project, on_line, on_progress, token)
            msg = f"{len(objs)} objects found and rated, {len(replace.suggest_targets(objs))} look broken"
        elif req[0] == "map":
            areas = replace.detail_map(project, on_line, on_progress, token)
            msg = f"weak ground mapped: {len(areas)} areas, {sum(a['area_m2'] for a in areas):,.0f} m2"
        elif req[0] == "apply":
            eds = replace.add_many(project, req[1], on_line)
            msg = f"{len(req[1])} edits applied ({len(eds)} in all)"
        else:
            eds = replace.undo(project, req[1], on_line)
            msg = f"undone; {len(eds)} edits left"
        return msg

    # --------------------------------------------------------------------------------------------- automatic
    def suggest(self):
        if not self.objs:
            self.g_status.content = "_Find and rate objects first._"
            return
        done = {e["object"] for e in replace.edits(self.app.path)}
        targets = [o for o in replace.suggest_targets(self.objs) if o["id"] not in done]
        excl = {o["id"] for o in targets}
        used, props = set(), []
        for o in targets[:MAX_ROWS]:
            d = None
            if self.g_auto_action.value.startswith("replace"):
                ds = replace.donors(self.objs, o, used, excl)
                d = ds[0]["id"] if ds else None
                if d is not None:
                    used.add(d)
            props.append({"object": o["id"], "donor": d})
        self.proposals = props
        self._draw_proposals()
        self._draw_markers()

    def _draw_proposals(self):
        for h in self._rows:
            h.remove()
        self._rows, by = [], self._objs_by_id()
        with self.g_auto_folder:
            if not self.proposals:
                self._rows.append(self.gui.add_markdown("_Nothing to propose._"))
            for p in self.proposals:
                o = by[p["object"]]
                what = f"→ donor #{p['donor']}" if p["donor"] is not None else "→ remove + fill"
                cb = self.gui.add_checkbox(f"#{o['id']} · {o['height']:.0f} m · sharp {o['sharp']:.2f} {what}",
                                           initial_value=True)
                p["check"] = cb
                b = self.gui.add_button(f"Go to #{o['id']}", icon=viser.Icon.FOCUS_2)
                b.on_click(lambda ev, k=o["id"]: self.go(k, ev.client))
                self._rows += [cb, b]

    def apply_checked(self):
        items = [(p["object"], p["donor"], True) for p in self.proposals if p.get("check") and p["check"].value]
        if items:
            self.proposals = []
            self._draw_proposals()
            self._start(("apply", items))

    # --------------------------------------------------------------------------------------------- manual
    def _arm(self, what="object"):
        md = self.g_pick_md if what == "object" else self.g_circle_md
        if not self.objs:
            md.content = "_Find and rate objects first._"
            return
        self.picking = what
        md.content = ("**Ctrl + click** a tree or bush in the 3D view." if what == "object" else
                      "**Ctrl + click** the ground in the 3D view.")

    def on_pick(self, origin, direction):
        """A Ctrl + click while picking: select the object under it. True if it used the click."""
        if not self.picking:
            return False
        what, self.picking = self.picking, None
        hit = self.app.ray_hit(origin, direction)
        if what == "circle":
            if hit is None:
                self.g_circle_md.content = "Nothing there: Pick a spot again."
                return True
            self.circle = (float(hit[0]), float(hit[1]))
            self.g_circle_md.content = (f"Circle at ({hit[0]:.1f}, {hit[1]:.1f}): set the radius, then Repair "
                                        "circle.")
            self._draw_markers()
            return True
        oid = replace.object_at(self.app.path, hit[0], hit[1]) if hit is not None else None
        if oid is None:
            self.g_pick_md.content = "No object there: Pick again and Ctrl + click on a tree or bush."
            return True
        self.select(oid)
        return True

    def select(self, oid):
        by = self._objs_by_id()
        o = by[oid]
        done = {e["object"] for e in replace.edits(self.app.path)}
        self.selected = oid
        self.donor_list = replace.donors(self.objs, o, exclude=done)
        sharp = f"{o['sharp']:.2f}" if o.get("sharp") is not None else "not rated"
        self.g_pick_md.content = (f"**#{oid}**: {o['height']:.1f} m tall, {2 * o['radius']:.1f} m wide, sharpness {sharp}"
                                  f"{' (already edited)' if oid in done else ''}; {len(self.donor_list)} donor candidates.")
        self.g_donor.options = tuple(f"#{d['id']} · {d['height']:.0f} m · sharp {d['sharp']:.2f}"
                                     for d in self.donor_list[:MAX_ROWS]) or ("-",)
        for h in getattr(self, "_drows", []):
            h.remove()
        self._drows = []
        with self.g_manual_folder:
            for d in self.donor_list[:8]:
                b = self.gui.add_button(f"Go to donor #{d['id']}", icon=viser.Icon.FOCUS_2)
                b.on_click(lambda ev, k=d["id"]: self.go(k, ev.client))
                self._drows.append(b)
        self._draw_markers()

    def manual(self, with_donor):
        if self.selected is None:
            self.g_pick_md.content = "_Pick an object first._"
            return
        donor = None
        if with_donor:
            v = self.g_donor.value
            if not v.startswith("#"):
                self.g_pick_md.content = "_No donor candidate fits this object._"
                return
            donor = int(v[1:].split()[0])
        self._start(("apply", [(self.selected, donor, True)]))

    # --------------------------------------------------------------------------------------------- edits
    def _draw_edits(self):
        for h in getattr(self, "_erows", []):
            h.remove()
        self._erows = []
        p = self.app.path
        eds = replace.edits(p) if p else []
        with self.g_edits_folder:
            if not eds:
                self._erows.append(self.gui.add_markdown("_None._"))
            areas = {a["id"]: a for a in self.areas}
            for e in eds:
                if e.get("kind") == "ground":
                    if e.get("circle"):
                        x, y, r = e["circle"]
                        what = f"ground circle {r:.1f} m"
                    else:
                        a = areas.get(e.get("area"), {"x": 0.0, "y": 0.0, "radius": 1.0})
                        x, y, r = a["x"], a["y"], a["radius"]
                        what = f"ground g{e.get('area')}"
                    go = (lambda c, x=x, y=y, r=r: self._go_ground(x, y, r, c))
                else:
                    what = (f"#{e['object']} → donor #{e['donor']}" if e.get("donor") is not None
                            else f"#{e['object']} removed")
                    go = (lambda c, o=e["object"]: self.go(o, c))
                row = self.gui.add_button_group(f"{what}", options=("Go", "Undo"))
                row.on_click(lambda ev, e=e, go=go: go(ev.client) if ev.target.value == "Go"
                             else self._start(("undo", e["id"])))
                self._erows.append(row)

    # --------------------------------------------------------------------------------------------- 3D
    def go(self, oid, client):
        o = self._objs_by_id().get(oid)
        if o is None or client is None:
            return
        target = np.array([o["x"], o["y"], o["ground_z"] + 0.5 * o["height"]])
        size = max(o["radius"], o["height"] / 2)
        pos = best_view(self.app.snap, target, self.app.fov, self.app.aspect, 1.5 * size)
        if pos is None:
            pos = target + np.array([0.0, -1.0, 0.8]) / np.linalg.norm([0.0, -1.0, 0.8]) * max(4 * size, 4.0)
        self.app.glide_to(client, pos, target)

    def _draw_markers(self):
        scene = self.app.server.scene
        scene.remove_by_name(NODE)
        by = self._objs_by_id()
        marks = [(p["object"], "proposal") for p in self.proposals]
        if self.selected is not None:
            marks.append((self.selected, "selected"))
            marks += [(d["id"], "donor") for d in self.donor_list[:8]]
        for oid, kind in marks:
            o = by.get(oid)
            if o is None:
                continue
            name = f"{NODE}/{kind}_{oid}"
            scene.add_frame(name, show_axes=False, position=(o["x"], o["y"], o["ground_z"]))
            scene.add_line_segments(f"{name}/cylinder", cylinder_segments(o["radius"] * 1.1, o["height"] * 1.05),
                                    colors=COLOURS[kind], thickness=3.0, thickness_units="screen")
            scene.add_label(f"{name}/label", f"#{oid}", position=(0.0, 0.0, o["height"] * 1.15))
        if self.circle is not None:
            x, y = self.circle
            r = float(self.g_radius.value)
            scene.add_frame(f"{NODE}/circle", show_axes=False, position=(x, y, self._ground_z(x, y)))
            scene.add_line_segments(f"{NODE}/circle/ring", cylinder_segments(r, 0.3, struts=4),
                                    colors=COLOURS["circle"], thickness=4.0, thickness_units="screen")
        if self.areas and self.g_show_areas.value:
            segs = self._area_outlines()
            if len(segs):
                scene.add_line_segments(f"{NODE}/areas", segs, colors=COLOURS["area"], thickness=2.5,
                                        thickness_units="screen")
            for a in self.areas[:MAX_ROWS]:
                scene.add_label(f"{NODE}/area_{a['id']}", f"g{a['id']}",
                                position=(a["x"], a["y"], self._ground_z(a["x"], a["y"]) + 0.5))

    def _area_outlines(self):
        """Line segments (N, 2, 3) along the weak areas' borders, 0.2 m above the ground."""
        p = os.path.join(self.app.path, replace.DIR)
        lab = np.load(os.path.join(p, "ground_labels.npy"))
        m = np.load(os.path.join(p, "detail.npz"))
        lo, cell = m["lo"], float(m["cell"])
        d = np.load(os.path.join(p, "objects.npz"))
        ground = replace.Ground(d["ground_grid"], d["ground_origin"])
        segs = []
        for a in self.areas:
            cs, _ = cv2.findContours((lab == a["id"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                q = c[:, 0, :].astype(float)  # (column, row) of border cells
                if len(q) < 2:
                    continue
                x, y = lo[0] + (q[:, 0] + 0.5) * cell, lo[1] + (q[:, 1] + 0.5) * cell
                pts = np.stack([x, y, ground(x, y) + 0.2], 1)
                segs.append(np.stack([pts, np.roll(pts, -1, 0)], 1))
        return np.concatenate(segs).astype(np.float32) if segs else np.zeros((0, 2, 3), np.float32)
