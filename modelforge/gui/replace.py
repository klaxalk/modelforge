"""The Replace tab (left panel): content-aware repair of the splat (modelforge/replace.py; the user asked, 2026-10-31).

Find and rate objects (a job), then either **automatic** (proposals: objects rendered much blurrier than their photos,
each to be removed with its ground filled, or replaced by a donor where one fits; tick them, Apply checked) or
**manual** (Pick, Ctrl + click an object; its donor candidates listed best first; remove + fill, or replace with the
chosen donor); **ground texture** (Map weak ground: a detail map of render vs photo; weak areas outlined in magenta,
ticked ones repaired; or an area drawn as a polygon by Ctrl + clicking its corners; optionally from a template: a
polygon of good ground drawn the same way, the only place the fill copies from). Every applied edit is listed with Go and Undo. Objects are marked by upright cylinders: orange the
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
           "circle": (255, 230, 0), "template": (40, 220, 40)}
MAX_ROWS = 40
COPY_FROM = {"the most similar detailed patch": "auto", "similar sharp ground anywhere": None, "my template": "mine"}  # Ground texture → Copy from


def poly_area(pts):
    """Area (m2) of a polygon [[x, y], ...] (shoelace)."""
    p = np.asarray(pts, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


class ReplacePanel:
    def __init__(self, app, gui):
        self.app, self.gui = app, gui
        self.lock = threading.RLock()
        self.objs, self.selected, self.donor_list, self.proposals = None, None, [], []
        self.picking = None  # "object" / "area" / "template": what the next Ctrl + click picks
        self.areas, self.area_props = [], []  # weak ground areas, their rows
        self.poly = {"area": [], "template": []}  # polygons being drawn / drawn: corners [x, y] (ENU)
        self.closed = {"area": False, "template": False}
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
            self.g_turn = gui.add_checkbox(
                "Turn donors to face the views", initial_value=False,
                hint="A tree in a splat is good only from where it was photographed: turns each donor about the "
                     "vertical so its photographed side faces the side its target was photographed from. The "
                     "sunlit side turns with it (fine under an overcast sky). Also for Manual.")
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
                                    "m2, no object on them. Finds the objects first if that has not run (their ground "
                                    "model; no rating). A minute or two.")
            b.on_click(lambda _: self._start(("map",)))
            self.g_area_md = gui.add_markdown("")
            with gui.add_folder("Template: good ground to copy from", expand_by_default=True):
                b = gui.add_button("Draw a template (Ctrl + click corners)", icon=viser.Icon.POLYGON,
                                   hint="Then Ctrl + click the corners of a patch of good ground of the kind the "
                                        "repair should have (e.g. the same road where it is sharp), and Finish. The "
                                        "repairs copy only from inside it (nothing standing on it is used).")
                b.on_click(lambda _: self._arm("template"))
                row = gui.add_button_group("", options=("Undo corner", "Finish", "Clear"))
                row.on_click(lambda ev: self._poly_action("template", ev.target.value))
                self.g_copy_from = gui.add_dropdown(
                    "Copy from", options=tuple(COPY_FROM), initial_value="the most similar detailed patch",
                    hint="Where Repair checked areas / Repair circle copy the ground from. The most similar detailed "
                         "patch: a disc of the repair's size where the splat is sharp, whose colour and kind (the "
                         "semantic features: road, gravel, grass ...) match the weak area's own best (its green ring "
                         "shows after the repair). Similar sharp ground anywhere: per cell, matched by the area's "
                         "surroundings. My template: the green polygon drawn above.")
                self.g_copy_from.on_update(lambda _: self._draw_markers())
                self.g_tmpl_md = gui.add_markdown("_No template._")
            self.g_show_areas = gui.add_checkbox("Show weak areas", initial_value=True,
                                                 hint="Magenta outlines on the ground, labelled g<id>.")
            self.g_show_areas.on_update(lambda _: self._draw_markers())
            self.g_area_folder = gui.add_folder("Weak areas (weakest first)", expand_by_default=True)
            b = gui.add_button("Repair checked areas", icon=viser.Icon.CHECK,
                               hint="Replaces the ground layer of the ticked areas (one job).")
            b.on_click(lambda _: self.repair_checked())
            b = gui.add_button("Draw the area to repair (Ctrl + click corners)", icon=viser.Icon.POLYGON,
                               hint="Then Ctrl + click the corners of the ground to repair, and Finish.")
            b.on_click(lambda _: self._arm("area"))
            row = gui.add_button_group("", options=("Undo corner", "Finish", "Clear"))
            row.on_click(lambda ev: self._poly_action("area", ev.target.value))
            self.g_area_draw_md = gui.add_markdown("")
            b = gui.add_button("Repair the drawn area", icon=viser.Icon.CIRCLE_CHECK,
                               hint="Replaces the ground layer inside the yellow polygon.")
            b.on_click(lambda _: self.repair_drawn())
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
            self.g_area_md.content = "_No weak ground mapped (Map weak ground)._"
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

    def _source(self):
        """What the ground repairs copy from: {"polygon": ...} (my template), "auto" or None (Copy from)."""
        how = COPY_FROM[self.g_copy_from.value]
        if how == "mine":
            if not self.closed["template"]:
                return None
            return {"polygon": [list(c) for c in self.poly["template"]]}
        return how

    def _poly_md(self, which):
        return self.g_area_draw_md if which == "area" else self.g_tmpl_md

    def _poly_status(self, which):
        pts = self.poly[which]
        md = self._poly_md(which)
        if not pts:
            md.content = "_No template._" if which == "template" else ""
        elif not self.closed[which]:
            md.content = f"{len(pts)} corners: **Ctrl + click** the next, then Finish."
        else:
            what = "repairs copy from the green polygon" if which == "template" else "Repair the drawn area"
            md.content = f"{len(pts)} corners, {poly_area(pts):,.1f} m²: {what}."

    def _poly_action(self, which, action):
        """Undo corner / Finish / Clear of the polygon being drawn (which: "area" or "template")."""
        pts = self.poly[which]
        if action == "Undo corner" and pts:
            pts.pop()
            self.closed[which] = False
            if self.picking is None and pts:
                self.picking = which
        elif action == "Finish":
            if len(pts) < 3:
                self._poly_md(which).content = "_A polygon needs at least 3 corners._"
                return
            self.closed[which] = True
            if self.picking == which:
                self.picking = None
            if which == "template":
                self.g_copy_from.value = "my template"
        elif action == "Clear":
            self.poly[which] = []
            self.closed[which] = False
            if self.picking == which:
                self.picking = None
            if which == "template" and self.g_copy_from.value == "my template":
                self.g_copy_from.value = "the most similar detailed patch"
        self._poly_status(which)
        self._draw_markers()

    def repair_checked(self):
        src = self._source()
        items = [{"area": aid, **({"source": src} if src else {})} for aid, cb in self.area_props if cb.value]
        if items:
            self._start(("apply", items))

    def repair_drawn(self):
        if not self.closed["area"]:
            self.g_area_draw_md.content = "_Draw the area first (at least 3 corners, then Finish)._"
            return
        pts = [list(c) for c in self.poly["area"]]
        self.poly["area"], self.closed["area"] = [], False
        self._poly_status("area")
        src = self._source()
        self._start(("apply", [{"polygon": pts, **({"source": src} if src else {})}]))

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

    def _queries(self):
        return {q.strip(): replace.QUERIES.get(q.strip(), 0.58) for q in self.g_kind.value.split(",") if q.strip()}

    def _start(self, request):
        self.request = request
        self.app.run_stage("replace")

    # --------------------------------------------------------------------------------------------- the job
    def run_job(self, project, on_line, on_progress, token):
        """Run the pending request in the GUI's job thread; returns the History line."""
        req = self.request or ("find",)
        self.request = None
        if req[0] == "find":
            objs = replace.find(project, self._queries(), on_line)
            objs = replace.rate(project, on_line, on_progress, token)
            msg = f"{len(objs)} objects found and rated, {len(replace.suggest_targets(objs))} look broken"
        elif req[0] == "map":
            if replace.objects(project) is None:  # the map uses the objects' ground and keeps off their footprints
                on_line("replace: finding the objects first (the map needs their ground model)")
                replace.find(project, self._queries(), on_line)
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
        turn = bool(self.g_turn.value)
        items = [(p["object"], p["donor"], True, turn) for p in self.proposals if p.get("check") and p["check"].value]
        if items:
            self.proposals = []
            self._draw_proposals()
            self._start(("apply", items))

    # --------------------------------------------------------------------------------------------- manual
    def _arm(self, what="object"):
        md = {"object": self.g_pick_md, "area": self.g_area_draw_md, "template": self.g_tmpl_md}[what]
        if not self.objs:
            md.content = ("_Find and rate objects first._" if what == "object" else
                          "_Map weak ground first (it sets up the ground model)._")
            return
        self.picking = what
        if what != "object":  # a new polygon
            self.poly[what], self.closed[what] = [], False
            self._draw_markers()
        md.content = ("**Ctrl + click** a tree or bush in the 3D view." if what == "object" else
                      "**Ctrl + click** the corners of the polygon in the 3D view, then Finish.")

    def on_pick(self, origin, direction):
        """A Ctrl + click while picking: select the object under it. True if it used the click."""
        if not self.picking:
            return False
        what = self.picking
        hit = self.app.ray_hit(origin, direction)
        if what in ("area", "template"):  # a corner; picking goes on until Finish
            if hit is not None:
                self.poly[what].append([float(hit[0]), float(hit[1])])
            self._poly_status(what)
            self._draw_markers()
            return True
        self.picking = None
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
        self._start(("apply", [(self.selected, donor, True, bool(self.g_turn.value))]))

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
                    if e.get("polygon"):
                        pts = np.asarray(e["polygon"], float)
                        x, y = pts.mean(0)
                        r = float(np.sqrt(poly_area(pts) / np.pi))
                        what = f"ground polygon {poly_area(pts):.0f} m²" + (" from template" if e.get("source") else "")
                    elif e.get("circle"):
                        x, y, r = e["circle"]
                        what = f"ground circle {r:.1f} m"
                        if e.get("source"):
                            what += " from template"
                    else:
                        a = areas.get(e.get("area"), {"x": 0.0, "y": 0.0, "radius": 1.0})
                        x, y, r = a["x"], a["y"], a["radius"]
                        what = f"ground g{e.get('area')}" + (" from template" if e.get("source") else "")
                    go = (lambda c, x=x, y=y, r=r: self._go_ground(x, y, r, c))
                else:
                    what = (f"#{e['object']} → donor #{e['donor']}{' (turned)' if e.get('turn') else ''}"
                        if e.get("donor") is not None else f"#{e['object']} removed")
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
        for which, colour in (("area", COLOURS["circle"]), ("template", COLOURS["template"])):
            pts = self.poly[which]
            if not pts:
                continue
            P = np.array([[x, y, self._ground_z(x, y) + 0.3] for x, y in pts], np.float32)
            for k, c in enumerate(P):  # the corners
                scene.add_icosphere(f"{NODE}/{which}_corner_{k}", radius=0.25, color=colour, position=tuple(c))
            if len(P) >= 2:
                seg = np.stack([P[:-1], P[1:]], 1)
                if self.closed[which]:
                    seg = np.concatenate([seg, np.stack([P[-1:], P[:1]], 1)])
                scene.add_line_segments(f"{NODE}/{which}_poly", seg, colors=colour, thickness=4.0,
                                        thickness_units="screen")
            if which == "template":
                scene.add_label(f"{NODE}/template_label", "template", position=tuple(P.mean(0) + [0, 0, 0.8]))
        st = os.path.join(self.app.path, replace.DIR, "state.json") if self.app.path else None
        if st and os.path.exists(st):  # the templates the applied ground repairs copied from
            import json
            for e in json.load(open(st)).get("edits", []):
                tp = (e.get("fill") or {}).get("template_polygon")
                if tp:
                    P = np.array([[x, y, self._ground_z(x, y) + 0.15] for x, y in tp], np.float32)
                    scene.add_line_segments(f"{NODE}/used_{e['id']}", np.stack([P, np.roll(P, -1, 0)], 1),
                                            colors=COLOURS["template"], thickness=2.0, thickness_units="screen")
                t = (e.get("fill") or {}).get("template_at")
                if t:
                    name = f"{NODE}/used_{e['id']}"
                    scene.add_frame(name, show_axes=False, position=(t[0], t[1], self._ground_z(t[0], t[1])))
                    scene.add_line_segments(f"{name}/ring", cylinder_segments(t[2], 0.15, struts=0),
                                            colors=COLOURS["template"], thickness=2.0, thickness_units="screen")
                    scene.add_label(f"{name}/label", f"source of edit {e['id']}", position=(0.0, 0.0, 0.6))
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
