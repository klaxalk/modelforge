"""The Rig tab (left panel, after Search): cut the semantic search's object out of the splat, hinge its parts with
joints picked by Ctrl + click, pose it with sliders and export the posed splat (modelforge/rig.py,
docs/todo_rigs.md).

In the 3D view a shown rig replaces its object: the splat is drawn without the object's Gaussians
(`decorate`, applied after the search's own decoration, also to the level-of-detail cells), the fill under it as
/rig/fill, and each part as its own splat /rig/part_k in its rest-pose world coordinates, moved by its node pose
(rig.transforms), so a slider only sends a pose.
"""

import os
import re
import threading
import traceback

import numpy as np
import viser

from modelforge import rig
from modelforge.gui import data as gdata

NODE = "/rig"
PART_COLOURS = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                         (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128)]) / 255.0
PICKS = ("a point on the hinge axis", "a second point on the axis", "a point on the moving part")
NONE = "-"


def _name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")[:40] or "object"


def _wxyz(R):
    from modelforge.export import splat_to_enu
    q = splat_to_enu.quat_from_rot(R)
    return tuple(float(x) for x in q / np.linalg.norm(q))


class RigPanel:
    def __init__(self, app, gui):
        self.app = app
        self.rig, self.name = None, None  # the open rig (rig.json) and its name
        self.idx = self.labels = self.fill = None  # object indices, part labels, fill arrays
        self.obj = None  # viser arrays (centers, cov, rgb, op) of the object's Gaussians, rest pose
        self.hidden = None  # bool per splat Gaussian: drawn by the rig instead of the splat
        self.handles, self.picks, self.lock = {}, [], threading.RLock()
        self._play = None  # threading.Event of the running playback
        self.g_md = gui.add_markdown("")
        self.g_name = gui.add_text("Name", initial_value="", hint="Name of a new rig (default: the query).")
        self.g_cut = gui.add_button(
            "Cut out the search's object", icon=viser.Icon.SCISSORS,
            hint="Search tab first: the selected object (Object list) is cut out of the splat: its matches, grown "
                 "over neighbouring Gaussians above the ground that match better than generic words, and everything "
                 "under them. The ground under it is filled from its surroundings (invented, not measured).")
        self.g_cut.on_click(lambda _: self.cut())
        self.g_rig = gui.add_dropdown("Rig", options=(NONE,), initial_value=NONE,
                                      hint="The project's rigs (rig/<name>/).")
        self.g_rig.on_update(lambda _: self.open(self.g_rig.value))
        self.g_show = gui.add_checkbox("Show the rig in the splat", initial_value=True,
                                       hint="Draw the rig's parts and fill instead of the object's Gaussians.")
        self.g_show.on_update(lambda _: self.redraw())
        self.g_colour = gui.add_checkbox("Colour the parts", initial_value=False,
                                         hint="Tint each part in its own colour, to check where the cuts went.")
        self.g_colour.on_update(lambda _: self.redraw())
        self.g_unirig = gui.add_button(
            "Propose joints (UniRig)", icon=viser.Icon.WAND,
            hint="UniRig (a network trained on artists' rigs) puts a skeleton into a mesh of the object, "
                 "with 10 seeds; every limb 2+ runs agree on that leaves the body becomes a hinge where it "
                 "leaves it (horizontal axis, a lift). About 2 min. Check the parts with Colour the parts; remove "
                 "or re-pick wrong ones. Press again while it runs to stop it. Needs UniRig installed "
                 "(scripts/install_unirig).")
        self.g_unirig.on_click(lambda _: self.propose())
        with gui.add_folder("Joints from a template (vision model)", expand_by_default=True):
            self.g_tmpl = gui.add_button(
                "Ask the vision model for the template", icon=viser.Icon.BRAIN,
                hint="The local vision model looks at photos of the object, names the machine and lists its parts "
                     "and joints in a chain (turntable / hinge / slide), each with where its pin is. Then you "
                     "Ctrl + click each pin and a point on the part it moves, as asked; the joints are built from "
                     "that (replacing the rig's joints). ~30 s; press again to stop.")
            self.g_tmpl.on_click(lambda _: self.ask_template())
            self.g_tmpl_md = gui.add_markdown("")
            self.g_tmpl_nav = gui.add_button_group("", options=("Undo last click", "Skip this one", "Cancel"),
                                                   hint="Undo: click that one again. Skip: a pin you cannot find "
                                                        "(that joint is left out). Cancel: stop clicking; the joints "
                                                        "stay as they were.")
            self.g_tmpl_nav.on_click(lambda e: self._guided_nav(e.target.value))
            self.g_tmpl_nav.visible = False
        with gui.add_folder("Parts (PartField)", expand_by_default=True):
            self.g_pf = gui.add_button(
                "Propose parts (PartField)", icon=viser.Icon.PUZZLE,
                hint="PartField (NVIDIA, non-commercial licence) splits the object into parts from its Gaussians; "
                     "touching parts get joints: a turntable where a part sits flat on another (an excavator's "
                     "house on its tracks), else a hinge at the contact. Replaces the rig's joints. ~10 s; press "
                     "again while it runs to stop it. Needs scripts/install_partfield.")
            self.g_pf.on_click(lambda _: self.propose_parts())
            self.g_pf_k = gui.add_slider("Parts", min=2, max=12, step=1, initial_value=4,
                                         hint="How many parts to split the object into (instant once proposed).")
            self.g_pf_k.on_update(lambda _: self._recluster())
            self.g_pf_part = gui.add_dropdown("Part to split", options=(NONE,), initial_value=NONE,
                                              hint="A part that should move as several (an excavator's arm: boom, "
                                                   "stick, bucket).")
            self.g_pf_split = gui.add_button("Split it in two", hint="Splits the chosen part only; the joints are "
                                                                      "proposed again.")
            self.g_pf_split.on_click(lambda _: self.split_part())
            self.g_pf_md = gui.add_markdown("")
        with gui.add_folder("New joint", expand_by_default=True):
            self.g_kind = gui.add_dropdown("Type", options=("revolute", "prismatic"), initial_value="revolute",
                                           hint="revolute: turns about the axis (degrees); prismatic: slides along it "
                                                "(metres).")
            self.g_parent = gui.add_dropdown("Parent part", options=("part 0 (base)",),
                                             initial_value="part 0 (base)",
                                             hint="The part the joint sits on; its new part is cut from this one.")
            self.g_pick = gui.add_button(
                "Pick the joint (Ctrl + click ×3)", icon=viser.Icon.POINTER,
                hint="Then Ctrl + click on the object: two points on the hinge axis (or along the slide), then a "
                     "point on the part that moves. The part = the parent's Gaussians beyond the plane through the "
                     "first point, facing the third, connected to the third.")
            self.g_pick.on_click(lambda _: self.start_pick())
            self.g_pick_md = gui.add_markdown("")
        self.g_joints = gui.add_folder("Joints")
        self.g_sliders = []
        self.g_undo = gui.add_button("Remove the last joint", hint="Its part joins its parent again.")
        self.g_undo.on_click(lambda _: self.remove_last())
        self.g_reset = gui.add_button("Rest pose", hint="Every joint back to 0.")
        self.g_reset.on_click(lambda _: self.reset())
        with gui.add_folder("Animate", expand_by_default=True):
            self.g_anim = gui.add_dropdown(
                "Motion", options=("demo", "keyframes"), initial_value="demo",
                hint="demo: each joint in turn through its limits and back, then all together. keyframes: the poses "
                     "added below, one after another, back to the first. Below them: the motions the vision model "
                     "planned.")
            self.g_plan = gui.add_button(
                "Plan motions (vision model)", icon=viser.Icon.BRAIN,
                hint="A local vision-language model (Qwen2.5-VL-7B, offline) looks at the object from 4 sides with "
                     "its joints marked, a close-up per joint and a photo, says what the machine and each part are, "
                     "and plans 2-3 motions it really performs; they appear under Motion. About 1 min, ~17 GB of GPU "
                     "memory. Press again while it runs to stop it. Needs scripts/install_unirig --vlm.")
            self.g_plan.on_click(lambda _: self.plan_motions())
            self.g_vlm_md = gui.add_markdown("")
            self.g_play = gui.add_button("▶ Play", hint="Play the motion in the 3D view (every browser sees it). "
                                                        "Moving a slider stops it.")
            self.g_play.on_click(lambda _: self.toggle_play())
            self.g_loop = gui.add_checkbox("Loop", initial_value=True, hint="Start again at the end.")
            self.g_speed = gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0,
                                          hint="Playback speed (1 = the demo's 30°/s, 0.5 m/s; keyframes as spaced).")
            self.g_add_key = gui.add_button("Add keyframe (the sliders' pose)",
                                            hint="The joints' current values as the next keyframe (stored with the "
                                                 "rig).")
            self.g_add_key.on_click(lambda _: self.add_keyframe())
            self.g_step = gui.add_number("Seconds between keyframes", initial_value=2.0, min=0.2, step=0.5)
            self.g_clear_keys = gui.add_button("Clear keyframes")
            self.g_clear_keys.on_click(lambda _: self.clear_keyframes())
            self.g_anim_md = gui.add_markdown("")
        self.g_export = gui.add_button(
            "Export the posed splat", icon=viser.Icon.DOWNLOAD,
            hint="export/<rig>_posed.ply: the whole splat with the object in this pose (positions, rotations and "
                 "view-dependent colour turned per part) and the fill.")
        self.g_export.on_click(lambda _: self.export())
        self.g_delete = gui.add_button("Delete the rig", color="red", hint="Removes rig/<name>/; the splat is "
                                                                             "untouched.")
        self.g_delete.on_click(lambda _: self.delete())

    # ------------------------------------------------------------ state
    @property
    def project(self):
        return self.app.path

    def refresh(self):
        """The project's state changed (a job ended, a project opened): list its rigs; the open one stays open
        while it exists (reloaded: its splat may have changed)."""
        names = rig.list_rigs(self.project) if self.project else []
        keep = self.name if self.name in names else NONE
        self.g_rig.options = (NONE,) + tuple(names)
        if keep == NONE:
            self.close()
        self.g_rig.value = keep
        if keep != NONE:
            self.open(keep)
        self._status()

    def _save(self):
        """Store the open rig, unless it was deleted meanwhile (a playback ending after Delete must not bring it
        back)."""
        if self.rig is not None and os.path.exists(os.path.join(rig.rig_dir(self.project, self.name), "rig.json")):
            rig.save(self.project, self.name, self.rig)

    def _status(self, text=None):
        if text is not None:
            self.g_md.content = text
            return
        if self.rig is None:
            self.g_md.content = ("Search for an object (Search tab), select it, then **Cut out the search's "
                                 "object**." if self.project else "")
            return
        n_parts = int(self.labels.max()) + 1 if len(self.labels) else 1
        stale = "" if rig.current(self.project, self.name) else " · ⚠️ the splat changed since: cut it out again"
        self.g_md.content = (f"**{self.name}** · {len(self.idx):,} Gaussians · {n_parts} part(s) · "
                             f"{len(self.rig['joints'])} joint(s) · fill {len(self.fill['means']):,}{stale}")

    def close(self):
        self.stop_play()
        self._guided = None
        if hasattr(self, "g_tmpl_nav"):
            self.g_tmpl_nav.visible = False
        self.rig = self.name = self.idx = self.labels = self.fill = self.obj = self.hidden = None
        self.picks = []
        self._clear()
        self._rebuild_sliders()

    def open(self, name):
        if name in (None, NONE):
            if self.rig is not None:
                self.close()
                self.app.redraw_splat_decorated()
            self._status()
            return
        try:
            with self.lock:
                self.rig, self.name = rig.load(self.project, name), name
                self.idx, self.labels, self.fill = rig.arrays(self.project, name)
                self._load_object()
            self._rebuild_sliders()
            self.redraw()
        except Exception as e:  # noqa: BLE001 - shown in the tab
            traceback.print_exc()
            self.close()
            self._status(f"❌ {e}")

    def _load_object(self):
        import plyfile
        v = plyfile.PlyData.read(self.app.dense["splat"])["vertex"].data
        self.obj = gdata.splat_arrays(v[self.idx], None)
        self.hidden = np.zeros(len(v), bool)
        self.hidden[self.idx] = True

    # ------------------------------------------------------------ cut out
    def cut(self):
        sem = self.app.semantic
        if sem.rel is None or not sem.objs or sem._emb is None:
            self._status("Search first (Search tab): the rig starts from the selected object.")
            return
        opts = list(sem.g_objects.options)
        k = opts.index(sem.g_objects.value) if sem.g_objects.value in opts else 0
        obj, query, thr = sem.objs[min(k, len(sem.objs) - 1)], sem.query, float(sem.g_thr.value)
        name = _name(self.g_name.value or query)
        self.g_cut.disabled = True
        self._status(f"⏳ cutting out **{query}** …")

        def work():
            try:
                import plyfile
                feats = sem._features()
                rel = feats.relevance(sem._emb[0], sem._emb[1:])  # all of it: the growing goes down to rig.GROW
                v = plyfile.PlyData.read(self.app.dense["splat"])["vertex"].data
                means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
                op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
                rgb = np.clip(0.5 + rig.SH_C0 * np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)
                others = [o for o in sem.objs if o is not obj]  # other instances: walls, never joined
                idx, ground = rig.cut_out(means, op, rel, thr, obj, others=others)
                mask = np.zeros(len(means), bool)
                mask[idx] = True
                rig.create(self.project, name, idx, rig.fill(means, rgb, op, mask, ground), query=query, threshold=thr)
                sem.clear()  # the red highlight would cover the cut
                self.g_rig.options = (NONE,) + tuple(rig.list_rigs(self.project))
                self.g_rig.value = name  # opens it
            except Exception as e:  # noqa: BLE001 - shown in the tab
                traceback.print_exc()
                self._status(f"❌ {e}")
            finally:
                self.g_cut.disabled = False
        threading.Thread(target=work, daemon=True).start()

    def delete(self):
        if self.name is None:
            return
        self.stop_play()
        rig.delete(self.project, self.name)
        self.refresh()
        self.app.redraw_splat_decorated()

    # ------------------------------------------------------------ joints
    def propose(self):
        """UniRig's limbs as joints (modelforge/rig_unirig.py), added after the rig's own; pressed while it runs:
        Stop."""
        from modelforge import jobs, rig_unirig
        if getattr(self, "_unirig_token", None) is not None:
            threading.Thread(target=self._unirig_token.cancel, daemon=True).start()
            self._status("⏹ stopping UniRig …")
            return
        if self.rig is None:
            self._status("Open or cut out a rig first.")
            return
        why = rig_unirig.available()
        if why:
            self._status(f"❌ {why}")
            return
        token = self._unirig_token = jobs.Token()
        self.g_unirig.label = "⏹ Stop UniRig"
        name = self.name

        def work():
            try:
                joints = rig_unirig.propose(self.project, name, on_line=lambda t: self._status(f"⏳ {t}"),
                                            token=token)
                if name != self.name:
                    return
                with self.lock:
                    for j in joints:
                        self.rig["joints"].append(j)
                    self._save()
                    self.labels = rig.relabel(self.project, self.name, self.obj[0].astype(np.float64))
                self._rebuild_sliders()
                self.redraw()
                moved = [int((self.labels == len(self.rig["joints"]) - len(joints) + k + 1).sum())
                         for k in range(len(joints))]
                self.g_pick_md.content = (f"UniRig: {len(joints)} joint(s) proposed, parts of "
                                          f"{', '.join(f'{m:,}' for m in moved)} Gaussians" if joints else
                                          "UniRig: no limb that most runs agree on leaves the body")
            except jobs.Cancelled:
                self._status("⏹ UniRig stopped: no joints added")
            except Exception as e:  # noqa: BLE001 - shown in the tab
                if token.cancelled:
                    self._status("⏹ UniRig stopped: no joints added")
                else:
                    traceback.print_exc()
                    self._status(f"❌ UniRig: {e}")
            finally:
                self._unirig_token = None
                self.g_unirig.label = "Propose joints (UniRig)"
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ PartField
    def _replace_joints(self, joints, note, md=None):
        """Replace the rig's joints by proposed ones (PartField parts, a template); the old ones kept in rig.json
        ("joints_before")."""
        with self.lock:
            if self.rig.get("joints"):
                self.rig["joints_before"] = self.rig["joints"]
            self.rig["joints"] = joints
            self._save()
            self.labels = rig.relabel(self.project, self.name, self.obj[0].astype(np.float64))
        self.stop_play()
        self._rebuild_sliders()
        if not self.g_colour.value:
            self.g_colour.value = True  # redraws; the parts are what to check now
        else:
            self.redraw()
        self._parts_options()
        (md or self.g_pf_md).content = note

    def _parts_options(self):
        from modelforge import rig_partfield
        parts = rig_partfield.load_parts(self.project, self.name) if self.rig else None
        if parts is None:
            self.g_pf_part.options, self.g_pf_part.value = (NONE,), NONE
            return
        names = {j["part"]: j["name"] for j in self.rig["joints"] if "part" in j}
        opts = tuple(f"part {p} · {int((parts == p).sum()):,} G · {names.get(p, 'base')}"
                     for p in range(int(parts.max()) + 1))
        cur = self.g_pf_part.value
        self.g_pf_part.options = opts
        self.g_pf_part.value = cur if cur in opts else opts[0]

    def propose_parts(self):
        from modelforge import jobs, rig_partfield
        if getattr(self, "_pf_token", None) is not None:
            threading.Thread(target=self._pf_token.cancel, daemon=True).start()
            self.g_pf_md.content = "⏹ stopping PartField …"
            return
        if self.rig is None:
            self._status("Open or cut out a rig first.")
            return
        why = rig_partfield.available()
        if why:
            self.g_pf_md.content = f"❌ {why}"
            return
        token = self._pf_token = jobs.Token()
        self.g_pf.label = "⏹ Stop PartField"
        name, k = self.name, int(self.g_pf_k.value)

        def work():
            try:
                if rig_partfield.load_features(self.project, name) is None or len(
                        rig_partfield.load_features(self.project, name)) != len(self.idx):
                    rig_partfield.features(self.project, name, on_line=lambda t: self._set_pf_md(f"⏳ {t}"),
                                           token=token)
                if name != self.name:
                    return
                self._parts_joints(k)
            except jobs.Cancelled:
                self.g_pf_md.content = "⏹ PartField stopped: joints unchanged"
            except Exception as e:  # noqa: BLE001 - shown in the tab
                if token.cancelled:
                    self.g_pf_md.content = "⏹ PartField stopped: joints unchanged"
                else:
                    traceback.print_exc()
                    self.g_pf_md.content = f"❌ PartField: {e}"
            finally:
                self._pf_token = None
                self.g_pf.label = "Propose parts (PartField)"
        threading.Thread(target=work, daemon=True).start()

    def _set_pf_md(self, text):
        self.g_pf_md.content = text

    def _parts_joints(self, k=None, parts=None):
        from modelforge import rig_partfield
        if parts is None:
            parts = rig_partfield.cluster(self.project, self.name, k)
        joints = rig_partfield.joints_from_parts(self.obj[0].astype(np.float64), parts)
        kinds = ", ".join(j["name"] for j in joints) or "none"
        self._replace_joints(joints, f"{int(parts.max()) + 1} parts → joints: {kinds}. Check them with Colour "
                                       "the parts; split a part that should move as several.")

    def _recluster(self):
        from modelforge import rig_partfield
        if self.rig is None or getattr(self, "_pf_token", None) is not None or \
                rig_partfield.load_features(self.project, self.name) is None:
            return
        k = int(self.g_pf_k.value)
        threading.Thread(target=lambda: self._safe(lambda: self._parts_joints(k)), daemon=True).start()

    def split_part(self):
        from modelforge import rig_partfield
        if self.rig is None or self.g_pf_part.value == NONE:
            return
        part = int(self.g_pf_part.value.split()[1])
        self._safe(lambda: self._parts_joints(parts=rig_partfield.split(self.project, self.name, part)))

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - shown in the tab
            traceback.print_exc()
            self.g_pf_md.content = f"❌ {e}"

    # ------------------------------------------------------------ template + guided clicks
    def ask_template(self):
        from modelforge import jobs, rig_vlm
        if getattr(self, "_tmpl_token", None) is not None:
            threading.Thread(target=self._tmpl_token.cancel, daemon=True).start()
            self.g_tmpl_md.content = "⏹ stopping the vision model …"
            return
        if self.rig is None:
            self._status("Open or cut out a rig first.")
            return
        why = rig_vlm.available()
        if why:
            self.g_tmpl_md.content = f"❌ {why}"
            return
        token = self._tmpl_token = jobs.Token()
        self.g_tmpl.label = "⏹ Stop the vision model"
        self.app.semantic.unload("the vision model")
        self._guided = None
        name = self.name

        def work():
            try:
                tmpl = rig_vlm.template(self.project, name, on_line=lambda t: setattr(self.g_tmpl_md, "content",
                                                                                     f"⏳ {t}"), token=token)
                if name != self.name:
                    return
                if not tmpl["joints"]:
                    self.g_tmpl_md.content = f"**{tmpl['machine']}**: the model listed no joints."
                    return
                steps = []
                for j in tmpl["joints"]:
                    if j.get("type") == "slew":  # its ring is hidden under the upper body: ask for the visible gap
                        steps.append(("pin", j["name"], f"the machine's side exactly at the gap where the {j['child']} "
                                                        f"sits on the {j['parent']} (the turntable's height)"))
                    else:
                        steps.append(("pin", j["name"], j["pin"]))
                    steps.append(("part", j["child"], f"a point in the middle of the {j['child']}"))
                self._guided = {"template": tmpl, "steps": steps, "i": 0, "pins": {}, "centres": {}, "marks": []}
                self.g_tmpl_nav.visible = True
                self._guided_prompt()
            except jobs.Cancelled:
                self.g_tmpl_md.content = "⏹ stopped"
            except Exception as e:  # noqa: BLE001 - shown in the tab
                if token.cancelled:
                    self.g_tmpl_md.content = "⏹ stopped"
                else:
                    traceback.print_exc()
                    self.g_tmpl_md.content = f"❌ vision model: {e}"
            finally:
                self._tmpl_token = None
                self.g_tmpl.label = "Ask the vision model for the template"
        threading.Thread(target=work, daemon=True).start()

    def _guided_prompt(self):
        g = self._guided
        t = g["template"]
        lines = [f"**{t['machine']}**: " + " → ".join(
            [t["joints"][0]["parent"]] + [f"{j['child']} ({j['type']})" for j in t["joints"]])]
        for j in t["joints"]:
            ok = "✅" if j["name"] in g["pins"] and j["child"] in g["centres"] else (
                "⏭" if any(s[1] == j["name"] for s in g["steps"][:g["i"]]) else "·")
            lines.append(f"{ok} {j['name']}: {j['type']}, {j['parent']} → {j['child']}")
        kind, key, phrase = g["steps"][g["i"]]
        lines.append(f"\n**Ctrl + click {phrase}** ({'pin of ' if kind == 'pin' else 'part: '}{key}; "
                     f"{g['i'] + 1} / {len(g['steps'])})")
        lines.append("A joint the model missed (e.g. an excavator's stick): add it afterwards with Pick the joint.")
        self.g_tmpl_md.content = "\n".join(lines)

    def _guided_click(self, origin, direction):
        g = self._guided
        hit = self.hit(origin, direction)
        if hit is None:
            self._guided_prompt()
            self.g_tmpl_md.content += "\n\n⚠️ That click missed the object: click on it."
            return
        kind, key, _ = g["steps"][g["i"]]
        (g["pins"] if kind == "pin" else g["centres"])[key] = hit
        g["marks"].append((hit, kind))
        self._draw_guided()
        self._guided_next()

    def _guided_next(self):
        g = self._guided
        g["i"] += 1
        if g["i"] >= len(g["steps"]):
            self._guided_finish()
        else:
            self._guided_prompt()

    def _guided_nav(self, what):
        if self._guided is None:
            return
        if what == "Cancel":
            self._guided = None
            self.g_tmpl_nav.visible = False
            self.app.server.scene.remove_by_name("/rig_guided")
            self.g_tmpl_md.content = "Cancelled: the joints are as they were."
            return
        g = self._guided
        if what == "Undo last click":
            if g["i"] == 0:
                return
            g["i"] -= 1
            kind, key, _ = g["steps"][g["i"]]
            (g["pins"] if kind == "pin" else g["centres"]).pop(key, None)
            if g["marks"]:
                g["marks"].pop()
            self._draw_guided()
            self._guided_prompt()
            return
        kind, key, _ = g["steps"][g["i"]]
        if kind == "pin":  # skipping a pin skips its part too
            g["i"] += 1
        self._guided_next()

    def _guided_finish(self):
        from modelforge import rig_vlm
        g, self._guided = self._guided, None
        self.g_tmpl_nav.visible = False
        self.app.server.scene.remove_by_name("/rig_guided")
        pins = dict(g["pins"])
        means = self.obj[0].astype(np.float64)
        for j in g["template"]["joints"]:  # a turntable: the click's height, centred in the upper body above it
            if j.get("type") == "slew" and j["name"] in pins:
                pins[j["name"]] = rig.ring(means, pins[j["name"]])
        joints = rig_vlm.template_joints(g["template"], pins, g["centres"])
        if not joints:
            self.g_tmpl_md.content = "No joint got both its pin and its part: nothing changed."
            return
        t = g["template"]
        self._replace_joints(joints, f"**{t['machine']}**: {len(joints)} joint(s) from the template: "
                                     + ", ".join(f"{j['name']} ({'turntable' if j['limits'][1] > 90 else j['type']})"
                                                 for j in joints) + ". Check with Colour the parts; move the sliders.",
                             md=self.g_tmpl_md)

    def _draw_guided(self):
        scene = self.app.server.scene
        scene.remove_by_name("/rig_guided")
        for i, (p, kind) in enumerate(self._guided["marks"] if self._guided else []):
            scene.add_icosphere(f"/rig_guided/p{i}", radius=0.15, color=(255, 220, 0) if kind == "pin" else (0, 220, 255),
                                position=tuple(float(x) for x in p))

    def start_pick(self):
        if self.rig is None:
            self._status("Open or cut out a rig first.")
            return
        self.picks = []
        self.g_pick_md.content = f"Ctrl + click {PICKS[0]}"
        self._draw_picks()

    def on_pick(self, origin, direction):
        """Ctrl + click while picking (a joint, or the template's guided clicks): the nearest object Gaussian along
        the ray. True when it was used."""
        if self.rig is not None and getattr(self, "_guided", None) is not None:
            self._guided_click(origin, direction)
            return True
        if self.rig is None or not self.g_pick_md.content.startswith("Ctrl + click"):
            return False
        hit = self.hit(origin, direction)
        if hit is None:
            self.g_pick_md.content = f"Ctrl + click {PICKS[len(self.picks)]} (missed the object: click on it)"
            return True
        self.picks.append(hit)
        self._draw_picks()
        if len(self.picks) < 3:
            self.g_pick_md.content = f"Ctrl + click {PICKS[len(self.picks)]}"
            return True
        try:
            parent = list(self.g_parent.options).index(self.g_parent.value)
            j = rig.new_joint(*self.picks, parent=parent, kind=self.g_kind.value,
                              name=f"joint {len(self.rig['joints']) + 1}")
            with self.lock:
                self.rig["joints"].append(j)
                self._save()
                self.labels = rig.relabel(self.project, self.name, self.obj[0].astype(np.float64))
            n = int((self.labels == len(self.rig["joints"])).sum())
            self.stop_play()
            self.g_pick_md.content = (f"✅ {j['name']}: {n:,} Gaussians move" if n else
                                      "⚠️ no Gaussians beyond that plane: remove the joint and pick again")
        except Exception as e:  # noqa: BLE001 - shown in the tab
            traceback.print_exc()
            self.g_pick_md.content = f"❌ {e}"
        self.picks = []
        self._draw_picks()
        self._rebuild_sliders()
        self.redraw()
        return True

    def hit(self, origin, direction):
        """Where a ray meets the object: the nearest of its opaque Gaussian centres within ~1.2 % of the distance
        (>= 15 cm) of the ray; the median of those within 30 cm behind it."""
        c, op = self.obj[0].astype(np.float64), self.obj[3][:, 0]
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        v = c - o
        t = v @ d
        off = np.linalg.norm(v - t[:, None] * d, axis=1)
        ok = (t > 0) & (op > 0.3) & (off < np.maximum(0.15, 0.012 * t))  # generous: pins are small
        if not ok.any():
            return None
        t0 = t[ok].min()
        near = ok & (t <= t0 + 0.3)
        return np.median(c[near], axis=0)

    def remove_last(self):
        if self.rig is None or not self.rig["joints"]:
            return
        self.stop_play()
        with self.lock:
            self.rig["joints"].pop()
            self._save()
            self.labels = rig.relabel(self.project, self.name, self.obj[0].astype(np.float64))
        self._rebuild_sliders()
        self.redraw()

    def reset(self):
        if self.rig is None:
            return
        for j in self.rig["joints"]:
            j["value"] = 0.0
        self._save()
        self._rebuild_sliders()
        self.pose()

    def _rebuild_sliders(self):
        for h in self.g_sliders:
            h.remove()
        self.g_sliders = []
        joints = self.rig["joints"] if self.rig else []
        parts = ["part 0 (base)"] + [f"part {k + 1} ({j['name']})" for k, j in enumerate(joints)]
        self.g_parent.options = tuple(parts)
        self.g_parent.value = parts[-1] if len(parts) > 1 else parts[0]
        with self.g_joints:
            for k, j in enumerate(joints):
                lo, hi = j["limits"]
                unit = "°" if j["type"] == "revolute" else " m"
                s = self.app.server.gui.add_slider(f"{j['name']}{unit}", min=float(lo), max=float(hi),
                                                   step=0.5 if j["type"] == "revolute" else 0.01,
                                                   initial_value=float(np.clip(j.get("value", 0.0), lo, hi)),
                                                   hint=f"{j['type']} joint {k + 1}: moves part {k + 1} "
                                                        f"(off part {j.get('parent', 0)}).")
                s.on_update(lambda e, k=k: self._slid(k, e.target.value))
                self.g_sliders.append(s)
        self._status()
        self._anim_status()
        self._motion_options()
        self._parts_options()

    def _slid(self, k, value):
        if self.rig is None or k >= len(self.rig["joints"]):
            return
        if self._play is not None and not getattr(self, "_playing_sets", False):
            self.stop_play()  # the user took a slider
        self.rig["joints"][k]["value"] = float(value)
        self.pose()
        self._save_later()

    def _save_later(self):
        """Store the pose a moment after the slider stops (not on every step)."""
        gen = self._save_gen = getattr(self, "_save_gen", 0) + 1

        def go():
            if gen == self._save_gen:
                self._save()
        threading.Timer(0.5, go).start()

    # ------------------------------------------------------------ drawing
    @property
    def shown(self):
        return self.rig is not None and self.g_show.value and self.obj is not None

    def decorate(self, centers, cov, rgb, op, idx, parent=None):
        """The splat arrays to send without the shown rig's object (idx: Gaussians of the file shown; with parent,
        a merged Gaussian goes when most of its sources are the object's)."""
        if not self.shown or self.hidden is None or not len(idx) or idx.max() >= len(self.hidden):
            return centers, cov, rgb, op
        h = self.hidden[idx]
        if parent is not None:
            share = np.bincount(parent, weights=h, minlength=len(centers)) / np.maximum(
                np.bincount(parent, minlength=len(centers)), 1)
            keep = share < 0.5
        else:
            keep = ~h
        return centers[keep], cov[keep], rgb[keep], op[keep]

    def _clear(self):
        scene = self.app.server.scene
        scene.remove_by_name(NODE)
        self.handles = {}

    def redraw(self):
        """Parts and fill again (a new rig, new joints, Show / Colour toggled) and the splat without the object."""
        with self.lock:
            self._clear()
            if self.shown:
                scene = self.app.server.scene
                f = self.fill
                if len(f["means"]):
                    rows = rig.fill_rows(np.dtype([(k, "f4") for k in (
                        "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "f_dc_0",
                        "f_dc_1", "f_dc_2", "opacity")]), f)
                    scene.add_gaussian_splats(f"{NODE}/fill", *gdata.splat_arrays(rows, None))
                c, cov, rgb, op = self.obj
                for k in range(int(self.labels.max()) + 1 if len(self.labels) else 1):
                    m = self.labels == k
                    if not m.any():
                        continue
                    col = rgb[m]
                    if self.g_colour.value:
                        col = (0.4 * col + 0.6 * PART_COLOURS[k % len(PART_COLOURS)]).astype(np.float32)
                    self.handles[k] = scene.add_gaussian_splats(f"{NODE}/part_{k}", centers=c[m], covariances=cov[m],
                                                                rgbs=col, opacities=op[m])
                self.pose()
        self._status()
        self.app.redraw_splat_decorated()

    def pose(self, values=None):
        """Move the parts' nodes to the joints' values (default: the stored ones)."""
        if not self.shown:
            return
        Ts = rig.transforms(self.rig["joints"], values)
        with self.app.server.atomic():
            for k, h in self.handles.items():
                T = Ts[k]
                h.wxyz, h.position = _wxyz(T[:3, :3]), tuple(float(x) for x in T[:3, 3])

    def _draw_picks(self):
        scene = self.app.server.scene
        scene.remove_by_name("/rig_picks")
        for i, p in enumerate(self.picks):
            scene.add_icosphere(f"/rig_picks/p{i}", radius=0.12, color=(255, 220, 0) if i < 2 else (0, 220, 255),
                                position=tuple(float(x) for x in p))
        if len(self.picks) >= 2:
            a, b = self.picks[0], self.picks[1]
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            scene.add_line_segments("/rig_picks/axis", np.array([[a - 2 * d, b + 2 * d]], np.float32),
                                    colors=(255, 220, 0), thickness=3, thickness_units="screen")

    # ------------------------------------------------------------ animation
    def _motion_options(self):
        """The Motion dropdown: demo, keyframes and the vision model's motions; the model's answer shown."""
        names = [m["name"] for m in (self.rig or {}).get("motions", [])]
        opts = ("demo", "keyframes") + tuple(f"planned: {n}" for n in names)
        cur = self.g_anim.value
        self.g_anim.options = opts
        self.g_anim.value = cur if cur in opts else "demo"
        v = (self.rig or {}).get("vlm")
        if not v:
            self.g_vlm_md.content = ""
            return
        lines = [f"**{v.get('machine') or '?'}** (vision model)"]
        for j in v.get("joints", []):
            ok = "" if j.get("axis_ok", True) else f" · ⚠️ axis: {j.get('suggested_axis') or 'not like this'}"
            lines.append(f"- {j.get('joint')}: {j.get('part')}: {j.get('function')}{ok}")
        for m in (self.rig or {}).get("motions", []):
            lines.append(f"- ▶ *{m['name']}*: {m.get('description', '')}")
        self.g_vlm_md.content = "\n".join(lines)

    def plan_motions(self):
        """Ask the vision model for motions; pressed while it runs: Stop."""
        from modelforge import jobs, rig_vlm
        if getattr(self, "_vlm_token", None) is not None:
            threading.Thread(target=self._vlm_token.cancel, daemon=True).start()
            self._anim_status("⏹ stopping the vision model …")
            return
        if self.rig is None or not self.rig["joints"]:
            self._anim_status("Add joints first (pick them or Propose joints).")
            return
        why = rig_vlm.available()
        if why:
            self._anim_status(f"❌ {why}")
            return
        token = self._vlm_token = jobs.Token()
        self.g_plan.label = "⏹ Stop the vision model"
        self.app.semantic.unload("the vision model")  # its GPU memory
        name = self.name

        def work():
            try:
                rig_vlm.plan(self.project, name, on_line=lambda t: self._anim_status(f"⏳ {t}"), token=token)
                if name == self.name:
                    fresh = rig.load(self.project, name)
                    self.rig["vlm"], self.rig["motions"] = fresh.get("vlm"), fresh.get("motions", [])
                    self._motion_options()
                    self._anim_status(f"{len(self.rig['motions'])} motion(s) planned: pick one under Motion, ▶ Play")
            except jobs.Cancelled:
                self._anim_status("⏹ vision model stopped: motions unchanged")
            except Exception as e:  # noqa: BLE001 - shown in the tab
                if token.cancelled:
                    self._anim_status("⏹ vision model stopped: motions unchanged")
                else:
                    traceback.print_exc()
                    self._anim_status(f"❌ vision model: {e}")
            finally:
                self._vlm_token = None
                self.g_plan.label = "Plan motions (vision model)"
        threading.Thread(target=work, daemon=True).start()

    def _keyframes(self):
        n = len(self.rig["joints"])
        return [(list(f) + [0.0] * n)[:n] for f in self.rig.get("keyframes", [])]  # joints added / removed since

    def add_keyframe(self):
        if self.rig is None:
            return
        self.rig.setdefault("keyframes", []).append([float(j.get("value", 0.0)) for j in self.rig["joints"]])
        self._save()
        self._anim_status()

    def clear_keyframes(self):
        if self.rig is None:
            return
        self.rig["keyframes"] = []
        self._save()
        self._anim_status()

    def _anim_status(self, text=None):
        if text is None:
            n = len(self.rig.get("keyframes", [])) if self.rig else 0
            text = f"{n} keyframe(s)" if self.rig else ""
        self.g_anim_md.content = text

    def toggle_play(self):
        if self._play is not None:
            self.stop_play()
        else:
            self.play()

    def play(self):
        if self.rig is None or not self.rig["joints"]:
            self._anim_status("No joints to move: add one first.")
            return
        n = len(self.rig["joints"])
        planned = {f"planned: {m['name']}": m for m in self.rig.get("motions", [])}
        if self.g_anim.value in planned:  # joints added / removed since the plan: padded / cut
            keys = [(float(t), (list(v) + [0.0] * n)[:n]) for t, v in planned[self.g_anim.value]["keys"]]
        elif self.g_anim.value == "demo":
            keys = rig.demo_track(self.rig["joints"])
        else:
            keys = rig.keyframe_track(self._keyframes(), float(self.g_step.value))
        if len(keys) < 2:
            self._anim_status("Add at least one keyframe (set the sliders, Add keyframe) and play again.")
            return
        joints = self.rig["joints"]
        if self.g_loop.value:
            keys = rig.closed(keys, joints)  # a motion ending elsewhere returns smoothly, no jump at the loop
        intro = rig.lead_in(keys, joints, [float(j.get("value", 0.0)) for j in joints])  # from the pose shown
        t_in = intro[-1][0] if intro else 0.0
        stop = self._play = threading.Event()
        self.g_play.label = "⏹ Stop"
        if not self.g_show.value:
            self.g_show.value = True  # nothing to see otherwise

        def run():
            import time
            t, last = 0.0, time.time()
            try:
                while not stop.is_set():
                    now = time.time()
                    t += (now - last) * float(self.g_speed.value)
                    last = now
                    if t < t_in:  # moving to the motion's first pose
                        self._apply(rig.sample(intro, t, loop=False))
                        self._anim_status(f"▶ to the start ({t_in - t:.1f} s)")
                        stop.wait(1 / 30)
                        continue
                    tm, end = t - t_in, keys[-1][0]
                    if tm >= end and not self.g_loop.value:
                        self._apply(rig.sample(keys, end, loop=False))
                        break
                    self._apply(rig.sample(keys, tm))
                    self._anim_status(f"▶ {tm % end:.1f} / {end:.1f} s")
                    stop.wait(1 / 30)
            except Exception:  # noqa: BLE001 - a rig closed under it
                traceback.print_exc()
            finally:
                if self._play is stop:  # ended by itself (not Loop); stop_play clears it otherwise
                    self._play = None
                if self._play is None:  # no newer playback started meanwhile
                    self.g_play.label = "▶ Play"
                    self._sync_sliders()
                    self._anim_status()
        self._player = threading.Thread(target=run, daemon=True)
        self._player.start()

    def _apply(self, values):
        if self.rig is None or len(values) != len(self.rig["joints"]):
            raise RuntimeError("the rig changed")
        for j, v in zip(self.rig["joints"], values):
            j["value"] = float(v)
        self.pose(values)

    def _sync_sliders(self):
        """The sliders to the pose the playback stopped at (without stopping anything)."""
        self._playing_sets = True
        try:
            for s, j in zip(self.g_sliders, self.rig["joints"] if self.rig else []):
                s.value = float(np.clip(j.get("value", 0.0), *j["limits"]))
        finally:
            self._playing_sets = False
        if self.rig is not None:
            self._save()

    def stop_play(self):
        """Stop a playback and wait for it to end (it stores the pose it stopped at: a Delete right after must not
        race it)."""
        if self._play is not None:
            self._play.set()
            self._play = None
            self.g_play.label = "▶ Play"
            player = getattr(self, "_player", None)
            if player is not None and player is not threading.current_thread():
                player.join(timeout=2.0)

    # ------------------------------------------------------------ export
    def export(self):
        if self.rig is None:
            return
        out = os.path.join(self.project, "export", f"{self.name}_posed.ply")
        self._status(f"⏳ writing {os.path.relpath(out, self.project)} …")

        def work():
            try:
                self._save()
                rig.posed_ply(self.project, self.name, out)
                self._status()
                self.g_md.content += f" · exported {os.path.relpath(out, self.project)}"
            except Exception as e:  # noqa: BLE001 - shown in the tab
                traceback.print_exc()
                self._status(f"❌ {e}")
        threading.Thread(target=work, daemon=True).start()
