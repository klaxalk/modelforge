# TODO: articulated and moving objects in the splat (rigs)

The user asked on 2026-10-27. Cut an object out of the splat (say an excavator), give it moving parts, and put it
back: an excavator you can pose, a tree that sways in the wind. The aim is that this works on any object, without a
template written for each kind.

## Status: first version built (2026-10-27, on kladno_dopravnik)

The user asked to work on it with the kladno_dopravnik world (a tracked mobile screener with three conveyors, 40
photos, 532k Gaussians). Built: steps 1 (cut-out + fill), 4 (joints by clicks, parts, review by colour) and 5 for
rigid parts (viser nodes, sliders, export), and UniRig joint proposals (step 3a, below). Not yet: soft motion / wind,
completion of unseen surfaces, the photo-mask cut-out, a local vision-language model for joint types / axes.

**Code:** `modelforge/rig.py` (cut-out, ground, fill, joints, parts, posing, PLY export), `modelforge/gui/rig.py`
(the **Rig** tab, left panel after Search), `App.decorate` (the search's highlight, then the shown rig's object
removed, also for level-of-detail cells), Ctrl + click goes to the Rig tab while it picks. Tests: `tests/test_rig.py`.
Experiment scripts: `build/rig/exp.py PROJECT NAME QUERY [GROW]` (cut-out + PLYs), `build/rig/look.sh NAME CAMS`
(renders object / removed / full at training cameras through `build/tile_diag/render.py`), `build/rig/gui_smoke.py`
(the tab headless: search, cut, a scripted joint, a slider, export).

**Use:** Search tab: query (e.g. "a machine"), Search, select the object → Rig tab: **Cut out the search's object**
→ Parent part, Type → **Pick the joint (Ctrl + click ×3)**: two points on the hinge axis, then one on the moving
part → its slider moves it; **Colour the parts** to check the cut; **Remove the last joint**, **Rest pose**,
**Export the posed splat** (`export/<rig>_posed.ply`, full SH turned per part).

**Cut-out, as built** (`rig.cut_out`): seeds = the object's matches (>= the search threshold, in its marker
cylinder); grown over 20 cm voxels above the ground through voxels whose best relevance is >= 0.5 (better than the
generic words); then the ground is estimated again without the grown core's footprint (filled in under it) and the
growing is repeated from the core within that footprint; finally every Gaussian in the grown voxels (dilated by one)
or under the object's voxel columns, more than 0.25 m above the ground, faint ones too. The search box grows x 1.5
while the object reaches its edge. kladno_dopravnik, "a machine" (threshold 0.64): 28,849 Gaussians in 0.6 s, the
whole screener with its three conveyors and the tracks; fill 14,423 Gaussians.
- What went wrong on the way (all fixed): a ground from all Gaussians sat on the machine (a cell under it sees mostly
  the machine), so its faint and lower parts stayed as a blurred ghost; a 2 m minimum filter went under the
  undulating floor, so the floor joined (237k Gaussians); `cv2.inpaint` returns garbage on float grids (±1 m) — the
  ground is filled by `rig.fill_holes` (normalized spreading + a Laplace relaxation).
- Left: a flat remnant at ground level where the tracks stood (the lowest 0.25 m), a few bits of gravel on the cut;
  the fill under the machine is the ground's colours inpainted (a dark smudge where its shadow was).

**Joints, as built:** a joint = type (revolute in degrees / prismatic in metres), parent part, axis (origin + the
direction to the second click), limits (±90° / ±2 m), value. Its part = the parent's Gaussians on the third click's
side of a plane through the axis (`rig.cut_normal`: the plane holds the axis, or lies across it when the click is
along the axis: a turntable), the 12 cm voxel-connected piece around the click. In viser each part is its own splat
node (`/rig/part_k`) moved by its node pose, so a slider sends only a pose; the splat is drawn without the object
and with the fill. Checked on kladno_dopravnik: a conveyor raised 30° (3,028 Gaussians) and the rear conveyor
(7,345), exported and rendered: the parts stay whole and their colour turns with them.

### Animation in the GUI (2026-10-27)

The user asked for automatic animation; "play in the GUI is fine now". Rig tab → **Animate**: Motion **demo** (each
joint in turn from rest to its upper limit, its lower limit and back at 30°/s or 0.5 m/s, a 0.5 s hold, then all
together; `rig.demo_track`) or **keyframes** (poses added with **Add keyframe** from the sliders, stored in rig.json
`keyframes`, `Seconds between keyframes` apart, back to the first; `rig.keyframe_track`), eased by smoothstep
(`rig.sample`), **▶ Play / ⏹ Stop**, Loop, Speed. A thread moves the parts' node poses at 30 Hz (every browser sees
it); moving a slider stops it; on stop the sliders take the pose it stopped at. Keyframes made before joints were
added or removed are padded / cut. **No jumps** (the user saw a planned motion snap back to its start at the
loop): with Loop on, a motion that ends in another pose is played backwards after a 0.5 s hold (`rig.closed`: it
unfolds the way it folded, at its own pace), and Play first moves smoothly from the pose shown to the motion's first
pose (`rig.lead_in`, at the demo's rates). On the user's "Elevate Conveyors" at 4x speed the largest step between
frames is 1.2°. Not yet: driving the whole object along a path, video export, motion from
observed poses or a vision-language model (see "Order").

### Template + guided clicks (2026-10-27)

After the pointing experiment (below) the user chose guided clicks. Rig tab → **Joints from a template (vision
model)** → **Ask the vision model for the template** (`rig_vlm.template`: up to 3 photos cropped around the object,
TEMPLATE_PROMPT with a worked crawler-crane example; ~30 s, Stop): the machine and its chain (parts, joints with
type slew / hinge / slide, parent, child, and a phrase naming where the pin is). The tab lists the chain and asks,
one at a time, "Ctrl + click <the pin phrase>" then "a point in the middle of <the part>"; **Skip this one** (a pin
not found: that joint is left out) / **Cancel**. After the last click `rig_vlm.template_joints` builds the joints
(slew: vertical axis; hinge: horizontal, across the direction from its pin to its part, snapped parallel to its
parent hinge within 30°) and replaces the rig's (old ones kept as `joints_before`); Colour the parts turns on.
A joint the model missed (it keeps leaving out an excavator's stick) is added with Pick the joint.
- **Turntable cut** (`rig.footprint`): a slew's plane is horizontal at the ring, but an excavator's arm hangs below
  it in front; the base keeps only what is below the ring *within* its footprint (the radius where the Gaussians
  below the ring thin out to < 10 % of their densest ring). Before, the arm below the ring stayed with the tracks
  and the bucket's part was empty.
- Headless on kladno_bagr's excavator with the model's real template ("Komatsu PC700LC": tracks → cabin slew →
  boom → bucket) and simulated clicks: 3 joints, the house and arm swing together, the boom lifts. Test:
  `test_turntable_keeps_the_hanging_arm_and_template_joints`.
- **The user's try: "hard to hit the pins, the body was split into two"** (fixed):
  - clicks count within ~1.2 % of the distance (>= 15 cm; was 0.4 % / 5 cm), **Undo last click**;
  - a turntable's pin is now asked as "the machine's side exactly at the gap where the upper body sits on the
    undercarriage" (the ring itself is hidden); `rig.ring`: the click's height, the axis through the upper body's
    centre just above it (median, re-centred: four clicks around the excavator agree within 5 cm). The shape shows
    no neck to find the ring by (cross-section ~14 m² from the tracks to the house);
  - the split: `cut_normal` turned the turntable's cut *vertical* (the "upper body" click was 2 m off the axis: the
    rule for a door's vertical hinge); slew joints now carry `"slew": true` and always cut across the axis; below
    the ring the base keeps only what stands on the floor within its footprint (`_on_floor`; a counterweight hangs).
    Clicks at -3.4 .. -2.6 m: the whole house + arm vs the tracks.

### Template + pointing by the vision model (2026-10-27, experiment, no button)

The user asked whether a small LLM could synthesize the excavator template itself, and to try "option 3": the model
writes the template and points to the pins in images, the points lifted to 3D. `rig_vlm.rig_from_template`
(TEMPLATE_PROMPT, POINT_PROMPT, `photo_views`, `lift`, `consensus_point`, `template_joints`); `export/vlm_ask.py` now
answers many queries per model load and reports each image's resized size (Qwen2.5-VL's points are pixels of it);
`export/rig_views.py` also writes the object's expected depth.
- **Pointing in renders** of the cut-out (grey background, 4 sides): useless; points bunch near the image centre,
  "the bucket" lands on the house, boom foot and bucket pin get the same pixel (`build/rig/point_test.py`).
- **Pointing in photos** (cropped around the object): much better (`build/rig/point_photo.py`): bucket, elbow and
  bucket pin roughly right in 2 of 3 photos, the boom foot wrong.
- **The whole chain** (kladno_bagr's yellow excavator; up to 6 photos >= 40° apart, only 4 qualified; each point the
  largest group of photos agreeing within 1 m, lifted by the object's depth rendered from the photo's camera): the
  model names the machine right ("Komatsu PC700LC", the photos read "PC 700") but its structure is wrong (slew between
  tracks and undercarriage, the cab hinged on the undercarriage, no stick); with a worked example (a crawler crane)
  in the prompt the chain is right (tracks -> upper body slew -> boom -> bucket) but the stick is still missing. Only
  the bucket and its pin get >= 2 photos agreeing (3 of 4); the slew centre and the boom foot are single guesses.
- Verdict: a 7B model is not enough to place pins. Next to try: a larger model (Qwen2.5-VL-32B in 4 bit to fit 24 GB),
  more photos per object, or the template with guided clicks (the model names the pins, the user clicks them).

### Parts from PartField (2026-10-27)

The user asked for something that skeletonizes an excavator automatically (UniRig gave it a tentacle-like chain of
34-37 short bones, or a "hand" of short leaves: no limbs). **PartField** (nv-tlabs, ICCV 2025; **NVIDIA License,
non-commercial use only**; `scripts/install_partfield` -> build/partfield, its own Python 3.11 + torch 2.7.1) splits
any shape into parts by a learned feature field; `modelforge/rig_partfield.py`, Rig tab → **Parts (PartField)**:
**Propose parts** (~14 s; replaces the rig's joints, the old ones kept as `joints_before_parts`; Stop while it
runs), **Parts** slider (instant re-clustering), **Split it in two** (one part only: an excavator's arm).
- Input: the object's Gaussian centres as a point cloud (`is_pc`): better than its TSDF mesh (decimated to 40k
  faces, 467k ran out of memory), which merged the boom with the house. Features (448-d per Gaussian) stored with the
  rig; parts by NumPy KMeans (k-means++, the best of 3) on the normalized features, as PartField clusters.
- **Joints from parts** (`joints_from_parts`): contact graph (0.3 m voxels), the largest part the base, the others hung
  on their strongest contact (>= 2 % of the smaller part's voxels); low parts at the floor (top < floor + 1.5 m,
  height < 1.5 m) stay with the base (gravel on the cut). Pin = centre of the child's Gaussians next to the parent;
  a flat horizontal contact covering > 30 % of the child's footprint = turntable (vertical axis), else a hinge
  across the child's direction. The joint stores its part; `rig.assign_parts` uses it instead of a plane cut.
- **kladno_bagr's yellow excavator** (23k Gaussians): 4 parts = undercarriage, house (slew), whole arm (hinge),
  gravel (base). Posed: the house turns on the tracks, the arm lifts at its foot. At 7 parts the boom separates
  from stick + bucket but the house and tracks fragment: Split is for that.
- **Not repeatable:** PartField's features differ slightly per run, and so does the clustering: a second run cut
  the house at its roof (−0.9..0.6 m instead of −1.8..0.6 m), which then is a hinge, not a turntable. Change the
  part count or split to steer; a joint-type override (hinge / slew) in the tab is open.

### Motions planned by a local vision model (2026-10-27)

The user asked for automatic, meaningful animation and chose the local vision-language model. **Qwen2.5-VL-7B-Instruct**
(Apache-2.0, offline, ~16 GB in build/hf_cache, `scripts/install_unirig --vlm`), run in UniRig's environment
(transformers 4.51, flash-attn; + accelerate) by `export/vlm_ask.py`; `modelforge/rig_vlm.py`; Rig tab → Animate →
**Plan motions (vision model)** (~1 min, ~17 GB of GPU memory; the Search tab's text model is unloaded first).
- **Images** (`views`, `export/rig_views.py` in nerfstudio's venv): the object alone from 4 sides (azimuths 45° off
  the axes, 35° up, framed to fill ~85 %), moving parts tinted, every hinge drawn white with a black outline and
  labelled J<k>; a close-up per joint (only its part, lightly tinted, from the side showing it largest); the project
  photo that shows the object largest.
- **Prompt**: the joints (type, which part, range), the sign convention (positive raises the far end; vertical axes
  counter-clockwise from above; slides outwards), then: think first (what machine, what each part is and how such a
  part moves, which motions in which order), then "JSON:" with machine, joints (part, function, axis ok?) and 2-3
  motions as steps of seconds + angles. `signs` maps the convention onto each joint (its seed turned +10°: up or
  down), `motions` clips to the limits and drops motions that move nothing. Stored in rig.json `vlm` / `motions`;
  the Motion dropdown lists them as "planned: <name>", the tab shows the answer.
- **On kladno_dopravnik's screener** (the user's rig, 2 hinges on the side conveyors + an empty one): first answer
  "Crusher", parts "green part", transport = conveyors *lowered* 60°, 90° beyond a ±60° limit. With per-joint
  close-ups and reasoning first: "Quarry Conveyor System", both parts "Conveyor Belt", "Raise Conveyors: folds the
  conveyors up for transport" (+, i.e. up, correct). A 7B model names parts plausibly but its motions stay simple
  (one or two moves); a larger model (Qwen2.5-VL-32B, 4-bit to fit 24 GB) is the next thing to try.
- Found on the way: a joint whose plane cuts nothing gets an empty part (the user's J3 on dopravnik); the prompt now
  tells the model to ignore such joints. The tab should warn about them (open).

### Network test: Articulate-Anything and UniRig (2026-10-27)

The user asked to try Articulate-Anything. **Not run:** it does not rig the given mesh; it assembles an object from
PartNet-Mobility parts (~2000 household objects) by looking for the most similar one, and needs a cloud
vision-language model (Gemini / GPT-4o / Claude API key; none here, and it would send renders of the site out).
Articulate AnyMesh (CoRL 2025, rigs any mesh) has no released code. Then, with the user's go-ahead: **UniRig**.

**UniRig, as built** (`modelforge/rig_unirig.py`, Rig tab **Propose joints (UniRig)**, `scripts/install_unirig`):
- Installed under build/unirig/ (gitignored): a standalone Python 3.11 through uv (bpy 4.2 needs 3.11, the image has
  3.12), torch 2.7.1 cu128, flash-attn 2.8.3 (prebuilt wheel), spconv cu126, torch_scatter / cluster, bpy 4.2;
  weights (VAST-AI/UniRig on Hugging Face) in build/hf_cache. Two snags: its extract script runs `pip install psutil`
  (falls through to the image's venv, refused, harmless; psutil is installed in its own), and its skin checkpoint
  needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (pickled config).
- Input: a TSDF mesh of the cut-out alone (`rig.object_mesh`: the splat-mesh code on a workspace whose splat is only
  the object; 459k faces, 4 cm voxel), as a Y-up GLB. Output mapped back to ENU by the similarity between the FBX's
  normalized mesh and UniRig's extracted mesh (same vertices; exact to 1e-5).
- **What UniRig makes of the screener:** a creature skeleton (16-22 bones), a spine through the body and limbs, three
  of them along the three conveyors (each split into 2-3 bones like a leg), plus "legs" down to the tracks. Its
  skinning (also run) paints the body in meaningless patches, so its weights are not used.
- **It varies a lot from run to run** (16 / 19 / 22 bones; each run finds 0-2 usable limbs). So `propose` runs it
  with 10 seeds (~3.5 min in all) and keeps limbs found in >= 2 runs (`consensus`: same signed direction, the
  other's seed within 1.5 m of the limb's line).
- **Bones to hinges** (`limbs`): every terminal chain (UniRig's limbs can be longer than its spine); sampled every
  0.5 m, the object's Gaussians counted in a 0.5 m slab across it, within 1 m of its line (its own) and 1-4 m beside
  it (the body). Clipped where its own run out (the bones overshoot tips); hinged just past the last sample with
  >= 10 % of the peak side count (where it leaves the body); dropped if it never gets clear or reaches < 2 m out.
  Axis horizontal across it, seed halfway out; the part is cut by `rig.assign_parts` as for clicked joints.
  A Gaussian-count sphere was tried first: FLANN's radius search caps at 64, and splats are hollow surfaces.
- **Result on kladno_dopravnik:** both side conveyors, each found in 4 of 10 runs, hinged where they leave the body
  (2,877 and 2,647 Gaussians); raised 35° they fold up whole (`build/unirig/out/posed.jpg`). The rear conveyor (over
  the body) and the screen box are not proposed. Only the lift axis is guessed: a slew (vertical axis) or a fold
  across would need re-picking. Test: `test_unirig_limbs_become_hinges_where_they_leave_the_body`.
- Diagnostics in build/unirig/: `draw2.py` (skeleton + hinges over the mesh), `paint.py` (Gaussians by bone),
  `profile2.py` (the slab counts).

## The goal in one picture

```
splat ──cut out──▶ object's Gaussians ──mesh──▶ rig (parts + joints, or skeleton + weights)
   │                       │                                  │
   └── hole filled         └──────── bound to control nodes ◀─┘
                                           │
                                 motion: joint sliders, a script, wind
                                           │
                               rendered: viser (rigid) / browser deformation / UE5
```

## Kinds of motion

| | Mechanical (excavator, crane, door) | Vegetation and soft (tree, bush, flag) |
|---|---|---|
| Bones | a few rigid parts in a chain | many branches in a hierarchy |
| Binding | each Gaussian to exactly one part | blended over nearby branches, plus some free motion per Gaussian (leaves) |
| Motion | joint angles (sliders, a script) | wind: procedural (or simulated) |
| Skeleton from | a network, a template, captured motion or clicks | a network, or grown in the canopy (space colonization) |

**One representation for both: sparse control nodes** (as SC-GS, Sparse-Controlled Gaussians): a few hundred nodes
inside the object, each Gaussian blended over its nearest ones (means and rotations moved, SH bands rotated: the
rotation code exists in `export/splat_to_enu.py`). Rigid articulation = nodes clustered into groups that move
together (the groups are the parts, their relative motion the joints); soft motion = nodes moving freely; static =
nodes that don't move. A template or a network's output is then only a constraint on the nodes.

## Steps

### 1. Cut out an object cleanly

- Start from the Search tab: a text query marks the Gaussians, `semantic.objects` gives the connected group
  (`modelforge/semantic.py`).
- Its edges are rough (2048 px RADIO patches; matches spill onto the ground around a machine, the tracks merge with
  the ground). Sharpen with SAM 2 masks in the photos, started from the semantic hit (a point or box prompt per
  photo), lifted to the Gaussians by voting across views (Gaussian Grouping / SAGA style).
- Output: the object as its own splat (ENU) + the rest of the scene without it.
- **Fill the hole:** the ground under it was never seen. Seed it from the SfM mesh (patched holes), refine with
  Difix (`modelforge/difix.py`). Its baked shadow stays.

### 2. A mesh of the object

The rigging networks take a mesh (or images), not a splat. Options:
- the splat mesh of just the object's Gaussians (`modelforge/splat_mesh.py`, TSDF from rendered depth);
- an image-to-3D mesh (TRELLIS, Hunyuan3D, ...) from the photos or renders, registered to the splat (ICP with
  scale). This one is complete, including the sides never seen (see step 6).

### 3. The rig

Sources, interchangeable; each gives parts / joints or a skeleton + weights:

**a. Networks that rig without being told what the object is** (they learned a prior from large training sets):

| Family | Examples | Output | Trained on | Fits |
|---|---|---|---|---|
| Auto-rigging of any mesh | RigNet (2020), MagicArticulate (2025, Articulation-XL from Objaverse), UniRig (2025, Tripo) | skeleton + skinning weights; no joint types / limits | artist rigs: characters, animals, some objects | trees, creatures; machines maybe (character-like skeletons, soft skinning) |
| Articulated objects (URDF) | URDFormer, Real2Code, SINGAPO, NAP, CAGE; Articulate-Anything, Articulate AnyMesh (with a vision-language model) | rigid parts + revolute / prismatic joints, axes, limits | mostly PartNet-Mobility (~2000 household objects, ~46 categories) | machines, doors; heavy machinery is out of their training range (the VLM-based ones generalize better) |
| Part motion | DragAPart, Puppet-Master (2024) | images / video of a part moving when dragged | Objaverse animations | a generic motion source for d. |

All research code: check dependencies, GPU memory and **licences** (some non-commercial) before anything goes into
the image. The field moves fast: search for newer models when this starts (this list is from mid-2026 knowledge).

**b. Captured motion (no prior at all).** On a construction site the same machine stands in different poses in
different passes or on different days. Two states of one object give its rigid parts (what moved together) and
joint axes (their relative motion; PARIS-style two-state reconstruction). Today the pipeline treats such changes as
errors (isolated photos, floaters, smears); here they would be the evidence. Only where the footage shows motion,
but then the result is real.

**c. Templates.** E.g. an excavator: undercarriage → upper body (turns about a vertical axis) → boom → stick →
bucket (hinges). Fitted to the cut-out: the turning axis from the body's symmetry, the pivots where parts meet, the
part boundaries from part queries ("bucket", "boom", "cab") or clicks. Robust, but one per kind; keep as an accuracy
upgrade, not the main path.

**d. A generated motion video.** Render the cut-out from a few views → an image-to-video model animates it ("moves
the way it normally does") → a point tracker (CoTracker) follows the pixels → tracks lifted onto the Gaussians → the
control nodes fitted (PhysDreamer, DreamGaussian4D work like this). Plausible motion, not necessarily right; can
differ between views and runs.

**e. A vision-language model naming the joints.** "Where does this bend and how?" Good as a first guess, vague about
axes.

**f. Trees without a network.** Space colonization: the canopy's Gaussians are attraction points, the trunk starts
where the tree meets the ground, branches grow towards the points. From a drone the trunk is mostly hidden under the
canopy, so the skeleton is plausible, not the real one; enough for natural sway.

### 4. Binding and the review editor

- Rig → control nodes → Gaussians: each Gaussian takes the weights / part label of the nearest mesh surface (hard for
  URDF parts, blended for an auto-rigger's skin).
- **Review editor** (the networks will be partly right): show the parts and axes in the 3D view; merge / split
  parts, drag an axis, pin a node, set limits. This more than any model makes the general version usable.

### 5. Motion and rendering

- **Rigid parts work in viser today:** one splat per part under a node hierarchy
  (`/object/house/boom/stick/bucket`); a joint moves one node's pose; a slider per joint. Check first that viser
  sorts several splat objects together (the parts interpenetrate at the joints).
- **Soft motion (wind) moves every Gaussian every frame:** sending 100k+ positions over the websocket at 30 Hz will
  not work. The deformation must run where the rendering does:
  - in the browser: send the rig (nodes, weights, wind parameters) once, the page computes the displacement each
    frame (a deeper patch than `docker/patch_viser_camera.py`);
  - in the UE5 plugin (`docs/todo_unreal_splat.md`): wind and skeletal animation exist there; the rig exported as
    a skeletal asset.
- **Wind** as in games (SpeedTree style): the trunk bends slowly; branches sway as damped oscillators, each level
  faster and smaller; leaf Gaussians flutter on their own; one gust field over the scene so neighbouring trees move
  together and gusts travel. Physical simulation (PhysGaussian: MPM on the Gaussians; learned material as in
  PhysDreamer) is an option for later: slow, research-grade.

### 6. Unseen surfaces

Moving a part reveals what no photo saw: inside the joints, under the boom, the ground under the bucket. Use the
image-to-3D mesh (step 2) only there: registered to the splat, seeded as Gaussians where the real splat has no
coverage. Real Gaussians wherever there are photos. Trees reveal little; machines need this most.

## Known limits

- Lighting is baked: a moving part keeps its old shading, a swaying tree's shadow stays still (gentle motion hides
  it; see `docs/todo_lighting.md`).
- A tree splat is blobs, not leaves: sway looks good, close-up flutter looks like a moving cloud.
- A machine that moved during the flight is already smeared in the splat (static-scene assumption); a 4D splat
  would be a much bigger project.
- Invented surfaces (steps 3d, 6) are plausible, not measured: mark them (a flag per Gaussian) so exports can say so.

## First test (the next step when this starts)

1. Cut the excavator out of kladno_dira (only its tile; no long runs on the whole project), or kladno_dopravnik's
   crusher (smaller, quicker, less articulated). A berlin tree for the vegetation side.
2. Make its mesh (splat mesh of the cut-out).
3. Run **UniRig** (automatic skeleton) and **Articulate-Anything** (parts + joints) on it; maybe MagicArticulate.
4. Compare: are the parts and axes where the machine bends? Which family to build on decides steps 3–4.

## Order

1. Cut-out + hole fill.
2. Control-node binding + rendering a fixed test motion (rigid in viser first).
3. Review editor.
4. Motion sources, cheapest and most reliable first: user clicks → captured two-state motion → networks → a
   generated video → a VLM's guess.
5. Soft motion and wind (browser deformation or UE5).
6. Completion of unseen surfaces.
