"""Joint proposals for a rig from UniRig (VAST-AI-Research/UniRig, MIT; the user asked, 2026-10-27): UniRig puts
a skeleton into any mesh, learned from artists' rigs (Articulation-XL), without being told what the object is.

**Pipeline** (`propose`): the rig's object as a mesh (rig.object_mesh: the splat-mesh TSDF of its Gaussians alone)
-> a Y-up GLB centred on its base -> UniRig's skeleton prediction (launch/inference/generate_skeleton.sh, ~5 s on
the GPU) -> the FBX read in UniRig's Python (export/unirig_fbx.py) -> the bones mapped back to ENU by the similarity
that takes the FBX's (normalized) mesh onto the mesh UniRig extracted (raw_data.npz, same vertices) -> joints.

**Bones to joints** (`limbs`, every terminal chain of the skeleton): UniRig rigs like a creature: a spine and
limbs. On kladno_dopravnik's screener its
spine ran through the body and three of its four limbs along the three conveyors (each split into 2-3 bones, as a
leg would be); its skin weights cut the body into meaningless patches. So every limb (a chain branching off the
spine, the longest root-to-leaf chain) that leaves the body and reaches out >= 2 m horizontally becomes one joint:
origin where it leaves the body (nothing of the object beside it any more; UniRig's limbs start at its spine, inside
the body), axis horizontal and across the limb (a lift hinge), the moving part's seed halfway out; UniRig's skeleton
differs from run to run, so it runs with 10 seeds and keeps the limbs found in >= 2 of them (`consensus`); the part itself is cut by rig.assign_parts (plane + connectivity), and the
user reviews it (Colour the parts) and removes or re-picks wrong ones.

**Install** (not in the image): `scripts/install_unirig` makes build/unirig/ with a standalone Python 3.11 (bpy 4.2
needs it; the image has 3.12), torch 2.7.1 cu128, spconv, torch_scatter / torch_cluster, flash-attn 2.8.3 (prebuilt
wheel), bpy 4.2 and the UniRig clone; weights from Hugging Face (VAST-AI/UniRig) into build/hf_cache.
MODELFORGE_UNIRIG overrides the folder.
"""

import json
import os
import shutil

import numpy as np

from modelforge import jobs, rig

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.environ.get("MODELFORGE_UNIRIG", os.path.join(_REPO, "build", "unirig"))
HF_HOME = os.path.join(_REPO, "build", "hf_cache")
LIMITS = [-60.0, 60.0]
SEEDS = (12345, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # UniRig's skeleton varies from run to run: limbs >= 2 runs agree on


def available():
    """None when UniRig is installed, else why not."""
    if not os.path.exists(os.path.join(HOME, "venv", "bin", "python")):
        return f"UniRig is not installed (run scripts/install_unirig; looked in {HOME})"
    if not os.path.exists(os.path.join(HOME, "UniRig", "run.py")):
        return f"the UniRig code is missing in {HOME}/UniRig"
    return None


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env.update(PATH=os.path.join(HOME, "venv", "bin") + os.pathsep + env.get("PATH", ""), HF_HOME=HF_HOME,
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")  # its checkpoints pickle their config (VAST-AI's weights)
    return env


def to_glb(mesh_ply, out):
    """ENU mesh -> Y-up GLB (x, z, -y) centred on its base; returns the offset (ENU = blender coords + offset; Blender
    reads the GLB back Z-up, i.e. as ENU - offset)."""
    import trimesh

    from modelforge import sfm_mesh
    d, f = sfm_mesh.read_mesh(mesh_ply)
    V = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    off = np.r_[np.median(V[:, :2], 0), V[:, 2].min()]
    P = V - off
    col = np.stack([d[k] for k in ("red", "green", "blue")], 1) if "red" in d.dtype.names else None
    trimesh.Trimesh(np.c_[P[:, 0], P[:, 2], -P[:, 1]], f, vertex_colors=col, process=False).export(out)
    return off


def similarity(A, B):
    """(s, R, a0, b0) with B ~ s R (A - a0) + b0 (Umeyama, no reflection)."""
    a0, b0 = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - a0).T @ (B - b0))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    s = float((S * np.diag(D)).sum() / ((A - a0) ** 2).sum())
    return s, R, a0, b0


def limbs(heads, tails, parents, means=None, min_reach=2.0, step=0.5, slab=0.25, near=1.0, side=4.0, clear=0.1):
    """Joint proposals from a skeleton in ENU (module doc): a list of dicts origin, axis, dir, seed, reach.

    Each limb (a terminal chain: from a bone whose parent branches down to a leaf; UniRig's limbs can be longer
    than its spine, so no spine is picked) is sampled every `step` metres along its line. With the object's
    Gaussian centres (means), per sample the Gaussians in a slab across the limb (|along| < slab) are counted within
    `near` of the limb's line (the limb itself) and `near`..`side` beside it (everything else: the body). The limb is
    clipped where it has no Gaussians of its own (UniRig's bones overshoot the tips); the hinge sits at the first
    sample after which the side count stays below `clear` x its peak: where the limb leaves the body. A limb that
    never gets clear lies along or over the body and is left out; so is one reaching out less than min_reach metres
    horizontally (UniRig's "legs" down to the tracks). Without means: the hinge at the limb's root."""
    n = len(parents)
    kids = [[c for c in range(n) if parents[c] == b] for b in range(n)]
    out = []
    for leaf in (b for b in range(n) if not kids[b]):
        chain = [leaf]  # up to the first bone whose parent branches (or the root's child)
        while parents[chain[-1]] >= 0 and len(kids[parents[chain[-1]]]) == 1 and parents[parents[chain[-1]]] >= 0:
            chain.append(int(parents[chain[-1]]))
        if parents[chain[-1]] < 0:  # an unbranched skeleton: no limbs
            continue
        chain = chain[::-1]
        poly = np.concatenate([heads[chain[:1]], tails[chain]])
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        if seg.sum() < min_reach:
            continue
        t = np.arange(0, seg.sum() + 1e-9, step)
        along = np.stack([np.interp(t, np.r_[0, np.cumsum(seg)], poly[:, k]) for k in range(3)], 1)
        u = poly[-1] - poly[0]
        u /= np.linalg.norm(u)
        o, end = along[0], along[-1]
        if means is not None and len(means):
            own, beside = [], []
            for q in along:
                dq = means - q
                a = dq @ u
                r = np.linalg.norm(dq - a[:, None] * u, axis=1)
                sl = np.abs(a) < slab
                own.append(int((sl & (r < near)).sum()))
                beside.append(int((sl & (r >= near) & (r < side)).sum()))
            own, beside = np.array(own), np.array(beside)
            has = np.flatnonzero(own >= 5)
            if not len(has):
                continue
            last = has[-1]
            busy = np.flatnonzero(beside[:last + 1] >= clear * max(beside.max(), 1))
            k = busy[-1] + 1 if len(busy) else 0
            if k >= last:  # never clear of the body
                continue
            o, end = along[k], along[last]
        d = end - o
        d[2] = 0.0
        reach = float(np.linalg.norm(d))
        if reach < min_reach:
            continue
        d /= reach
        axis = np.cross(d, [0.0, 0.0, 1.0])
        seed = (o + end) / 2
        out.append({"origin": o, "axis": axis, "dir": d, "seed": seed, "reach": reach,
                    "bones": [int(x) for x in chain]})
    return out


def consensus(runs, near=1.5, min_runs=2):
    """Limbs found in at least min_runs of the runs (lists of limbs): grouped greedily, farthest reaching first, by
    direction (within ~37°, cos > 0.8; opposite limbs share a hinge axis up to its sign) and line (the other's seed within `near` metres of the limb's
    line; the seed slides along a limb from run to run), at most one per run; each group averaged."""
    flat = sorted(((r, p) for r, ps in enumerate(runs) for p in ps), key=lambda x: -x[1]["reach"])
    used, out = [False] * len(flat), []
    for i, (ri, pi) in enumerate(flat):
        if used[i]:
            continue
        group, runs_in = [pi], {ri}
        used[i] = True
        for j, (rj, pj) in enumerate(flat):
            if used[j] or rj in runs_in:
                continue
            if pj["dir"] @ pi["dir"] > 0.8 and abs((pj["seed"] - pi["origin"])[:2] @ pi["axis"][:2]) < near:
                group.append(pj)
                runs_in.add(rj)
                used[j] = True
        if len(runs_in) >= min_runs:
            d = np.sum([g["dir"] for g in group], 0)
            d /= np.linalg.norm(d)
            out.append({"origin": np.mean([g["origin"] for g in group], 0), "dir": d,
                        "axis": np.cross(d, [0.0, 0.0, 1.0]),
                        "seed": np.mean([g["seed"] for g in group], 0), "reach": float(np.mean([g["reach"] for g in group])),
                        "runs": len(runs_in)})
    return out


def propose(project, name, on_line=print, token=None, seeds=SEEDS, min_runs=2):
    """Run UniRig on the rig's object with several seeds and return the joints of the limbs most runs agree on
    (rig.new_joint dicts, rest pose), farthest reaching first. Files in rig/<name>/unirig/."""
    import plyfile
    import shlex

    from modelforge import splat_mesh
    why = available()
    if why:
        raise RuntimeError(why)
    token = token or jobs.Token()
    work = os.path.join(rig.rig_dir(project, name), "unirig")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    on_line("unirig: a mesh of the object (TSDF of its Gaussians)")
    mesh = rig.object_mesh(project, name, on_line=lambda s: None, token=token)
    glb = os.path.join(work, "object.glb")
    off = to_glb(mesh, glb)
    v = plyfile.PlyData.read(splat_mesh.splat_path(project))["vertex"].data
    idx = rig.arrays(project, name)[0]
    means = np.stack([v["x"][idx], v["y"][idx], v["z"][idx]], 1)
    runs, log = [], []
    for k, seed in enumerate(seeds):
        token.check()
        on_line(f"unirig: skeleton {k + 1} / {len(seeds)}")
        fbx = os.path.join(work, f"skeleton_{seed}.fbx")
        try:
            token.run(["bash", "-c", f"cd {shlex.quote(os.path.join(HOME, 'UniRig'))} && bash "
                       f"launch/inference/generate_skeleton.sh --input {shlex.quote(os.path.abspath(glb))} "
                       f"--output {shlex.quote(os.path.abspath(fbx))} --seed {int(seed)}"], log.append, env=_env(),
                      group=True)  # its launch scripts start Python children: Stop ends them all
        finally:
            open(os.path.join(work, "unirig.log"), "w").write("\n".join(log))
        if not os.path.exists(fbx):
            raise RuntimeError("UniRig wrote no skeleton (see rig/<name>/unirig/unirig.log)")
        raw = np.load(os.path.join(work, "object", "raw_data.npz"), allow_pickle=True)["vertices"]
        npz = fbx[:-4] + ".npz"
        token.run([os.path.join(HOME, "venv", "bin", "python"), os.path.join(_REPO, "modelforge", "export",
                   "unirig_fbx.py"), fbx, npz], env=_env())
        sk = np.load(npz)
        if len(sk["vertices"]) != len(raw):
            raise RuntimeError("UniRig's mesh does not match the extracted one: cannot map its frame back")
        sc, R, a0, b0 = similarity(sk["vertices"], raw)
        heads, tails = (sc * (P - a0) @ R.T + b0 + off for P in (sk["heads"], sk["tails"]))  # -> Blender -> ENU
        np.savez(npz[:-4] + "_enu.npz", heads=heads, tails=tails, parents=sk["parents"])
        runs.append(limbs(heads, tails, sk["parents"], means))
    props = sorted(consensus(runs, min_runs=min(min_runs, len(runs))), key=lambda p: -p["reach"])
    j = lambda d: {k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in d.items()}  # noqa: E731
    json.dump({"runs": [[j(p) for p in r] for r in runs], "limbs": [j(p) for p in props]},
              open(os.path.join(work, "limbs.json"), "w"), indent=1)
    on_line(f"unirig: {len(props)} limb(s) found in >= {min(min_runs, len(runs))} of {len(runs)} runs "
            f"(per run: {', '.join(str(len(r)) for r in runs)})")
    return [rig.new_joint(p["origin"], p["origin"] + p["axis"], p["seed"], limits=list(LIMITS),
                          name=f"unirig {k + 1}") for k, p in enumerate(props)]
