"""Read a UniRig result (FBX) in UniRig's own Python (Blender as a module, bpy 4.2): the armature's bones (names,
parent index, head and tail in world coordinates) and the mesh's vertices (to map UniRig's normalized frame back),
plus the skin weights per bone when the FBX has them (modelforge/rig_unirig.py runs it).

    python unirig_fbx.py IN.fbx OUT.npz
"""
import sys

import bpy
import numpy as np


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=sys.argv[1])
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    M = arm.matrix_world
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    out = {"names": np.array(names),
           "parents": np.array([names.index(b.parent.name) if b.parent else -1 for b in bones]),
           "heads": np.array([list(M @ b.head_local) for b in bones]),
           "tails": np.array([list(M @ b.tail_local) for b in bones])}
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if meshes:
        o = meshes[0]
        out["vertices"] = np.array([list(o.matrix_world @ v.co) for v in o.data.vertices])
        groups = {g.index: g.name for g in o.vertex_groups}
        W = np.zeros((len(out["vertices"]), len(names)), np.float32)
        for v in o.data.vertices:
            for g in v.groups:
                if groups.get(g.group) in names:
                    W[v.index, names.index(groups[g.group])] = g.weight
        if W.any():
            out["weights"] = W
    np.savez(sys.argv[2], **out)
    print(f"[unirig] {len(names)} bones", flush=True)


if __name__ == "__main__":
    main()
