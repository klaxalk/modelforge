"""Dump the Gaussians of a splatfacto checkpoint to .npz, for the GUI's live preview.

Runs in nerfstudio's venv (/opt/nerfstudio/bin/python), so it imports only torch and numpy, not
modelforge. Output arrays are in nerfstudio's normalized frame, in the same conventions as
ns-export's PLY: means, log-scales, wxyz quaternions, opacity logits, SH band 0 (features_dc).

Usage:
    /opt/nerfstudio/bin/python modelforge/export/ckpt_dump.py step-000002000.ckpt out.npz
"""

import os
import sys

import numpy as np
import torch

KEYS = ("means", "scales", "quats", "opacities", "features_dc")


def main(ckpt, out):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    params = {k.rsplit(".", 1)[-1]: v for k, v in d["pipeline"].items() if ".gauss_params." in k}
    arrays = {k: params[k].float().numpy() for k in KEYS}
    tmp = out + ".tmp.npz"
    np.savez(tmp, step=np.int64(d.get("step", -1)), **arrays)
    os.replace(tmp, out)  # the GUI never sees a half-written file


if __name__ == "__main__":
    main(*sys.argv[1:3])
