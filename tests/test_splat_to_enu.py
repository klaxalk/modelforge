"""splat_to_enu: the converted splat must render identically in the original frame."""

import json

import numpy as np
import plyfile

from modelforge.export import splat_to_enu as S


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_to_rot(q):  # wxyz rows -> (N, 3, 3)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def make_splat(rng, n=200, degree=3):
    K = (degree + 1) ** 2 - 1
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * K)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    for nm in names:
        v[nm] = rng.normal(size=n)
    return v, K


def covariances(v):
    Rg = quat_to_rot(np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64))
    S2 = np.exp(2 * np.stack([v[f"scale_{i}"] for i in range(3)], -1).astype(np.float64))
    return np.einsum("nij,nj,nkj->nik", Rg, S2, Rg)


def sh_colors(v, K, d):
    degree = int(round(np.sqrt(K + 1))) - 1
    F = np.stack([v[f"f_rest_{i}"] for i in range(3 * K)], -1).astype(np.float64).reshape(-1, 3, K)
    return np.einsum("nck,mk->nmc", F, S.sh_basis(d, degree))  # view-dependent part only


def test_splat_maps_back_exactly(tmp_path):
    rng = np.random.default_rng(0)
    R, t, s = random_rotation(rng), rng.normal(size=3), 0.0123
    v, K = make_splat(rng)
    w = S.transform_vertices(v, R, t, s)

    X_ns = np.stack([v[c] for c in "xyz"], -1).astype(np.float64)
    X_w = np.stack([w[c] for c in "xyz"], -1).astype(np.float64)
    assert np.allclose(s * (X_w @ R.T + t), X_ns, atol=1e-4)  # forward map reproduces the input

    # covariance: Sigma_ns = s^2 R Sigma_w R^T
    assert np.allclose(s ** 2 * np.einsum("ij,njk,lk->nil", R, covariances(w), R), covariances(v), rtol=1e-3, atol=1e-6)

    # view-dependent colour: c_w(d) == c_ns(R d) for any direction d in the original frame
    d = rng.normal(size=(50, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.allclose(sh_colors(w, K, d), sh_colors(v, K, d @ R.T), atol=1e-4)
    for nm in ("f_dc_0", "opacity"):
        assert np.array_equal(w[nm], v[nm])

    # file round trip through the CLI entry point
    src, tr, dst = tmp_path / "s.ply", tmp_path / "t.json", tmp_path / "o.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(src)
    json.dump({"transform": np.c_[R, t].tolist(), "scale": s}, open(tr, "w"))
    S.convert(str(src), str(tr), str(dst))
    out = plyfile.PlyData.read(dst)["vertex"].data
    assert np.allclose(out["x"], w["x"]) and np.allclose(out["f_rest_20"], w["f_rest_20"])
