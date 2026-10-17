"""Map a nerfstudio splatfacto export (`ns-export gaussian-splat`) back to the project's
ENU world, so the splat lines up with the COLMAP dense mesh and the refined cameras.

nerfstudio normalizes poses with dataparser_transforms.json: x_ns = s * (R x + t).
Inverting that per Gaussian:
    position     x = R^T (x_ns / s - t)
    log-scale    log_scale - log(s)
    rotation     q = q(R^T) (x) q_ns          (wxyz, as gsplat stores them)
    SH colour    band 0 unchanged; band l rotated by D_l with Y_l(R d) = D_l Y_l(d),
                 solved by least squares over sampled directions (exact for SH, and it
                 sidesteps sign conventions of closed-form Wigner matrices)
The real SH basis is the 3DGS/gsplat one (degree <= 3); f_rest is channel-major
(f_rest_{c * K + k}), as nerfstudio's exporter writes it.

Usage:
    python -m modelforge.export.splat_to_enu splat.ply dataparser_transforms.json -o splat_enu.ply
"""

import argparse
import json

import numpy as np
import plyfile

C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(d, degree):
    """Real SH basis without band 0, (N, (degree+1)^2 - 1), 3DGS ordering."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = []
    if degree >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        cols += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)]
    if degree >= 3:
        cols += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                 C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                 C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(cols, -1)


def sh_rotation(R, degree, n=500, seed=0):
    """Block-diagonal X (K x K) such that the SH coefficients f in frame 'ns' become X @ f
    in the frame where directions are d = R^T d_ns, i.e. c'(d) = c(R d)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A, B = sh_basis(d, degree), sh_basis(d @ R.T, degree)  # Y(d), Y(R d)
    K = A.shape[1]
    X = np.zeros((K, K))
    start = 0
    for l in range(1, degree + 1):
        sl = slice(start, start + 2 * l + 1)
        X[sl, sl] = np.linalg.lstsq(A[:, sl], B[:, sl], rcond=None)[0]  # = D_l^T
        start += 2 * l + 1
    return X


def quat_from_rot(R):  # wxyz
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return np.array(q)


def quat_mul(a, b):  # wxyz, broadcasting over rows of b
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def load_transform(path):
    d = json.load(open(path))
    T = np.array(d["transform"], float)
    R, t, s = T[:3, :3], T[:3, 3], float(d["scale"])
    if np.linalg.det(R) < 0:
        raise ValueError("dataparser transform is a reflection; SH rotation assumes a proper rotation")
    return R, t, s


def transform_vertices(v, R, t, s):
    """Return a new structured vertex array mapped from nerfstudio to the original frame."""
    out = v.copy()
    X = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    Xw = (X / s - t) @ R  # = R^T (x/s - t) per row
    for i, c in enumerate("xyz"):
        out[c] = Xw[:, i]
    for i in range(3):
        if f"scale_{i}" in v.dtype.names:
            out[f"scale_{i}"] = v[f"scale_{i}"] - np.log(s)
    if "rot_0" in v.dtype.names:
        q = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
        qn = quat_mul(quat_from_rot(R.T), q)
        for i in range(4):
            out[f"rot_{i}"] = qn[:, i]
    rest = sorted((n for n in v.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest:
        K = len(rest) // 3
        degree = int(round(np.sqrt(K + 1))) - 1
        if (degree + 1) ** 2 - 1 != K:
            raise ValueError(f"{len(rest)} f_rest fields do not match an SH degree")
        X_sh = sh_rotation(R, degree)
        F = np.stack([v[n] for n in rest], -1).astype(np.float64).reshape(-1, 3, K)  # channel-major
        Fn = F @ X_sh.T
        Fn = Fn.reshape(-1, 3 * K)
        for i, n in enumerate(rest):
            out[n] = Fn[:, i]
    return out


def convert(splat_path, transforms_path, out_path):
    R, t, s = load_transform(transforms_path)
    ply = plyfile.PlyData.read(splat_path)
    v = ply["vertex"].data
    new = transform_vertices(v, R, t, s)
    plyfile.PlyData([plyfile.PlyElement.describe(new, "vertex")], text=False).write(out_path)
    return {"gaussians": len(new), "scale": s, "path": out_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("splat")
    ap.add_argument("dataparser_transforms")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    print(convert(args.splat, args.dataparser_transforms, args.output))


if __name__ == "__main__":
    main()
