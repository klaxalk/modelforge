"""View-dependent splat colours in the GUI (View -> Splat colours): the SH bands packed after viser's splat buffer
(gui.data.pack_splats), the browser's merge of splat objects (docker/patch_viser_camera.py MERGE_JS, in node) and its
SH sum (gui.data.sh_view_colour, which mirrors the shader) against gsplat's."""
import importlib.util
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from modelforge.gui import data as gdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vertices(n, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    c = (degree + 1) ** 2 - 1
    names = (["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1",
              "f_dc_2", "opacity"] + [f"f_rest_{i}" for i in range(3 * c)])
    v = np.zeros(n, [(k, "f4") for k in names])
    for k in names:
        v[k] = rng.normal(0, 0.3, n)
    v["rot_0"] += 1
    return v


def test_sh_columns_and_packing():
    v = _vertices(50)
    assert gdata.sh_degree_of(v) == 3
    base = gdata.splat_arrays(v, None)[2]
    one = gdata.splat_arrays(v, None, sh_degree=1)[2]
    full = gdata.splat_arrays(v, None, sh_degree=3)[2]
    assert base.shape == (50, 3) and one.shape == (50, 12) and full.shape == (50, 48)
    assert np.array_equal(one, full[:, :12])  # a lower degree is a column prefix
    # coefficient-major, channels interleaved; the PLY is channel-major (15 per channel)
    assert full[7, 3 + 3 * 4 + 2] == v[7]["f_rest_34"]
    q = gdata.sh_dequantize(gdata.sh_quantize(full[:, 3:]))
    assert np.abs(q - full[:, 3:]).max() <= np.abs(full[:, 3:]).max() / 127 * 0.51 + 1e-3

    c, cov, rgb, op = gdata.splat_arrays(v, None, sh_degree=3)
    plain = gdata.pack_splats(c, cov, rgb[:, :3], op)
    buf = gdata.pack_splats(c, cov, rgb, op)
    assert plain.shape == (50, 8) and buf.ndim == 1
    head = int(buf[3])
    k = head & 255
    assert head >> 16 == gdata.SH_TAG and (head >> 8) & 255 == 3 and k == 12 and len(buf) == 50 * (8 + k)
    main = buf[:400].reshape(50, 8).copy()
    main[0, 3] = 0
    assert np.array_equal(main, plain)  # the 8-word block is viser's
    words = buf[400:].reshape(50, k).view(np.uint8)[:, :47]
    assert np.allclose(gdata.sh_dequantize(words), q)


def test_view_colour_matches_gsplat():
    """The shader's sum (sh_view_colour) = gsplat's spherical harmonics + 0.5, clamped (as its rasterizer)."""
    py = "/opt/nerfstudio/bin/python"
    if not os.path.exists(py):
        pytest.skip("needs nerfstudio's venv (CUDA image)")
    v = _vertices(200, seed=1)
    v["f_dc_0"] = v["f_dc_1"] = v["f_dc_2"] = 0.1  # base colours inside [0, 1]: the browser gets them clamped
    rgb = gdata.splat_arrays(v, None, sh_degree=3)[2]
    dirs = np.random.default_rng(2).normal(size=(200, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for degree in (1, 3):
        cols = 3 + 3 * ((degree + 1) ** 2 - 1)
        ours = gdata.sh_view_colour(rgb[:, :cols], dirs)
        coeffs = np.zeros((200, 16, 3), np.float32)
        coeffs[:, 0] = (rgb[:, :3] - 0.5) / gdata.SH_C0
        coeffs[:, 1:(degree + 1) ** 2] = rgb[:, 3:cols].reshape(200, -1, 3)
        code = ("import sys, json, numpy as np, torch, gsplat\n"
                "d = json.load(sys.stdin)\n"
                "c = gsplat.spherical_harmonics(d['deg'], torch.tensor(d['dirs']).float().cuda(), "
                "torch.tensor(d['coeffs']).float().cuda())\n"
                "print(json.dumps(torch.clamp(c + 0.5, 0, 1).cpu().numpy().tolist()))")
        env = {k: x for k, x in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        r = subprocess.run([py, "-c", code], input=json.dumps({"deg": degree, "dirs": dirs.tolist(),
                                                                "coeffs": coeffs.tolist()}),
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        theirs = np.array(json.loads(r.stdout.strip().splitlines()[-1]))
        assert np.abs(ours - theirs).max() < 1e-4, (degree, np.abs(ours - theirs).max())


def test_browser_merge_keeps_viser_layout():
    """MERGE_JS (node): the 8-word block of every object for the sorter, the SH words gathered at the widest stride,
    zeros for objects without them, the tag's word replaced by the group index."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    spec = importlib.util.spec_from_file_location("pv", os.path.join(ROOT, "docker", "patch_viser_camera.py"))
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    js = pv.MERGE_JS.replace('<script id="mf-sh">', "").replace("</script>", "").replace("window.", "globalThis.")
    v = _vertices(5)
    c, cov, rgb, op = gdata.splat_arrays(v, None, sh_degree=1)
    a = gdata.pack_splats(c, cov, rgb, op)  # degree 1: 3 words per Gaussian
    c3, cov3, rgb3, op3 = gdata.splat_arrays(v[:3], None, sh_degree=3)
    b = gdata.pack_splats(c3, cov3, rgb3, op3)  # degree 3: 12 words
    p = gdata.pack_splats(c[:2], cov[:2], rgb[:2, :3], op[:2]).ravel()  # no SH
    code = js + """
const parse = s => Uint32Array.from(JSON.parse(s));
const m = globalThis.__mfMerge({a: parse(process.argv[1]), p: parse(process.argv[2]), b: parse(process.argv[3])});
console.log(JSON.stringify({n: m.numGaussians, g: m.numGroups, buf: Array.from(m.gaussianBuffer),
  gi: Array.from(m.groupIndices), sh: Array.from(m.gaussianBuffer.mfSh), t: m.gaussianBuffer.mfShT,
  d: m.gaussianBuffer.mfDeg}));"""
    r = subprocess.run([node, "-e", code, json.dumps(a.tolist()), json.dumps(p.tolist()), json.dumps(b.tolist())],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.loads(r.stdout)
    assert m["n"] == 10 and m["g"] == 3 and m["t"] == 3 and m["d"] == 3
    assert m["gi"] == [0] * 5 + [1] * 2 + [2] * 3
    buf = np.array(m["buf"], np.uint32).reshape(10, 8)
    assert list(buf[:, 3]) == m["gi"]
    exp = np.r_[a[:40].reshape(5, 8), p.reshape(2, 8), b[:24].reshape(3, 8)]
    exp[:, 3] = m["gi"]
    assert np.array_equal(buf, exp)
    sh = np.array(m["sh"], np.uint32).reshape(10, 12)
    assert np.array_equal(sh[:5, :3], a[40:].reshape(5, 3)) and not sh[:5, 3:].any()
    assert not sh[5:7].any() and np.array_equal(sh[7:], b[24:].reshape(3, 12))


def test_lod_keeps_the_bands(tmp_path):
    """The level of detail stores the higher bands per level (merged like the colour); load_cell gives them by
    degree."""
    import plyfile
    from modelforge import splat_lod
    v = _vertices(20000, seed=3)
    v["x"] = np.random.default_rng(4).uniform(0, 10, 20000)
    v["y"] = np.random.default_rng(5).uniform(0, 10, 20000)
    v["scale_0"] = v["scale_1"] = v["scale_2"] = -4
    os.makedirs(tmp_path / "splat")
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(str(tmp_path / "splat" / "splat_enu.ply"))
    idx = splat_lod.build(str(tmp_path), cell_max=12000)
    cell = idx["cells"][0]
    base = splat_lod.load_cell(str(tmp_path), 0, cell)[2]
    one = splat_lod.load_cell(str(tmp_path), 0, cell, sh_degree=1)[2]
    full = splat_lod.load_cell(str(tmp_path), 0, cell, sh_degree=3)[2]
    assert base.shape[1] == 3 and one.shape[1] == 12 and full.shape[1] == 48
    src, _ = splat_lod.cell_sources(str(tmp_path), 0, cell)
    want = gdata.splat_arrays(v, None, sh_degree=3)[2][src]
    assert np.abs(full[:, 3:] - want[:, 3:]).max() < np.abs(want[:, 3:]).max() / 100
    coarse = splat_lod.load_cell(str(tmp_path), len(cell["levels"]) - 1, cell, sh_degree=3)[2]
    assert coarse.shape[1] == 48 and len(coarse) < len(full)
