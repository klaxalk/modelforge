"""Splat Stop / Pause / Resume: a checkpoint at the current step, the partial export, resume.json.

The trainer and ns-export are stand-ins (a fake nerfstudio bin folder), so this runs without a GPU;
nsplugin/mf_train.py itself needs nerfstudio's venv."""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import plyfile
import pytest

from modelforge import jobs, splat
from tests.test_splat_to_enu import make_splat

FAKE_TRAIN = textwrap.dedent('''\
    #!{python}
    # stand-in for `python -m mf_train <ns-train args>`: progress rows, a checkpoint on request
    import json, os, sys, time
    args = sys.argv[3:]
    json.dump(args, open(os.path.join(os.path.dirname(sys.argv[0]), "last_args.json"), "w"))
    ctrl = os.environ["MF_SPLAT_CONTROL"]
    out, method = args[args.index("--output-dir") + 1], args[args.index("--method-name") + 1]
    run = os.path.join(out, "data", method, str(time.time_ns()))
    os.makedirs(os.path.join(run, "nerfstudio_models"))
    open(os.path.join(run, "config.yml"), "w").write("fake")
    json.dump({{"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "scale": 1.0}},
              open(os.path.join(run, "dataparser_transforms.json"), "w"))
    start = 0
    if "--load-dir" in args:
        d = args[args.index("--load-dir") + 1]
        start = max(int(f[5:-5]) for f in os.listdir(d)) + 1
    total = int(args[args.index("--max-num-iterations") + 1])
    for step in range(start, total):
        print(f"{{step}} ({{100 * step / total:.2f}}%)   1 ms", flush=True)
        time.sleep(0.005)
        if step % 10 == 0 and os.path.exists(ctrl):
            open(os.path.join(run, "nerfstudio_models", f"step-{{step:09d}}.ckpt"), "w").write("x")
            json.dump({{"step": step, "dir": os.path.join(run, "nerfstudio_models")}}, open(ctrl + ".done", "w"))
            sys.exit(0)
    ''')

FAKE_EXPORT = textwrap.dedent('''\
    #!{python}
    import shutil, sys
    args = sys.argv[1:]
    shutil.copy({src!r}, args[args.index("--output-dir") + 1] + "/splat.ply")
    ''')


@pytest.fixture
def fake_ns(tmp_path, monkeypatch):
    project, bin_dir = tmp_path / "p", tmp_path / "bin"
    os.makedirs(project)  # no splat/ yet, as in a fresh project
    os.makedirs(bin_dir)
    open(project / "transforms.json", "w").write(json.dumps({"frames": []}))
    open(project / "sparse_pc.ply", "w").write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
                                               "property float y\nproperty float z\nend_header\n0 0 0\n")
    v, _ = make_splat(np.random.default_rng(0), n=50)
    src = str(tmp_path / "src.ply")
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(src)
    for name, text in (("python", FAKE_TRAIN.format(python=sys.executable)),
                       ("ns-export", FAKE_EXPORT.format(python=sys.executable, src=src))):
        open(bin_dir / name, "w").write(text)
        os.chmod(bin_dir / name, 0o755)
    monkeypatch.setattr(splat, "NS_BIN", str(bin_dir))
    monkeypatch.setattr(splat, "available", lambda: None)
    return str(project), str(bin_dir)


def stop_when(job, steps, at, pause=None):
    def watch():
        while not steps or steps[-1] < at:
            time.sleep(0.01)
        if pause is not None:
            pause.set()
        job.cancel()
    threading.Thread(target=watch, daemon=True).start()


def test_pause_resume_stop(fake_ns):
    project, bin_dir = fake_ns
    steps = []
    job, pause = jobs.Token(), threading.Event()
    stop_when(job, steps, 100, pause)
    with pytest.raises(splat.Paused, match="paused at step"):
        splat.run(project, "splatfacto", 5000, job=job, pause=pause, on_line=lambda m: None,
                  on_progress=lambda s, p: steps.append(s), schedule={"strategy": "mcmc", "max_gaussians": 10})
    info = splat.resume_info(project)
    assert info and info["step"] >= 100 and info["step"] % 10 == 0 and info["iterations"] == 5000
    assert info["schedule"]["strategy"] == "mcmc"
    assert not os.path.exists(os.path.join(project, "splat", "splat_enu.ply"))  # paused: no export
    assert not os.path.exists(os.path.join(project, splat.CONTROL))

    # resume with its own settings from its checkpoint; Stop exports the partial splat
    paused_at, steps[:] = info["step"], []
    job = jobs.Token()
    stop_when(job, steps, paused_at + 50)
    with pytest.raises(splat.StoppedPartial, match="partial"):
        splat.resume(project, job=job, on_line=lambda m: None, on_progress=lambda s, p: steps.append(s))
    args = json.load(open(os.path.join(bin_dir, "last_args.json")))
    assert args[args.index("--load-dir") + 1] == info["load_dir"]
    assert args[args.index("--pipeline.model.strategy") + 1] == "mcmc"
    assert steps[0] == paused_at + 1  # continued, not restarted
    report = json.load(open(os.path.join(project, "report.json")))["splat"]
    assert report["stopped_at_step"] > paused_at and report["resumed_from"] == info["load_dir"]
    assert os.path.exists(os.path.join(project, "splat", "splat_enu.ply")) and report["num_gaussians"] == 50
    assert splat.resume_info(project)["step"] == report["stopped_at_step"]  # can continue again

    # a changed transforms.json (new cameras / scene frame) cannot continue the old run
    t = os.path.join(project, "transforms.json")
    os.utime(t, (time.time() + 10, time.time() + 10))
    assert splat.resume_info(project) is None


def test_finished_run_clears_resume(fake_ns):
    project, _ = fake_ns
    os.makedirs(os.path.join(project, "splat"))
    json.dump({"load_dir": "/nonexistent"}, open(os.path.join(project, splat.RESUME), "w"))
    stats = splat.run(project, "splatfacto", 30, on_line=lambda m: None)
    assert "stopped_at_step" not in stats and not os.path.exists(os.path.join(project, splat.RESUME))


def test_stop_before_checkpoint_is_a_plain_stop(fake_ns):
    project, _ = fake_ns
    job = jobs.Token()
    job.cancel()
    with pytest.raises(jobs.Cancelled) as e:
        splat.run(project, "splatfacto", 30, job=job, on_line=lambda m: None)
    assert type(e.value) is jobs.Cancelled and splat.resume_info(project) is None


def test_token_soft_stop(tmp_path):
    """cancel() asks a process started with soft_stop and waits for it instead of terminating it."""
    flag, done = str(tmp_path / "flag"), str(tmp_path / "done")
    code = (f"import os, time\nwhile not os.path.exists({flag!r}): time.sleep(0.01)\n"
            f"time.sleep(0.3); open({done!r}, 'w').write('saved')\nprint('bye')")
    job = jobs.Token()
    threading.Timer(0.3, job.cancel).start()
    with pytest.raises(jobs.Cancelled):
        job.run([sys.executable, "-c", code], soft_stop=lambda: open(flag, "w").write("1"), soft_timeout=10)
    assert open(done).read() == "saved"  # it finished by itself
    job = jobs.Token()  # without soft_stop: terminated
    threading.Timer(0.3, job.cancel).start()
    t0 = time.time()
    with pytest.raises(jobs.Cancelled):
        job.run([sys.executable, "-c", "import time; time.sleep(30)"])
    assert time.time() - t0 < 10


def test_train_command_load_dir():
    cmd = splat.train_command("/p", iterations=100, load_dir="/p/nerfstudio/x/nerfstudio_models")
    assert cmd[cmd.index("--load-dir") + 1] == "/p/nerfstudio/x/nerfstudio_models"
    assert cmd[-3] == "nerfstudio-data"


@pytest.mark.skipif(not os.path.exists("/opt/nerfstudio/bin/python"), reason="needs nerfstudio's venv")
def test_mf_train_imports_in_nerfstudio_venv():
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
    env["PYTHONPATH"] = splat.PLUGIN_DIR
    r = subprocess.run(["/opt/nerfstudio/bin/python", "-c",
                        "import mf_train, nerfstudio.engine.trainer as t; "
                        "assert t.Trainer.train_iteration is mf_train.train_iteration; "
                        "assert t.Trainer._load_checkpoint is mf_train.load_checkpoint"],
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_empty_sparse_cloud_is_a_clear_error(fake_ns):
    """SfM without 3D points (too little overlap) used to surface as nerfstudio's IndexError."""
    project, _ = fake_ns
    open(os.path.join(project, "sparse_pc.ply"), "w").write(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with pytest.raises(RuntimeError, match="no 3D points"):
        splat.run(project, "splatfacto", 30, on_line=lambda m: None)


def test_failure_names_the_module(tmp_path):
    with pytest.raises(RuntimeError, match="exited with code") as e:
        jobs.Token().run([sys.executable, "-m", "json.tool", str(tmp_path / "missing.json")])
    assert str(e.value).startswith("json.tool ")  # not "python -m"
