"""Stopping jobs: modelforge.jobs.Token, and bundle adjustment aborting on it."""

import threading
import time

import numpy as np
import pytest

from modelforge import jobs
from modelforge.sfm import refine, robust
from tests.test_m3_robust import make_priors, make_scene, set_poses


def test_run_streams_and_fails():
    t = jobs.Token()
    lines = []
    assert t.run(["sh", "-c", "echo a; echo b"], lines.append) == ["a", "b"] and lines == ["a", "b"]
    with pytest.raises(RuntimeError, match="exited with code 3"):
        t.run(["sh", "-c", "echo oops; exit 3"])


def test_cancel_kills_the_subprocess():
    t = jobs.Token()
    threading.Timer(0.5, t.cancel).start()
    t0 = time.time()
    with pytest.raises(jobs.Cancelled):
        t.run(["sleep", "30"])
    assert time.time() - t0 < 5
    with pytest.raises(jobs.Cancelled):  # a cancelled token refuses new work
        t.run(["true"])
    jobs.check(None)  # no token: nothing to check


def test_ba_aborts_on_stop():
    rec, gt = make_scene(seed=7)
    priors = make_priors(gt, np.random.default_rng(7), pos_sigma=0.05, rot_sigma_deg=0.2)
    set_poses(rec, priors)
    t = jobs.Token()
    seen = []

    def on_iteration(summary):
        seen.append(summary.iteration)
        if len(seen) == 3:
            t.event.set()  # Stop pressed during the solve

    ba = refine.prior_ba(rec, priors, robust.PriorModel(), on_iteration=on_iteration, token=t, settle=None)
    assert ba["iterations"] == 3 and ba["termination"] == "USER_FAILURE"


def test_ba_skip_keeps_the_current_state():
    """Skip ends the solve successfully (the state so far is kept); Stop aborts it."""
    rec, gt = make_scene(seed=8)
    priors = make_priors(gt, np.random.default_rng(8), pos_sigma=0.05, rot_sigma_deg=0.2)
    set_poses(rec, priors)
    skip = threading.Event()

    def on_iteration(summary):
        if summary.iteration == 2:
            skip.set()

    ba = refine.prior_ba(rec, priors, robust.PriorModel(), on_iteration=on_iteration, skip=skip, settle=None)
    assert ba["termination"] == "SKIPPED" and ba["usable"] and ba["iterations"] == 3
