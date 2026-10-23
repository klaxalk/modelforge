"""Job functions for tests/test_worker.py (run in a spawned process: must be importable)."""
import os
import signal
import time


def talk(project, progress=None, token=None):
    from modelforge import pipeline
    pipeline.log("hello from the job")
    for k in range(3):
        progress({"phase": "step", "k": k, "project": project})
    token.check()


def wait_for_cancel(project, token=None, skip=None):
    for _ in range(600):
        token.check()  # raises Cancelled once the GUI's token is cancelled
        if skip is not None and skip.is_set():
            return
        time.sleep(0.05)


def stuck(project, token=None):
    time.sleep(600)  # ignores the token: an uninterruptible call


def die(project):
    os.kill(os.getpid(), signal.SIGKILL)  # as the kernel's OOM killer does


def fail(project):
    raise ValueError("no images")
