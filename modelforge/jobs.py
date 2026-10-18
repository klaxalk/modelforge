"""Stopping a running job (GUI Stop button): a token checked between steps, which also terminates
the subprocesses it started.

Long stages run as subprocesses where possible (PatchMatch stereo, Delaunay meshing, texturing,
nerfstudio) so that Stop is immediate. In-process pycolmap calls (feature extraction, matching,
triangulation, fusion) cannot be interrupted; the token is checked as soon as they return. Bundle
adjustment checks it every iteration (its Ceres callback aborts the solve).
"""

import subprocess
import threading


class Cancelled(Exception):
    """The job was stopped by the user."""


class Token:
    def __init__(self):
        self.event = threading.Event()
        self._procs = set()
        self._lock = threading.Lock()

    @property
    def cancelled(self):
        return self.event.is_set()

    def check(self):
        if self.event.is_set():
            raise Cancelled("stopped")

    def cancel(self):
        """Request a stop and terminate the running subprocesses (killed if they ignore it)."""
        self.event.set()
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    def run(self, cmd, on_line=None, env=None):
        """Run cmd, stream its output lines to on_line; Cancelled if stopped, RuntimeError on failure."""
        self.check()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        with self._lock:
            self._procs.add(p)
        tail = []
        try:
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    tail = (tail + [line])[-20:]
                    if on_line:
                        on_line(line)
            rc = p.wait()
        finally:
            with self._lock:
                self._procs.discard(p)
        self.check()
        if rc:
            raise RuntimeError(f"{cmd[0].rsplit('/', 1)[-1]} {cmd[1] if len(cmd) > 1 else ''} exited with code {rc}: "
                               + (tail[-1] if tail else ""))
        return tail


def check(token):
    """token.check() for an optional token."""
    if token is not None:
        token.check()
