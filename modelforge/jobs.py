"""Stopping a running job (GUI Stop button): a token checked between steps, which also terminates
the subprocesses it started.

Long stages run as subprocesses where possible (PatchMatch stereo, Delaunay meshing, texturing,
nerfstudio) so that Stop is immediate. In-process pycolmap calls (feature extraction, matching,
triangulation, fusion) cannot be interrupted; the token is checked as soon as they return. Bundle
adjustment checks it every iteration (its Ceres callback aborts the solve).
"""

import contextlib
import os
import re
import signal
import subprocess
import sys
import threading

_TAP = threading.Lock()


class Cancelled(Exception):
    """The job was stopped by the user."""


class Token:
    def __init__(self):
        self.event = threading.Event()
        self._procs = set()
        self._soft = {}  # process -> (callable asking it to end by itself, seconds to wait for that)
        self._groups = set()  # processes started in their own process group: stopped as a group
        self._lock = threading.Lock()

    @property
    def cancelled(self):
        return self.event.is_set()

    def check(self):
        if self.event.is_set():
            raise Cancelled("stopped")

    def cancel(self):
        """Request a stop and terminate the running subprocesses (killed if they ignore it). Processes
        started with soft_stop are asked first and get its timeout to end by themselves."""
        self.event.set()
        with self._lock:
            procs, soft = list(self._procs), dict(self._soft)
        for p in procs:
            if p.poll() is None and p not in soft:
                self._signal(p, signal.SIGTERM)
        for p in procs:
            if p in soft and p.poll() is None:
                ask, timeout = soft[p]
                try:
                    ask()
                    p.wait(timeout=timeout)
                except Exception:  # noqa: BLE001 - asking failed or took too long: terminate as usual
                    p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._signal(p, signal.SIGKILL)

    def _signal(self, p, sig):
        """Send sig to a process, or to its whole process group when it was started with group=True."""
        try:
            if p in self._groups:
                os.killpg(p.pid, sig)
            else:
                p.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            pass

    def run(self, cmd, on_line=None, env=None, soft_stop=None, soft_timeout=300, group=False):
        """Run cmd, stream its output lines to on_line; Cancelled if stopped, RuntimeError on failure.
        soft_stop: called on cancel instead of terminating, then up to soft_timeout s to exit by itself
        (splat training saves a checkpoint first). group: start it in its own process group and stop the whole
        group (a shell script's children, e.g. UniRig's launch scripts, would outlive the shell)."""
        self.check()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                             start_new_session=group)
        with self._lock:
            self._procs.add(p)
            if group:
                self._groups.add(p)
            if soft_stop:
                self._soft[p] = (soft_stop, soft_timeout)
        tail = []
        try:
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    if re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip():  # not just a colour reset
                        tail = (tail + [line])[-20:]
                    if on_line:
                        on_line(line)
            rc = p.wait()
        finally:
            with self._lock:
                self._procs.discard(p)
                self._groups.discard(p)
                self._soft.pop(p, None)
        self.check()
        if rc:
            prog = cmd[0].rsplit("/", 1)[-1]
            if len(cmd) > 2 and cmd[1] == "-m":  # python -m module: name the module
                prog, cmd = cmd[2], cmd[2:]
            raise RuntimeError(f"{prog} {cmd[1] if len(cmd) > 1 else ''} exited with code {rc}: "
                               + (tail[-1] if tail else ""))
        return tail


@contextlib.contextmanager
def stderr_lines(on_line):
    """While inside, every line written to file descriptor 2 (COLMAP's log from in-process pycolmap calls, which
    report progress nowhere else) also goes to on_line; it still reaches the real stderr. Process-wide: a second
    tap while one is open does nothing."""
    if not _TAP.acquire(blocking=False):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)

    def pump():
        buf = b""
        with os.fdopen(r, "rb", buffering=0) as f:
            while True:
                chunk = f.read(65536)
                if not chunk:  # fd 2 restored: the pipe's last writer is gone
                    break
                os.write(saved, chunk)
                *lines, buf = (buf + chunk).split(b"\n")
                for line in lines:
                    try:
                        on_line(line.decode(errors="replace"))
                    except Exception:  # noqa: BLE001 - a progress display must not break the job
                        pass
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        th.join(timeout=5)
        os.close(saved)
        _TAP.release()


def check(token):
    """token.check() for an optional token."""
    if token is not None:
        token.check()
