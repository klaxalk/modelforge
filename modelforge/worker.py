"""Heavy in-process stages in a job process of their own (docs/todo_robustness.md §1).

Why (2026-10-23): the kernel OOM-killed the GUI twice with kladno_dira's refine inside it; the GUI and the job died
together. Now `run` starts the stage function in a spawned child process: an out-of-memory kill takes only the job
(the GUI reports it), the GUI's own memory stays small, and Stop also ends pycolmap calls that cannot be interrupted
(features, matching, triangulation, fusion: the process is terminated after STOP_GRACE s).

The GUI side calls `run(target, kwargs, callbacks, token, events)`:
  target     "module:function", e.g. "modelforge.pipeline:stage_refine"
  kwargs     picklable arguments; the placeholders REPORT / PRIORS / META / TOKEN are built in the child (a Report of
             kwargs["project"], the project's priors / metadata, the child's jobs.Token)
  callbacks  {argument name: GUI function}: the child passes a proxy that forwards each call through a queue; a
             pycolmap.Reconstruction in the arguments becomes {"poses": {image name: cam_from_world params}} (live BA
             events at most every LIVE_EVERY s)
  events     {argument name: threading.Event} (Skip, Skip this BA round): set in the GUI, set in the child
The child's log lines (pipeline.LISTENERS) arrive through the GUI's on_log. Cancelling the token cancels the child's
(a BA aborts within an iteration); a child that has not ended STOP_GRACE s later is terminated.

A job outlives the GUI (2026-10-24): killed by a signal, Python skips its exit handlers, so the child is not
terminated and finishes as an orphan (only its pipes to the GUI break). With a project the child keeps
PROJECT/job.json (JOB: pid, target, label, start, then the result) and PROJECT/job.log (its log lines), and watches
PROJECT/job.cancel (Stop without the GUI that started it). `detached(project)` tells a restarted GUI about it
(running, or finished meanwhile); `FileToken` stops it; `clear` forgets a finished one.
"""

import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback

REPORT, PRIORS, META, TOKEN = "<report>", "<priors>", "<meta>", "<token>"
LIVE_EVERY = 0.25  # s between forwarded live BA events
ONE_SHOT = {"skip_round"}  # events the job clears itself after use: cleared in the GUI once forwarded
STOP_GRACE = 60.0  # s a cancelled job may take to stop by itself before it is terminated


JOB, LOG, CANCEL = "job.json", "job.log", "job.cancel"  # in the project folder


def _write_job(project, d):
    path = os.path.join(project, JOB)
    with open(path + ".part", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(path + ".part", path)


def alive(pid):
    """The process exists and is not a zombie."""
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def detached(project):
    """PROJECT/job.json as a dict with "state": "running" (its process lives), "finished" (a result is recorded) or
    "lost" (the process is gone without one: killed); None without a job file."""
    try:
        d = json.load(open(os.path.join(project, JOB)))
    except (OSError, ValueError, TypeError):
        return None
    d["state"] = "finished" if d.get("result") else "running" if alive(d.get("pid", -1)) else "lost"
    return d


def clear(project):
    for name in (JOB, CANCEL):
        try:
            os.remove(os.path.join(project, name))
        except OSError:
            pass


class FileToken:
    """A jobs.Token-like stop for a job the GUI did not start (re-attached): cancel() asks it through job.cancel."""

    def __init__(self, project):
        self.project, self.cancelled = project, False

    def cancel(self):
        self.cancelled = True
        open(os.path.join(self.project, CANCEL), "w").close()


class Killed(RuntimeError):
    """The job process died without a word (killed by the system: out of memory?)."""


def _sanitize(x):
    import pycolmap
    if isinstance(x, pycolmap.Reconstruction):
        return {"poses": {x.image(i).name: list(x.frame(x.image(i).frame_id).rig_from_world.params)
                          for i in x.reg_image_ids()}}
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_sanitize(v) for v in x)
    return x


def _child(target, kwargs, callbacks, events, out, inbox, label=None):
    from modelforge import jobs, pipeline
    token = jobs.Token()
    flags = {name: threading.Event() for name in events}
    project = kwargs.get("project")
    job, logf = None, None
    if project and os.path.isdir(project):  # survives the GUI: job.json + job.log, Stop through job.cancel
        clear(project)
        job = {"pid": os.getpid(), "target": target, "label": label or target.split(":")[1], "started": time.time(),
               "log": LOG}
        _write_job(project, job)
        logf = open(os.path.join(project, LOG), "w", buffering=1)

    def listen():
        while True:
            try:
                msg = inbox.get()
            except (EOFError, OSError):  # the GUI is gone: the job goes on
                return
            if msg == "cancel":
                token.cancel()
            elif msg in flags:
                flags[msg].set()

    def watch_cancel():
        while not token.cancelled:
            if os.path.exists(os.path.join(project, CANCEL)):
                token.cancel()
            time.sleep(1.0)

    threading.Thread(target=listen, daemon=True).start()
    if job is not None:
        threading.Thread(target=watch_cancel, daemon=True).start()

    def on_line(line):
        out.put(("log", line))
        if logf is not None:
            try:
                logf.write(line.rstrip("\n") + "\n")
            except (OSError, ValueError):
                pass
    pipeline.LISTENERS.append(on_line)
    pri = meta = None
    if PRIORS in kwargs.values() or META in kwargs.values():
        from modelforge.sfm import priors as priors_mod
        pri, meta = priors_mod.load(project)
    real = {REPORT: lambda: pipeline.Report(project), PRIORS: lambda: pri, META: lambda: meta, TOKEN: lambda: token}
    kw = {k: real[v]() if isinstance(v, str) and v in real else v for k, v in kwargs.items()}
    last = {}

    def proxy(name):
        def call(*args, **kw2):
            ev = args[0] if args and isinstance(args[0], dict) else None
            if ev is not None and ev.get("phase") == "ba":  # live BA poses: at most every LIVE_EVERY s
                now = time.time()
                if now - last.get(name, 0.0) < LIVE_EVERY:
                    return None
                last[name] = now
            out.put(("call", name, _sanitize(args), _sanitize(kw2)))
            return None
        return call
    for name in callbacks:
        kw[name] = proxy(name)
    for name in events:
        kw[name] = flags[name]
    module, fn = target.split(":")
    result = ("error", "ended without a result", "")
    try:
        import importlib
        getattr(importlib.import_module(module), fn)(**kw)
        result = ("done",)
    except jobs.Cancelled:
        result = ("cancelled",)
    except BaseException as e:  # noqa: BLE001 - everything goes back to the GUI
        result = ("error", f"{type(e).__name__}: {e}", traceback.format_exc())
    finally:
        if job is not None:  # for a GUI that restarted meanwhile
            _write_job(project, {**job, "result": result[0], "error": result[1] if len(result) > 1 else "",
                                 "ended": time.time()})
        out.put(result)
        out.put(("exit",))


def run(target, kwargs, callbacks=None, token=None, events=None, on_log=None, label=None):
    """Run target(**kwargs) in a job process (see the module doc); returns when it ended. Raises jobs.Cancelled when
    stopped, Killed when the process died (signal), RuntimeError with the child's message when it failed."""
    from modelforge import jobs
    callbacks, events = dict(callbacks or {}), dict(events or {})
    ctx = mp.get_context("spawn")
    out, inbox = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_child, args=(target, kwargs, list(callbacks), list(events), out, inbox, label),
                    daemon=True)
    p.start()
    sent, cancel_at, result = set(), None, None
    try:
        while True:
            try:
                msg = out.get(timeout=0.2)
            except queue.Empty:
                msg = None
            if msg is not None:
                kind = msg[0]
                if kind == "log":
                    if on_log:
                        on_log(msg[1])
                elif kind == "call":
                    try:
                        callbacks[msg[1]](*msg[2], **msg[3])
                    except Exception:  # noqa: BLE001 - a GUI callback never stops the job
                        traceback.print_exc()
                elif kind == "exit":
                    break
                else:
                    result = msg
                continue
            for name, ev in events.items():  # Skip / Skip this BA round pressed in the GUI
                if ev.is_set() and name not in sent:
                    inbox.put(name)
                    if name in ONE_SHOT:
                        ev.clear()
                    else:
                        sent.add(name)
            if token is not None and token.cancelled and cancel_at is None:
                inbox.put("cancel")
                cancel_at = time.time()
            if cancel_at is not None and time.time() - cancel_at > STOP_GRACE and p.is_alive():
                p.terminate()  # stuck in a call that cannot be interrupted
                p.join(5)
                raise jobs.Cancelled()
            if not p.is_alive():
                while True:  # what it said before it went
                    try:
                        m = out.get_nowait()
                    except queue.Empty:
                        break
                    if m[0] == "log" and on_log:
                        on_log(m[1])
                    elif m[0] in ("done", "cancelled", "error"):
                        result = m
                break
    finally:
        p.join(10)
        if p.is_alive():
            p.kill()
        if kwargs.get("project"):  # the GUI that started it saw the end: nothing to re-attach to
            d = detached(kwargs["project"])
            if d is not None and d.get("pid") == p.pid:
                clear(kwargs["project"])
    if result is None:
        code = p.exitcode
        if cancel_at is not None:
            raise jobs.Cancelled()
        sig = -code if code is not None and code < 0 else None
        name = signal.Signals(sig).name if sig else f"exit code {code}"
        raise Killed(f"the job process was killed ({name}): out of memory? The GUI kept running; see `dmesg` / "
                     "the memory monitor" if sig == signal.SIGKILL else f"the job process ended without a result ({name})")
    if result[0] == "cancelled":
        raise jobs.Cancelled()
    if result[0] == "error":
        print(result[2], flush=True)  # the child's traceback, in the server console
        raise RuntimeError(result[1])
