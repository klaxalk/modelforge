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
"""

import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback

REPORT, PRIORS, META, TOKEN = "<report>", "<priors>", "<meta>", "<token>"
LIVE_EVERY = 0.25  # s between forwarded live BA events
ONE_SHOT = {"skip_round"}  # events the job clears itself after use: cleared in the GUI once forwarded
STOP_GRACE = 60.0  # s a cancelled job may take to stop by itself before it is terminated


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


def _child(target, kwargs, callbacks, events, out, inbox):
    from modelforge import jobs, pipeline
    token = jobs.Token()
    flags = {name: threading.Event() for name in events}

    def listen():
        while True:
            msg = inbox.get()
            if msg == "cancel":
                token.cancel()
            elif msg in flags:
                flags[msg].set()

    threading.Thread(target=listen, daemon=True).start()
    pipeline.LISTENERS.append(lambda line: out.put(("log", line)))
    project = kwargs.get("project")
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
    try:
        import importlib
        getattr(importlib.import_module(module), fn)(**kw)
        out.put(("done",))
    except jobs.Cancelled:
        out.put(("cancelled",))
    except BaseException as e:  # noqa: BLE001 - everything goes back to the GUI
        out.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))
    finally:
        out.put(("exit",))


def run(target, kwargs, callbacks=None, token=None, events=None, on_log=None):
    """Run target(**kwargs) in a job process (see the module doc); returns when it ended. Raises jobs.Cancelled when
    stopped, Killed when the process died (signal), RuntimeError with the child's message when it failed."""
    from modelforge import jobs
    callbacks, events = dict(callbacks or {}), dict(events or {})
    ctx = mp.get_context("spawn")
    out, inbox = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_child, args=(target, kwargs, list(callbacks), list(events), out, inbox), daemon=True)
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
