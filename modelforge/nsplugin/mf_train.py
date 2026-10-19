"""ns-train with two fixes for pausing and resuming splat training (run in /opt/nerfstudio by
modelforge/splat.py as `python -m mf_train <ns-train args>`; nothing installed).

1. Checkpoint on request. When the file named by $MF_SPLAT_CONTROL appears, the trainer saves a
   checkpoint at the current step, writes {"step", "dir"} to $MF_SPLAT_CONTROL.done and exits 0.
   nerfstudio saves only every steps_per_save and nothing on Ctrl-C, so Stop / Pause would otherwise
   lose up to that many steps (or everything before the first save).
2. Resume (--load-dir). splatfacto's load_state_dict replaces its Gaussian parameters with new
   tensors of the checkpoint's size, but the optimizers built before still hold the initial tensors,
   and the loaded Adam state is attached to those: the first step fails on a shape mismatch or trains
   tensors nobody renders. The optimizers are rebound to the model's parameters, keeping that state.
"""

import json
import os
import sys

from nerfstudio.engine import trainer as _trainer

CONTROL = os.environ.get("MF_SPLAT_CONTROL")
CHECK_EVERY = 10  # steps between looks at the control file

_train_iteration = _trainer.Trainer.train_iteration
_load_checkpoint = _trainer.Trainer._load_checkpoint


def train_iteration(self, step):
    out = _train_iteration(self, step)
    if CONTROL and step % CHECK_EVERY == 0 and os.path.exists(CONTROL):
        self.save_checkpoint(step)
        with open(CONTROL + ".done", "w") as f:
            json.dump({"step": step, "dir": str(self.checkpoint_dir)}, f)
        print(f"[modelforge] checkpoint saved at step {step}; training ends here", flush=True)
        sys.stdout.flush()
        os._exit(0)  # no final save, eval or viewer wait
    return out


def load_checkpoint(self):
    model = getattr(self.pipeline, "module", self.pipeline).model
    params = getattr(model, "gauss_params", None)
    before = {}
    if params is not None:
        for name, opt in self.optimizers.optimizers.items():
            if name in params:
                before[name] = opt.param_groups[0]["params"][0]
    _load_checkpoint(self)
    for name, old in before.items():
        new = params[name]
        if new is old:
            continue
        opt = self.optimizers.optimizers[name]
        state = opt.state.pop(old, None)
        opt.param_groups[0]["params"] = [new]
        if state is not None:
            opt.state[new] = state


_trainer.Trainer.train_iteration = train_iteration
_trainer.Trainer._load_checkpoint = load_checkpoint

if __name__ == "__main__":
    from nerfstudio.scripts.train import entrypoint
    sys.argv[0] = "ns-train"
    entrypoint()
