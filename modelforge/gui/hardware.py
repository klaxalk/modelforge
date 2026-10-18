"""Hardware tab: live CPU, RAM, GPU and GPU-memory use, sampled once per second.

No extra packages: CPU and RAM come from /proc (the host's figures, also inside the container),
the GPU from `nvidia-smi` (absent on CPU-only machines: the GPU rows then say so).
"""

import collections
import os
import subprocess
import threading
import time

import plotly.graph_objects as go
import viser

PERIOD = 1.0  # s between samples
HISTORY = 120  # samples kept for the plot (2 min)
GPU_FIELDS = ("index", "name", "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw")


def cpu_times():
    """(busy, total) jiffies over all cores from /proc/stat."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
    total = sum(v[:8])  # guest time is already counted in user / nice
    return total - idle, total


def cpu_percent(prev, cur):
    db, dt = cur[0] - prev[0], cur[1] - prev[1]
    return 100.0 * db / dt if dt > 0 else 0.0


def memory():
    """(used, total) bytes; used = MemTotal − MemAvailable (page cache counts as free)."""
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0]) * 1024
    return m["MemTotal"] - m.get("MemAvailable", m["MemFree"]), m["MemTotal"]


def _num(s):
    try:
        return float(s)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


def parse_gpus(text):
    """nvidia-smi CSV (GPU_FIELDS, noheader, nounits) -> list of dicts; memory in bytes."""
    out = []
    for line in text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != len(GPU_FIELDS):
            continue
        mib = 1024 ** 2
        used, total = _num(f[3]), _num(f[4])
        out.append({"index": int(f[0]), "name": f[1], "util": _num(f[2]),
                    "mem_used": used * mib if used is not None else None,
                    "mem_total": total * mib if total is not None else None,
                    "temp": _num(f[5]), "power": _num(f[6])})
    return out


def gpus():
    """Per-GPU stats, or None without nvidia-smi / driver."""
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_gpus(r.stdout) if r.returncode == 0 else None


def _gb(b):
    return f"{b / 1024 ** 3:.1f} GB"


class HardwarePanel:
    """Bars + numbers + a 2-minute history; a daemon thread samples and updates them."""

    def __init__(self, app, gui, plot_aspect=0.7):
        self.app = app
        self.gui = gui
        self.hist = collections.deque(maxlen=HISTORY)  # (t, cpu %, ram %, [gpu util %], [gpu mem %])
        self.g_md = gui.add_markdown("_sampling …_")
        gui.add_markdown("**CPU**")
        self.g_cpu = gui.add_progress_bar(0.0, color="blue")
        gui.add_markdown("**RAM**")
        self.g_ram = gui.add_progress_bar(0.0, color="teal")
        self.g_gpu_md = gui.add_markdown("")
        self.g_gpu = []  # (utilization bar, memory bar) per GPU found at start
        for g in gpus() or []:
            gui.add_markdown(f"**GPU {g['index']}** utilization / memory")
            self.g_gpu.append((gui.add_progress_bar(0.0, color="green"), gui.add_progress_bar(0.0, color="orange")))
        self.g_plot = gui.add_plotly(go.Figure(), aspect=plot_aspect)
        self.g_plot_on = gui.add_checkbox("History plot", initial_value=True,
                                          hint="Plot the last 2 minutes (CPU, RAM, GPU, GPU memory, in %). "
                                               "Untick to save bandwidth.")
        self.g_plot_on.on_update(lambda _: setattr(self.g_plot, "visible", self.g_plot_on.value))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = cpu_times()
        n = 0
        while True:
            time.sleep(PERIOD)
            try:
                cur = cpu_times()
                cpu, prev = cpu_percent(prev, cur), cur
                self._show(cpu, memory(), gpus(), plot=self.g_plot_on.value and n % 2 == 0)
            except Exception as e:  # a display glitch must not end the sampling
                self.g_md.content = f"⚠ hardware stats: {e}"
            n += 1

    def _show(self, cpu, mem, gpu, plot):
        used, total = mem
        ram = 100.0 * used / total
        self.g_cpu.value = cpu
        self.g_ram.value = ram
        self.g_md.content = (f"CPU **{cpu:.0f} %** of {os.cpu_count()} threads · RAM **{_gb(used)}** of {_gb(total)} "
                             f"({ram:.0f} %)")
        gutil, gmem = [], []
        if gpu is None:
            self.g_gpu_md.content = "**GPU**: no NVIDIA GPU / `nvidia-smi` here"
        else:
            lines = []
            for k, g in enumerate(gpu):
                u = g["util"] or 0.0
                m = 100.0 * g["mem_used"] / g["mem_total"] if g["mem_used"] is not None and g["mem_total"] else 0.0
                if k < len(self.g_gpu):
                    self.g_gpu[k][0].value, self.g_gpu[k][1].value = u, m
                gutil.append(u)
                gmem.append(m)
                extra = "".join(s for s in (f" · {g['temp']:.0f} °C" if g["temp"] is not None else "",
                                            f" · {g['power']:.0f} W" if g["power"] is not None else ""))
                mem_s = f"{_gb(g['mem_used'])} of {_gb(g['mem_total'])}" if g["mem_used"] is not None else "n/a"
                lines.append(f"GPU {g['index']} ({g['name']}): **{u:.0f} %** · memory **{mem_s}** ({m:.0f} %){extra}")
            self.g_gpu_md.content = "  \n".join(lines)
        self.hist.append((time.time(), cpu, ram, gutil, gmem))
        if plot:
            self.g_plot.figure = self._figure()

    def _figure(self):
        t0 = self.hist[-1][0]
        x = [h[0] - t0 for h in self.hist]
        fig = go.Figure()
        fig.add_scatter(x=x, y=[h[1] for h in self.hist], name="CPU", line_color="#4c78a8")
        fig.add_scatter(x=x, y=[h[2] for h in self.hist], name="RAM", line_color="#2a9d8f")
        ngpu = len(self.hist[-1][3])
        for i in range(ngpu):
            sfx = f" {i}" if ngpu > 1 else ""
            fig.add_scatter(x=x, y=[h[3][i] if i < len(h[3]) else None for h in self.hist], name="GPU" + sfx,
                            line_color="#54a24b")
            fig.add_scatter(x=x, y=[h[4][i] if i < len(h[4]) else None for h in self.hist], name="GPU mem" + sfx,
                            line_color="#f58518", line_dash="dot")
        fig.update_layout(template=self.app.plot_template(), margin=dict(l=40, r=10, t=10, b=30),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(range=[0, 100], title="%"),
                          xaxis=dict(range=[-PERIOD * HISTORY, 0], title="s"))
        return fig

