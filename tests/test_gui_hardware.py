"""Hardware tab: /proc + nvidia-smi parsing and the live panel on a real viser server."""

import time

import viser

from modelforge.gui import hardware


def test_parse_gpus():
    text = ("0, NVIDIA GeForce RTX 3090, 36, 5230, 24576, 54, 112.30\n"
            "1, Tesla T4, [N/A], 0, 15360, [N/A], [Not Supported]\n")
    g = hardware.parse_gpus(text)
    assert [x["index"] for x in g] == [0, 1]
    assert g[0]["util"] == 36 and g[0]["mem_used"] == 5230 * 1024 ** 2 and g[0]["power"] == 112.3
    assert g[1]["util"] is None and g[1]["temp"] is None and g[1]["mem_total"] == 15360 * 1024 ** 2
    assert hardware.parse_gpus("") == []


def test_cpu_and_memory():
    a = hardware.cpu_times()
    sum(i * i for i in range(10 ** 6))
    b = hardware.cpu_times()
    assert 0.0 <= hardware.cpu_percent(a, b) <= 100.0
    assert hardware.cpu_percent(a, a) == 0.0
    used, total = hardware.memory()
    assert 0 < used < total


class _App:
    def plot_template(self):
        return "plotly_white"


def test_panel_updates():
    server = viser.ViserServer(port=0, verbose=False)
    try:
        p = hardware.HardwarePanel(_App(), server.gui)
        deadline = time.time() + 10
        while len(p.hist) < 3 and time.time() < deadline:
            time.sleep(0.2)
        assert len(p.hist) >= 3
        assert "CPU" in p.g_md.content and "RAM" in p.g_md.content
        assert 0 < p.g_ram.value < 100
        if hardware.gpus():
            assert p.g_gpu and p.g_gpu[0][1].value > 0  # the desktop / driver always hold some GPU memory
            assert "GPU 0" in p.g_gpu_md.content
        else:
            assert "no NVIDIA GPU" in p.g_gpu_md.content
        assert len(p.g_plot.figure.data) == 2 + 2 * len(p.g_gpu)
    finally:
        server.stop()
