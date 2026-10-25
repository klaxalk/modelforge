"""Saved camera views (modelforge/gui/views.py): per project in views.json; save, update, rename, delete, go."""
import json
from types import SimpleNamespace

import pytest

from modelforge.gui import views


def test_views_file(tmp_path):
    p = str(tmp_path)
    assert views.load(p) == []
    a = views.add(p, "", views.pose((1, 2, 3), (0, 0, 0), fov=0.9))
    b = views.add(p, "  pit  ", views.pose((4, 5, 6), (1, 1, 0)))
    c = views.add(p, "pit", views.pose((7, 8, 9), (1, 1, 0)))
    assert (a, b, c) == ("View 1", "pit", "pit (2)")
    v = views.load(p)
    assert [x["name"] for x in v] == ["View 1", "pit", "pit (2)"]
    assert v[0]["position"] == [1.0, 2.0, 3.0] and v[0]["fov"] == 0.9 and v[0]["up"] == [0.0, 0.0, 1.0]
    views.update(p, "pit", views.pose((0, 0, 50), (0, 0, 0)))
    assert views.load(p)[1]["position"] == [0.0, 0.0, 50.0]
    assert views.rename(p, "pit (2)", "pit") == "pit (2)"  # taken: kept unique
    assert views.rename(p, "pit (2)", "north wall") == "north wall"
    with pytest.raises(ValueError):
        views.rename(p, "north wall", " ")
    views.delete(p, "View 1")
    assert [x["name"] for x in views.load(p)] == ["pit", "north wall"]
    with pytest.raises(KeyError):
        views.delete(p, "View 1")
    json.dump({"views": [{"name": "broken"}]}, open(tmp_path / "views.json", "w"))
    assert views.load(p) == []  # records without a pose are skipped


class _Handle:
    def __init__(self, value=""):
        self.value, self.content = value, ""

    def remove(self):
        pass

    def on_click(self, f):
        self.click = f

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Gui:
    def __init__(self):
        self.groups = []

    def add_markdown(self, text=""):
        return _Handle()

    def add_text(self, *a, initial_value="", **k):
        return _Handle(initial_value)

    def add_button(self, *a, **k):
        return _Handle()

    def add_folder(self, *a, **k):
        return _Handle()

    def add_button_group(self, label, options, **k):
        h = _Handle()
        h.options = options
        self.groups.append(h)
        return h


def test_views_panel(tmp_path):
    """Save from a browser's camera, then ▶ glides there with the saved up and field of view."""
    glides = []
    app = SimpleNamespace(path=str(tmp_path), glide_to=lambda client, pos, look, up=None, fov=None:
                          glides.append((tuple(pos), tuple(look), tuple(up), fov)))
    gui = _Gui()
    panel = views.ViewsPanel(app, gui)
    client = SimpleNamespace(camera=SimpleNamespace(position=(10.0, 0, 5), look_at=(0, 0, 0), up_direction=(0, 0, 1),
                                                    fov=0.7))
    panel.g_name.value = "overview"
    panel.save(client)
    assert views.load(str(tmp_path))[0]["name"] == "overview" and panel.g_name.value == ""
    row = gui.groups[-1]
    assert row.options[0] == "▶ overview"
    row.click(SimpleNamespace(target=SimpleNamespace(value="▶ overview"), client=client))
    assert glides == [((10.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.7)]
    client.camera.position = (0, 20.0, 5)
    gui.groups[-1].click(SimpleNamespace(target=SimpleNamespace(value="⟳"), client=client))
    assert views.load(str(tmp_path))[0]["position"] == [0.0, 20.0, 5.0]
