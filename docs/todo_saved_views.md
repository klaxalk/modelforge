# TODO: saved camera views

Priority: **low** (the user asked, 2026-10-24). Status: **done 2026-10-25**.

A tab in the left panel: **Save view** stores the current camera pose (position, look-at, up, field of view) of the
browser that pressed it under a name the user types; a list of saved views, each with **Go** (glide there, as
`App.glide_to` does for the semantic objects), rename and delete. Saved per project (e.g. `PROJECT/views.json`, so it
moves with renames and is kept by Clean; positions in the project's world frame).

## Done 2026-10-25

- Left panel → **Views** (after View; `modelforge/gui/views.py`): Name + **Save current view** (the pressing browser's
  position, look-at, up and vertical field of view; an empty name becomes "View N", a taken one gets " (2)"), then
  one row per view: **▶ name** glides there (`App.glide_to(up=, fov=)`: up from the start, the field of view at the
  end), **⟳** replaces it with the current camera, **✏** rename (modal), **🗑** delete (modal).
- `PROJECT/views.json` ({"views": [{name, position, look_at, up, fov, saved}]}, world frame): listed as a final
  result in Projects, kept by Clean and by re-extraction (a new extraction may move the world frame: the views then
  point elsewhere). The list follows a project switch.
- Tests: `tests/test_views.py` (the file, the panel with a stand-in browser).
