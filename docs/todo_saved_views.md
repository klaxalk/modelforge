# TODO: saved camera views

Priority: **low** (the user asked, 2026-10-24). Status: not started.

A tab in the left panel: **Save view** stores the current camera pose (position, look-at, up, field of view) of the
browser that pressed it under a name the user types; a list of saved views, each with **Go** (glide there, as
`App.glide_to` does for the semantic objects), rename and delete. Saved per project (e.g. `PROJECT/views.json`, so it
moves with renames and is kept by Clean; positions in the project's world frame).
