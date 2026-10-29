# A classical 2D map: done 2026-10-27

The user asked on 2026-10-27 for a 2D map that shows the flight path and the current position of the 3D camera.

## Core
- A floating map window, like the photo window, or a tab.
- Content:
  - each source's flight path in its colour;
  - the registered cameras as dots;
  - the 3D view's camera as a marker with a heading / field-of-view cone, updated from the camera callbacks, also
    during Fly and Orbit.
- Background tiles: OpenStreetMap or aerial imagery. Projects have `enu_origin_lla`, so ENU → WGS84 is known.
  - Video-kind projects have no metric frame and get a plain top-down plot without tiles.
  - ROS bags have a world frame without a geo origin, so the same applies to them.
- Open question: tiles need internet access from the browser, which works for OSM. An offline fallback would be a
  top-down render of the splat or the SfM mesh, as an orthophoto.

## Chosen uses (the user, 2026-10-27)
1. **Click to go:** a click on the map glides the 3D camera there (`App.glide_to`), looking down or along the path.
2. **Orthophoto layer:** a top-down render of the splat (or the coloured SfM mesh) as a georeferenced image. Also
   useful as an export (GeoTIFF).
3. **Coverage / quality heat map:**
   - photos per ground cell and the keyframes' overlap;
   - the isolated photos left out;
   - the Difix thin spots;
   - held-out PSNR per area.
   These show where to fly again.
4. **Tiles on the map:** the tile regions / cells with their status, with the click to leave a tile out (today in
   the 3D view).
5. **Ingest on the map:**
   - the planned image positions and each source's time range; dragging the range ends on the path picks the piece
     of the flight to extract;
   - **draw a rectangle:** the parts of the flights that see the rectangle become the sources (one piece per stretch
     of time). Open: what "see" means, the drone's position inside the rectangle or the look-at point inside it
     (probably the look-at, as the tiles do), and a minimum stretch length so short passes do not split into many
     pieces.
6. **Images used / Register more:** draw a polygon on the map to select the photos of an area (a spatial subset
   instead of a time range). It goes next to every N-th / time range / keyframes.
7. **Semantic search results** as pins (the found objects); a click goes there.
8. **Saved views and Difix target views** as pins with their direction; a click glides there.

Not wanted: revisit / flight-alignment display, re-flight mission export.

## Built (2026-10-27)

- **Window:** quick bar **Map** opens the floating Map window (`modelforge/gui/map.py`, `MapPanel`). The server draws
  a 640 px north-up image; clicks come back through the page's `img[data-mf-click]` hook (as the Tie point view's), so
  no page patch was needed. Zoom ＋ / －, Fit, Follow (keeps the 3D camera on the map). It redraws on camera moves
  (≤ 5 Hz) and when anything it shows changes on disk (checked every 2 s).
- **Background** (`modelforge/map2d.py`): Esri aerial imagery (default), OpenStreetMap streets or none. The server
  fetches the tiles (one thread, a User-Agent, cached under `ROOT/.map_tiles/`) and maps the ENU pixel grid to Web
  Mercator: exact on a 17 × 17 grid, a fitted quadratic in between. Interpolating that grid with `cv2.resize`
  clamped the edges into streaks. Projects without a GPS origin show a plain background.
- **Orthophoto** (`modelforge/ortho.py`, `export/ortho_render.py`): the splat rendered straight down with gsplat's
  orthographic camera → `splat/ortho.png` (RGBA) + `ortho.json` (bounds, m/px, the splat's size + mtime, so a new
  splat makes it stale). Job stage `ortho` (Map → Background and layers → Render orthophoto). Limits:
  - the area is the flown one (cameras + their look-at points + 15 %); the sparse points' extent reached the
    horizon (fullrun_test 300 m for a 50 m orbit);
  - left out are Gaussians outside it, above the points' 99.5 % height + 2 m, or wider (3σ) than 2 % of the area
    (≥ 1.5 m). Without that, the background's huge far blobs covered everything.
  - fullrun_test: 3814 × 4096 px at 1.9 cm/px in seconds.
  **Export GeoTIFF** writes `export/ortho.tif`: UTM of the origin (EPSG 326xx / 327xx), an affine
  ModelTransformationTag fitted to the ENU → UTM mapping (the meridian convergence rotates it), written with PIL
  (no GDAL / tifffile in the image).
- **Live camera:** a red dot with a field-of-view wedge, from the browsers' camera updates.
- **Layers** (Background and layers):
  - flight path per source (cut at pauses > 30 s) and the cameras;
  - photos the export left out as isolated (✕, at their prior positions);
  - splat tiles (outlines / rectangles, status colours, "(out)");
  - the Ingest plan (tracks bright in range, grey outside; the planned images);
  - search results (yellow circles, #k);
  - saved views and Difix views (pins with an arrow).
- **Heat maps:**
  - **photos per area:** registered photos whose frame holds a ground cell (the terrain of the snapshot's points;
    no occlusion; red few … green many over the 2–98 % range);
  - **overlap of neighbouring photos:** the analysis' inliers / keypoints between consecutive photos in use,
    drawn on the path (red < 0.06 … green ≥ 0.2);
  - **held-out quality:** evaluation/metrics.json PSNR per photo at its look-at;
  - **Difix thin spots:** the mesh samples where the splat is thin (difix.suggest_views' measure), magenta.
- **Click** (picker):
  - **Go there:** the 3D view glides to look at the spot, keeping its offset. A pin within 14 px goes to that
    object / view instead.
  - **Centre the map**.
  - **Tile:** leave out / take back (`App._toggle_tile`).
  - **Ingest rectangle:** two corners. `ingest.sources_in_area` takes each video's whole clip; a frame counts when
    its look-at (the optical axis on the takeoff plane, as the flight check does) or, optionally, the drone lies
    inside. Stretches with gaps < 3 s are joined and those < 5 s dropped (both settable). One Source per stretch
    keeps the video's lead / SRT / altitude offset. Apply replaces the Ingest sources; Undo brings the previous
    ones back. kladno_dira, 100 × 100 m in the pit: 11 pieces in 5 s.
  - **Ingest start / end:** the nearest source track point (≤ 20 px) sets its range.
  - **Area:** polygon corners, then Apply. `Selection.set_area` gets the polygon and the extracted photos whose
    look-at (on the sparse points' terrain) lies inside. Images used and Register more photos then choose only
    among them, under either selection mode. A checkbox there ignores the area, and the setting records it.
- Tests: `tests/test_map.py`:
  - geodesy round trip, UTM against a known value;
  - view, runs, polygon, cached tiles, coverage;
  - GeoTIFF tags and staleness;
  - the rectangle on a fake clip;
  - the window on a refined project: coverage drawn, centre click, area → Images used / Register, quick bar.
- Not done: zoom with the mouse wheel / dragging the map (clicks only), a Map entry in the job queue, and the
  orthophoto of the SfM mesh for projects without a splat.
