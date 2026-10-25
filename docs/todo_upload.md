# TODO: streaming, resumable uploads of large files

Priority: **medium** (the user asked, 2026-10-25). Status: **done 2026-10-25** (see the end).

Uploads (Videos tab ＋ Upload, Ingest tab) go through viser's file upload widget: the browser sends the file over the
GUI's websocket and viser holds it **in server memory until it is complete** (a 1.2 GB clip needs ~2.4 GB of RAM at
peak; `App._watch_uploads` only shows progress), then hands it to Python, which writes it (`.part`, then renamed). An
interrupted upload (network drop, closed tab, GUI restart) starts from zero, and files of tens of GB (long 4K clips,
ROS bags) do not fit at all.

Wanted: uploads that stream to disk as they arrive, survive interruptions and resume where they stopped, for files of
any size.

Ideas:
- A separate upload path outside the websocket: an HTTP endpoint (viser's server only serves its build folder, so
  either a small server of our own on another port, or a `process_request` hook in viser's websockets server) taking
  chunks with their offset (e.g. the tus protocol, or plain `PUT ?offset=`), written straight to `<dest>.part` with
  fsync per chunk.
- Resume: the server answers the size of the `.part` for an upload id (file name + size + mtime from the browser, or a
  hash of the first MB); the browser continues from there. The browser side: `File.slice()` per chunk (e.g. 16 MB),
  retries with backoff, progress in the GUI.
- The page gets a small upload widget through the existing page patch (`docker/patch_viser_page.py`) or a hidden
  input like Fly's report channel; viser's own widget stays for small files.
- At the end: size (and optionally a checksum) checked, renamed into the library folder; `.part` files of abandoned
  uploads listed / cleaned in the Videos tab.
- Bag folders: one upload per file of the folder, or a tar streamed and unpacked as it arrives.

## Done 2026-10-25

Over the GUI's own websocket (port 8080; no new port, no container change), with viser's file transfer messages but
our own browser and server sides:
- **Browser** (`docker/patch_viser_page.py` UPLOADER, a page script): a file picked in a `data-mf-upload` element
  (`gui/upload.widget`: Videos tab "＋ Upload a large file here (streamed, resumable)" into the current folder, Ingest
  tab into the library) goes out in 1 MiB parts read with `File.slice`, at most 32 MiB beyond the server's
  acknowledgement (no more browser memory than that), several files one after the other; progress, rate and "resumed
  at" under the picker; no answer for 8 s (a dropped connection) → start again → the server answers what it has →
  continue from there; leaving the page during an upload asks first. `docker/patch_viser_camera.py` exposes viser's
  message sender (`window.__mfV`, set every frame) and routes our acknowledgements to the uploader (`window.__mfAck`).
- **Server** (`modelforge/gui/upload.py` StreamUploads, registered on viser's websocket handlers; viser's own upload
  handlers ignore these messages): a writer thread per upload writes each part at its offset into
  `.<name>.mfupload` in the target folder, acknowledges the gap-free length to that browser only, and every 64 MiB
  fsyncs and saves it in `.<name>.mfupload.json` (key = name + size + the file's lastModified). A start with a known
  key resumes from the saved length (a dropped connection; a reload or GUI restart when the same file is picked
  again; at most the last 64 MiB are sent again). Complete: `videos.finish_upload` (renamed into place, archives of
  bag folders unpacked from the file: `extract_bags` takes a path now), the Videos tab's message, list and Ingest's
  video list refreshed. Refused (a name that may not be uploaded, an existing file): acknowledged with -1, the reason
  in the Videos tab.
- Interrupted uploads are listed in the Videos tab's folder ("⏸ name · done / size MB", 🗑 deletes what arrived).
  viser's own upload buttons were removed once the user had tried it (2026-10-25): this is the only upload now, with
  their labels and hover hints.
- **Checked:** `tests/test_upload.py` (parts out of order, a transfer resumed, a server restart resumed from the saved
  length, refused names, a bag archive unpacked); the page script in node against a mock server that drops the
  connection for 10 s mid-upload (resumed, identical file); a 3.2 MB upload through the live GUI's websocket from
  Python, interrupted after the first part and resumed on a new connection (identical in the library). Not yet seen
  in a real browser.
