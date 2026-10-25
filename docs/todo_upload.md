# TODO: streaming, resumable uploads of large files

Priority: **medium** (the user asked, 2026-10-25). Status: not started.

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
