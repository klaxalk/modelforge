#!/bin/sh
# viser's camera keys, changed in its client bundle (zstd-compressed inside index.html, so its loader edits the code
# after unpacking; each pattern occurs once, in the camera controls):
#  - Q / E move down / up; the user wants F / R (the movement key set and the elevate calls);
#  - the move speed: viser's fixed 2 m/s (`let i=2*t` with t the frame time) becomes the GUI's View → Move speed,
#    read from a hidden element (#mf-move-speed, data-speed) while a key is held; 5 m/s until the page has one;
#  - View → Orbit: data-orbit (deg/s, 0 = off) turns the view every frame about its look-at point, as the arrow keys
#    do, so the mouse wheel and drags keep working while it orbits (the server-driven orbit overwrote them).
# Fails if viser's loader changed (the decoded code is checked by the GUI's hint text only: a changed pattern
# leaves viser's own keys and speed).
set -e
f="$1"
grep -q 'const code=td.decode(heap.subarray(up+ss,up+sz));' "$f"
sed -i 's|const code=td.decode(heap.subarray(up+ss,up+sz));|const code=td.decode(heap.subarray(up+ss,up+sz)).replaceAll("`KeyQ`","`KeyF`").replaceAll("`KeyE`","`KeyR`").replace("if(n===null\|\|r.size===0\|\|!n.enabled)return;let i=2*t,o=50*Yt.DEG2RAD*t;","let mfE=document.getElementById(`mf-move-speed`),mfO=+(mfE?.dataset.orbit\|\|0);if(n===null\|\|r.size===0\&\&!mfO\|\|!n.enabled)return;mfO\&\&n.rotate(mfO*Yt.DEG2RAD*Math.min(t,.1),0,!0);let i=(window.__mfSpeed=+(mfE?.dataset.speed\|\|window.__mfSpeed\|\|5))*t,o=50*Yt.DEG2RAD*t;");|' "$f"
grep -q 'replaceAll("`KeyQ`","`KeyF`")' "$f"
grep -q 'mf-move-speed' "$f"
