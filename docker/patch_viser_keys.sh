#!/bin/sh
# viser's camera keys: Q / E move down / up; the user wants F / R. The client bundle is zstd-compressed inside
# index.html, so the key names are swapped by its loader after unpacking (they occur only in the camera controls:
# the movement key set and the elevate calls). Fails if viser's loader changed.
set -e
f="$1"
grep -q 'const code=td.decode(heap.subarray(up+ss,up+sz));' "$f"
sed -i 's|const code=td.decode(heap.subarray(up+ss,up+sz));|const code=td.decode(heap.subarray(up+ss,up+sz)).replaceAll("`KeyQ`","`KeyF`").replaceAll("`KeyE`","`KeyR`");|' "$f"
grep -q 'replaceAll("`KeyQ`","`KeyF`")' "$f"
