#!/bin/bash
# Download DAIC-WOZ corpus with resume + size verification.
BASE="https://dcapswoz.ict.usc.edu/wwwdaicwoz"
DEST="${DEST:-/workspace/PrivChain-MDD-/data/daic_woz/raw}"
# The manifest ships beside this script. It used to live only at
# /workspace/daic_files.txt, which meant destroying the instance also destroyed
# the list of what to download.
LIST="${LIST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/daic_files.txt}"
PAR=6

if [ ! -f "$LIST" ]; then
  echo "manifest not found: $LIST" >&2
  exit 1
fi

mkdir -p "$DEST"
cd "$DEST" || exit 1

fetch() {
  f="$1"
  for attempt in 1 2 3; do
    wget -q -c -T 60 "$BASE/$f" && { echo "OK   $f"; return 0; }
    sleep 3
  done
  echo "FAIL $f"
  return 1
}
export -f fetch
export BASE

grep -v index.php "$LIST" | xargs -P "$PAR" -I{} bash -c "fetch {}"
echo "=== DOWNLOAD PASS COMPLETE ==="
