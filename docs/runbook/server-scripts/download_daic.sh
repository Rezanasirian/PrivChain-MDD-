#!/bin/bash
# Download DAIC-WOZ corpus with resume + size verification.
BASE="https://dcapswoz.ict.usc.edu/wwwdaicwoz"
DEST="/workspace/PrivChain-MDD-/data/daic_woz/raw"
LIST="/workspace/daic_files.txt"
PAR=6

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
