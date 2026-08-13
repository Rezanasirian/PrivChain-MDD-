#!/bin/bash
# Extract each participant archive into data/daic_woz/{pid}_P/.
# CLNF_hog.* is skipped: ~87 GB of the ~118 GB total and unused by the pipeline
# (video modality reads CLNF_AUs). The zips are kept, so hog stays recoverable.
RAW="/workspace/PrivChain-MDD-/data/daic_woz/raw"
DEST="/workspace/PrivChain-MDD-/data/daic_woz"

one() {
  z="$1"
  id=$(basename "$z" _P.zip)
  d="$DEST/${id}_P"
  mkdir -p "$d"
  if unzip -o -qq -j "$RAW/$z" -x "*CLNF_hog*" -d "$d" 2>/dev/null; then
    echo "OK   $id  ($(ls "$d" | wc -l) files)"
  else
    echo "FAIL $id"
  fi
}
export -f one; export RAW DEST

cd "$RAW" || exit 1
ls *_P.zip | grep -v "^440_P.zip$" | xargs -P 6 -I{} bash -c "one {}"
echo "=== EXTRACT PASS COMPLETE ==="
