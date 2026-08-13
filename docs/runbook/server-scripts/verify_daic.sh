#!/bin/bash
DEST="/workspace/PrivChain-MDD-/data/daic_woz/raw"
cd "$DEST" || exit 1
missing=0; badsize=0
while read -r f exp; do
  [ "$f" = "index.php" ] && continue
  if [ ! -f "$f" ]; then echo "MISSING  $f"; missing=$((missing+1)); continue; fi
  act=$(stat -c %s "$f")
  if [ "$act" != "$exp" ]; then echo "SIZEBAD  $f  expected=$exp actual=$act"; badsize=$((badsize+1)); fi
done < /workspace/sizes.txt
echo "----"
echo "missing=$missing  size_mismatch=$badsize"
