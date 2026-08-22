#!/bin/bash
# Fresh-instance data pipeline: deps -> download -> extract -> layout -> cache.
set -uo pipefail
R=/workspace/PrivChain-MDD-
S=$R/docs/runbook/server-scripts

echo "### apt $(date -u +%H:%M:%S)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq wget unzip >/dev/null

echo "### download $(date -u +%H:%M:%S)"
bash $S/download_daic.sh 2>&1 | grep -c "^OK" | sed "s/^/downloaded_ok=/"

echo "### extract $(date -u +%H:%M:%S)"
bash $S/extract_daic.sh 2>&1 | tail -3

echo "### split csvs $(date -u +%H:%M:%S)"
cp -v $R/data/daic_woz/raw/*.csv $R/data/daic_woz/ 2>&1 | wc -l | sed "s/^/csvs=/"

echo "### text cache: none — the committed .npy cache was purged (see incident note);"
echo "### text embeddings rebuild on GPU during the first run."
mkdir -p $R/data/daic_woz/_feature_cache

echo "### done $(date -u +%H:%M:%S)"
df -h / | tail -1
