#!/bin/bash
# Python environment. Deliberately NOT installing flwr here: it pins numpy<2 and
# breaks torch/scipy/mypy. It goes to /workspace/flwr-libs on PYTHONPATH instead.
#
# On a vast.ai PyTorch image the preinstalled torch lives in the /venv/main
# virtualenv, NOT in the system interpreter — installing into system python there
# gets you a second, CPU-only torch and a box that silently trains on CPU. Prefer
# the venv when it exists; fall back to the system interpreter otherwise.
set -uo pipefail
cd /workspace/PrivChain-MDD-
export DEBIAN_FRONTEND=noninteractive

if [ -x /venv/main/bin/python ]; then
  PY=/venv/main/bin/python
  PIP_ARGS=""
else
  PY=python3
  PIP_ARGS="--break-system-packages"
fi
echo "### interpreter: $PY"

echo "### pip $(date -u +%H:%M:%S)"
$PY -m pip install -q $PIP_ARGS -e ".[audio,nlp,ml,viz,dev]" 2>&1 | tail -5
echo "### versions $(date -u +%H:%M:%S)"
$PY - <<PY
import numpy, torch, scipy
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("scipy", scipy.__version__)
import opacus, transformers, sklearn
print("opacus", opacus.__version__, "transformers", transformers.__version__, "sklearn", sklearn.__version__)
PY
echo "### done $(date -u +%H:%M:%S)"
