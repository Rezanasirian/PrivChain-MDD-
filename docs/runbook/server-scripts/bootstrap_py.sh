#!/bin/bash
# Python environment. Deliberately NOT installing flwr here: it pins numpy<2 and
# breaks torch/scipy/mypy. It goes to /workspace/flwr-libs on PYTHONPATH instead.
set -uo pipefail
cd /workspace/PrivChain-MDD-
export DEBIAN_FRONTEND=noninteractive
echo "### pip $(date -u +%H:%M:%S)"
pip install -q --break-system-packages -e ".[audio,nlp,ml,viz,dev]" 2>&1 | tail -5
echo "### versions $(date -u +%H:%M:%S)"
python3 - <<PY
import numpy, torch, scipy
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("scipy", scipy.__version__)
import opacus, transformers, sklearn
print("opacus", opacus.__version__, "transformers", transformers.__version__, "sklearn", sklearn.__version__)
PY
echo "### done $(date -u +%H:%M:%S)"
