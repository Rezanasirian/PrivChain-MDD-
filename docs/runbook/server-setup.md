# Server runbook — rebuilding the environment from scratch

Everything the project needs is either in this repository or reproducible from a
script here. This is the order to do it in, and what each step costs.

The GPU box is a rented vast.ai container. **Stopping** it keeps the filesystem;
**destroying or recycling** it does not, and the 119 GB DAIC-WOZ corpus is the
only thing that is genuinely expensive to replace.

## What lives only on the server

| item | size | replaceable? |
|---|---|---|
| `data/daic_woz/` (raw corpus) | 119 GB | Yes, but needs the DUA and a long download |
| `data/daic_woz/_feature_cache/` | 416 MB | Yes — regenerated on first run, costs GPU time for the text embeddings |
| `/workspace/fabric`, `gopath`, `gocache`, `flwr-libs` | ~1.4 GB | Yes — `docs/runbook/server-scripts/00_toolchain.sh` and `pip install` |
| `.fabric/` (Fabric network state) | ~33 MB | Yes — `scripts/fabric/setup_network.sh` regenerates crypto and ledger |

Everything else — code, configs, ADRs, experiment results — is committed.

## 1. Repository and Python

```bash
git clone <repo> /workspace/PrivChain-MDD- && cd /workspace/PrivChain-MDD-
# The env used so far: numpy>=2.1 (torch 2.12 requires it), opacus, transformers.
# `flwr` must NOT go in the main environment: it pins numpy<2 and breaks torch,
# scipy and mypy. Install it to a separate directory (see step 5).
```

Verify with `ruff check src scripts tests`, `mypy --strict src`, `pytest -q`.

## 2. Data

`bootstrap_data.sh` chains this whole section — apt deps, download, extract, the
split-CSV layout the config expects, and the text-embedding restore below — and
is what was actually used to rebuild on 2026-08-14. It took **~15 minutes** on a
fresh box at ~100 MB/s. Run it detached:

```bash
setsid nohup bash docs/runbook/server-scripts/bootstrap_data.sh \
    > /workspace/bootstrap.log 2>&1 < /dev/null &
```

`bootstrap_py.sh` does the Python environment in parallel — it is deliberately
separate so the long download is never blocked behind a pip resolve.

The individual steps, if you need them one at a time:

```bash
bash docs/runbook/server-scripts/download_daic.sh   # ~119 GB, 6 parallel streams
bash docs/runbook/server-scripts/extract_daic.sh
```

The manifest (`daic_files.txt`, 197 entries) sits beside the script and is
committed — it previously existed only at `/workspace/daic_files.txt`, so
destroying the instance would have destroyed the list of what to download.
No credentials are needed; the USC host serves the archive directly.

Participant 440's archive is truncated at source and is excluded in
`configs/daic_woz.yaml` (ADR-0010).

**Restore the text embeddings before the first run** — they are the only cached
artifact that costs GPU time and the only one that survives a re-download:

```bash
mkdir -p data/daic_woz/_feature_cache
cp docs/runbook/text-embedding-cache/*.npy data/daic_woz/_feature_cache/
```

The audio/video cache is deliberately not kept: its key includes each source
file's mtime, so a fresh download invalidates it anyway. It rebuilds on first use
from CPU-bound CSV parsing.

## 3. Reproducing the results

Each script writes to `experiments/<phase>/<run-id>/`, and the committed copies
of those directories are the evidence behind the ADRs.

```bash
python scripts/train_baseline.py            --daic-config configs/daic_woz.yaml
python scripts/run_modality_ablation.py     --daic-config configs/daic_woz.yaml --normalization session
python scripts/run_dp_sweep.py              --daic-config configs/daic_woz.yaml
python scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
python scripts/run_reid_risk.py             --daic-config configs/daic_woz.yaml
python scripts/run_federated_comparison.py  --daic-config configs/daic_woz.yaml --partition iid
```

Long runs must be detached — a dropped SSH connection kills a foreground job:

```bash
setsid nohup bash <script> > /workspace/<name>.log 2>&1 < /dev/null &
```

## 4. Fabric (Phase 5)

No Docker on this container, by design — see ADR-0022. The network runs natively.

```bash
bash docs/runbook/server-scripts/00_toolchain.sh   # Go + Fabric binaries
bash scripts/fabric/teardown.sh --purge
bash scripts/fabric/setup_network.sh
bash scripts/fabric/deploy_chaincode.sh
bash docs/runbook/server-scripts/start_gateway.sh
bash scripts/fabric/verify_ledger.sh
```

Then a federated round against the real ledger with
`--blockchain-config` pointing at a copy of `configs/blockchain.yaml` whose
`backend` is `fabric_rest`. The committed default stays `mock` so CI is offline.

Chaincode checks: `gofmt -l .`, `go vet ./...`, `go test ./...` in
`chaincode/privchain-cc`.

## 5. Flower (parity check only)

```bash
pip install --target /workspace/flwr-libs "flwr[simulation]"
rm -rf /workspace/flwr-libs/{numpy*,scipy*,torch*}   # never shadow the main env
PYTHONPATH=/workspace/flwr-libs python scripts/run_flower_parity.py --daic-config configs/daic_woz.yaml
```

## Traps that cost time before

- **Foreground SSH jobs die.** Detach with `setsid nohup`.
- **`flwr` in the main environment** downgrades numpy and breaks torch, scipy and
  mypy. Keep it in `/workspace/flwr-libs` on `PYTHONPATH`.
- **Ray workers get no GPU** unless `client_resources` asks for a share, and a
  CUDA-built client then fails to deserialize — silently, with Flower carrying on
  from untrained weights.
- **Shell scripts must be LF.** A `\r` after `set -euo pipefail` makes bash
  reject the option. `.gitattributes` pins this.
- **`No space left on device` is usually the *local* machine**, not the server:
  `tail`/`head` in an `ssh ... | tail` pipeline run locally.
