#!/usr/bin/env bash
# Install the Go toolchain and the Fabric binaries under /workspace (no root needed).
set -euo pipefail

GO_VERSION=1.23.4
FABRIC_VERSION=2.5.10
ROOT=/workspace/fabric
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -x "$ROOT/go/bin/go" ]; then
  echo "--- downloading Go ${GO_VERSION} ---"
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o go.tgz
  tar -xzf go.tgz && rm go.tgz
fi
"$ROOT/go/bin/go" version

if [ ! -x "$ROOT/bin/peer" ]; then
  echo "--- downloading Fabric ${FABRIC_VERSION} binaries ---"
  curl -fsSL "https://github.com/hyperledger/fabric/releases/download/v${FABRIC_VERSION}/hyperledger-fabric-linux-amd64-${FABRIC_VERSION}.tar.gz" -o fabric.tgz
  tar -xzf fabric.tgz && rm fabric.tgz
fi
ls "$ROOT/bin"
"$ROOT/bin/peer" version | head -3
echo "TOOLCHAIN_OK"
