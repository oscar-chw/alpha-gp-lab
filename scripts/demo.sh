#!/usr/bin/env bash
# The demo, offline and about 15 seconds: the SYNTHETIC regime-change run, then its full replay
# (hashes, recomputation, SQLite read-back). Standard library only. The real-data runs are not part
# of the demo because the Binance CSVs are never committed (see docs/reproduce.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3.11 || command -v python3)}"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$PY" -m alpha_gp_lab demo --out "$out/demo"
"$PY" -m alpha_gp_lab verify "$out/demo" > /dev/null
echo "demo.sh: SYNTHETIC demo ran and its replay matched"
