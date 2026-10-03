#!/usr/bin/env bash
# Full gate: unit tests, the Fraction hand cases, then every documented run plus its replay.
# Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3.11 || command -v python3)}"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1

"$PY" -m unittest discover -s tests
"$PY" tests/hand_cases.py > /dev/null

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$PY" -m alpha_gp_lab demo --out "$out/demo"
"$PY" -m alpha_gp_lab verify "$out/demo" > /dev/null
"$PY" -m alpha_gp_lab run --config fixtures/demo_no_seeds_config.json --out "$out/no-seeds"
"$PY" -m alpha_gp_lab walkforward --out "$out/walk-forward"
"$PY" -m alpha_gp_lab verify "$out/walk-forward" > /dev/null
echo "check.sh: all passed"
