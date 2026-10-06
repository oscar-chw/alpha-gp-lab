#!/usr/bin/env bash
# Full gate: unit tests, the Fraction hand cases, the demo, then every documented run plus its replay.
# Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

"$PY" -m unittest discover -s tests
"$PY" tests/hand_cases.py > /dev/null
bash scripts/demo.sh

out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$PY" -m alpha_gp_lab run --config fixtures/demo_no_seeds_config.json --out "$out/no-seeds"
"$PY" -m alpha_gp_lab walkforward --out "$out/walk-forward"
"$PY" -m alpha_gp_lab verify "$out/walk-forward" > /dev/null

# Real data is never committed. Without it these steps skip; with it the files must match the pinned hashes.
if [ -d data/binance-daily ]; then
  "$PY" -m alpha_gp_lab verify-data --data data/binance-daily > /dev/null
  # The pre-registered main run must reproduce the committed result (floats to 1e-9 relative).
  "$PY" -m alpha_gp_lab run --config fixtures/binance_daily_config.json --out "$out/binance" > "$out/binance.json"
  "$PY" - "$out/binance.json" results/binance_daily_main.json <<'PYEOF'
import json, math, sys
def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b
got, want = (json.load(open(p)) for p in sys.argv[1:])
sys.exit(0 if same(got, want) else 'real-data main run does not reproduce results/binance_daily_main.json')
PYEOF
else
  echo "check.sh: data/binance-daily not present; real-data steps skipped (see docs/reproduce.md)"
fi
echo "check.sh: all passed"
