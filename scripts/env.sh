# Sourced by check.sh and demo.sh. Interpreter: $PYTHON if set (as in CI), else $PORTFOLIO_VENV/bin/python, else
# python3.11, else python3. Standard library only, so any 3.11 will do; one that was named but does not run stops here.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${PORTFOLIO_VENV:-}" ]; then PY="$PORTFOLIO_VENV/bin/python"
else PY="$(command -v python3.11 || command -v python3 || true)"
fi
if [ -z "$PY" ] || ! command -v "$PY" > /dev/null; then
  echo "error: no Python interpreter at '${PY}' (set PYTHON, or PORTFOLIO_VENV to a venv with bin/python)" >&2
  exit 3
fi
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
