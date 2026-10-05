# Reproduce every run

The README's Quick start runs the SYNTHETIC demo and the full gate. This page has every command,
including the optional real-data download and the follow-up scripts.

## Synthetic runs (under 5 minutes)

Python 3.11 (the only version tested), standard library only. From the repository root:

```sh
export PYTHONPATH=src
python3.11 -m alpha_gp_lab demo --out runs/demo          # single split, SYNTHETIC, ~10 s here
python3.11 -m alpha_gp_lab verify runs/demo              # hashes + full recomputation + SQLite read-back
python3.11 -m alpha_gp_lab walkforward --out runs/wf     # rolling folds, about 20 s here
python3.11 -m alpha_gp_lab seeds                         # replayed LLM proposals and their verdicts
bash scripts/demo.sh                                     # the demo plus its replay, in a temp folder
bash scripts/check.sh                                    # everything above plus the test suite
```

Output directories must not exist yet; an existing one is verified and reused only if the
inputs and code are identical.

## Real data

Optional; the download takes longer than five minutes and the CSVs are never
committed). The configs read `data/binance-daily/`, which `.gitignore` excludes:

```sh
python3 scripts/fetch_binance_daily.py --start 2020-01 --end 2026-08 --out data/binance-daily \
  --symbols "$(python3 -c 'import json; print(",".join(json.load(open("fixtures/binance_universe.json"))["symbols"]))')"
python3.11 -m alpha_gp_lab verify-data                   # SHA-256, rows and dates against the pinned universe
python3.11 -m alpha_gp_lab run --config fixtures/binance_daily_config.json --out runs/binance-main          # ~90 s here
python3.11 -m alpha_gp_lab run --config fixtures/binance_walkforward_config.json --out runs/binance-wf      # minutes
```

The follow-up analysis, the post-hoc diagnostics and the figures (the analysis takes about 6 minutes on
10 processes, the diagnostics a few seconds; the plot
needs matplotlib, here from the shared venv in `PORTFOLIO_VENV`):

```sh
python3.11 scripts/analyze_binance.py --config fixtures/binance_analysis_config.json --out results/binance_analysis.json
python3.11 scripts/diagnose_binance.py --config fixtures/binance_diagnostics_config.json --out results/binance_diagnostics.json
"$PORTFOLIO_VENV/bin/python" scripts/plot_binance.py   # writes docs/figures/*.png
```

`check.sh` runs `verify-data` and re-runs the main real-data config against
`results/binance_daily_main.json` when `data/binance-daily/` exists, and prints a skip message
when it does not. `seeds --live --config <config>` refreshes a replay entry with one request to
the pinned OpenRouter model; it needs `OPENROUTER_API_KEY` (or the file
`~/.config/openrouter/api_key`) and fails loudly without it.

## Tests and runtime

- `PYTHONPATH=src python3.11 -m unittest discover -s tests` prints `Ran 121 tests` and `OK`.
- `python3.11 tests/hand_cases.py` re-derives the evaluator's arithmetic in exact Fractions:
  for example IC 2/5, turnover 2, gross 1/20 and net 49/1000 on a four-asset case.
- On the development machine the demo printed `demo finished in 9.6s` to stderr on its last run, and
  `scripts/check.sh` (tests, hand cases, `scripts/demo.sh`, the other two runs in [synthetic-results.md](synthetic-results.md), two replays, then
  `verify-data` and the main real-data run when the data is present) exited 0 both with and
  without `data/binance-daily/`. Runtime varies by machine.
