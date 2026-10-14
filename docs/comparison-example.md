# Inspect a small GP-versus-random comparison

This example runs the actual GP and random-search implementations on a small generated panel,
using the existing validation rule and drifted-book transaction costs. It writes both the inputs
and the outputs so a reader can inspect why a candidate was chosen or rejected. It needs Python 3.11
and the standard library; no download, credentials, model call or optional dependency.

From the repository root, choose a fresh output directory:

```sh
export PYTHONPATH=src
python3.11 scripts/example_comparison.py --out runs/comparison-small
python3.11 -m alpha_gp_lab verify runs/comparison-small/primary
cat runs/comparison-small/report.md
```

The wrapper runs the normal GP/store path, including input hashes, SQLite lineage and replay,
then the existing analysis script with one process and two seeds. It does not read historical
Binance results as its example input and does not change them. Existing output directories are
refused without changing their files. A failure keeps its partial directory and stderr log for
inspection; correct the cause and use a new directory.

## Inputs and evaluation roles

The source is [comparison_small_config.json](../fixtures/comparison_small_config.json), with
the generator implemented in [data.py](../src/alpha_gp_lab/data.py). These are deliberately
planted synthetic effects, with 12 assets, 3 groups and 120 daily observations. Ninety observations
use the reversal regime and the final 30 use the adverse regime. This is a current mechanics
example, **not a preregistered financial study or evidence of a market edge**.

| Setting | Value and interpretation |
|---|---|
| Dataset | Fixed generator seed 11; generated bars are stored in primary/panel.json |
| Train | Index 20..70; GP fitness/parents and random-search ranking only |
| Validation | Index 71..95; qualification, correlation filtering and final selection |
| Test | Index 96..119; each selected pick scored after selection |
| Search seeds | 20261003 and 20261004, fixed before running |
| Declared budget | 12 members × 3 generations = 36 train scores per search; random control receives the same cap |
| Actual work | Every seed's accepted train scores and validation count are reported; GP counts include copied elites and random search seeks distinct signals |
| Execution | Delay 1; close-to-close hypothetical returns, unit-gross dollar-neutral weights |
| Costs | 5 bps per side × drifted-book turnover; no spread, slippage, borrow or funding |
| Controls | Fixed reversal and momentum baselines plus a 5-day range factor, evaluated with the same timing and fee |
| Inference | Existing Welch/2-SE rule describes variability across selected optimizer seeds on this fixed panel/split, not uncertainty across markets or future regimes |

The primary GP run is rerun by the analysis and must reproduce its saved expression and metrics.
The comparison is refused before writing output if the saved primary record disagrees. Selecting
no qualifying candidate is a valid outcome: no GP pick, GP-test score or GP-relative inference is
invented. Controls and per-seed rejection counts remain available; comparisons without enough
selected seeds report an undecided rule. `trials.test` counts the primary GP pick's test, not all
control/seed evaluations.

## Inspect the evidence

| File | What to inspect |
|---|---|
| config.json / analysis_plan.json | Exact inputs, split spans, selection thresholds, search seeds/budget and statistical settings |
| primary/config.json / panel.json / llm.json | Inputs to the primary search; LLM seeds are disabled |
| primary/report.md / report.json | GP generations, validation and selected test outcome |
| primary/state.sqlite / completion.json | Append-only search lineage and checked bundle hashes |
| audited_comparison.json | Canonical machine-readable analysis with budget-gated verdict, complete numeric results and explicitly unadjusted arithmetic |
| comparison.json | Raw legacy analysis retained for diagnosis; use the manifest's canonical_result for decisions |
| budget_audit.json | Declared versus realized admitted-slot budgets, eligibility and withheld winner reason; raw statistics retained |
| report.md | Readable seed/status/IC/net-return/work table, with the original rule and limitations |
| example_manifest.json | Input/panel/source hashes and hashes of all preceding output files |
| analysis.stdout.json / analysis.stderr.log | Analysis output and failure diagnostics |

For a rejection case, copy the small config to a new file, set `selection.min_ic` to 1 and
`selection.max_turnover` to 0, and pass `--config <copied-file>` with another fresh `--out`.
This intentionally strict rule demonstrates refusal; it does not improve the search. The example
refuses live/replayed LLM seeds and configs outside its small synthetic bounds (16 assets,
200 dates, 128 train scores, `max_attempts<=10`, one split). The retry cap also bounds rejected
candidate generation, rather than counting only successful scores. Use the general
`run` and analysis commands for other studies; do not retune the historical study from this example.

Baseline names `gp_pick`, `random_search_pick` and `range_5d` are reserved by this example's
result labels and generated control. A collision is refused before generating inputs or creating
an output directory, rather than failing after the primary search has already run.

`comparison_scope` in the canonical result and CLI makes the uncertainty scope explicit. More
search seeds refine the conditional optimizer comparison; they do not create new market windows
or remove survivorship, reused-holdout and cost-model limitations. Admitted slots, including
copied elites and cached scores, establish this budget rule's accounting basis rather than equal
compute. Numeric differences, standard errors and the original decision rule are preserved.

A valid retry limit can also leave the declared search budget incomplete. The wrapper now withholds
an equal-admitted-budget winner verdict in that case, while retaining the raw difference and standard
error. [Budget audit, concrete shortfall and accounting limits](budget-audit.md).

## Historical real-data result stays separate

The original study used 34 surviving Binance pairs over the recorded 6.7-year panel, with the
test window 2025-01..2026-08. Its source assumptions and survivorship limits are in
[reproduce.md](reproduce.md), [limits.md](limits.md) and
[the pinned universe](../fixtures/binance_universe.json). The original configs and all result JSON
are preserved. [results.md](results.md) retains the neutral/negative finding: test rank IC near 0.08
was matched by random search, and the corrected drifted-book cost model did not establish GP
superiority on test net return. The follow-up was planned after the main result, the test window
was reused, and the cost amendment is disclosed there. Nothing in this small example adds a new
profitability, generalisation or historical preregistration claim.
