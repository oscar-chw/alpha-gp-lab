# Synthetic results

The three SYNTHETIC runs: the regime-change demo, its no-seeds ablation and a four-regime
walk-forward. The real-data results are in [results.md](results.md).

All results on this page are **SYNTHETIC**, with 5 bps per unit of turnover, rerun on 2026-10-06
after the cost amendment ([results.md](results.md#amendment-2026-10-06-costs-charged-on-the-drifted-book)):
the demo's net figures moved in the fourth digit, the ablation chose a different expression, and
the walk-forward is unchanged.

## 1. Demo: a regime change between validation and test (negative result)

Command: `PYTHONPATH=src python3.11 -m alpha_gp_lab demo --out runs/demo`.
Config: `fixtures/demo_config.json`. Search seed 20261003, data seed 11, 40 assets in
4 industries, 300 days: 240 days `reversal`, then 60 days `adverse`. Splits by date index:
train [30, 180], validation [181, 239], test [240, 299]. GP: population 64, 10 generations,
tournament 4, elitism 4, max depth 6, max 15 nodes, hall of fame 16. Fitness penalties:
turnover 0.02, complexity 0.001. Selection: min IC 0.02, max turnover 1.6, max correlation 0.7,
no existing alphas.

| Field (printed) | Value |
|---|---|
| regimes (train / validation / test) | reversal / reversal / adverse |
| selected | `group_neutralize(-returns, industry)` |
| selected_origin | `llm_seed`, generation 0 (the hand-written fixture) |
| validation mean_ic | 0.133 (ic_tstat 6.937, 58 intervals) |
| validation mean_turnover / mean_net | 1.341 / 0.001961 |
| **test mean_ic** | **-0.1877** (ic_tstat -11.44, 59 intervals) |
| **test mean_net** | **-0.004343** (mean_turnover 1.323) |
| occurrences / unique_expressions | 640 / 473 |
| rejected: duplicate / equivalent / degenerate / limits | 115 / 54 / 22 / 16 |
| correlation_rejected | 15 (shortlist: 1) |
| LLM proposals accepted / rejected | 6 / 3 |

Validation chose a reversal alpha that held up on validation and then lost on the adverse
test period, where the reversal effect flips sign. The split discipline cannot prevent this:
nothing in train or validation reveals a regime that starts afterwards. The other 15
hall-of-fame members also qualified on validation, but each was correlated above
`max_corr` = 0.7 with the pick (they are listed in the run's `report.json`), so the
correlation filter left a shortlist of one. `verify runs/demo` exits 0 and prints the
identical summary.

## 2. Ablation: the same demo without LLM seeds

Command: `PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/demo_no_seeds_config.json --out runs/no-seeds`
(identical config except `"use_seeds": false`).

| Field (printed) | Value |
|---|---|
| selected | `group_neutralize(-returns, industry)` |
| selected_origin | `crossover`, generation 3 |
| validation mean_ic | 0.133 (ic_tstat 6.937) |
| test mean_ic | -0.1877 (ic_tstat -11.44) |
| test mean_net | -0.004343 |
| occurrences / unique_expressions | 640 / 491 |

Without seeds the GP bred the seeded run's pick, `group_neutralize(-returns, industry)`, on its
own by generation 3, so it scored the same and failed on the adverse test period in the same way.
(Before the cost amendment the unseeded run picked `ts_rank(-returns, 20)`, validation IC 0.1413,
test IC -0.153.) So the seeded
win in the demo shows that the plumbing works. It is not evidence that an LLM helps: the
fixture is hand-written and no real LLM has been run.

## 3. Walk-forward over four regimes

Command: `PYTHONPATH=src python3.11 -m alpha_gp_lab walkforward --out runs/wf`.
Config: `fixtures/walkforward_config.json`. Search seed 20261003, data seed 11, 30 assets in
3 industries, 520 days: 200 `reversal`, 120 `momentum`, 100 `noise`, 100 `adverse`. Folds:
warmup 30, train 100, validation 40, test 40, step 40 days. GP: population 32, 6 generations,
hall of fame 8; other settings as in the demo.

| Fold | Test regime(s) | Selected | Origin | Validation IC | Test IC |
|---:|---|---|---|---:|---:|
| 0 | momentum, reversal | `group_neutralize(-returns, industry)` | llm_seed | 0.164 | 0.05229 |
| 1 | momentum | `-group_zscore(returns, industry)` | mutation:industry | 0.06204 | -0.1065 |
| 2 | momentum | none qualified | | | |
| 3 | momentum, noise | `ts_mean(returns, 5)` | llm_seed | 0.117 | 0.1843 |
| 4 | noise | `ts_mean(returns, 5)` | llm_seed | 0.191 | -0.02103 |
| 5 | noise | none qualified | | | |
| 6 | adverse, noise | `ts_mean(zscore(returns), 10)` | mutation:point | 0.02027 | 0.06895 |
| 7 | adverse | `ts_min(returns, 10)` | mutation:subtree | 0.06067 | 0.05969 |

Printed summary: 8 folds, 6 with a selection, mean test IC 0.03963, 4 folds
positive and 2 negative. In fold 1 a reversal alpha was chosen on a validation window that
was still mostly reversal and then tested on pure momentum; in fold 4 a momentum alpha was
chosen on a validation window that was partly momentum and then tested on pure noise.
