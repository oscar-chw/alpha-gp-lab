# Real-data runs: ledger and full-precision tables

The README and [results.md](results.md) round to 3-4 significant figures or fewer. This page keeps
the audit detail: every real-data command with its code version and exit code, and the main and
walk-forward tables at the full precision the JSON files hold. When each plan was committed
relative to the results it could have seen is in [evidence.md](evidence.md#timeline-what-each-plan-could-have-seen).

## Every real-data run

| # | Command | Code | Exit | What happened |
|---:|---|---|---:|---|
| 1 | `verify-data` (several times, and inside `check.sh`) | from `4f9c276` on | 0 | 34 files match the pinned SHA-256s, rows and dates |
| 2 | `seeds --live` (main config) | `4f9c276` | 1 | CLI not signed in; no LLM output |
| 3 | `run` main config | `168fa31` | 0 | the main-run numbers; printed before `net_tstat` existed; output overwritten by run 4 |
| 4 | `run` main config | `2061c41` | 0 | identical numbers plus `net_tstat`; overwritten by run 6 |
| 5 | `run` walk-forward config | `2061c41` | 0 | identical numbers, without `mean_gross` and `valid_ic_intervals`; overwritten by run 7 |
| 6 | `run` main config | `eb05550` | 0 | committed as `results/binance_daily_main.json` |
| 7 | `run` walk-forward config | `eb05550` | 0 | committed as `results/binance_walkforward.json` |
| 8 | `scripts/analyze_binance.py` (follow-up plan) | `405e288` | 0 | committed as `results/binance_analysis.json`; 20 GP and 20 random-search seeds |
| 9 | `scripts/diagnose_binance.py` (POST-HOC diagnostics) | `00c3cbe` | 0 | committed as `results/binance_diagnostics.json`; overwritten by run 10 |
| 10 | `scripts/diagnose_binance.py` (adds up/down t-statistics and two neutralised reference rankings) | `482f022` | 0 | every run-9 number reproduced; overwritten by run 18 |
| 11 | `run` main config, 2026-10-06 cost amendment, first try | `60b9ffa` (working tree) | 0 | byte-identical to run 15 |
| 12 | `run` walk-forward config, same | working tree, edited during the run | 1 | the run's closing self-replay refused the changed code identity; output discarded |
| 13 | `scripts/analyze_binance.py`, same | `60b9ffa` (working tree) | 0 | byte-identical to run 17 |
| 14 | `scripts/diagnose_binance.py`, same | working tree | 0 | byte-identical to run 18 |
| 15 | `run` main config | `a86a5ca` | 0 | committed as `results/binance_daily_main.json` |
| 16 | `run` walk-forward config | `a86a5ca` | 0 | committed as `results/binance_walkforward.json` |
| 17 | `scripts/analyze_binance.py` | `a86a5ca` | 0 | committed as `results/binance_analysis.json` |
| 18 | `scripts/diagnose_binance.py` | `a86a5ca` | 0 | committed as `results/binance_diagnostics.json` |
| 19 | `run` walk-forward config | `60b9ffa` | 0 | byte-identical to run 16: the later robustness fixes change no number |

Runs 11 to 19 regenerate every result under the corrected cost model of the 2026-10-06 amendment
([results.md](results.md#amendment-2026-10-06-costs-charged-on-the-drifted-book)); the main config
and every follow-up plan are unchanged, and runs 15 to 18 used the code of `a86a5ca`, which adds
only robustness fixes to the cost-model commit `60b9ffa` (run 19 checks that they change no number).
`scripts/plot_binance.py` redrew the figures from run 17. Before that, runs 4 to 7 were repeated only to print more statistics (`net_tstat`, then `mean_gross` and
`valid_ic_intervals`); every number that existed before was reproduced exactly, and the selection
does not read the added fields. Each `run` also replays itself once inside `verify`, and
`scripts/check.sh` re-runs the main config and compares it with the committed file. A timing run
on SYNTHETIC data shaped like the real panel (34 assets, 2,435 days) was made before
pre-registration; it used no real data.

## Main run and follow-up rows, full precision

Sources: `results/binance_daily_main.json` (first three rows) and `results/binance_analysis.json`
(`table`, last two rows), as regenerated on 2026-10-06.

| Signal | Validation IC | Test IC (t-stat) | Test turnover | Test gross / day | Test net / day (t-stat) | Test net, sum over 607 days |
|---|---:|---:|---:|---:|---:|---:|
| GP pick `abs(ts_decay_linear((low / close), 10))` | 0.07148111461736868 | 0.08208763518369619 (7.106267209479283) | 0.28342370123071803 | 0.0003534791635240535 | 7.005546229333541e-05 (0.1901215321505456) | 0.0425236656120546 |
| momentum_20d `(close / ts_delay(close, 20))` | -0.017496146192444408 | -0.006158512090458223 (-0.5588498549217263) | 0.31571927623831575 | 0.0003768912600628651 | 6.117198382454931e-05 (0.15525285173780218) | 0.03713139418150143 |
| reversal_1d `-returns` | 0.012864710545246051 | 0.011485586060223782 (1.1150804633424585) | 1.311764328819241 | -0.0002194546543761369 | -0.001531218983195378 (-4.093907881235119) | -0.9294499227995945 |
| *random search, equal budget* `ts_decay_linear(-ts_std(returns, 5), 20)` | 0.04790283617149554 | 0.08001626010548919 (6.7788549855658005) | 0.12770424905105066 | 0.0002515778361908061 | 0.0001238735871397555 (0.33053334844907367) | 0.07519126739383158 |
| *simple control* range_10d `-ts_mean(((close - low) / close), 10)` | 0.0675223594147245 | 0.08168754151237534 (7.120987588283992) | 0.2499825752058029 | 0.00039088590783317885 | 0.00014090333262737598 (0.3937453307017764) | 0.08552832290481722 |

## Walk-forward, full precision

Source: `results/binance_walkforward.json`.

| Fold | Train / validation / test | GP pick (origin) | Validation IC | Test IC (t-stat) | Test net / day (t-stat) | Days with a defined IC (validation / test) | Test IC: momentum_20d / reversal_1d | Test net / day: momentum_20d / reversal_1d |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 2020–2021 / 2022 / 2023-01 to 2023-12 | `rank(ts_decay_linear(group_zscore(((low + open) / high), industry), 20))` (mutation:industry, gen 7) | 0.09880980010601817 | 0.05585083710164717 (3.4870264087344167) | -1.4231718011469857e-05 (-0.030175877665870918) | 364 of 364 / 364 of 364 | -0.04321781216397266 / 0.028308526631794412 | -0.0002936909233211895 / -0.0005544295921700096 |
| 1 | 2021–2022 / 2023 / 2024-01 to 2024-12 | `-volume` (crossover, gen 2) | 0.020381524654208727 | 0.0355939810291763 (3.468360207546086) | -0.000607884777647931 (-1.4466024956697374) | 364 of 364 / 365 of 365 | -0.017496146192444408 / 0.012864710545246051 | 0.0006717461375297615 / -0.0017446884189456892 |
| 2 | 2022–2023 / 2024 / 2025-01 to 2025-12 | `(-high / ts_min(low, 3))` (crossover, gen 8) | 0.060016410970508915 | 0.05176398919516002 (3.514106111893519) | -0.0011636075129319628 (-2.4119062894780527) | 365 of 365 / 364 of 364 | 0.006810627137847665 / 0.03445099387783291 | 0.0006391237551602504 / -0.0009770227332941985 |
| 3 | 2023–2024 / 2025 / 2026-01 to 2026-08 | `-volume` (mutation:subtree, gen 8) | 0.053565582492736796 | 0.05431174436708224 (4.246070500741888) | 0.0006210992728437737 (1.3387869466427704) | 364 of 364 / 242 of 242 | -0.02305913358334184 / -0.02376317181800733 | -0.0007040917314473729 / -0.002402150660760874 |

## Picks that rank raw price or volume levels

`coin_units` (`src/alpha_gp_lab/grammar.py`) flags an expression whose per-coin unit does not
cancel. Applied to the picks already on disk:

```sh
PYTHONPATH=src python3.11 -c "
import json; from alpha_gp_lab.grammar import parse, coin_units
a = json.load(open('results/binance_analysis.json')); w = json.load(open('results/binance_walkforward.json'))
for k in ('gp', 'random'): print(k, [r['selected'] for r in a['per_seed'] if r['kind'] == k and coin_units(parse(r['selected'])) != 0])
print('walk-forward folds', [f['fold'] for f in w['folds'] if coin_units(parse(f['selected'])) != 0])"
```

On the regenerated files it prints 7 of the 20 GP seed picks, 4 of the 20 random-search picks and
walk-forward folds 1 and 3 (before the 2026-10-06 amendment: 6, 4 and fold 3). The main pick is
unit-free.
