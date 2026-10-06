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
| 10 | `scripts/diagnose_binance.py` (adds up/down t-statistics and two neutralised reference rankings) | `482f022` | 0 | every run-9 number reproduced; committed as `results/binance_diagnostics.json` |

Runs 4 to 7 were repeated only to print more statistics (`net_tstat`, then `mean_gross` and
`valid_ic_intervals`); every number that existed before was reproduced exactly, and the selection
does not read the added fields. Each `run` also replays itself once inside `verify`, and
`scripts/check.sh` re-runs the main config and compares it with the committed file. A timing run
on SYNTHETIC data shaped like the real panel (34 assets, 2,435 days) was made before
pre-registration; it used no real data.

## Main run and follow-up rows, full precision

Sources: `results/binance_daily_main.json` (first three rows) and `results/binance_analysis.json`
(`table`, last two rows).

| Signal | Validation IC | Test IC (t-stat) | Test turnover | Test gross / day | Test net / day (t-stat) | Test net, sum over 607 days |
|---|---:|---:|---:|---:|---:|---:|
| GP pick `abs(ts_decay_linear((low / close), 10))` | 0.07148111461736868 | 0.08208763518369619 (7.106267209479283) | 0.27298586844370465 | 0.0003534791635240535 | 8.049329508034882e-05 (0.2184396051637011) | 0.04885943011377173 |
| momentum_20d `(close / ts_delay(close, 20))` | -0.017496146192444408 | -0.006158512090458223 (-0.5588498549217263) | 0.3045894780045946 | 0.0003768912600628651 | 7.23017820582705e-05 (0.1835034941876255) | 0.04388718170937019 |
| reversal_1d `-returns` | 0.012864710545246051 | 0.011485586060223782 (1.1150804633424585) | 1.3103524132462043 | -0.0002194546543761369 | -0.0015298070676223413 (-4.090451699976516) | -0.9285928900467612 |
| *random search, equal budget* `ts_decay_linear(-ts_std(returns, 5), 20)` | 0.04790283617149554 | 0.08001626010548919 (6.7788549855658005) | 0.11008818683981006 | 0.0002515778361908061 | 0.00014148964935099608 (0.37756295049535377) | 0.08588421715605463 |
| *simple control* range_10d `-ts_mean(((close - low) / close), 10)` | 0.0675223594147245 | 0.08168754151237534 (7.120987588283992) | 0.23781374164163194 | 0.00039088590783317885 | 0.0001530721661915469 (0.42775063095956656) | 0.09291480487826898 |

## Walk-forward, full precision

Source: `results/binance_walkforward.json`.

| Fold | Train / validation / test | GP pick (origin) | Validation IC | Test IC (t-stat) | Test net / day (t-stat) | Days with a defined IC (validation / test) | Test IC: momentum_20d / reversal_1d | Test net / day: momentum_20d / reversal_1d |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 2020–2021 / 2022 / 2023-01 to 2023-12 | `abs(ts_decay_linear(((low + low) / high), 20))` (mutation:subtree, gen 9) | 0.0972391015759644 | 0.04926120863890851 (2.991478763240003) | -0.0001906847106677105 (-0.3942232962089944) | 364 of 364 / 364 of 364 | -0.04321781216397266 / 0.028308526631794412 | -0.0002851170095445421 / -0.0005510324356024837 |
| 1 | 2021–2022 / 2023 / 2024-01 to 2024-12 | `ts_min(ts_mean(ts_corr(low, low, 3), 3), 3)` (crossover, gen 7) | 0.022632785897698147 | -0.040810366686938666 (-0.7607723819039659) | -1.587844156803073e-06 (-0.06876089594446956) | 10 of 364 / 5 of 365 | -0.017496146192444408 / 0.012864710545246051 | 0.0006836112765127413 / -0.001741581359885962 |
| 2 | 2022–2023 / 2024 / 2025-01 to 2025-12 | `ts_sum(ts_sum(ts_sum(ts_min(returns, 5), 20), 5), 20)` (crossover, gen 8) | 0.0394824080783983 | 0.06430799835763458 (3.495329956753379) | 0.0008274287240752346 (1.570693234427387) | 365 of 365 / 364 of 364 | 0.006810627137847665 / 0.03445099387783291 | 0.0006513061922729902 / -0.0009755359373508695 |
| 3 | 2023–2024 / 2025 / 2026-01 to 2026-08 | `ts_min((ts_min(low, 20) - group_neutralize(zscore(returns), industry)), 20)` (crossover, gen 8) | 0.0739947424704937 | 0.09319068722568953 (6.035930760406955) | 0.0008513965938700916 (1.620099777311947) | 364 of 364 / 242 of 242 | -0.02305913358334184 / -0.02376317181800733 | -0.0006945371542677393 / -0.0024008227684519583 |

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

It prints 6 of the 20 GP seed picks, 4 of the 20 random-search picks and walk-forward fold 3. The
main pick is unit-free.
