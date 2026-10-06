# Real-data results in full

Every real-data result behind the README's Results table: the main run, the POST-HOC diagnostics,
the follow-up analyses and the walk-forward, each with its command and source file. All data here
is REAL (Binance daily spot bars); the SYNTHETIC runs are in [synthetic-results.md](synthetic-results.md).

Numbers on this page are rounded to 3-4 significant figures; the cited JSON files hold full
precision, and [real-data-runs.md](real-data-runs.md) repeats the main tables at full
precision with the run ledger. "net" is the mean hypothetical return per daily
interval of a unit-gross, dollar-neutral portfolio after the fee on turnover; "gross" is the same
before the fee; "sum" adds daily values without compounding; ICIR is the mean daily IC over its
standard deviation. How each result file is produced, checked and quoted here is drawn in
[DIAGRAMS.md](DIAGRAMS.md#5-how-a-result-reaches-the-readme).

## Summary

**Binance daily, 34 coins, 2020–2026.** 6.7 years of Binance daily bars, 34 coins (2020-01 to 2026-08). Splits fixed in a
committed config before any run: train 2020-2023, validation 2024, test 2025-01 to 2026-08.
Signals use data through the previous day only (delay 1), costs are 10 bps per side, and 176
looks at the test window are counted (walk-forward fold 3's validation and the post-hoc data
splits are not; see below). On the test window the pick's rank IC is 0.082, but
most of it is a fixed tilt (a ranking frozen at end-2024 scores 87% of it, a constant low-beta
ranking 81%) in a market where the daily-rebalanced equal-weight basket of the 34 coins fell
65.8%, and its returns are indistinguishable from zero before and after costs (gross t 0.96,
Newey-West 0.91; net t 0.22, Newey-West 0.21). The GP, an equal-budget random search and a
one-line range factor all score about 0.08. After removing beta, 65% of the IC remains; after
beta and size, IC 0.049 (60%), and it comes from down days, consistent with the known
low-volatility effect (a 60-day low-volatility ranking scores 0.045 under the same
neutralisation). Sources: `results/binance_*.json` (diagnostics post-hoc); survivorship-biased
universe, test window reused, other data SYNTHETIC ([limits.md](limits.md)).

## Universe, data and pre-registered setup

**Survivorship bias: these 34 coins are USDT pairs still trading in 2026, chosen in 2026.** The
rule, recorded in `fixtures/binance_universe.json`: the orchestrating agent picked, on 2026-10-03,
large USDT spot pairs listed on Binance by 2020-01 and still trading; no numeric size threshold
was recorded, so "large" is a judgement, not a reproducible screen. KNCUSDT met the rule but its
download failed. Which delisted pairs a point-in-time rule would have added has not been checked
(that needs a new download). One venue, daily bars, no borrow or funding costs (see [limits.md](limits.md)).

*Data.* Binance spot 1d klines from data.binance.vision, 2,435 days per coin, 2020-01-01 to
2026-08-31, pinned in `fixtures/binance_universe.json` (symbol list, date range, SHA-256 of each
CSV). `PYTHONPATH=src python3.11 -m alpha_gp_lab verify-data` prints `"ok": true` and exits 0 on
the local copy. The CSVs are not in the repository.

*Pre-registered setup.* `fixtures/binance_daily_config.json` and
`fixtures/binance_walkforward_config.json` were committed in `4f9c276`, before any GP or baseline
was run on the real data, and were not changed afterwards (commit timeline and hash map in
[evidence.md](evidence.md)). The configs fix:
- train 2020-01-01 to 2023-12-31, validation 2024, test 2025-01-01 to 2026-08-31 (607 daily
  intervals);
- delay 1 (signal from data through day t-1, held from the close of t to the close of t+1);
- costs 10 bps per side, i.e. 10 bps on every unit of notional traded;
- one group: crypto has no industries, so the industry operators act on the whole cross-section;
- GP seed 20261003, population 64, 10 generations, hall of fame 16, fitness penalties
  0.02 x turnover and 0.001 x size; selection: validation IC >= 0.01, turnover <= 1.0,
  correlation <= 0.7;
- baselines in the same run: `momentum_20d` = `close / ts_delay(close, 20)` and
  `reversal_1d` = `-returns`.

## Main run

No LLM seeds:
`PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/binance_daily_config.json --out runs/binance-main`
→ `results/binance_daily_main.json`.

| Signal | Validation IC | Test IC (t) | Test ICIR | Test turnover | Test gross / day | Test net / day (t) | Test net, sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| GP pick `abs(ts_decay_linear((low / close), 10))` | 0.0715 | 0.0821 (7.11) | 0.288 | 0.273 | 3.53e-4 | 8.05e-5 (0.22) | 0.0489 |
| momentum_20d `close / ts_delay(close, 20)` | -0.0175 | -0.00616 (-0.56) | -0.023 | 0.305 | 3.77e-4 | 7.23e-5 (0.18) | 0.0439 |
| reversal_1d `-returns` | 0.0129 | 0.0115 (1.12) | 0.045 | 1.31 | -2.19e-4 | -1.53e-3 (-4.09) | -0.929 |
| *random search, equal budget* `ts_decay_linear(-ts_std(returns, 5), 20)` | 0.0479 | 0.0800 (6.78) | 0.275 | 0.110 | 2.52e-4 | 1.41e-4 (0.38) | 0.0859 |
| *simple control* range_10d `-ts_mean(((close - low) / close), 10)` | 0.0675 | 0.0817 (7.12) | 0.289 | 0.238 | 3.91e-4 | 1.53e-4 (0.43) | 0.0929 |

The first three rows are the pre-registered main run. The two rows in italics come from the
follow-up analysis below (`results/binance_analysis.json`, `table`) and did not influence the
pick. ICIR comes from `results/binance_diagnostics.json` (`icir`).

- The pick was made by crossover in generation 6. Its validation net was -4.11e-4 per day
  (t -0.67): the selection rule ranks by IC-based fitness and does not look at net returns.
- The 14 candidates the correlation filter rejected are all variants of the same daily-range
  ratios (`low / high`, `low / close`, `open / high`), each correlated above 0.7 with the pick: the
  hall of fame was essentially one idea.
- `reversal_1d` has a positive test IC but loses after costs (turnover 1.31, net t -4.09); under
  delay 1 it also skips a day before trading.

## What the test IC is (POST-HOC diagnostics)

These were written after the test result, the follow-up analysis and a reviewer's probes had all
been seen; the test window is spent, so they explain the result and are not new evidence.
Command (exit 0, about 6 s):
`PYTHONPATH=src python3.11 scripts/diagnose_binance.py --config fixtures/binance_diagnostics_config.json --out results/binance_diagnostics.json`.
The script refuses to run unless its own portfolio loop reproduces the pick's committed gross and
net. Every value below is in `results/binance_diagnostics.json`.

**The market fell, and the IC came from the down days.** The equal-weighted, daily-rebalanced 34-coin basket rose
996% over train, rose 52.2% over validation and fell 65.8% over test (`basket_return`). The pick's
daily IC has correlation -0.615 with the basket's daily return; its mean IC is 0.250 on the 307
down days and -0.0893 on the 300 up days. The random-search pick (-0.635; 0.257 / -0.101) and the
range control (-0.629; 0.253 / -0.0933) behave the same (`market_state`).

**Most of it is a fixed tilt.** Two constant rankings, chosen without any test data and never
updated, score (`constant_rankings`):

| Constant ranking | Test IC (t; Newey-West t) | 95% block-bootstrap interval | Share of the pick's 0.0821 |
|---|---:|---:|---:|
| the pick's mean rank over validation (2024), frozen | 0.0714 (5.29; 6.14) | 0.0522 to 0.0906 | 87% |
| low beta: minus each coin's beta to the basket over train | 0.0665 (5.25; 6.01) | 0.0474 to 0.0850 | 81% |

The book is long large, low-beta coins and short small alts (`legs`, mean test weight and days in
the leg out of 607):

| Long leg | Weight | Days long | Test return | Short leg | Weight | Days short | Test return |
|---|---:|---:|---:|---|---:|---:|---:|
| TRXUSDT | 0.0536 | 601 | +30.0% | ONEUSDT | -0.0367 | 582 | -97.2% |
| BTCUSDT | 0.0487 | 601 | -16.9% | ZECUSDT | -0.0342 | 502 | +1,357% |
| BNBUSDT | 0.0442 | 579 | -2.3% | FETUSDT | -0.0299 | 518 | -88.4% |
| LTCUSDT | 0.0192 | 473 | -53.9% | THETAUSDT | -0.0289 | 555 | -92.8% |
| XRPUSDT | 0.0154 | 433 | -40.9% | ENJUSDT | -0.0188 | 444 | -88.8% |

**Part of it survives neutralisation, on down days only.** Each test day, the signal's ranks and the next-day returns
are residualised cross-sectionally on each coin's trailing beta (120 days) and/or log dollar
volume (20 days), both computed only from data the signal could see, and the rank IC is taken
between the residuals (`neutralised_ic`):

| Neutralised on | Test IC (t; Newey-West t) | 95% block-bootstrap interval | Mean IC (t), up days | Mean IC (t), down days |
|---|---:|---:|---:|---:|
| nothing (the pick as tested) | 0.0821 (7.11; 7.60) | 0.0637 to 0.100 | -0.0893 | 0.250 |
| trailing beta | 0.0534 (5.98; 5.89) | 0.0345 to 0.0727 | 0.0164 (1.22) | 0.0896 (7.81) |
| log dollar volume (size) | 0.0768 (6.85; 7.17) | 0.0578 to 0.0950 | -0.0797 (-6.05) | 0.230 (17.5) |
| beta and size | 0.0489 (5.56; 5.46) | 0.0311 to 0.0664 | 0.0113 (0.87) | 0.0857 (7.45) |

There are 300 up days and 307 down days. Beta, not size, carries most of the tilt. After removing
beta and size, IC 0.0489 remains (about 60% of the raw IC), but on up days it is 0.0113 with t 0.87,
zero within noise: **the residual comes from the down days.**

**The residual resembles the known low-volatility effect.** Under the same beta-and-size neutralisation
(`neutralised_references_beta_and_size`):

| Reference ranking, neutralised on beta and size | Test IC (t; Newey-West t) | 95% interval | Up days (t) | Down days (t) |
|---|---:|---:|---:|---:|
| plain low volatility, `-ts_std(returns, 60)` | 0.0451 (5.01; 4.69) | 0.0239 to 0.0659 | 0.0145 (1.09) | 0.0751 (6.32) |
| the pick's 2024 ranking, frozen | 0.0197 (2.30; 2.40) | 0.00309 to 0.0364 | -0.0258 (-2.30) | 0.0643 (5.18) |

A textbook 60-day volatility ranking scores almost the same residual IC as the GP's pick, with the
same up/down pattern. Even a ranking frozen at the end of 2024 keeps 0.0197 after neutralisation,
so part of the residual is still a fixed tilt rather than daily information. The evidence is
resemblance: the pick was not neutralised on volatility, and the overlap of the two residuals was
not measured. Either way it looks like a known effect measured on one falling window, not evidence
of a new or tradable signal.

**Why the P&L is about zero, before costs as well as after.** Test gross is 3.53e-4 per day with
t 0.959 (Newey-West 0.910): not significant before costs either. One coin explains it: ZECUSDT rose
1,357% over test while sitting in the short leg on 502 of 607 days, and its contribution to mean
gross was -4.47e-4 per day, larger than the whole gross. Dropping ZEC from the cross-section (a
diagnostic chosen after seeing it, not a strategy) gives gross 8.49e-4 (t 2.48) and net 5.71e-4
(t 1.66) per day (`without_largest_negative_contributor`). Rank IC ignores magnitudes; a 34-name
equal-rank book does not, so a single short squeeze outweighs hundreds of correct small ranks.

**Shorting would cost more than the net.** The short leg is half of unit gross, so a borrow cost
above 5.88% a year on it wipes out the test net of 8.05e-5 per day
(`short_borrow_break_even_per_year`; net / 0.5 x 365). Margin borrow on small alts is often above
that, and spread and slippage on the short leg come on top.

## Follow-up analyses (pre-registered after the main result)

Plan: `fixtures/binance_analysis_config.json` and `scripts/analyze_binance.py`, committed in
`405e288` at 18:15:39, before either was run on real data but **19 minutes after the main test
results were committed** (`8306c42`, 17:56:41). Its seeds, budget, bootstrap and decision rules
were fixed before it ran, but it was written knowing the pick and its test score; in particular
the range control was chosen because the pick looked like a range measure
([timeline](evidence.md#timeline-what-each-plan-could-have-seen)). Command (run once,
exit 0, about 6 minutes on 10 processes):
`PYTHONPATH=src python3.11 scripts/analyze_binance.py --config fixtures/binance_analysis_config.json --out results/binance_analysis.json`.
It refuses to run unless its rerun of the main seed reproduces `results/binance_daily_main.json`.
Every value below is in `results/binance_analysis.json`.

**1. Equal-budget random search.** `gp.random_search` draws random expressions from the GP's own
generation-0 generator, admits them through the same size, duplicate, degenerate and equivalence
filter until it holds 640, takes the 16 best on train, then applies the GP's validation rule and
one test score: the GP's budget without breeding. Repeats are refused across the whole sample, so
random search scores 640 distinct expressions to the GP's 487, which slightly favours it.

- Same seed as the main run: random-search test IC 0.0800 against the GP's 0.0821; the mean daily
  difference, GP minus random, is 0.00207 (95% block-bootstrap interval -0.00992 to 0.0138). The
  random pick earned more net (1.41e-4 vs 8.05e-5 per day; difference interval -4.36e-4 to
  3.37e-4), also within noise.
- Over 20 pre-registered seeds (20261003 to 20261022; every search selected a candidate):

| Search | Test IC: mean (sd) | Test IC: min / max | Test IC > 0 | Test net / day: mean (sd) | Test net > 0 |
|---|---:|---:|---:|---:|---:|
| GP | 0.0739 (0.0137) | 0.0463 / 0.0951 | 20 of 20 | 2.37e-4 (3.44e-4) | 16 of 20 |
| Random search | 0.0668 (0.0261) | -0.00536 / 0.0867 | 19 of 20 | -8.90e-5 (3.31e-4) | 8 of 20 |
| GP minus random (Welch SE) | 0.00708 (0.00659) | | | 3.26e-4 (1.07e-4) | |

  By the pre-registered rule (a difference above 2 standard errors), **GP does not beat random
  search on test IC** (1.07 SE). It passes the rule on test net (3.06 SE), but that is **not
  evidence of out-of-sample net alpha**: all 40 searches share one test window, so the SE measures
  how consistently GP picks smoother signals, not whether smoother signals earn more in another
  period. A post-hoc description: 7 of the 20 random picks are raw one- or two-day range ratios
  such as `low / high`, with a mean test turnover of 0.373 against 0.142 for the GP picks (means of
  `per_seed[].test.mean_turnover`); the GP's net edge looks like smoothing, not a better signal.

**2. Serial dependence and the multiple-testing family.**

- The evaluator's `ic_tstat` and `net_tstat` use the i.i.d. standard error, so they **do not
  account for autocorrelation**. Newey-West t-statistics (Bartlett kernel; 5 lags by the usual rule
  of thumb for 607 days, and 20): test IC 7.60 and 8.75, test net 0.207 and 0.214. The daily ICs
  are slightly negatively autocorrelated, so the correction raises the IC t. Neither correction
  touches the bigger issue above: the whole window is one draw of "alts fell against majors".
- Block bootstrap (circular, 20-day blocks, 10,000 resamples, seed 20261003; IC and net use the same
  resampled days), 95% intervals of the test mean (`inference`):

| Signal | Test IC interval | Test net / day interval | Share of resampled net means <= 0 |
|---|---:|---:|---:|
| GP pick | 0.0637 to 0.100 | -7.06e-4 to 7.68e-4 | 0.395 |
| Random-search pick | 0.0624 to 0.0974 | -7.40e-4 to 9.27e-4 | 0.347 |
| Simple control (range_10d) | 0.0626 to 0.100 | -5.92e-4 to 8.19e-4 | 0.318 |
| momentum_20d | -0.0294 to 0.0170 | -7.48e-4 to 9.83e-4 | 0.451 |
| reversal_1d | -0.0135 to 0.0374 | -2.37e-3 to -6.98e-4 | 1.0 |

- The family that matters is the looks at the 2025-01 to 2026-08 window, not the expressions the
  search tried on train and validation (those never saw test). Not counted: walk-forward fold 3,
  which scored its 16 validation candidates on 2025, inside this window, and the post-hoc data
  splits (up/down days, the without-ZEC test at gross t 2.48). Counted in
  `fixtures/binance_diagnostics_config.json`: 3 in the main run (pick and 2 baselines), 40 in the
  follow-up (19 more GP seeds, 20 random-search seeds, the range control), 6 from walk-forward
  folds 2 and 3 (sub-windows of this one), 7 post-hoc diagnostics, and the 120 grid hypotheses of
  the sibling repo asof-research on the same pairs and window: **176 looks**. Bonferroni over 176
  (two-sided normal p; `test_window_looks`): the pick's raw test IC gives 2.10e-10 (i.i.d. t) and
  5.24e-12 (Newey-West t); the neutralised ICs give 6.64e-7 (beta), 1.28e-10 (size) and 8.20e-6
  (beta and size). The net was never significant, so there is nothing to correct there. Bonferroni
  treats each look as independent evidence; it cannot repair the single-window problem.

**3. What the pick is.** `abs(ts_decay_linear((low / close), 10))`: `low / close` lies in (0, 1]
for positive prices, so `abs` does nothing (GP seed 20261019 found the same expression without
`abs`, with identical test numbers). It is a linearly weighted 10-day mean of how close each day's
close sat to its low: high for coins with small recent `(close - low) / close`, a downside-range
measure. On train and validation only, its mean cross-sectional rank correlation is 0.939 / 0.944
with the 10-day mean range and 0.687 / 0.720 with negative 20-day realised volatility
(`interpretation`). Two of the 20 GP seeds picked negative 20-day volatility itself. The control,
the plain 10-day mean of `(close - low) / close` negated, does as well: test IC 0.0817 vs 0.0821
(difference interval -0.00445 to 0.00525), net 1.53e-4 vs 8.05e-5 per day (difference interval
-2.25e-4 to 7.96e-5) (`pick_minus.range_10d`). **A one-line, non-GP version captures the pick**,
and the diagnostics above say what both capture on this window: mostly a fixed tilt (frozen
ranking 87%, constant low-beta ranking 81%) in a falling alt market, with 60% of the IC left after
beta and size neutralisation.

**4. Figures** (`scripts/plot_binance.py`, run with matplotlib from the shared venv; the library
does not need it). Rolling 60-day mean test IC:

![Rolling 60-day test rank IC: the GP pick, the random-search pick and the simple control move together between about 0 and 0.15; momentum and reversal swing around zero](figures/rolling_ic.png)

Cumulative net return on test, after 10 bps per side:

![Cumulative net return on the test period: the GP pick, the random-search pick, the simple control and 20-day momentum all end between +0.04 and +0.10 summed over 607 days, while 1-day reversal loses 0.93](figures/cumulative_net.png)

## Walk-forward

Four rolling yearly folds (two years train, one validation, one test), same settings:
`PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/binance_walkforward_config.json --out runs/binance-wf`
→ `results/binance_walkforward.json` (full-precision table in
[real-data-runs.md](real-data-runs.md#walk-forward-full-precision)).

| Fold | Test window | GP pick | Validation IC | Test IC (t) | Test net / day (t) | Test IC: momentum / reversal |
|---:|---|---|---:|---:|---:|---:|
| 0 | 2023 | `abs(ts_decay_linear(((low + low) / high), 20))` | 0.0972 | 0.0493 (2.99) | -1.91e-4 (-0.39) | -0.0432 / 0.0283 |
| 1 | 2024 | `ts_min(ts_mean(ts_corr(low, low, 3), 3), 3)` | 0.0226 | -0.0408 (-0.76) | -1.59e-6 (-0.07) | -0.0175 / 0.0129 |
| 2 | 2025 | `ts_sum(ts_sum(ts_sum(ts_min(returns, 5), 20), 5), 20)` | 0.0395 | 0.0643 (3.50) | 8.27e-4 (1.57) | 0.00681 / 0.0345 |
| 3 | 2026-01 to 2026-08 | `ts_min((ts_min(low, 20) - group_neutralize(zscore(returns), industry)), 20)` | 0.0740 | 0.0932 (6.04) | 8.51e-4 (1.62) | -0.0231 / -0.0238 |

Mean test IC 0.0415; 3 folds positive, 1 negative. Fold 1's pick (`ts_corr(low, low, 3)` underneath,
the correlation of a series with itself) is constant on almost every day, so its IC rests on 10
validation days and 5 test days: the rule had no minimum-coverage check. Fold 3's pick subtracts a
z-score from `ts_min(low, 20)`, a raw price, so it largely ranks by price level, a size proxy (see
[limits.md](limits.md)). Folds 2 and 3 overlap the main test window; they are separate searches, not extra
evidence for the main pick.
