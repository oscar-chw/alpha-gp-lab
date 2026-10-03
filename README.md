# alpha-gp-lab

that proposes seed expressions, and strict train / validation / test roles, in the Python
standard library only.

Results come from two sources, always labelled: real Binance spot daily bars for 34 coins
(2020-01-01 to 2026-08-31, a **survivorship-biased** universe, see Limits) and SYNTHETIC data

## The problem

factors (`rank(-ts_delta(close, 5))` and the like) for ones that predict next-period
cross-sectional returns. Search is cheap; honest evaluation is not. A search that is allowed
to look at the data it is judged on will always find something, and an LLM asked for
"good alphas" adds a second, unaudited source of ideas.

This lab asks a narrow question: if a search may only *breed* on train data, may only
*choose* on validation data, and must accept a single untouched test score, what does it
actually find, and does that survive a regime change?

## Approach (methods and algorithms)

**Grammar.** Expressions are parsed with Python's `ast` into frozen trees and never passed
to `eval`. Fields: `open, high, low, close, volume, returns`. Operators: `rank, zscore, abs,
sign, log, winsorize`, three industry-relative operators (`group_rank / group_zscore /
group_neutralize(x, industry)`), ten time-series operators (`ts_delta, ts_delay, ts_mean,
ts_sum, ts_std, ts_rank, ts_min, ts_max, ts_decay_linear, ts_corr`) and `+ - * /` (protected
division). Unknown fields, operators, windows or syntax are refused with a reason. The
semantics are local definitions written down in `src/alpha_gp_lab/evaluate.py`; they do not

**Timing and metrics.** The interval at date t opens at `close[t]` and closes at
`close[t+1]`; its signal is the expression's row at `t - delay` (delay 1 in every run here).
Each day the signal's cross-sectional ranks are centred and scaled to a dollar-neutral,
unit-gross portfolio. Per split the evaluator reports mean rank IC (Spearman, average ties),
its t-statistic, mean daily turnover (`sum |w_t - w_{t-1}|`, between 0 and 2), and mean
gross and net interval return (net = gross - `fee_bps` x turnover). A constant signal
abstains into cash.

**Genetic programming** (`src/alpha_gp_lab/gp.py`), with the budget set by config:
- generation 0: the LLM's accepted seeds, then ramped half-and-half random trees;
- tournament selection, elitism, subtree crossover, and five mutations (subtree, point,
  window, hoist, and wrapping a subtree in one of five industry-relative templates);
- depth and node-count limits; every offspring is re-parsed by the same parser as any input;
- a duplicate filter (canonical form, so `a + b` equals `b + a`) and a semantic-equivalence
  filter (a hash of the signal's ranks on every train day, so `rank(x)`, `zscore(x)` and `x`
  count as one candidate);
- fitness = mean rank IC - `turnover_penalty` x turnover - `complexity_penalty` x tree size.

**Splits.** A config gives index spans or ISO-date spans (`["2024-01-01", "2024-12-31"]`
resolves to the first and last panel dates inside it), as one fold, an explicit list of folds,
or a rolling `walk_forward` in days.

**Split roles.**
- *Train* fitness drives tournaments, elitism and the hall of fame (best train score per
  distinct signal).
- *Validation* re-scores the hall of fame, keeps candidates with IC >= `min_ic` and turnover
  <= `max_turnover`, ranks them by the same penalised fitness, and builds a shortlist with a
  correlation filter: a candidate is rejected if its mean cross-sectional signal correlation
  exceeds `max_corr` with an already-selected alpha, meaning one listed in the config's
  `existing_alphas` (alphas chosen before this run) or one already on the shortlist. The top
  of the shortlist is the final pick. The shipped configs list no existing alphas, so there
  the filter only shapes the shortlist and the pick is the best validation score; a test
  shows an existing alpha pushing the pick elsewhere.
- *Test* scores only that final pick, once.

Each evaluator is built on the panel truncated at its split's last label, so later bars do not
exist inside it.

**Walk-forward.** A list of folds, or `walk_forward` in the config, rolls train / validation / test
windows forward and reruns the whole search per fold, with no shared state; the report gives
each fold's test IC.

**Baselines.** A config may name fixed control expressions (`"baselines"`). Each fold scores
them on train, validation and test with the same evaluators, timing and costs as the GP's pick.
They never enter the GP, the hall of fame or the selection. The printed summary also counts
`validation_candidates`, every candidate the pick was chosen from on validation.

**LLM seed proposer** (`src/alpha_gp_lab/llm_seed.py`). A prompt containing a short research
brief and the grammar (never data) asks for N expressions. Responses are cached in a JSON
replay file keyed by the SHA-256 of the exact prompt. Every proposed line is parsed; invalid
lines are logged, rejected and counted (a line starting `- ` is rejected too, because a list
bullet and a minus sign cannot be told apart). The demo and the tests read only the replay file.
`seeds --live` calls the local `claude -p` CLI with the prompt on stdin, every tool disabled and
an empty working directory, and saves the answer labelled **REAL LLM OUTPUT** with the CLI
version, model and date. A run with `use_seeds: false` reads no LLM output at all. The one live
attempt so far failed before reaching a model (see Results). The only shipped replay entry is
labelled **HAND-WRITTEN FIXTURE (not real LLM output)**. It was written while building this repo,
by someone who knew how the synthetic data is generated, and it deliberately contains one
duplicate and three invalid lines. It is used by the SYNTHETIC demo only.

**Data adapters** (`src/alpha_gp_lab/data.py`):
- a SYNTHETIC generator whose next-day returns load on lagged reversal and momentum
  features according to a regime per segment: `reversal`, `momentum`, `noise`, and `adverse`
  (reversal with its sign flipped);
- a loader for a directory of daily OHLCV CSVs (one per symbol, optional `groups.csv`);
- `scripts/fetch_binance_daily.py`, which downloads Binance public spot daily klines from
  data.binance.vision and verifies each archive's SHA-256 against its `.CHECKSUM` file. It was
  run once, with permission, for the universe pinned in `fixtures/binance_universe.json`
  (symbols, date range and each CSV's SHA-256). `verify-data` checks a local directory against
  that file. The CSVs are not committed. The fetcher's URL building, checksum logic and kline
  conversion are unit-tested offline.

in-memory fake transport that returns fixture responses, and a submit / poll / fetch client
that parses the expression locally before anything reaches the transport. There is no
endpoint URL anywhere in the package.

**Persistence and replay** (`src/alpha_gp_lab/store.py`). Each run writes a fresh directory:
exact config bytes, the panel, the LLM seed record, the code hashes, an append-only SQLite
lineage (nodes, parent edges, per-split results, selections, LLM verdicts; triggers abort any
UPDATE, DELETE or key-colliding INSERT, so `INSERT OR REPLACE` cannot rewrite a row either),
the report, and last a hash manifest. `verify` checks the hashes, the code and platform
identity, the inputs (a synthetic panel must be exactly what its config generates), compares
the stored schema and trigger SQL with the expected DDL, recomputes the entire experiment and
compares it byte for byte, then reads every SQLite row back. A directory without the manifest (an interrupted run) is kept
but refused for reuse.

## Results (real numbers with their source; synthetic clearly labelled)

Values are copied from the JSON each command prints (full precision). "net" is the mean
hypothetical return per daily interval of a unit-gross, dollar-neutral portfolio after the fee
on turnover; "sum" adds the daily values without compounding.

### Real data (Binance daily, 34 coins, 2020–2026)

**Survivorship bias: these 34 coins are USDT pairs still trading in 2026, chosen in 2026.
Every number in this subsection is biased toward coins that survived.** KNCUSDT failed to
download and is excluded. One venue, daily bars, no borrow or funding costs (see Limits).

*Data.* Binance spot 1d klines from data.binance.vision, 2,435 days per coin, 2020-01-01 to
2026-08-31, pinned in `fixtures/binance_universe.json` (symbol list, date range, SHA-256 of
each CSV). `PYTHONPATH=src python3.11 -m alpha_gp_lab verify-data` prints `"ok": true` and exits
0 on the local copy. The CSVs are not in the repository.

*Pre-registered setup.* `fixtures/binance_daily_config.json` and
`fixtures/binance_walkforward_config.json` were committed in `3efa07d`, before any GP or
baseline was run on the real data, and were not changed afterwards:
- train 2020-01-01 to 2023-12-31, validation 2024-01-01 to 2024-12-31, test 2025-01-01 to
  2026-08-31 (607 daily intervals);
- delay 1 (signal from data through day t-1, held from the close of t to the close of t+1);
- costs 10 bps per side, i.e. 10 bps on every unit of notional traded;
- one group: crypto has no industries, so the industry operators act on the whole cross-section;
- GP seed 20261003 (the only seed tried), the synthetic demo's budget (population 64,
  10 generations, hall of fame 16), fitness penalties 0.02 x turnover and 0.001 x size;
- selection: validation IC >= 0.01, turnover <= 1.0, correlation <= 0.7;
- baselines in the same run: `momentum_20d` = `close / ts_delay(close, 20)` (the 20-day return)
  and `reversal_1d` = `-returns`, on the same splits, timing and costs.

**Main run** (no LLM seeds). Command:
`PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/binance_daily_config.json --out runs/binance-main`.
Output: `results/binance_daily_main.json`.

| Signal | Validation IC | Test IC (t-stat) | Test turnover | Test gross / day | Test net / day (t-stat) | Test net, sum over 607 days |
|---|---:|---:|---:|---:|---:|---:|
| GP pick `abs(ts_decay_linear((low / close), 10))` | 0.07148111461736868 | 0.08208763518369619 (7.106267209479283) | 0.27298586844370465 | 0.0003534791635240535 | 8.049329508034882e-05 (0.2184396051637011) | 0.04885943011377173 |
| momentum_20d `(close / ts_delay(close, 20))` | -0.017496146192444408 | -0.006158512090458223 (-0.5588498549217263) | 0.3045894780045946 | 0.0003768912600628651 | 7.23017820582705e-05 (0.1835034941876255) | 0.04388718170937019 |
| reversal_1d `-returns` | 0.012864710545246051 | 0.011485586060223782 (1.1150804633424585) | 1.3103524132462043 | -0.0002194546543761369 | -0.0015298070676223413 (-4.090451699976516) | -0.9285928900467612 |

- The pick was made by crossover in generation 6. Its validation net was
  -0.00041132983838806654 per day (t-stat -0.6699945071164971): the selection rule ranks by
  IC-based fitness and does not look at net returns.
- **Against the controls:** the pick beats both baselines on test IC. On test net it is above
  both, but only just above `momentum_20d` (8.049329508034882e-05 vs 7.23017820582705e-05 per
  day), and its net t-stat of 0.2184396051637011 cannot be told apart from zero. Momentum earned slightly *more* gross (0.0003768912600628651
  vs 0.0003534791635240535 per day); the pick's edge in net comes from lower turnover. On this
  sample the test net is positive and above both baselines, but it is not evidence of a
  profitable strategy.
- `reversal_1d` has a positive test IC but loses after costs (turnover 1.3103524132462043,
  net t-stat -4.090451699976516); under delay 1 it also skips a day before trading.
- The 14 candidates the correlation filter rejected are all variants of the same daily-range
  ratios (`low / high`, `low / close`, `open / high`), each correlated above 0.7 with the pick
  (`correlation_rejected` in the run's `report.json` lists them with their correlations): the
  hall of fame was essentially one idea.

**Walk-forward**, four rolling yearly folds (two years train, one validation, one test), same
settings. Command:
`PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/binance_walkforward_config.json --out runs/binance-wf`.
Output: `results/binance_walkforward.json`.

| Fold | Train / validation / test | GP pick (origin) | Validation IC | Test IC (t-stat) | Test net / day (t-stat) | Days with a defined IC (validation / test) | Test IC: momentum_20d / reversal_1d | Test net / day: momentum_20d / reversal_1d |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 2020–2021 / 2022 / 2023-01 to 2023-12 | `abs(ts_decay_linear(((low + low) / high), 20))` (mutation:subtree, gen 9) | 0.0972391015759644 | 0.04926120863890851 (2.991478763240003) | -0.0001906847106677105 (-0.3942232962089944) | 364 of 364 / 364 of 364 | -0.04321781216397266 / 0.028308526631794412 | -0.0002851170095445421 / -0.0005510324356024837 |
| 1 | 2021–2022 / 2023 / 2024-01 to 2024-12 | `ts_min(ts_mean(ts_corr(low, low, 3), 3), 3)` (crossover, gen 7) | 0.022632785897698147 | -0.040810366686938666 (-0.7607723819039659) | -1.587844156803073e-06 (-0.06876089594446956) | 10 of 364 / 5 of 365 | -0.017496146192444408 / 0.012864710545246051 | 0.0006836112765127413 / -0.001741581359885962 |
| 2 | 2022–2023 / 2024 / 2025-01 to 2025-12 | `ts_sum(ts_sum(ts_sum(ts_min(returns, 5), 20), 5), 20)` (crossover, gen 8) | 0.0394824080783983 | 0.06430799835763458 (3.495329956753379) | 0.0008274287240752346 (1.570693234427387) | 365 of 365 / 364 of 364 | 0.006810627137847665 / 0.03445099387783291 | 0.0006513061922729902 / -0.0009755359373508695 |
| 3 | 2023–2024 / 2025 / 2026-01 to 2026-08 | `ts_min((ts_min(low, 20) - group_neutralize(zscore(returns), industry)), 20)` (crossover, gen 8) | 0.0739947424704937 | 0.09319068722568953 (6.035930760406955) | 0.0008513965938700916 (1.620099777311947) | 364 of 364 / 242 of 242 | -0.02305913358334184 / -0.02376317181800733 | -0.0006945371542677393 / -0.0024008227684519583 |

Printed summary: 4 folds, 4 with a selection, mean test IC 0.04148738188382349, 3 folds
positive and 1 negative. The GP's test IC is above both baselines in folds 0, 2 and 3 and below
both in fold 1. Fold 1's pick, `ts_corr(low, low, 3)` underneath, is constant on almost every
day (the correlation of a series with itself), so it abstains into cash and its IC rests on
10 validation days and 5 test days: the qualification rule has no minimum-coverage check. Fold 3's
pick subtracts a z-score from `ts_min(low, 20)`, a raw price in USDT, so for most coins it ranks by
price level. Folds 2 and 3 overlap the main run's test period; they are separate searches on
different training windows, not extra evidence for the main pick.

**LLM seeds: not run.** The single permitted live call,
`PYTHONPATH=src python3.11 -m alpha_gp_lab seeds --live --config fixtures/binance_daily_config.json`,
exited 1 before reaching a model: `claude auth status` reports the local CLI is not signed in,
and the CLI's JSON shows 0 input and 0 output tokens. Nothing was saved as LLM output, and no
retry was made (1 of the 3 allowed calls used). Record: `results/llm_live_attempt.json`. So the
seeded run (`fixtures/binance_daily_seeded_config.json`, pre-registered in `3efa07d`, identical
to the main config except `use_seeds`) has not been run, and there is no seeded-vs-ablation
comparison on real data. The ablation without seeds is the main run above.

**Every real-data run, and the multiple-testing count.**

| # | Command | Code | Exit | What happened |
|---:|---|---|---:|---|
| 1 | `verify-data` (several times, and inside `check.sh`) | from `3efa07d` on | 0 | 34 files match the pinned SHA-256s, rows and dates |
| 2 | `seeds --live` (main config) | `3efa07d` | 1 | CLI not signed in; no LLM output |
| 3 | `run` main config | `90337d2` | 0 | the numbers above; printed before `net_tstat` existed; output overwritten by run 4 |
| 4 | `run` main config | `2ba1d3a` | 0 | identical numbers plus `net_tstat`; overwritten by run 6 |
| 5 | `run` walk-forward config | `2ba1d3a` | 0 | identical numbers, without `mean_gross` and `valid_ic_intervals`; overwritten by run 7 |
| 6 | `run` main config | `6352651` | 0 | committed as `results/binance_daily_main.json` |
| 7 | `run` walk-forward config | `6352651` | 0 | committed as `results/binance_walkforward.json` |

Runs 4 to 7 were repeated only to print two more statistics (`net_tstat`, then `mean_gross` and
`valid_ic_intervals`); every number that existed before was reproduced exactly, and the
selection does not read the added fields. Each `run` also replays itself once inside `verify`,
and `scripts/check.sh` re-runs the main config and compares it with the committed file. A
timing run on SYNTHETIC data shaped like the real panel (34 assets, 2,435 days) was made before
pre-registration; it used no real data.

Multiple testing: 1 main config, 1 walk-forward config and 1 GP seed were tried; no config,
seed or threshold was changed after a real-data result. In the main run the GP scored 640
candidate occurrences (487 distinct expressions) on train, 16 hall-of-fame candidates on
validation, and tested 1. The walk-forward scored 16 candidates on validation in each of its 4
folds (64 in total; 479, 471, 498 and 488 distinct expressions on train). The 2 baselines were
also scored on validation and test in every fold, as controls that cannot be selected. None of
the printed t-statistics adjusts for this.

### Synthetic results

All results in this subsection are **SYNTHETIC**, with 5 bps per unit of turnover.

#### 1. Demo: a regime change between validation and test (negative result)

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
| validation mean_ic | 0.1330271074594035 (ic_tstat 6.937494613005247, 58 intervals) |
| validation mean_turnover / mean_net | 1.3405172413793103 / 0.0019615477294169584 |
| **test mean_ic** | **-0.18768721976659142** (ic_tstat -11.444822278839396, 59 intervals) |
| **test mean_net** | **-0.004342244755687273** (mean_turnover 1.3214406779661017) |
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

#### 2. Ablation: the same demo without LLM seeds

Command: `PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/demo_no_seeds_config.json --out runs/no-seeds`
(identical config except `"use_seeds": false`).

| Field (printed) | Value |
|---|---|
| selected | `ts_rank(-returns, 20)` |
| selected_origin | `crossover`, generation 4 |
| validation mean_ic | 0.1412517609812856 (ic_tstat 5.774237143687695) |
| test mean_ic | -0.1530285151206077 (ic_tstat -7.127309136838554) |
| test mean_net | -0.00372178882397133 |
| occurrences / unique_expressions | 640 / 470 |

Without seeds the GP found a reversal expression on its own, with a validation IC close to
the seeded run's, and it failed on the adverse test period in the same way. So the seeded
win in the demo shows that the plumbing works. It is not evidence that an LLM helps: the
fixture is hand-written and no real LLM has been run.

#### 3. Walk-forward over four regimes

Command: `PYTHONPATH=src python3.11 -m alpha_gp_lab walkforward --out runs/wf`.
Config: `fixtures/walkforward_config.json`. Search seed 20261003, data seed 11, 30 assets in
3 industries, 520 days: 200 `reversal`, 120 `momentum`, 100 `noise`, 100 `adverse`. Folds:
warmup 30, train 100, validation 40, test 40, step 40 days. GP: population 32, 6 generations,
hall of fame 8; other settings as in the demo.

| Fold | Test regime(s) | Selected | Origin | Validation IC | Test IC |
|---:|---|---|---|---:|---:|
| 0 | momentum, reversal | `group_neutralize(-returns, industry)` | llm_seed | 0.16400444938820913 | 0.052291434927697444 |
| 1 | momentum | `-group_zscore(returns, industry)` | mutation:industry | 0.06203559510567297 | -0.10646273637374862 |
| 2 | momentum | none qualified | | | |
| 3 | momentum, noise | `ts_mean(returns, 5)` | llm_seed | 0.11698553948832036 | 0.18432703003337042 |
| 4 | noise | `ts_mean(returns, 5)` | llm_seed | 0.19098998887652946 | -0.02103448275862069 |
| 5 | noise | none qualified | | | |
| 6 | adverse, noise | `ts_mean(zscore(returns), 10)` | mutation:point | 0.020266963292547274 | 0.0689543937708565 |
| 7 | adverse | `ts_min(returns, 10)` | mutation:subtree | 0.06066740823136818 | 0.05968854282536151 |

Printed summary: 8 folds, 6 with a selection, mean test IC 0.0396273637374861, 4 folds
positive and 2 negative. In fold 1 a reversal alpha was chosen on a validation window that
was still mostly reversal and then tested on pure momentum; in fold 4 a momentum alpha was
chosen on a validation window that was partly momentum and then tested on pure noise.

### Tests and runtime

- `PYTHONPATH=src python3.11 -m unittest discover -s tests` prints `Ran 75 tests` and `OK`.
- `python3.11 tests/hand_cases.py` re-derives the evaluator's arithmetic in exact Fractions:
  for example IC 2/5, turnover 2, gross 1/20 and net 49/1000 on a four-asset case.
- On the development machine the demo printed `demo finished in 9.6s` to stderr on its last run, and
  `scripts/check.sh` (tests, hand cases, the three synthetic runs above, two replays, then
  `verify-data` and the main real-data run when the data is present) exited 0 both with and
  without `data/binance-daily/`. Runtime varies by machine.

## How to run (under 5 minutes)

Python 3.11 (the only version tested), standard library only. From the repository root:

```sh
export PYTHONPATH=src
python3.11 -m alpha_gp_lab demo --out runs/demo          # single split, SYNTHETIC, ~10 s here
python3.11 -m alpha_gp_lab verify runs/demo              # hashes + full recomputation + SQLite read-back
python3.11 -m alpha_gp_lab walkforward --out runs/wf     # rolling folds, about 20 s here
python3.11 -m alpha_gp_lab seeds                         # replayed LLM proposals and their verdicts
bash scripts/check.sh                                    # everything above plus the test suite
```

Output directories must not exist yet; an existing one is verified and reused only if the
inputs and code are identical.

Real data (optional; the download takes longer than five minutes and the CSVs are never
committed). The configs read `data/binance-daily/`, which `.gitignore` excludes:

```sh
python3 scripts/fetch_binance_daily.py --start 2020-01 --end 2026-08 --out data/binance-daily \
  --symbols "$(python3 -c 'import json; print(",".join(json.load(open("fixtures/binance_universe.json"))["symbols"]))')"
python3.11 -m alpha_gp_lab verify-data                   # SHA-256, rows and dates against the pinned universe
python3.11 -m alpha_gp_lab run --config fixtures/binance_daily_config.json --out runs/binance-main          # ~90 s here
python3.11 -m alpha_gp_lab run --config fixtures/binance_walkforward_config.json --out runs/binance-wf      # minutes
```

`check.sh` runs `verify-data` and re-runs the main real-data config against
`results/binance_daily_main.json` when `data/binance-daily/` exists, and prints a skip message
when it does not. `seeds --live --config <config>` refreshes a replay entry through the local
`claude -p` CLI; it spends LLM quota and needs the CLI to be signed in.

## Architecture

```
src/alpha_gp_lab/
  grammar.py       parser (ast, no eval), frozen trees, canonical form, industry templates
  evaluate.py      operator semantics, timing, rank IC, turnover, net, fitness, fingerprints
  gp.py            random trees, crossover, mutation, generations, validation selection, folds
  data.py          Panel, SYNTHETIC regime generator, OHLCV CSV loader, pinned-universe check
  llm_seed.py      prompt, hash-keyed replay cache, grammar validation, optional claude -p
  store.py         run bundle, append-only SQLite lineage, manifest, verify
  config.py        strict config validation, index or date splits, fold lists, walk-forward folds
  cli.py           demo | walkforward | run | verify | seeds | verify-data
scripts/           check.sh, fetch_binance_daily.py
fixtures/          configs (SYNTHETIC and Binance), binance_universe.json, LLM replay
results/           printed summaries of the real-data runs and the failed live-LLM attempt
tests/             unittest suite and hand_cases.py (Fractions, no evaluator import)
```

Diagrams of the data flow, the split roles and the lineage schema are in
[docs/architecture.md](docs/architecture.md).

## Limits

- **Survivorship bias.** The 34 real-data coins are pairs still trading on Binance in 2026,
  chosen in 2026. Coins that were delisted or collapsed between 2020 and 2026 are absent, and
  KNCUSDT is missing because its download failed. Every real-data number in this README is
  biased toward survivors.
- **One venue.** Binance spot prices and volumes only, with no cross-check against other
  exchanges.
- **Daily bars.** UTC close-to-close days. Delay 1 means a signal uses data through the close of
  day t-1 and is held from the close of day t to the close of day t+1, so anything faster than a
  day is invisible and the "1-day reversal" baseline skips a day before trading.
- **Cost model.** A linear fee on traded notional: 10 bps per side on the real data (a round trip
  costs 20 bps), 5 bps per side on the synthetic data. No spread, slippage, market impact, fee
  tiers or capacity limit.
- **No shorting constraints or funding costs.** The portfolio is dollar-neutral, so about half of
  it is short spot coins. Shorting spot needs margin borrowing or perpetual futures; borrow
  rates, borrow availability and funding payments are not modelled.
- **The test period is a single regime.** The main test window, 2025-01-01 to 2026-08-31, is one
  stretch of one market. The walk-forward adds four yearly test windows, still from one
  market's history.
- **Simple portfolio model.** Dollar-neutral, unit gross, daily rebalance, no risk model and no
  compounding. "net" is a hypothetical interval return, not a backtest.
- **Selection and multiple testing.** Validation picks the best of up to 16 hall-of-fame
  candidates, which the GP bred from hundreds of train-scored expressions. The printed
  t-statistics adjust for neither, nor for autocorrelated daily values.
- **No real LLM output.** The only replay entry is hand-written by someone who knew the
  synthetic generator, and the one live attempt failed. On real data an LLM's prior knowledge
  of published anomalies would itself be information from outside the sample period.
- **Synthetic effects are planted by construction**, so finding them shows that the machinery
  works, not that there is an edge anywhere.
- **Scale.** Pure Python, single process: the main real-data run takes about 90 s for 640
  candidate occurrences on 34 coins. Universes of thousands of instruments would need
  vectorised code.
- **Correlation filter scope.** It compares candidates within one run; there is no persistent
  library of previously selected alphas.
- **Replay is platform-bound.** The code identity includes the Python version and platform, and
  `verify` refuses a bundle written elsewhere, because maths-library differences can change
  low-order float bits. The numbers above were produced on one machine only.
- **Replay is not a signature.** `verify` detects accidental or casual alteration, but someone
  who controls the code and every file can forge a consistent bundle.
- The live LLM path is tested only against a fake `claude` executable.

## What I learned

Candidate lessons drawn from the real-data results above. None is Oscar's own conclusion until
he confirms or rewrites it.

1. A strong rank IC is not a tradable return. The main pick's test IC was 0.08208763518369619
   (t-stat 7.106267209479283), yet its net was 8.049329508034882e-05 per day with a t-stat of
   0.2184396051637011 after 10 bps per side.

DRAFT — Oscar to confirm

2. Selecting on IC-based fitness can choose a signal that lost money on validation (net
   -0.00041132983838806654 per day for the main pick). If net return is the goal, it has to be
   in the selection rule.

DRAFT — Oscar to confirm

3. Without diversity pressure the GP converges on one family: 14 of the 16 hall-of-fame
   members in the main run were near-copies of the pick's daily-range ratio, so "16 validation
   candidates" overstates how many different ideas were tested.

DRAFT — Oscar to confirm

4. A qualification rule needs a coverage check: walk-forward fold 1 selected a signal defined
   on 10 of 364 validation days.

DRAFT — Oscar to confirm

5. The textbook controls did poorly under delay 1 and 10 bps per side: 1-day reversal lost
   money on test (net t-stat -4.090451699976516) and 20-day momentum had a test IC of
   -0.006158512090458223. Beating them is a low bar, not a result.

DRAFT — Oscar to confirm

## Attribution

Implemented with AI coding agents under Oscar's design and review.

- The genetic-algorithm / bandit idea for searching alpha expressions and simulation settings
  (Apache-2.0, commit `dead3cc70a7b3c6a8bfd4849ef141690c2eaec19`). This repo implements the
  genetic-programming part; no code from that project is reused.
- The multi-agent research pattern (one role proposes hypotheses, another implements and
  evaluates them, held-out results decide) comes from
  [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) (MIT), via Oscar's fork
  [hihihhi/RD-Agent](https://github.com/hihihhi/RD-Agent). Here the LLM proposes, the GP
  develops, and the validation and test splits decide; no code is reused.
- This repo grows out of Oscar's earlier offline search workflow, a toy-scale single-generation
  version with the same no-`eval` parsing, split roles and SQLite replay ideas, which is
  rewritten here. The lower-bound-IC / upper-bound-turnover selection rule, the five
  industry-relative templates and the simulation-settings field set are clean
  blanket licence; none of that code is copied.

Licence: MIT (see [LICENSE](LICENSE)).
