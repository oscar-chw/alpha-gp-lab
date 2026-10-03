# alpha-gp-lab

that proposes seed expressions, and strict train / validation / test roles, in the Python
standard library only.


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

**Walk-forward.** `walk_forward` in the config rolls train / validation / test windows forward
and reruns the whole search per fold, with no shared state; the report gives each fold's test IC.

**LLM seed proposer** (`src/alpha_gp_lab/llm_seed.py`). A prompt containing a short research
brief and the grammar (never data) asks for N expressions. Responses are cached in a JSON
replay file keyed by the SHA-256 of the exact prompt. Every proposed line is parsed; invalid
lines are logged, rejected and counted (a line starting `- ` is rejected too, because a list
bullet and a minus sign cannot be told apart). The demo and the tests read only the replay file.
`seeds --live` would call the local `claude -p` CLI to refresh the cache; it has not been
run. The shipped replay entry is labelled **HAND-WRITTEN FIXTURE (not real LLM output)**. It
was written while building this repo, by someone who knew how the synthetic data is
generated, and it deliberately contains one duplicate and three invalid lines.

**Data adapters** (`src/alpha_gp_lab/data.py`):
- a SYNTHETIC generator whose next-day returns load on lagged reversal and momentum
  features according to a regime per segment: `reversal`, `momentum`, `noise`, and `adverse`
  (reversal with its sign flipped);
- a loader for a directory of daily OHLCV CSVs (one per symbol, optional `groups.csv`);
- `scripts/fetch_binance_daily.py`, which would download Binance public spot daily klines from
  data.binance.vision and verify each archive's SHA-256 against its `.CHECKSUM` file. It is
  **not run**: downloading waits for permission. Its URL building, checksum logic and kline
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

All results are **SYNTHETIC**. Values are copied from the JSON each command prints (full
precision); "net" is the mean hypothetical return per daily interval of a unit-gross
portfolio after 5 bps per unit of turnover.

### 1. Demo: a regime change between validation and test (negative result)

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

### 2. Ablation: the same demo without LLM seeds

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

### 3. Walk-forward over four regimes

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

- `PYTHONPATH=src python3.11 -m unittest discover -s tests` prints `Ran 59 tests` and `OK`.
- `python3.11 tests/hand_cases.py` re-derives the evaluator's arithmetic in exact Fractions:
  for example IC 2/5, turnover 2, gross 1/20 and net 49/1000 on a four-asset case.
- On the development machine the demo printed `demo finished in 9.6s` to stderr on its last run, and
  `scripts/check.sh` (tests, hand cases, the three runs above and two replays) exited 0.
  Runtime varies by machine.

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

Real daily data, once downloading is approved (not run so far):

```sh
python3 scripts/fetch_binance_daily.py --symbols BTCUSDT,ETHUSDT --start 2024-01 --end 2024-06 --out data/binance
```

Then point a config at it with `"data": {"kind": "csv", "path": "<directory relative to the config>"}`
and run `python3.11 -m alpha_gp_lab run --config <that config> --out <fresh dir>`.
`seeds --live` refreshes the LLM replay cache through the local `claude -p` CLI; it spends
LLM quota and has not been run.

## Architecture

```
src/alpha_gp_lab/
  grammar.py       parser (ast, no eval), frozen trees, canonical form, industry templates
  evaluate.py      operator semantics, timing, rank IC, turnover, net, fitness, fingerprints
  gp.py            random trees, crossover, mutation, generations, validation selection, folds
  data.py          Panel, SYNTHETIC regime generator, OHLCV CSV loader
  llm_seed.py      prompt, hash-keyed replay cache, grammar validation, optional claude -p
  store.py         run bundle, append-only SQLite lineage, manifest, verify
  config.py        strict config validation, walk-forward folds
  cli.py           demo | walkforward | run | verify | seeds
scripts/           check.sh, fetch_binance_daily.py (not run)
tests/             unittest suite and hand_cases.py (Fractions, no evaluator import)
```

Diagrams of the data flow, the split roles and the lineage schema are in
[docs/architecture.md](docs/architecture.md).

## Limits

  The synthetic effects are planted by construction, so finding them shows that the machinery
  works, not that there is an edge anywhere.
- **The LLM has not been run.** The only replay entry is hand-written by someone who knew the
  generator. On real data, an LLM's prior knowledge of published anomalies is itself
  information from outside the sample period.
- **Selection bias.** Validation picks the best of up to 16 hall-of-fame candidates; the
  printed IC t-statistics do not adjust for that choice or for autocorrelated daily ICs.
- **Simple portfolio model.** Dollar-neutral, unit gross, daily rebalance, linear fee on
  turnover. No slippage, borrow cost, capacity, risk model or compounding. "net" is a
  hypothetical interval return, not a backtest.
- **Scale.** Pure Python, single process: the demo evaluates 640 candidate occurrences on
  40 assets. Universes of thousands of instruments would need vectorised code.
- **Correlation filter scope.** It compares candidates within one run; there is no persistent
  library of previously selected alphas.
- **Replay is platform-bound.** The code identity includes the Python version and platform, and
  `verify` refuses a bundle written elsewhere, because maths-library differences can change
  low-order float bits. The numbers above were produced on one machine only.
- **Replay is not a signature.** `verify` detects accidental or casual alteration, but someone
  who controls the code and every file can forge a consistent bundle.
- The Binance fetcher and the live LLM path are only tested offline, against fakes.

## What I learned

TODO-OSCAR: Oscar's own conclusions from building and running this lab (none are recorded in
the sources yet).

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
