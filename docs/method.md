# How the search works

The grammar, the timing contract, the genetic programming loop, the split roles, the controls and
the run bundle, in the order a run uses them. The diagrams are in [DIAGRAMS.md](DIAGRAMS.md); the
module table and lineage schema are in [architecture.md](architecture.md).

**Grammar.** Expressions are parsed with Python's `ast` into frozen trees and never passed
to `eval`. Fields: `open, high, low, close, volume, returns`. Operators: `rank, zscore, abs,
sign, log, winsorize`, three industry-relative operators (`group_rank / group_zscore /
group_neutralize(x, industry)`), ten time-series operators (`ts_delta, ts_delay, ts_mean,
ts_sum, ts_std, ts_rank, ts_min, ts_max, ts_decay_linear, ts_corr`) and `+ - * /` (protected
division). Unknown fields, operators, windows or syntax are refused with a reason. The
semantics are local definitions written down in `src/alpha_gp_lab/evaluate.py`.

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
The generation loop with the main run's budget (population 64, 10 generations, hall of fame 16)
is drawn in [DIAGRAMS.md, diagram 3](DIAGRAMS.md#3-the-gp-generation-loop).

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

The split roles on the real data, with the dates from the committed configs, are drawn in
[DIAGRAMS.md, diagram 2](DIAGRAMS.md#2-split-roles-on-the-real-data); walk-forward folds 2 and 3
fall inside the main test window.

**Walk-forward.** A list of folds, or `walk_forward` in the config, rolls train / validation / test
windows forward and reruns the whole search per fold, with no shared state; the report gives
each fold's test IC.

**Baselines.** A config may name fixed control expressions (`"baselines"`). Each fold scores
them on train, validation and test with the same evaluators, timing and costs as the GP's pick.
They never enter the GP, the hall of fame or the selection. The printed summary also counts
`validation_candidates`, every candidate the pick was chosen from on validation.

**Controls and inference** (`gp.random_search`, `src/alpha_gp_lab/stats.py`). Random search draws
the GP's budget of random expressions from its own generator and filter, with no breeding, and
goes through the same validation rule and single test score. For the real-data pick, Newey-West
t-statistics, a circular block bootstrap over days, Bonferroni over the counted looks at the test window,
and post-hoc diagnostics (IC by market state, constant-ranking controls, IC after cross-sectional
residualisation on trailing beta and size, per-coin P&L contributions) sit beside the i.i.d.
t-statistic the evaluator prints.

**Opt-in rules for future windows.** `gp.unit_check` rejects any candidate whose per-coin unit
does not cancel (`grammar.coin_units`: a price is USDT per coin, a volume is coins, so ranking raw
`close` or `volume` across coins ranks by the arbitrary size of a coin unit). `selection.min_coverage`
refuses a candidate whose IC is defined on less than that share of validation days. Both are off
in every committed config so the recorded runs replay; a result under them needs data after
2026-08-31.

**LLM seed proposer** (`src/alpha_gp_lab/llm_seed.py`). Optional seeds for generation 0, parsed
by the same grammar; how they are requested, cached and refused is in [llm-seeds.md](llm-seeds.md).

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

**Persistence and replay** (`src/alpha_gp_lab/store.py`). Each run writes a fresh directory:
exact config bytes, the panel, the LLM seed record, the code hashes, an append-only SQLite
lineage (nodes, parent edges, per-split results, selections, LLM verdicts; triggers abort any
UPDATE, DELETE or key-colliding INSERT, so `INSERT OR REPLACE` cannot rewrite a row either),
the report, and last a hash manifest. `verify` checks the hashes, the code and platform
identity, the inputs (a synthetic panel must be exactly what its config generates), compares
the stored schema and trigger SQL with the expected DDL, recomputes the entire experiment and
compares it byte for byte, then reads every SQLite row back. A directory without the manifest (an interrupted run) is kept
but refused for reuse.
