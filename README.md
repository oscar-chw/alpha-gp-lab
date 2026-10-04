# alpha-gp-lab

*choose* on validation data and take one untouched test score, with pre-registered real-data runs
and controls. Python standard library only.

**Real data:** 6.7 years of Binance daily bars, 34 coins (2020-01 to 2026-08). Splits fixed in a
committed config before any run: train 2020-2023, validation 2024, test 2025-01 to 2026-08.
Signals use data through the previous day only (delay 1), costs are 10 bps per side, and every
look at the test window is counted.

**Real-data result (Binance daily, 34 coins, test 2025-01 to 2026-08): test rank IC 0.082, but it
is mostly a low-beta tilt in a market where the coin basket fell 66%, and its returns are
indistinguishable from zero before and after costs.** The GP, an equal-budget random search and a one-line range factor all score
about 0.08. After removing beta and size, IC 0.049 remains, and it comes from down days: the known
low-volatility effect. Sources: `results/binance_*.json` (diagnostics post-hoc); survivorship-biased

![Cumulative net return on the test period: the GP pick, the random-search pick, the simple control and 20-day momentum all end between +0.04 and +0.10 summed over 607 days, while 1-day reversal loses 0.93](docs/figures/cumulative_net.png)

```sh
export PYTHONPATH=src
python3.11 -m alpha_gp_lab demo --out runs/demo    # SYNTHETIC regime-change demo, ~10 s
bash scripts/check.sh                              # tests, demo, replays; real-data steps if data/ exists
```

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

**Controls and inference** (`gp.random_search`, `src/alpha_gp_lab/stats.py`). Random search draws
the GP's budget of random expressions from its own generator and filter, with no breeding, and
goes through the same validation rule and single test score. For the real-data pick, Newey-West
t-statistics, a circular block bootstrap over days, Bonferroni over every look at the test window,
and post-hoc diagnostics (IC by market state, constant-ranking controls, IC after cross-sectional
residualisation on trailing beta and size, per-coin P&L contributions) sit beside the i.i.d.
t-statistic the evaluator prints.

**Opt-in rules for future windows.** `gp.unit_check` rejects any candidate whose per-coin unit
does not cancel (`grammar.coin_units`: a price is USDT per coin, a volume is coins, so ranking raw
`close` or `volume` across coins ranks by the arbitrary size of a coin unit). `selection.min_coverage`
refuses a candidate whose IC is defined on less than that share of validation days. Both are off
in every committed config so the recorded runs replay; a result under them needs data after
2026-08-31.

**LLM seed proposer** (`src/alpha_gp_lab/llm_seed.py`). A prompt containing a short research
brief and the grammar (never data) asks for N expressions. Responses are cached in a JSON
replay file keyed by the SHA-256 of the exact prompt. Every proposed line is parsed; invalid
lines are logged, rejected and counted (a line starting `- ` is rejected too, because a list
bullet and a minus sign cannot be told apart). The demo and the tests read only the replay file.
`seeds --live` makes one request to a free hosted open-weight model on OpenRouter, pinned as
`qwen/qwen3.8-27b:free` (weights: Hugging Face `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0; created on Hugging Face 2026-08-05,
listed by OpenRouter 2026-08-14; sources: [Hugging Face model API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B),
[OpenRouter model list](https://openrouter.ai/api/v1/models)). No Anthropic or OpenAI model is
used, by Oscar's decision. The request is temperature 0 with reasoning off and bounded output.
Any non-200 status (429 is the rate limit), error field, unfinished answer, empty content,
oversized body or response from another model is refused and nothing is saved; a refusal is not
retried, since the free tier allows 20 requests a minute and 50 a day without purchased credits.
An accepted answer is saved labelled **REAL LLM OUTPUT** with the response's model, provider,
response id and date. The key is read from `OPENROUTER_API_KEY` and never saved. A run with
`use_seeds: false` reads no LLM output at all. The one live attempt so far (through `claude -p`,
before the switch) failed before reaching a model (see Results). The only shipped replay entry is
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

### Design decisions and trade-offs

- **Parse, never `eval`.** LLM text and offspring go through the same `ast`-based parser, so a
  hostile line cannot run. The cost is a hand-maintained grammar: anything outside it is
  refused, not guessed.
- **Standard library only.** It runs on a bare Python 3.11 with nothing to install. The cost is
  speed: one real-data run takes about 90 s in pure Python, and nothing is vectorised.
- **Split roles enforced by construction.** Each evaluator only holds the panel up to its own
  split's end, so test bars cannot reach the search. The protection is against leakage, not
  against a regime that starts after validation (the synthetic demo shows that).
- **Pre-registered real-data configs.** The configs were committed before any real-data run, so
  thresholds could not be tuned on the results. The cost: guesses such as min IC 0.01 stay
  fixed even where they turn out weak (walk-forward fold 1).
- **Rank IC as fitness and selection score.** It is robust to fat-tailed returns and needs no
  position sizing. The cost: it ignores costs and dollar P&L, and on the real data it picked a
  signal whose validation net was negative.
- **A grammar without units, so far.** Any field may meet any operator, which keeps the search
  space simple but lets raw price and volume levels be compared across coins. In a 34-coin
  universe these act as size proxies: 6 of the 20 GP seed picks, 4 of the 20 random-search picks
  and walk-forward fold 3 rank partly by them
  ([command](docs/real-data-runs.md#picks-that-rank-raw-price-or-volume-levels)). The fix,
  `gp.unit_check`, exists but stays off for the spent window; switching it on is for the next one.
- **Random search shares the GP's filter and validation rule.** The control differs from the GP
  only in having no breeding, so a difference between them is the GP's contribution. The cost:
  it inherits the GP's generator, so it says nothing about a smarter non-evolutionary search.
- **Delay 1, close-to-close.** A signal never sees the bar it trades on. The cost: a day is
  skipped, which handicaps short-horizon reversal.
- **Append-only lineage and full recomputation in `verify`.** Every candidate and score can be
  replayed exactly. The costs: each run computes twice, and a bundle only verifies on the
  platform that wrote it.
- **Real data pinned by hash, never committed.** The repository stays small and redistributes no
  exchange data, while `verify-data` proves which files were used. The cost: readers download
  the data themselves.

## Results (real numbers with their source; synthetic clearly labelled)

Numbers in this README are rounded to 3-4 significant figures; the cited JSON files hold full
precision, and [docs/real-data-runs.md](docs/real-data-runs.md) repeats the main tables at full
precision with the run ledger and commit timeline. "net" is the mean hypothetical return per daily
interval of a unit-gross, dollar-neutral portfolio after the fee on turnover; "gross" is the same
before the fee; "sum" adds daily values without compounding; ICIR is the mean daily IC over its
standard deviation.

### Real data (Binance daily, 34 coins, 2020–2026)

**Survivorship bias: these 34 coins are USDT pairs still trading in 2026, chosen in 2026.** The
rule, recorded in `fixtures/binance_universe.json`: the orchestrating agent picked, on 2026-10-03,
large USDT spot pairs listed on Binance by 2020-01 and still trading; no numeric size threshold
was recorded, so "large" is a judgement, not a reproducible screen. KNCUSDT met the rule but its
download failed. Which delisted pairs a point-in-time rule would have added has not been checked
(that needs a new download). One venue, daily bars, no borrow or funding costs (see Limits).

*Data.* Binance spot 1d klines from data.binance.vision, 2,435 days per coin, 2020-01-01 to
2026-08-31, pinned in `fixtures/binance_universe.json` (symbol list, date range, SHA-256 of each
CSV). `PYTHONPATH=src python3.11 -m alpha_gp_lab verify-data` prints `"ok": true` and exits 0 on
the local copy. The CSVs are not in the repository.

*Pre-registered setup.* `fixtures/binance_daily_config.json` and
`fixtures/binance_walkforward_config.json` were committed in `3efa07d`, before any GP or baseline
was run on the real data, and were not changed afterwards:
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

**Main run** (no LLM seeds):
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

#### What the test IC is (POST-HOC diagnostics)

These were written after the test result, the follow-up analysis and a reviewer's probes had all
been seen; the test window is spent, so they explain the result and are not new evidence.
Command (exit 0, about 6 s):
`PYTHONPATH=src python3.11 scripts/diagnose_binance.py --config fixtures/binance_diagnostics_config.json --out results/binance_diagnostics.json`.
The script refuses to run unless its own portfolio loop reproduces the pick's committed gross and
net. Every value below is in `results/binance_diagnostics.json`.

**The market fell, and the IC came from the down days.** The equal-weighted 34-coin basket rose
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

**The residual is the known low-volatility effect.** Under the same beta-and-size neutralisation
(`neutralised_references_beta_and_size`):

| Reference ranking, neutralised on beta and size | Test IC (t; Newey-West t) | 95% interval | Up days (t) | Down days (t) |
|---|---:|---:|---:|---:|
| plain low volatility, `-ts_std(returns, 60)` | 0.0451 (5.01; 4.69) | 0.0239 to 0.0659 | 0.0145 (1.09) | 0.0751 (6.32) |
| the pick's 2024 ranking, frozen | 0.0197 (2.30; 2.40) | 0.00309 to 0.0364 | -0.0258 (-2.30) | 0.0643 (5.18) |

A textbook 60-day volatility ranking scores almost the same residual IC as the GP's pick, with the
same up/down pattern. Even a ranking frozen at the end of 2024 keeps 0.0197 after neutralisation,
so part of the residual is still a fixed tilt rather than daily information. It is a known effect
measured on one falling window, not evidence of a new or tradable signal.

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

#### Follow-up analyses (pre-registered after the main result)

Plan: `fixtures/binance_analysis_config.json` and `scripts/analyze_binance.py`, committed in
`c5e212e` at 18:15:39, before either was run on real data but **19 minutes after the main test
results were committed** (`2eeb94a`, 17:56:41). Its seeds, budget, bootstrap and decision rules
were fixed before it ran, but it was written knowing the pick and its test score; in particular
the range control was chosen because the pick looked like a range measure
([timeline](docs/real-data-runs.md#timeline-what-each-plan-could-have-seen)). Command (run once,
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
  period. A post-hoc description: the random picks are mostly raw one-day range ratios such as
  `low / high`, with a mean test turnover of 0.373 against 0.142 for the GP picks (means of
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

- The family that matters is every look at the 2025-01 to 2026-08 window, not the expressions the
  search tried on train and validation (those never saw test). Counted in
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
and the diagnostics above say what both capture on this window: mostly low beta in a falling
alt market.

**4. Figures** (`scripts/plot_binance.py`, run with matplotlib from the shared venv; the library
does not need it). Rolling 60-day mean test IC:

![Rolling 60-day test rank IC: the GP pick, the random-search pick and the simple control move together between about 0 and 0.15; momentum and reversal swing around zero](docs/figures/rolling_ic.png)

The cumulative net return on test, after 10 bps per side, is the figure at the top of this README
(`docs/figures/cumulative_net.png`).

#### Walk-forward

Four rolling yearly folds (two years train, one validation, one test), same settings:
`PYTHONPATH=src python3.11 -m alpha_gp_lab run --config fixtures/binance_walkforward_config.json --out runs/binance-wf`
→ `results/binance_walkforward.json` (full-precision table in
[docs/real-data-runs.md](docs/real-data-runs.md#walk-forward-full-precision)).

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
Limits). Folds 2 and 3 overlap the main test window; they are separate searches, not extra
evidence for the main pick.

#### LLM seeds: not run

**Live LLM seeds pending: CLI login expired on 2026-10-03.** The single permitted live call,
`PYTHONPATH=src python3.11 -m alpha_gp_lab seeds --live --config fixtures/binance_daily_config.json`,
exited 1 before reaching a model (0 input and 0 output tokens; `results/llm_live_attempt.json`).
No retry was made. The seeded config (`fixtures/binance_daily_seeded_config.json`, pre-registered
in `3efa07d`) has not been run, so there is no seeded-vs-ablation comparison on real data; the main
run is the ablation without seeds. The live path now uses the pinned open-weight model
`qwen/qwen3.8-27b:free` on OpenRouter, by Oscar's decision on 2026-10-05.

*Note added 2026-10-05, before any live call reached a model:* the `3efa07d` pre-registration
described the live seeds as coming from `claude -p`. The proposer model has changed to the pinned
open-weight model above; the seeded config's bytes (brief, n, splits, GP settings) are unchanged.

#### Every real-data run

Ten real-data commands so far, all listed with their code version and exit code in
[docs/real-data-runs.md](docs/real-data-runs.md#every-real-data-run): `verify-data`, one failed
live LLM call, three main runs and two walk-forward runs (repeats only added printed fields and
reproduced every earlier number), the follow-up analysis and two runs of the post-hoc diagnostics (the second added fields and
reproduced the first). One main
config, one walk-forward config and one main GP seed were pre-registered; no config, seed or
threshold was changed after a real-data result.

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
| validation mean_ic | 0.133 (ic_tstat 6.937, 58 intervals) |
| validation mean_turnover / mean_net | 1.341 / 0.001962 |
| **test mean_ic** | **-0.1877** (ic_tstat -11.44, 59 intervals) |
| **test mean_net** | **-0.004342** (mean_turnover 1.321) |
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
| validation mean_ic | 0.1413 (ic_tstat 5.774) |
| test mean_ic | -0.153 (ic_tstat -7.127) |
| test mean_net | -0.003722 |
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

### Tests and runtime

- `PYTHONPATH=src python3.11 -m unittest discover -s tests` prints `Ran 104 tests` and `OK`.
- `python3.11 tests/hand_cases.py` re-derives the evaluator's arithmetic in exact Fractions:
  for example IC 2/5, turnover 2, gross 1/20 and net 49/1000 on a four-asset case.
- On the development machine the demo printed `demo finished in 9.6s` to stderr on its last run, and
  `scripts/check.sh` (tests, hand cases, `scripts/demo.sh`, the other two synthetic runs above, two replays, then
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
bash scripts/demo.sh                                     # the demo plus its replay, in a temp folder
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

## Architecture

```
src/alpha_gp_lab/
  grammar.py       parser (ast, no eval), frozen trees, canonical form, industry templates, coin units
  evaluate.py      operator semantics, timing, rank IC, turnover, net, fitness, fingerprints
  gp.py            random trees, crossover, mutation, generations, validation selection, folds,
                   equal-budget random search
  stats.py         Newey-West t, circular block bootstrap, normal p-value, Bonferroni, OLS residuals
  data.py          Panel, SYNTHETIC regime generator, OHLCV CSV loader, pinned-universe check
  llm_seed.py      prompt, hash-keyed replay cache, grammar validation, optional OpenRouter call
  store.py         run bundle, append-only SQLite lineage, manifest, verify
  config.py        strict config validation, index or date splits, fold lists, walk-forward folds
  cli.py           demo | walkforward | run | verify | seeds | verify-data
scripts/           check.sh, demo.sh, fetch_binance_daily.py, analyze_binance.py,
                   diagnose_binance.py (post-hoc), plot_binance.py
.github/workflows/ ci.yml: runs check.sh on Python 3.11 (not yet run on GitHub)
fixtures/          configs (SYNTHETIC and Binance), binance_universe.json, LLM replay
results/           printed summaries of the real-data runs, the follow-up analysis, the post-hoc
                   diagnostics, the failed live-LLM attempt
docs/              architecture.md, real-data-runs.md (ledger, timeline, full precision),
                   figures/ drawn from results/binance_analysis.json
tests/             unittest suite and hand_cases.py (Fractions, no evaluator import)
```

Diagrams of the data flow, the split roles and the lineage schema are in
[docs/architecture.md](docs/architecture.md).

## Limits

- **Survivorship bias and a judgement-based universe.** The 34 real-data coins were picked on
  2026-10-03 by the orchestrating agent as large USDT pairs listed by 2020-01 and still trading,
  with no recorded size threshold. Coins that were delisted or collapsed between 2020 and 2026
  are absent, and KNCUSDT is missing because its download failed. Every real-data number in this
  README is biased toward survivors; a point-in-time universe with delisted pairs needs a new
  download.
- **The test window is not an untouched holdout.** 2025-01-01 to 2026-08-31 was scored by the main
  run and then again by this repo's follow-up analysis (20 GP and 20 random-search seeds, the
  range-factor control, the bootstrap and the comparisons above). Across the portfolio, the sibling
  repo asof-research ran its own pre-registered study on the same 34 pairs, splits, costs and
  baselines, committed shortly after this repo's real-data results. Counted together, the window
  has had 176 looks (see Results). The follow-up plan was committed 19 minutes after the main test
  result and the diagnostics after a reviewer's probes, so neither was blind to it. Repeated looks
  at one window weaken it as out-of-sample evidence, and the window is one draw of a market in
  which small alts fell 66% against majors.
- **One venue.** Binance spot prices and volumes only, with no cross-check against other
  exchanges.
- **Daily bars.** UTC close-to-close days. Delay 1 means a signal uses data through the close of
  day t-1 and is held from the close of day t to the close of day t+1, so anything faster than a
  day is invisible and the "1-day reversal" baseline skips a day before trading.
- **Cost model.** A linear fee on traded notional: 10 bps per side on the real data (a round trip
  costs 20 bps), 5 bps per side on the synthetic data. No spread, slippage, market impact, fee
  tiers or capacity limit.
- **No shorting constraints or funding costs.** The portfolio is dollar-neutral, so half of it is
  short spot coins. Shorting spot needs margin borrowing or perpetual futures; borrow rates,
  borrow availability and funding payments are not modelled. A borrow cost above 5.88% a year
  wipes out the pick's test net.
- **Single-name tails.** A 34-name equal-rank book is exposed to one coin's squeeze: ZEC's
  +1,357% in the short leg outweighed the whole test gross. Rank IC does not see this.
- **The test period is a single regime.** The main test window, 2025-01-01 to 2026-08-31, is one
  stretch of one market. The walk-forward adds four yearly test windows, still from one
  market's history.
- **Simple portfolio model.** Dollar-neutral, unit gross, daily rebalance, no risk model and no
  compounding. "net" is a hypothetical interval return, not a backtest.
- **Selection and multiple testing.** Validation picks the best of up to 16 hall-of-fame
  candidates, which the GP bred from hundreds of train-scored expressions; that selection used
  data the test never saw. The `run` output's t-statistics assume independent days; the follow-up
  adds Newey-West t-statistics, block-bootstrap intervals and a Bonferroni bound over every look at
  the test window, for the main pick only. No White Reality Check or SPA test has been run over
  those looks, and none of these corrections handles a single-window market tilt.
- **The GP adds little here.** On this data an equal-budget random search matches it on test IC,
  and a one-line range measure matches its pick. What remains is mostly a low-beta tilt; the main
  evidence is about that, not about genetic programming.
- **The grammar has no units in the recorded runs.** Raw price and volume levels can be ranked
  across coins, where they act as size proxies (6 of 20 GP seed picks, 4 of 20 random-search
  picks, walk-forward fold 3). Proposed fix, implemented but off: `gp.unit_check`, with
  `selection.min_coverage` for candidates defined on few days (walk-forward fold 1). Results under
  either must come from a window after 2026-08-31, which needs new data.
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
- The live LLM path is tested only against canned OpenRouter responses and a loopback HTTP
  server; no request has reached OpenRouter yet.

## What I learned

Lessons drawn from the real-data results above, confirmed by Oscar on 2026-10-03. They predate
the post-hoc diagnostics, which later located the gap between the pick's IC and its return mainly
in single-name tails (gross t 0.96 before any fee).

1. A strong rank IC is not a tradable return. The main pick's test IC was 0.08208763518369619
   (t-stat 7.106267209479283), yet its net was 8.049329508034882e-05 per day with a t-stat of
   0.2184396051637011 after 10 bps per side.

2. Selecting on IC-based fitness can choose a signal that lost money on validation (net
   -0.00041132983838806654 per day for the main pick). If net return is the goal, it has to be
   in the selection rule.

3. Without diversity pressure the GP converges on one family: 14 of the 16 hall-of-fame
   members in the main run were near-copies of the pick's daily-range ratio, so "16 validation
   candidates" overstates how many different ideas were tested.

4. A qualification rule needs a coverage check: walk-forward fold 1 selected a signal defined
   on 10 of 364 validation days.

5. The textbook controls did poorly under delay 1 and 10 bps per side: 1-day reversal lost
   money on test (net t-stat -4.090451699976516) and 20-day momentum had a test IC of
   -0.006158512090458223. Beating them is a low bar, not a result.

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
