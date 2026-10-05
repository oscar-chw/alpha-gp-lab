# Limits in full

Every known limit of the method and the results. The README lists the six that matter most.

- **Survivorship bias and a judgement-based universe.** The 34 real-data coins were picked on
  2026-10-03 by the orchestrating agent as large USDT pairs listed by 2020-01 and still trading,
  with no recorded size threshold. Coins that were delisted or collapsed between 2020 and 2026
  are absent, and KNCUSDT is missing because its download failed. Every real-data number in this
  repo is biased toward survivors; a point-in-time universe with delisted pairs needs a new
  download.
- **The test window is not an untouched holdout.** 2025-01-01 to 2026-08-31 was scored by the main
  run and then again by this repo's follow-up analysis (20 GP and 20 random-search seeds, the
  range-factor control, the bootstrap and the comparisons in [results.md](results.md)). Across the portfolio, the sibling
  repo asof-research ran its own pre-registered study on the same 34 pairs, splits, costs and
  baselines, committed shortly after this repo's real-data results. Counted together, the window
  has had at least 176 looks (see [results.md](results.md#summary) for what is not counted). The follow-up plan was committed 19 minutes after the main test
  result and the diagnostics after a reviewer's probes, so neither was blind to it. Repeated looks
  at one window weaken it as out-of-sample evidence, and the window is one draw of a market in
  which the daily-rebalanced equal-weight 34-coin basket fell 65.8%.
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
  adds Newey-West t-statistics, block-bootstrap intervals and a Bonferroni bound over the 176 counted
  looks at the test window, for the main pick only. No White Reality Check or SPA test has been run over
  those looks, and none of these corrections handles a single-window market tilt.
- **The GP adds little here.** On this data an equal-budget random search matches it on test IC,
  and a one-line range measure matches its pick. What remains is mostly a fixed tilt (frozen ranking
  87%, constant low-beta ranking 81%; 60% survives beta and size neutralisation); the main
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
  low-order float bits. The recorded numbers were produced on one machine only.
- **Replay is not a signature.** `verify` detects accidental or casual alteration, but someone
  who controls the code and every file can forge a consistent bundle.
- The live LLM path is tested only against canned OpenRouter responses and a loopback HTTP
  server; no request has reached OpenRouter yet.
