# Design decisions and trade-offs

Why the lab is built the way it is, and what each choice costs.

## Decisions

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
  universe these act as size proxies: 7 of the 20 GP seed picks, 4 of the 20 random-search picks
  and walk-forward folds 1 and 3 rank partly by them
  ([command](real-data-runs.md#picks-that-rank-raw-price-or-volume-levels)). The fix,
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
