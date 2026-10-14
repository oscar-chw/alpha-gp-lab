# Small GP-versus-random comparison

SYNTHETIC mechanics example; not evidence of a market edge.

Delay: 1 bars. Fee: 5 bps per side, charged on drifted-book turnover. No spread, slippage, borrow or funding.

Train breeds/ranks; validation qualifies/selects; test evaluates each selected pick.
No qualifying candidate means no pick or test score, never a zero return.

Winner verdict: WITHHELD because declared admitted-score budgets were incomplete.
Accounting includes GP elites; not equal compute. See budget_audit.json for every seed.

| Search | Seed | Status | Expression | Test rank IC | Net / interval | Train scores |
|---|---:|---|---|---:|---:|---:|
| gp | 20261003 | SELECTED | `(ts_corr(open, low, 20) * log(close))` | 0.06232897537245363 | 0.0016462579433186204 | 29 |
| gp | 20261004 | SELECTED | `ts_delta(-returns, 2)` | -0.15019762845849802 | -0.004093290894351737 | 34 |
| random | 20261003 | SELECTED | `(ts_mean(group_neutralize(high, industry), 3) - ((returns + close) / sign(low)))` | -0.3070842201276984 | -0.006173512626196336 | 24 |
| random | 20261004 | SELECTED | `ts_rank(low, 20)` | 0.23526062792851052 | 0.004461234016298119 | 24 |

GP minus random across selected seeds (original 2-SE rule; two seeds are a mechanics sample):

```json
{
  "mean_ic": {
    "difference": -0.008022530443428239,
    "gp_beats_random": null,
    "standard_error": 0.2912496744843806,
    "undecided": "incomplete admitted-score budgets; equal-budget winner verdict withheld"
  },
  "mean_net": {
    "difference": -0.0003673771705674498,
    "gp_beats_random": null,
    "standard_error": 0.006042355852853468,
    "undecided": "incomplete admitted-score budgets; equal-budget winner verdict withheld"
  }
}
```

Read audited_comparison.json for the canonical controls, daily series and effective inference.
comparison.json is the raw legacy analysis; its winner arithmetic is unadjusted.
Read primary/report.md and primary/state.sqlite for the primary search and lineage.
Inputs, selection thresholds and split spans are in config.json and analysis_plan.json.
Historical real-data conclusions remain in docs/results.md and results/binance_analysis.json.
