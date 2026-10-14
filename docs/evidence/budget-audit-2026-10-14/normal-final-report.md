# Small GP-versus-random comparison

SYNTHETIC mechanics example; not evidence of a market edge.

Delay: 1 bars. Fee: 5 bps per side, charged on drifted-book turnover. No spread, slippage, borrow or funding.

Train breeds/ranks; validation qualifies/selects; test evaluates each selected pick.
No qualifying candidate means no pick or test score, never a zero return.

Admitted-score budgets: COMPLETE.
Accounting includes GP elites; not equal compute. See budget_audit.json for every seed.

| Search | Seed | Status | Expression | Test rank IC | Net / interval | Train scores |
|---|---:|---|---|---:|---:|---:|
| gp | 20261003 | SELECTED | `(ts_corr(open, low, 3) * abs(close))` | 0.18516266342353296 | 0.0040683613613591175 | 36 |
| gp | 20261004 | SELECTED | `ts_rank(ts_delta(group_zscore(-returns, industry), 2), 20)` | -0.15063983772715533 | -0.004142803090205392 | 36 |
| random | 20261003 | SELECTED | `(open - close)` | -0.4660991182730313 | -0.01013886270553494 | 36 |
| random | 20261004 | SELECTED | `ts_rank(low, 20)` | 0.23526062792851052 | 0.004461234016298119 | 36 |

GP minus random across selected seeds (original 2-SE rule; two seeds are a mechanics sample):

```json
{
  "mean_ic": {
    "difference": 0.1326806580204492,
    "gp_beats_random": false,
    "standard_error": 0.38880226766666026
  },
  "mean_net": {
    "difference": 0.002801593480195273,
    "gp_beats_random": false,
    "standard_error": 0.008375351424530154
  }
}
```

Read audited_comparison.json for the canonical controls, daily series and effective inference.
comparison.json is the raw legacy analysis; its winner arithmetic is unadjusted.
Read primary/report.md and primary/state.sqlite for the primary search and lineage.
Inputs, selection thresholds and split spans are in config.json and analysis_plan.json.
Historical real-data conclusions remain in docs/results.md and results/binance_analysis.json.
