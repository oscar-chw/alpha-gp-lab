# Evidence: what was fixed when, and how to check it

Which plans were committed before which results, how commit hashes map after the re-attribution,
and where each kind of evidence lives. The results themselves are in [results.md](results.md).

## Pre-registration

- `fixtures/binance_daily_config.json` and `fixtures/binance_walkforward_config.json` were
  committed in `91823e6`, before any GP or baseline was run on the real data, and were not changed
  afterwards. What they fix is listed in [results.md](results.md#universe-data-and-pre-registered-setup).
- The seeded config `fixtures/binance_daily_seeded_config.json` was pre-registered in the same
  commit and has not been run; the proposer-model change of 2026-10-05 is recorded in
  [llm-seeds.md](llm-seeds.md#status-on-real-data-not-run).
- The follow-up plan (`fixtures/binance_analysis_config.json`, `scripts/analyze_binance.py`) was
  committed after the main test result; the diagnostics are POST-HOC. See the timeline below.

Ten real-data commands so far, all listed with their code version and exit code in
[real-data-runs.md](real-data-runs.md#every-real-data-run): `verify-data`, one failed
live LLM call, three main runs and two walk-forward runs (repeats only added printed fields and
reproduced every earlier number), the follow-up analysis and two runs of the post-hoc diagnostics
(the second added fields and reproduced the first). One main config, one walk-forward config and
one main GP seed were pre-registered; no pre-registered config, seed or threshold was changed after
a real-data result. The post-hoc diagnostics config was extended once after its first real-data
run (round 2: two neutralised reference rankings, 5 to 7 looks).

## Timeline: what each plan could have seen

`git log --date=iso` gives these commit times (local time, 2026-10-03):

| Commit | Time | What |
|---|---|---|
| `91823e6` | 17:37:30 | main and walk-forward configs pre-registered, before any real-data run |
| `fea7780` | 17:56:41 | main-run and walk-forward test results committed |
| `881147e` | 18:15:39 | follow-up plan (random search, seeds, bootstrap, range control) committed, 19 minutes **after** the main test result |
| `5f6480e` | 18:21:48 | follow-up results committed |
| `f93540f` | 18:53:44 | post-hoc diagnostics script committed, after a reviewer had probed the test window |
| `bf8af3c` | 18:54:13 | post-hoc diagnostics results committed |
| `14b5834` | 19:16:29 | two more post-hoc diagnostics (up/down t, neutralised reference rankings) committed before running, after a second review |

So the follow-up plan fixed its seeds, budget, decision rules and bootstrap before it ran, but it
was written knowing the main pick and its test score. In particular the range control was chosen
because the pick looked like a range measure. That choice can only hurt the GP's case, but the
follow-up is not blind to the test result, and the diagnostics are plainly post-hoc.

## Commit hashes

Commit hashes were re-issued on 2026-10-04 when commit authorship was re-attributed to Oscar's
GitHub account; trees and dates are unchanged, and the old→new mapping is in
[commit-hash-map.txt](commit-hash-map.txt). Every hash on these pages is a new one.

## How a result reaches the README

Every real-data number is quoted from a committed `results/*.json` file written by a command over
a committed config; the follow-up and diagnostic scripts refuse to run unless they reproduce the
committed main result, and `scripts/check.sh` re-runs the main config against it whenever the data
is present. Drawn in [DIAGRAMS.md](DIAGRAMS.md#5-how-a-result-reaches-the-readme).

## Data, model and CI provenance

- Data: the 34 CSVs are pinned by SHA-256 in `fixtures/binance_universe.json` and checked by
  `verify-data`; the universe rule and its survivorship bias are in
  [results.md](results.md#universe-data-and-pre-registered-setup).
- Model: the public API responses behind the pinned seed model are committed in
  [model-evidence/](model-evidence/README.md).
- CI: `.github/workflows/ci.yml` runs `scripts/check.sh` on Python 3.11 (passed on main at
  `9ecf2ce`, 2026-10-04; the real-data steps skip there).
