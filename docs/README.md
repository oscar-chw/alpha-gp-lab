# alpha-gp-lab docs

Start with the [project README](../README.md); every page below answers one question in more depth.

## Understand it

| Page | What it answers |
|---|---|
| [method.md](method.md) | How does the search work: grammar, timing, GP loop, split roles, controls, run bundle? |
| [architecture.md](architecture.md) | Which module does what, and what does the append-only lineage record hold? |
| [DIAGRAMS.md](DIAGRAMS.md) | All five diagrams, numbered: overview, split roles, GP loop, LLM seed path, result flow |
| [llm-seeds.md](llm-seeds.md) | How are LLM seeds requested, cached and refused, and why has none run on real data? |

## Check the evidence

| Page | What it answers |
|---|---|
| [results.md](results.md) | Every real-data result in full: main run, POST-HOC diagnostics, follow-up analyses, walk-forward |
| [synthetic-results.md](synthetic-results.md) | What do the SYNTHETIC demo, its no-seeds ablation and the four-regime walk-forward show? |
| [real-data-runs.md](real-data-runs.md) | Which real-data commands ran, with which code, and the main tables at full precision |
| [evidence.md](evidence.md) | What was committed before which result, and how do the commit hashes map? |
| [limits.md](limits.md) | What can these results not tell you? Every known limit |
| [model-evidence/README.md](model-evidence/README.md) | Where do the pinned seed model's facts come from? |
| [commit-hash-map.txt](commit-hash-map.txt) | Old to new commit hashes after the 2026-10-04 re-attribution |

## Reference

| Page | What it answers |
|---|---|
| [reproduce.md](reproduce.md) | How do I rerun everything, including the real-data download and the follow-up scripts? |
| [figures/](figures/) | The two real-data figures, drawn by `scripts/plot_binance.py` |
