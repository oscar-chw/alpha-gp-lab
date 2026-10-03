# Architecture

## Data flow and split roles

```mermaid
flowchart LR
    subgraph inputs[Inputs]
        CFG[config JSON<br/>seed, splits or walk_forward,<br/>GP budget, penalties]
        SYN[data.synthetic_panel<br/>SYNTHETIC regimes]
        CSV[data.load_csv_dir<br/>daily OHLCV CSVs]
        BIN[scripts/fetch_binance_daily.py<br/>pinned by fixtures/binance_universe.json]
        LLM[llm_seed.propose<br/>replay file by default<br/>--live: claude -p]
    end
    BIN -. writes .-> CSV
    LLM -->|every line parsed;<br/>invalid lines logged + rejected| SEEDS[accepted seed expressions]
    SYN --> PANEL[Panel]
    CSV --> PANEL
    PANEL -->|head through train end| TRAIN[Evaluator: train]
    PANEL -->|head through validation end| VAL[Evaluator: validation]
    PANEL -->|head through test end| TEST[Evaluator: test]
    SEEDS --> GP
    CFG --> GP
    subgraph GP[gp.search]
        INIT[generation 0:<br/>seeds + ramped half-and-half] --> LOOP[tournament, elitism,<br/>subtree crossover, 5 mutations,<br/>depth/size limits,<br/>duplicate + equivalence filter]
        LOOP --> HOF[hall of fame<br/>best train score per fingerprint]
    end
    TRAIN -->|fitness = rank IC - turnover - complexity| GP
    HOF --> PICK[qualify, rank by validation fitness,<br/>correlation filter, shortlist]
    RS[gp.random_search<br/>equal budget, same filter,<br/>no breeding: the control] --> PICK
    VAL --> PICK
    PICK -->|final pick only| TEST
    TEST --> REPORT[report]
    GP --> STORE[(store: append-only SQLite<br/>+ hash manifest)]
    REPORT --> STORE
    STORE -->|verify: hashes, read-back,<br/>full recomputation| REPORT
```

Each `Evaluator` is constructed on `panel.head(end + 1)` for the span it scores, so a later
split's bars do not exist inside an earlier split's evaluator. The tests perturb every bar
after the validation span (and, separately, after the train span) and require the search and
selection to come out byte-identical.

## Modules

| Module | Responsibility |
|---|---|
| `grammar.py` | AST-based parser (no `eval`), frozen `Node` trees, canonical form for duplicate detection, the five industry-relative templates, the grammar text sent to the LLM |
| `evaluate.py` | Local operator semantics, the delay-1 timing contract, rank IC, rebalancing turnover, net return, fitness, signal fingerprints and pairwise signal correlation |
| `gp.py` | Random trees, crossover, mutation, the generation loop, hall of fame, validation selection with the correlation filter, fixed baseline expressions scored on the same splits, walk-forward folds, and `random_search`: the equal-budget control that shares the admission filter and the validation rule |
| `stats.py` | Newey-West t-statistic, circular block bootstrap of a mean, normal p-value, Bonferroni bound |
| `data.py` | `Panel`, the SYNTHETIC multi-regime generator, the CSV-directory loader, the pinned-universe check behind `verify-data` |
| `llm_seed.py` | Prompt, prompt-hash replay cache, grammar validation of every proposed line, optional live refresh through `claude -p` |
| `store.py` | Run bundle writer, append-only SQLite schema with triggers, completion manifest, `verify` replay |
| `config.py` | Strict config validation and fold construction (index or ISO-date spans; one fold, a listed set, or rolling) |
| `cli.py` | `demo`, `walkforward`, `run`, `verify`, `seeds`, `verify-data` |
| `scripts/analyze_binance.py` | The pre-registered follow-up analyses of the main real-data run (random search and GP over 20 seeds, inference, the simple control); refuses to run unless the main pick reproduces |
| `scripts/plot_binance.py` | The README figures from `results/binance_analysis.json`; the only file that needs matplotlib |

## Lineage record

```mermaid
erDiagram
    nodes ||--o{ edges : "child"
    nodes ||--o{ edges : "parent"
    nodes ||--o{ results : "scored on"
    selection }o--|| nodes : "selected_id"
    nodes {
        text id PK "sha256 of fold, generation, index, expression, parents, operation, settings"
        int fold
        int generation
        text record "full JSON"
    }
    edges {
        text child FK
        int ordinal
        text parent FK
    }
    results {
        text node FK
        text split "train | validation | test"
        text record "metrics, score"
    }
    selection {
        int fold PK
        text record "shortlist, rejections, pick"
    }
    llm_proposals {
        int ordinal PK
        text status "accepted | rejected"
        text expression
        text reason
    }
```

Every table has `BEFORE UPDATE` and `BEFORE DELETE` triggers that abort, plus a `BEFORE INSERT`
trigger that aborts when the key already exists (otherwise `INSERT OR REPLACE` would delete
and rewrite a row without firing the delete trigger). `verify` compares the stored schema
and trigger SQL with the expected DDL, so a renamed or emptied trigger is caught. Each generation
is committed before the next one starts, so an interrupted run leaves its finished
generations on disk and no `completion.json`; such a directory is refused for reuse.
