"""Genetic programming over factor expressions, with strict split roles.

* train      — fitness for tournaments, elitism, the duplicate/equivalence filter and the
               hall of fame. Every GP decision reads only an Evaluator built on train dates.
* validation — re-scores the hall of fame, applies the qualification rule and the
               correlation filter against already-selected alphas, and picks the final alpha.
* test       — scores the final pick, once. Nothing reads it back.

All randomness comes from one ``random.Random`` seeded from (config seed, fold), so a config
and a panel determine the report exactly.

The GA idea (population of expressions, crossover and mutation, settings search) is adapted
from public formulaic-alpha miners; no code from them is reused.
"""
import random

from .config import canonical, digest, folds
from .evaluate import Evaluator, score
from .grammar import (BINARY, FIELDS, GROUP, MIN_WINDOW, TS, UNARY, Node, canonical as canonical_form, depth,
                      industry_variants, parse, positions, replace_at, size)

# Operators that can replace one another in a point mutation without changing arity or parameters.
_SWAPPABLE = [FIELDS, UNARY + ('neg',), GROUP, TS, tuple(BINARY)]


def _window(rng, op, windows, current=None):
    allowed = [w for w in windows if w >= MIN_WINDOW.get(op, 1) and w != current]
    return rng.choice(allowed) if allowed else None


def random_tree(rng, max_depth, windows, full=False):
    """'Grow' (full=False) or 'full' random tree, as in ramped half-and-half initialisation."""
    if max_depth <= 1 or (not full and rng.random() < 0.3):
        return Node(rng.choice(FIELDS))
    kind = rng.choices(('unary', 'group', 'winsor', 'ts', 'corr', 'binary'), weights=(3, 1, 0.5, 4, 0.5, 2))[0]
    sub = lambda: random_tree(rng, max_depth - 1, windows, full)  # noqa: E731
    if kind == 'unary':
        return Node(rng.choice(UNARY + ('neg',)), (sub(),))
    if kind == 'group':
        return Node(rng.choice(GROUP), (sub(),))
    if kind == 'winsor':
        return Node('winsorize', (sub(),), std=rng.choice((2.0, 3.0, 4.0)))
    if kind == 'ts':
        op = rng.choice(TS)
        return Node(op, (sub(),), window=_window(rng, op, windows))
    if kind == 'corr':
        return Node('ts_corr', (sub(), sub()), window=_window(rng, 'ts_corr', windows))
    return Node(rng.choice(tuple(BINARY)), (sub(), sub()))


def crossover(a, b, rng):
    """Subtree crossover: a random subtree of ``a`` is replaced by a random subtree of ``b``."""
    path, _ = rng.choice(positions(a))
    _, donor = rng.choice(positions(b))
    return replace_at(a, path, donor)


def mutate(tree, rng, windows):
    """One mutation; returns (child, kind). Inapplicable kinds fall back to subtree mutation."""
    kind = rng.choice(('subtree', 'point', 'window', 'hoist', 'industry'))
    spots = positions(tree)
    if kind == 'point':
        options = [(p, n, cls) for p, n in spots for cls in _SWAPPABLE if n.op in cls and len(cls) > 1]
        if options:
            path, node, cls = rng.choice(options)
            op = rng.choice([o for o in cls if o != node.op])
            win = node.window
            if win is not None and win < MIN_WINDOW.get(op, 1):
                win = MIN_WINDOW[op]
            return replace_at(tree, path, Node(op, node.args, win, node.std)), kind
    if kind == 'window':
        options = [(p, n) for p, n in spots if n.window is not None and _window(rng, n.op, windows, n.window)]
        if options:
            path, node = rng.choice(options)
            return replace_at(tree, path, Node(node.op, node.args, _window(rng, node.op, windows, node.window))), kind
    if kind == 'hoist' and len(spots) > 1:
        return rng.choice(spots[1:])[1], kind
    if kind == 'industry':
        path, node = rng.choice(spots)
        return replace_at(tree, path, rng.choice(industry_variants(node))), kind
    path, _ = rng.choice(spots)
    return replace_at(tree, path, random_tree(rng, rng.randint(1, 3), windows)), 'subtree'


def _regimes(panel, start, end):
    # The interval at t is labelled by the return into t+1, generated under regimes[t+1].
    return sorted(set(panel.regimes[start + 1:end + 1])) if panel.regimes else None


def search(panel, split, config, seeds, fold, settings_sha, on_generation=None):
    gp, fit, sel, ev = config['gp'], config['fitness'], config['selection'], config['evaluation']
    (ts, te), (vs, ve), (xs, xe) = split['train'], split['validation'], split['test']
    rng = random.Random(f'{config["seed"]}:{fold}')
    pen = (fit['turnover_penalty'], fit['complexity_penalty'])
    train = Evaluator(panel.head(te + 1), ev['delay'], ev['fee_bps'])
    counts = dict(occurrences=0, rejected_limits=0, rejected_duplicate=0, rejected_equivalent=0,
                  rejected_degenerate=0, llm_seeds_admitted=0, population_shortfall=0)
    assessed = {}

    def assess(tree):
        key = str(tree)
        if key not in assessed:
            m = train.metrics(tree, ts, te)
            s = score(m, tree, *pen)
            assessed[key] = (m, s, train.fingerprint(tree, ts, te) if s is not None else None)
        return assessed[key]

    def admit(tree, taken):
        """The duplicate / semantic-equivalence filter, against the generation being built."""
        if depth(tree) > gp['max_depth'] or size(tree) > gp['max_nodes']:
            counts['rejected_limits'] += 1
            return None
        canon = canonical_form(tree)
        if canon in taken:
            counts['rejected_duplicate'] += 1
            return None
        m, s, fp = assess(tree)
        if s is None:
            counts['rejected_degenerate'] += 1
            return None
        if fp in taken:
            counts['rejected_equivalent'] += 1
            return None
        taken.update((canon, fp))
        return m, s, fp

    nodes, results, history, generations = [], [], [], []

    def add(pop, generation, tree, parents, operation, assessed_):
        m, s, fp = assessed_
        expression = str(tree)
        content = dict(fold=fold, generation=generation, index=len(pop), expression=expression,
                       parents=parents, operation=operation, settings_sha256=settings_sha)
        record = dict(id=digest(canonical(content))[:20], **content, depth=depth(tree), size=size(tree))
        member = dict(record=record, tree=tree, score=s, fp=fp)
        pop.append(member)
        nodes.append(record)
        results.append(dict(node=record['id'], split='train', metrics=m, score=s, fingerprint=fp[:16]))
        counts['occurrences'] += 1

    def emit(generation, pop, first_node, first_result):
        scores = [p['score'] for p in pop]
        best = max(pop, key=lambda p: (p['score'], -len(p['record']['expression']))) if pop else None
        generations.append(dict(generation=generation, size=len(pop),
                                best_train_score=best['score'] if best else None,
                                best_expression=best['record']['expression'] if best else None,
                                mean_train_score=sum(scores) / len(scores) if scores else None))
        if on_generation:
            on_generation(fold, generation, nodes[first_node:], results[first_result:])

    # Generation 0: admissible LLM seeds first, then ramped half-and-half random trees.
    pop, taken = [], set()
    for expr in seeds:
        tree = parse(expr)
        a = admit(tree, taken)
        if a:
            add(pop, 0, tree, [], 'llm_seed', a)
            counts['llm_seeds_admitted'] += 1
    lo, hi = gp['init_depth']
    tries = 0
    while len(pop) < gp['population'] and tries < gp['population'] * gp['max_attempts']:
        tries += 1
        tree = random_tree(rng, rng.randint(lo, hi), gp['windows'], full=rng.random() < 0.5)
        a = admit(tree, taken)
        if a:
            add(pop, 0, tree, [], 'random_init', a)
    counts['population_shortfall'] += gp['population'] - len(pop)
    emit(0, pop, 0, 0)
    history.extend(pop)

    def order(p):
        return (-p['score'], p['record']['expression'], p['record']['id'])

    def tournament(pop):
        return min(rng.sample(pop, min(gp['tournament'], len(pop))), key=order)

    for generation in range(1, gp['generations']):
        if not pop:
            break
        first_node, first_result = len(nodes), len(results)
        new, taken = [], set()
        for elite in sorted(pop, key=order)[:gp['elitism']]:
            add(new, generation, elite['tree'], [elite['record']['id']], 'elite', admit(elite['tree'], taken))
        tries = 0
        while len(new) < gp['population'] and tries < gp['population'] * gp['max_attempts']:
            tries += 1
            if rng.random() < gp['crossover_prob']:
                a, b = tournament(pop), tournament(pop)
                child, op, parents = crossover(a['tree'], b['tree'], rng), 'crossover', [a['record']['id'], b['record']['id']]
            else:
                a = tournament(pop)
                child, kind = mutate(a['tree'], rng, gp['windows'])
                op, parents = 'mutation:' + kind, [a['record']['id']]
            try:   # every offspring must survive the same parser as any input; only depth/length can fail
                child = parse(str(child), gp['max_depth'])
            except ValueError:
                counts['rejected_limits'] += 1
                continue
            verdict = admit(child, taken)
            if verdict:
                add(new, generation, child, parents, op, verdict)
        counts['population_shortfall'] += gp['population'] - len(new)
        pop = new
        emit(generation, pop, first_node, first_result)
        history.extend(pop)

    # Hall of fame: best train score per distinct fingerprint across every generation.
    best = {}
    for p in sorted(history, key=order):
        best.setdefault(p['fp'], p)
    hall = sorted(best.values(), key=order)[:gp['hall_of_fame']]

    report = dict(fold=fold, splits=split, split_dates={k: [panel.dates[a], panel.dates[b]] for k, (a, b) in split.items()},
                  generations=generations, counts=counts,
                  regimes={k: _regimes(panel, *split[k]) for k in ('train', 'validation', 'test')},
                  hall_of_fame=[p['record']['id'] for p in hall], shortlist=[], correlation_rejected=[],
                  selected_id=None, selected_expression=None, selected_origin=None, validation=None, test=None)
    val = Evaluator(panel.head(ve + 1), ev['delay'], ev['fee_bps'])
    qualified = []
    for p in hall:
        m = val.metrics(p['tree'], vs, ve)
        s = score(m, p['tree'], *pen)
        results.append(dict(node=p['record']['id'], split='validation', metrics=m, score=s))
        if s is not None and m['mean_ic'] >= sel['min_ic'] and m['mean_turnover'] <= sel['max_turnover']:
            qualified.append((s, p, m))
    qualified.sort(key=lambda x: (-x[0], x[1]['record']['expression'], x[1]['record']['id']))
    # Correlation filter: against alphas selected before this run (config existing_alphas) and
    # against those already shortlisted. Only the existing alphas can change the final pick.
    existing = [parse(e) for e in sel['existing_alphas']]
    shortlist = []
    for s, p, m in qualified:
        if len(shortlist) == sel['select_k']:
            break
        selected = existing + [q['tree'] for _, q, _ in shortlist]
        corrs = [val.correlation(p['tree'], q, vs, ve) for q in selected]
        worst = max((abs(c) for c in corrs if c is not None), default=0.0)
        entry = dict(id=p['record']['id'], expression=p['record']['expression'], validation_score=s,
                     validation=m, max_abs_corr_to_selected=worst)
        if worst > sel['max_corr']:
            report['correlation_rejected'].append(entry)
        else:
            shortlist.append((s, p, m))
            report['shortlist'].append(entry)
    tester = Evaluator(panel.head(xe + 1), ev['delay'], ev['fee_bps'])
    # Fixed control expressions, scored on the same splits, timing and costs as the GP's pick.
    # They are not candidates: they never enter the GP, the hall of fame or the selection.
    report['baselines'] = {name: dict(expression=str(parse(expr)), train=train.metrics(parse(expr), ts, te),
                                      validation=val.metrics(parse(expr), vs, ve), test=tester.metrics(parse(expr), xs, xe))
                           for name, expr in sorted(config.get('baselines', {}).items())}
    if shortlist:
        _, pick, m = shortlist[0]
        test = tester.metrics(pick['tree'], xs, xe, detail=True)
        results.append(dict(node=pick['record']['id'], split='test', metrics=test, score=score(test, pick['tree'], *pen)))
        origin = pick['record']
        by_id = {n['id']: n for n in nodes}
        while origin['operation'] == 'elite':   # follow elite copies back to where the expression was made
            origin = by_id[origin['parents'][0]]
        report.update(selected_id=pick['record']['id'], selected_expression=pick['record']['expression'],
                      selected_origin=dict(operation=origin['operation'], generation=origin['generation']),
                      validation=m, test=test)
    report['status'] = 'SELECTED' if shortlist else 'NO_QUALIFYING_CANDIDATE'
    report['unique_expressions'] = len({n['expression'] for n in nodes})
    report['nodes'], report['results'] = nodes, results
    return report


def compute_report(config, panel, llm_record, on_generation=None):
    """The whole experiment as a pure function of (config, panel, LLM record)."""
    settings_sha = digest(canonical(config))
    seeds = llm_record['accepted'] if config['llm']['use_seeds'] else []
    fold_reports = [search(panel, split, config, seeds, i, settings_sha, on_generation)
                    for i, split in enumerate(folds(config, panel.dates))]
    tested = [f['test']['mean_ic'] for f in fold_reports if f['test'] and f['test']['mean_ic'] is not None]
    summary = dict(folds=len(fold_reports), folds_selected=sum(f['status'] == 'SELECTED' for f in fold_reports),
                   mean_test_ic=sum(tested) / len(tested) if tested else None,
                   folds_test_ic_positive=sum(x > 0 for x in tested),
                   folds_test_ic_negative=sum(x < 0 for x in tested))
    return dict(schema_version=1, name=config['name'], mode='single' if isinstance(config.get('splits'), dict) else 'walk_forward',
                data=dict(label=panel.label, synthetic=panel.synthetic, symbols=len(panel.symbols),
                          dates=len(panel.dates), first_date=panel.dates[0], last_date=panel.dates[-1]),
                settings_sha256=settings_sha,
                llm=dict(used_as_seeds=config['llm']['use_seeds'], source=llm_record['source'],
                         prompt_sha256=llm_record['prompt_sha256'],
                         accepted=llm_record['accepted'], rejected=llm_record['rejected'],
                         duplicates=llm_record['duplicates']),
                folds=fold_reports, summary=summary)
