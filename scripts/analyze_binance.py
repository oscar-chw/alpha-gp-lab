"""Pre-registered follow-up analyses of the main real-data run (fixtures/binance_analysis_config.json).

1. Equal-budget random search, the GP over the same seeds, and the decision rule between them.
2. Multiple-testing accounting: trial counts, Newey-West t, block-bootstrap intervals, Bonferroni.
3. One simple control for the interpretation, plus train/validation correlations with references.
4. The daily test series the figures are drawn from.

    PYTHONPATH=src python3.11 scripts/analyze_binance.py --config fixtures/binance_analysis_config.json \
        --out results/binance_analysis.json

Standard library only. Exits non-zero if the primary-seed GP rerun does not reproduce the
committed main result, so the analysis can never describe a different pick from the README's.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys

from alpha_gp_lab.cli import load_inputs
from alpha_gp_lab.config import canonical, digest, folds, read_json, validate
from alpha_gp_lab.evaluate import Evaluator
from alpha_gp_lab.gp import random_search, search
from alpha_gp_lab.grammar import parse
from alpha_gp_lab.stats import block_bootstrap_ci, bonferroni, newey_west_tstat, two_sided_p

KEEP = ('mean_ic', 'ic_tstat', 'mean_turnover', 'mean_gross', 'mean_net', 'net_tstat', 'sum_net', 'intervals',
        'valid_ic_intervals')
_STATE = {}


def _init(main_config):
    _, panel, _ = load_inputs(main_config)
    config = validate(read_json(Path(main_config).read_bytes()))
    _STATE.update(panel=panel, config=config, split=folds(config, panel.dates)[0])


def _search(task):
    """One search on the main config's split: ('gp', seed) or ('random', seed, budget)."""
    kind, seed = task[0], task[1]
    panel, split = _STATE['panel'], _STATE['split']
    config = copy.deepcopy(_STATE['config'])
    config['seed'] = seed
    if kind == 'random':
        out = random_search(panel, split, config, task[2], seed)
        trials = dict(train=out['counts']['occurrences'], distinct=out['counts']['occurrences'],
                      validation=out['validation_candidates'])
    else:
        out = search(panel, split, config, [], 0, digest(canonical(config)))
        trials = dict(train=out['counts']['occurrences'], distinct=out['unique_expressions'],
                      validation=sum(r['split'] == 'validation' for r in out['results']))
        out = dict(status=out['status'], selected=out['selected_expression'], validation=out['validation'], test=out['test'])
    return dict(kind=kind, seed=seed, status=out['status'], selected=out['selected'], trials=trials,
                validation={k: out['validation'][k] for k in KEEP} if out['validation'] else None,
                test=out['test'])


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def _inference(ic, net, plan):
    """Newey-West t and block-bootstrap interval for the mean of each daily series (IC days where defined)."""
    boot = plan['bootstrap']
    out = {}
    for name, xs in (('ic', [x for x in ic if x is not None]), ('net', net)):
        out[name] = dict(mean=math.fsum(xs) / len(xs), days=len(xs),
                         newey_west_tstat={str(L): newey_west_tstat(xs, L) for L in plan['newey_west_lags']},
                         bootstrap=block_bootstrap_ci(xs, boot['block'], boot['reps'], boot['seed'], boot['level']))
    return out


def _difference(a, b, plan):
    """Bootstrap interval of the mean daily difference a - b; IC on days where both are defined."""
    boot = plan['bootstrap']
    ic = [x - y for x, y in zip(a['ic_series'], b['ic_series']) if x is not None and y is not None]
    net = [x - y for x, y in zip(a['net_series'], b['net_series'])]
    return {name: dict(mean=math.fsum(xs) / len(xs), days=len(xs),
                       bootstrap=block_bootstrap_ci(xs, boot['block'], boot['reps'], boot['seed'], boot['level']))
            for name, xs in (('ic', ic), ('net', net))}


def _over_seeds(rows):
    """Per-metric summary of the seeds that selected a pick; None for a metric when no seed did."""
    picked = [r for r in rows if r['test']]
    out = dict(seeds=len(rows), no_qualifying_candidate=len(rows) - len(picked))
    for m in ('mean_ic', 'mean_net'):
        xs = [r['test'][m] for r in picked]
        out[m] = dict(mean=statistics.fmean(xs), sd=statistics.stdev(xs) if len(xs) > 1 else None,
                      min=min(xs), max=max(xs), positive=sum(x > 0 for x in xs), values=xs) if xs else None
    return out


def _welch(a, b):
    """Difference of means, its Welch standard error and the pre-registered 2-SE rule. With fewer than two
    picks on either side there is no standard error: the rule cannot be applied, and the reason is reported."""
    if a is None or b is None or a['sd'] is None or b['sd'] is None:
        return dict(difference=a['mean'] - b['mean'] if a and b else None, standard_error=None, gp_beats_random=None,
                    undecided='too few picks: GP {}, random search {}'.format(*(len(x['values']) if x else 0 for x in (a, b))))
    se = math.sqrt(a['sd'] ** 2 / len(a['values']) + b['sd'] ** 2 / len(b['values']))
    return dict(difference=a['mean'] - b['mean'], standard_error=se, gp_beats_random=a['mean'] - b['mean'] > 2 * se)


def _p(t):
    """Two-sided p of a t-statistic; None when the t-statistic is (zero spread, or under two days)."""
    return None if t is None else two_sided_p(t)


def _bonferroni(p, m):
    return None if p is None else bonferroni(p, m)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    plan_path = Path(args.config)
    plan = read_json(plan_path.read_bytes())
    main_config = plan_path.parent / plan['main_config']
    committed = json.loads((plan_path.parent / plan['main_result']).read_text())['folds'][0]
    _init(main_config)
    panel, config, split = _STATE['panel'], _STATE['config'], _STATE['split']
    (ts, te), (vs, ve), (xs, xe) = split['train'], split['validation'], split['test']

    tasks = [('gp', s) for s in plan['seeds']] + [('random', s, plan['random_search_budget']) for s in plan['seeds']]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(tasks))), initializer=_init,
                             initargs=(main_config,)) as pool:
        rows = list(pool.map(_search, tasks))
    gp_rows = [r for r in rows if r['kind'] == 'gp']
    rs_rows = [r for r in rows if r['kind'] == 'random']
    primary = plan['primary_seed']
    gp0 = next(r for r in gp_rows if r['seed'] == primary)
    rs0 = next(r for r in rs_rows if r['seed'] == primary)

    # The analysis must describe the committed pick, or nothing it says applies to the README.
    if gp0['selected'] != committed['selected'] or not all(_same(gp0['test'][k], committed['test'][k])
                                                             for k in ('mean_ic', 'mean_net')):
        sys.exit('primary-seed GP rerun does not reproduce ' + plan['main_result'])

    tester = Evaluator(panel.head(xe + 1), config['evaluation']['delay'], config['evaluation']['fee_bps'])
    series = {'gp_pick': gp0['test']}
    if rs0['test']:
        series['random_search_pick'] = rs0['test']
    for name, expr in sorted(plan['control'].items()):
        series[name] = tester.metrics(parse(expr), xs, xe, detail=True)
    for name, expr in sorted(config['baselines'].items()):
        series[name] = tester.metrics(parse(expr), xs, xe, detail=True)
        if not _same(series[name]['mean_ic'], committed['baselines'][name]['test']['mean_ic']):
            sys.exit(f'baseline {name} does not reproduce ' + plan['main_result'])

    inference = {n: _inference(s['ic_series'], s['net_series'], plan) for n, s in series.items()}
    p_iid = _p(gp0['test']['ic_tstat'])
    significance = dict(ic_tstat_iid=gp0['test']['ic_tstat'], p_iid=p_iid, newey_west={})
    counts = dict(validation_candidates=gp0['trials']['validation'], unique_expressions=gp0['trials']['distinct'])
    for lag, t in inference['gp_pick']['ic']['newey_west_tstat'].items():
        p = _p(t)
        significance['newey_west'][lag] = dict(t=t, p=p, bonferroni={c: dict(m=counts[c], p=_bonferroni(p, counts[c]))
                                                                      for c in plan['bonferroni_counts']})
    significance['bonferroni_iid'] = {c: dict(m=counts[c], p=_bonferroni(p_iid, counts[c])) for c in plan['bonferroni_counts']}

    gp_seeds, rs_seeds = _over_seeds(gp_rows), _over_seeds(rs_rows)
    seeds = dict(gp=gp_seeds, random_search=rs_seeds,
                 gp_minus_random={m: _welch(gp_seeds[m], rs_seeds[m]) for m in ('mean_ic', 'mean_net')})

    validation_ev = Evaluator(panel.head(ve + 1), config['evaluation']['delay'], config['evaluation']['fee_bps'])
    pick = parse(gp0['selected'])
    interpretation = {name: dict(expression=str(parse(expr)),
                                 train_corr=validation_ev.correlation(pick, parse(expr), ts, te),
                                 validation_corr=validation_ev.correlation(pick, parse(expr), vs, ve))
                      for name, expr in sorted(plan['interpretation_references'].items())}

    expressions = dict(gp_pick=gp0['selected'], random_search_pick=rs0['selected'], **plan['control'], **config['baselines'])
    table = {n: dict(expression=str(parse(expressions[n])), test={k: s[k] for k in KEEP},
                     validation={k: v for k, v in validation_ev.metrics(parse(expressions[n]), vs, ve).items() if k in KEEP})
             for n, s in series.items()}
    report = dict(
        name=plan['name'], main_config=plan['main_config'], primary_seed=primary,
        test_dates=[panel.dates[t + 1] for t in range(xs, xe)],   # interval t closes at close[t + 1]
        fee_bps_per_side=config['evaluation']['fee_bps'], table=table,
        trials=dict(gp=gp0['trials'], random_search=rs0['trials'], test=1),
        inference=inference, significance=significance,
        pick_minus={n: _difference(series['gp_pick'], s, plan) for n, s in series.items() if n != 'gp_pick'},
        seeds=seeds, per_seed=[{k: r[k] for k in ('kind', 'seed', 'status', 'selected', 'trials', 'validation')}
                               | dict(test={k: r['test'][k] for k in KEEP} if r['test'] else None) for r in rows],
        interpretation=interpretation,
        series={n: dict(ic=s['ic_series'], net=s['net_series']) for n, s in series.items()})
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(table={n: {k: r['test'][k] for k in ('mean_ic', 'mean_net')} | dict(expression=r['expression'])
                                 for n, r in report['table'].items()},
                          significance=significance, seeds={k: v for k, v in seeds.items() if k == 'gp_minus_random'},
                          interpretation=interpretation), indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
