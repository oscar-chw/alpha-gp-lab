"""Audit declared admitted-score budgets without changing underlying search statistics."""
import copy


def audit_budget(report, expected_seeds, budget):
    """Withhold the original winner rule if any predeclared search underfilled its budget.

    This accounts for admitted slots, including GP elite copies. It does not establish equal
    unique evaluations, compute time, validation effort or statistical power.
    """
    if (not isinstance(expected_seeds, list) or not expected_seeds
            or any(type(s) is not int or s < 0 for s in expected_seeds)
            or len(set(expected_seeds)) != len(expected_seeds)):
        raise ValueError('expected seeds must be distinct non-negative integers')
    if type(budget) is not int or budget <= 0:
        raise ValueError('budget must be a positive integer')
    if not isinstance(report, dict) or not isinstance(report.get('per_seed'), list):
        raise ValueError('per_seed rows are required')
    rows = {}
    for row in report['per_seed']:
        if not isinstance(row, dict):
            raise ValueError('per_seed row must be an object')
        kind, seed = row.get('kind'), row.get('seed')
        if type(seed) is not int or seed not in expected_seeds:
            raise ValueError(f'unexpected seed in per_seed row: {seed!r}')
        if kind not in ('gp', 'random'):
            raise ValueError(f'unexpected kind in per_seed row: {kind!r}')
        key = (kind, seed)
        if key in rows:
            raise ValueError(f'duplicate per_seed row: {key}')
        trials = row.get('trials')
        count = trials.get('train') if isinstance(trials, dict) else None
        if type(count) is not int or not 0 <= count <= budget:
            raise ValueError(f'train count outside declared budget for {key}: {count!r}')
        status = row.get('status')
        if status not in ('SELECTED', 'NO_QUALIFYING_CANDIDATE'):
            raise ValueError(f'invalid qualification status in per_seed row: {key}')
        if ((status == 'SELECTED' and not isinstance(row.get('test'), dict))
                or (status == 'NO_QUALIFYING_CANDIDATE' and row.get('test') is not None)):
            raise ValueError(f'qualification status and test disagree in per_seed row: {key}')
        rows[key] = row
    missing = [(kind, seed) for seed in expected_seeds for kind in ('gp', 'random') if (kind, seed) not in rows]
    if missing:
        raise ValueError(f'missing per_seed rows: {missing}')
    summary = report.get('seeds')
    raw = summary.get('gp_minus_random') if isinstance(summary, dict) else None
    if not isinstance(raw, dict) or any(not isinstance(raw.get(m), dict) for m in ('mean_ic', 'mean_net')):
        raise ValueError('gp_minus_random requires mean_ic and mean_net comparison objects')
    per_seed, reasons = [], []
    for seed in expected_seeds:
        gp, random = rows['gp', seed], rows['random', seed]
        g, r = gp['trials']['train'], random['trials']['train']
        per_seed.append(dict(seed=seed, gp=g, random=r, gp_status=gp['status'], random_status=random['status']))
        if g != budget or r != budget:
            reasons.append(dict(seed=seed, gp_admitted=g, random_admitted=r, required_each=budget))
    effective = copy.deepcopy(raw)
    if reasons:
        for metric in effective.values():
            metric['gp_beats_random'] = None
            metric['undecided'] = 'incomplete admitted-score budgets; equal-budget winner verdict withheld'
    return dict(schema='alpha-gp-budget-audit/v1', budget_basis='admitted training slots including GP elites; not equal compute',
                expected_seeds=list(expected_seeds), nominal_per_search=budget, per_seed=per_seed,
                eligible=not reasons, reasons=reasons, comparison=effective,
                unadjusted_comparison=copy.deepcopy(raw))
