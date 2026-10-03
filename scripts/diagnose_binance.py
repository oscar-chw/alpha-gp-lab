"""POST-HOC diagnostics of the main real-data pick (fixtures/binance_diagnostics_config.json).

Written after the test result was seen, to answer a reviewer: how much of the test IC is the
market moving, a fixed cross-sectional tilt, or exposure to beta and size, and why the P&L
disagrees with the IC. The test window is already spent, so none of this is new evidence.

    PYTHONPATH=src python3.11 scripts/diagnose_binance.py --config fixtures/binance_diagnostics_config.json \
        --out results/binance_diagnostics.json

Standard library only. Refuses to run unless its own portfolio loop reproduces the committed
gross and net of the pick, so the decomposition is of the same numbers the README reports.
"""
import argparse
import json
import math
from pathlib import Path
import sys

from alpha_gp_lab.cli import load_inputs
from alpha_gp_lab.config import folds, read_json, validate
from alpha_gp_lab.evaluate import Evaluator, ranks, spearman
from alpha_gp_lab.grammar import parse
from alpha_gp_lab.stats import block_bootstrap_ci, bonferroni, newey_west_tstat, residualise, two_sided_p


def mean(xs):
    return math.fsum(xs) / len(xs)


def tstat(xs):
    m, n = mean(xs), len(xs)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))
    return m / (sd / math.sqrt(n)) if sd > 0 else None


def corr(x, y):
    mx, my = mean(x), mean(y)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(math.fsum((a - mx) ** 2 for a in x) * math.fsum((b - my) ** 2 for b in y))


def compound(rets):
    total = 1.0
    for r in rets:
        total *= 1 + r
    return total - 1


def portfolio(rows, labels, start, end, delay, fee, keep):
    """The evaluator's book (centred signal ranks, unit gross, abstain on a constant row) over the
    coins in ``keep``: daily weights, gross and net for intervals t in [start, end)."""
    weights = {}
    for t in range(start - 1, end):
        s = rows[t - delay]
        w = None
        if s is not None:
            rk = ranks([s[i] for i in keep])
            if rk.count(rk[0]) != len(rk):
                m = mean(rk)
                centred = [v - m for v in rk]
                g = math.fsum(abs(v) for v in centred)
                w = [v / g for v in centred]
        weights[t] = w
    zeros = [0.0] * len(keep)
    gross, net, book = [], [], []
    for t in range(start, end):
        w, prev = weights[t] or zeros, weights[t - 1] or zeros
        g = math.fsum(a * labels[t][i] for a, i in zip(w, keep))
        gross.append(g)
        net.append(g - fee * math.fsum(abs(a - b) for a, b in zip(w, prev)))
        book.append(w)
    return gross, net, book


def summary(xs, plan):
    b = plan['bootstrap']
    return dict(mean=mean(xs), days=len(xs), tstat=tstat(xs), newey_west_tstat=newey_west_tstat(xs, plan['newey_west_lags']),
                bootstrap=block_bootstrap_ci(xs, b['block'], b['reps'], b['seed'], b['level']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    plan_path = Path(args.config)
    plan = read_json(plan_path.read_bytes())
    main_config = plan_path.parent / plan['main_config']
    committed = json.loads((plan_path.parent / plan['main_result']).read_text())['folds'][0]
    analysis = json.loads((plan_path.parent / plan['analysis_result']).read_text())
    _, panel, _ = load_inputs(main_config)
    config = validate(read_json(main_config.read_bytes()))
    split = folds(config, panel.dates)[0]
    (ts, te), (vs, ve), (xs, xe) = split['train'], split['validation'], split['test']
    delay, fee = config['evaluation']['delay'], config['evaluation']['fee_bps'] / 10000
    symbols, n = panel.symbols, len(panel.symbols)
    close, volume = panel.field('close'), panel.field('volume')
    labels = [[b / a - 1 for a, b in zip(p, q)] for p, q in zip(close, close[1:])]   # interval t: close[t] -> close[t+1]
    market = [mean(r) for r in labels]
    ev = Evaluator(panel.head(xe + 1), delay, config['evaluation']['fee_bps'])
    pick = committed['selected']
    rows = ev.signal(parse(pick))
    test = range(xs, xe)

    gross, net, book = portfolio(rows, labels, xs, xe, delay, fee, list(range(n)))
    if not (math.isclose(mean(gross), committed['test']['mean_gross'], rel_tol=1e-9)
            and math.isclose(mean(net), committed['test']['mean_net'], rel_tol=1e-9)):
        sys.exit('portfolio loop does not reproduce the committed gross and net of ' + pick)

    # 1. What the market did, per split (equal-weighted basket of all coins, compounded).
    basket = {name: compound(market[a:b]) for name, (a, b) in (('train', (ts, te)), ('validation', (vs, ve)), ('test', (xs, xe)))}

    # 2. IC by market state, for the pick and the signals that look like it.
    def daily_ic(r):
        return [spearman(r[t - delay], labels[t]) if r[t - delay] is not None else None for t in test]

    market_state = {}
    for name in plan['market_state_signals']:
        expr = analysis['table'][name]['expression']
        ic = daily_ic(ev.signal(parse(expr)))
        pairs = [(i, market[t]) for i, t in zip(ic, test) if i is not None]
        up, down = [i for i, m in pairs if m > 0], [i for i, m in pairs if m <= 0]
        market_state[name] = dict(expression=expr, corr_ic_market=corr([i for i, _ in pairs], [m for _, m in pairs]),
                                  up_days=len(up), mean_ic_up=mean(up), down_days=len(down), mean_ic_down=mean(down))

    # 3. Constant rankings: no daily information at all, only a fixed tilt chosen before test.
    def constant_ic(scores):
        return [spearman(scores, labels[t]) for t in test]

    val_ranks = [_unit_ranks(rows[t - delay]) for t in range(vs, ve) if rows[t - delay] is not None]
    frozen = [mean([r[i] for r in val_ranks]) for i in range(n)]
    beta_train = _betas(labels, market, range(ts, te), n)
    constant = dict(frozen_validation_rank=dict(rule="the pick's mean cross-sectional rank over validation, frozen",
                                                ic=summary(constant_ic(frozen), plan),
                                                scores={s: v for s, v in zip(symbols, frozen)}),
                    low_train_beta=dict(rule='minus each coin\'s beta to the equal-weighted market over train',
                                        ic=summary(constant_ic([-b for b in beta_train]), plan)))

    # 4. Neutralised IC: residualise the signal's ranks and the next-day returns, each day, on
    #    trailing beta and/or log dollar volume, using only data the signal could see (through t - delay).
    bw, sw = plan['beta_window'], plan['size_window']

    def trailing_beta(t):
        return _betas(labels, market, range(t - delay - bw, t - delay), n)   # intervals closing by close[t - delay]

    def log_dollar_volume(t):
        days = range(t - delay - sw + 1, t - delay + 1)
        return [math.log(mean([close[d][i] * volume[d][i] for d in days])) for i in range(n)]

    refs = {name: ev.signal(parse(expr)) for name, expr in sorted(plan['neutralised_references'].items())}
    neutral = {k: [] for k in ('beta', 'size', 'beta_and_size')}
    ref_neutral = {name: [] for name in list(refs) + ['frozen_validation_rank']}
    raw = []
    for t in test:
        s = rows[t - delay]
        rk = ranks(s)
        raw.append(spearman(s, labels[t]))
        beta, size = trailing_beta(t), log_dollar_volume(t)
        for k, cols in (('beta', [beta]), ('size', [size]), ('beta_and_size', [beta, size])):
            neutral[k].append(spearman(residualise(rk, cols), residualise(labels[t], cols)))
        # Reference rankings under the same beta-and-size neutralisation: is the residual a known effect?
        cols, y = [beta, size], residualise(labels[t], [beta, size])
        for name, r in refs.items():
            ref_neutral[name].append(spearman(residualise(ranks(r[t - delay]), cols), y))
        ref_neutral['frozen_validation_rank'].append(spearman(residualise(ranks(frozen), cols), y))
    up = [m > 0 for m in market[xs:xe]]

    def by_state(v):
        out = dict(ic=summary(v, plan))
        for state, keep in (('up', True), ('down', False)):
            xs_ = [x for x, u in zip(v, up) if u == keep]
            out[f'mean_ic_{state}'], out[f'tstat_{state}'], out[f'days_{state}'] = mean(xs_), tstat(xs_), len(xs_)
        return out

    neutralised = {k: by_state(v) for k, v in neutral.items()}
    neutralised_references = {name: dict(expression=plan['neutralised_references'].get(name, 'constant: ' + constant['frozen_validation_rank']['rule']),
                                         **by_state(v)) for name, v in ref_neutral.items()}

    # 5. Who is in the book, and who made or lost the money (gross, before costs).
    coins = []
    for i, sym in enumerate(symbols):
        w = [b[i] for b in book]
        coins.append(dict(symbol=sym, mean_weight=mean(w), days_long=sum(x > 0 for x in w), days_short=sum(x < 0 for x in w),
                          contribution_to_mean_gross=mean([b[i] * labels[t][i] for b, t in zip(book, test)]),
                          test_return=close[xe][i] / close[xs][i] - 1))
    coins.sort(key=lambda c: c['mean_weight'])
    worst = min(coins, key=lambda c: c['contribution_to_mean_gross'])
    keep = [i for i, s in enumerate(symbols) if s != worst['symbol']]
    g_ex, n_ex, _ = portfolio(rows, labels, xs, xe, delay, fee, keep)

    looks = sum(plan['test_window_looks'].values())
    p_nw = two_sided_p(newey_west_tstat(raw, plan['newey_west_lags']))
    report = dict(
        name=plan['name'], post_hoc=True, pick=pick, test_dates=[panel.dates[xs + 1], panel.dates[xe]],
        basket_return=basket,
        pick_ic=summary(raw, plan), pick_gross=summary(gross, plan), pick_net=summary(net, plan),
        icir={name: _icir(analysis['series'][name]['ic']) for name in analysis['table']},
        market_state=market_state, constant_rankings=constant, neutralised_ic=neutralised,
        neutralised_references_beta_and_size=neutralised_references,
        legs=dict(long=coins[::-1][:5], short=coins[:5], all=coins),
        largest_negative_contributor=worst['symbol'],
        without_largest_negative_contributor=dict(excluded=worst['symbol'], gross=summary(g_ex, plan), net=summary(n_ex, plan)),
        short_borrow_break_even_per_year=mean(net) / 0.5 * 365,   # the short leg is half of unit gross; crypto trades 365 days
        test_window_looks=dict(items=plan['test_window_looks'], total=looks,
                               pick_ic_bonferroni=dict(p_iid=bonferroni(two_sided_p(tstat(raw)), looks), p_newey_west=bonferroni(p_nw, looks)),
                               neutralised_ic_bonferroni={k: bonferroni(two_sided_p(v['ic']['newey_west_tstat']), looks)
                                                          for k, v in neutralised.items()}))
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('basket_return', 'market_state', 'neutralised_ic', 'neutralised_references_beta_and_size', 'largest_negative_contributor',
                                             'short_borrow_break_even_per_year', 'test_window_looks')}
                     | dict(constant_ic={k: v['ic']['mean'] for k, v in constant.items()},
                            pick_gross_t=report['pick_gross']['tstat']), indent=1, sort_keys=True))
    return 0


def _betas(labels, market, span, n):
    """Each coin's OLS beta to the equal-weighted market over the intervals in ``span``."""
    ms = [market[u] for u in span]
    mm = mean(ms)
    vm = math.fsum((m - mm) ** 2 for m in ms)
    out = []
    for i in range(n):
        ri = [labels[u][i] for u in span]
        mi = mean(ri)
        out.append(math.fsum((r - mi) * (m - mm) for r, m in zip(ri, ms)) / vm)
    return out


def _unit_ranks(row):
    return [(r - 1) / (len(row) - 1) for r in ranks(row)]


def _icir(ic):
    xs = [x for x in ic if x is not None]
    m = mean(xs)
    return m / math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


if __name__ == '__main__':
    sys.exit(main())
