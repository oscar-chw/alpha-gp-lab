"""Local operator semantics, split metrics and fitness.

Timing contract: the interval at date index t opens at close[t] and closes at close[t+1].
Its signal is the expression's row at t - delay (delay >= 1), so it uses data through t - 1
at the latest. An ``Evaluator`` is built on ``panel.head(end + 1)`` for the split it scores,
so dates after that split's last label do not exist inside it.
"""
from collections import OrderedDict
import hashlib
import math

from .grammar import BINARY, FIELDS, GROUP, TS, TS2, UNARY, size

DIV_EPS = 1e-12


def ranks(values):
    """1-based ranks; tied values share the average of the ranks they span."""
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    n = len(x)
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def _cs_rank(row):
    n = len(row)
    return [(r - 1) / (n - 1) for r in ranks(row)] if n > 1 else [0.5]


def _cs_zscore(row):
    m = math.fsum(row) / len(row)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in row) / len(row))
    return [(x - m) / sd if sd > 0 else 0.0 for x in row]


def _cs_winsorize(row, k):
    m = math.fsum(row) / len(row)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in row) / len(row))
    return [min(m + k * sd, max(m - k * sd, x)) for x in row]


def _neutralize(row):
    m = math.fsum(row) / len(row)
    return [x - m for x in row]


_ROW = {'rank': _cs_rank, 'zscore': _cs_zscore,
        'abs': lambda r: [abs(x) for x in r],
        'sign': lambda r: [(x > 0) - (x < 0) + 0.0 for x in r],
        'log': lambda r: [math.copysign(math.log1p(abs(x)), x) for x in r]}   # signed log1p: defined everywhere
_GROUP = {'group_rank': _cs_rank, 'group_zscore': _cs_zscore, 'group_neutralize': _neutralize}
_BIN = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b if abs(b) > DIV_EPS else 0.0}   # protected division, as usual in GP


def _std(w):
    m = math.fsum(w) / len(w)
    return math.sqrt(math.fsum((x - m) ** 2 for x in w) / len(w))


def _ts_rank(w):
    last = w[-1]
    return (sum(x < last for x in w) + (sum(x == last for x in w) - 1) / 2) / (len(w) - 1)


_WINDOW = {'ts_mean': lambda w: math.fsum(w) / len(w), 'ts_sum': math.fsum, 'ts_std': _std,
           'ts_rank': _ts_rank, 'ts_min': min, 'ts_max': max,
           'ts_decay_linear': lambda w: math.fsum((i + 1) * x for i, x in enumerate(w)) / (len(w) * (len(w) + 1) / 2)}


def _first(rows):
    return next((i for i, r in enumerate(rows) if r is not None), len(rows))


def _timeseries(rows, op, d):
    f, total = _first(rows), len(rows)
    lead = d if op in ('ts_delta', 'ts_delay') else d - 1
    if f + lead >= total:
        return [None] * total
    out = []
    for col in zip(*rows[f:]):
        if op == 'ts_delta':
            out.append([col[j] - col[j - d] for j in range(lead, len(col))])
        elif op == 'ts_delay':
            out.append([col[j - d] for j in range(lead, len(col))])
        else:
            fn = _WINDOW[op]
            out.append([fn(col[j - d + 1:j + 1]) for j in range(lead, len(col))])
    return [None] * (f + lead) + [list(r) for r in zip(*out)]


def _ts_corr(a, b, d):
    f, total = max(_first(a), _first(b)), len(a)
    if f + d - 1 >= total:
        return [None] * total
    out = []
    for ca, cb in zip(zip(*a[f:]), zip(*b[f:])):
        out.append([pearson(ca[j - d + 1:j + 1], cb[j - d + 1:j + 1]) or 0.0 for j in range(d - 1, len(ca))])
    return [None] * (f + d - 1) + [list(r) for r in zip(*out)]


class Evaluator:
    """Signals and metrics on one panel view, with an LRU cache of subtree signals.

    The cache only saves time; results are identical with any cache size.
    """

    def __init__(self, panel, delay, fee_bps, cache_size=512):
        if type(delay) is not int or delay < 1:
            raise ValueError('delay must be an integer >= 1')
        self.panel, self.delay, self.fee = panel, delay, fee_bps / 10000
        self._cache, self._cache_size = OrderedDict(), cache_size
        close = panel.bars['close']
        self._label_ranks = [ranks([b / a - 1 for a, b in zip(p, q)]) for p, q in zip(close, close[1:])]
        self._labels = [[b / a - 1 for a, b in zip(p, q)] for p, q in zip(close, close[1:])]

    def signal(self, node):
        if node.op in FIELDS:
            return self.panel.field(node.op)
        key = str(node)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        args = [self.signal(c) for c in node.args]
        op = node.op
        if op == 'neg':
            out = [None if r is None else [-x for x in r] for r in args[0]]
        elif op in UNARY:
            out = [None if r is None else _ROW[op](r) for r in args[0]]
        elif op == 'winsorize':
            out = [None if r is None else _cs_winsorize(r, node.std) for r in args[0]]
        elif op in GROUP:
            fn, idx = _GROUP[op], self.panel.group_index
            out = []
            for r in args[0]:
                if r is None:
                    out.append(None)
                    continue
                row = list(r)
                for members in idx:
                    for i, v in zip(members, fn([r[i] for i in members])):
                        row[i] = v
                out.append(row)
        elif op in BINARY:
            fn = _BIN[op]
            out = [None if x is None or y is None else [fn(u, v) for u, v in zip(x, y)] for x, y in zip(*args)]
        elif op in TS:
            out = _timeseries(args[0], op, node.window)
        elif op in TS2:
            out = _ts_corr(args[0], args[1], node.window)
        else:
            raise ValueError('unknown operator ' + op)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def _check(self, start, end):
        if not 0 <= start < end < len(self.panel.dates):
            raise ValueError('split outside this evaluator\'s view')

    def metrics(self, node, start, end, detail=False):
        """Mean rank IC, rebalancing turnover and net return over intervals t in [start, end)."""
        self._check(start, end)
        rows = self.signal(node)
        weights = {}
        for t in range(start - 1, end):
            s = rows[t - self.delay] if t - self.delay >= 0 else None
            rk = ranks(s) if s is not None else None
            if rk is None or rk.count(rk[0]) == len(rk):
                weights[t] = (None, None)   # warm-up or constant signal: abstain into cash
                continue
            m = math.fsum(rk) / len(rk)
            centred = [v - m for v in rk]
            gross = math.fsum(abs(v) for v in centred)
            weights[t] = ([v / gross for v in centred], rk)
        zeros = [0.0] * len(self.panel.symbols)
        ics, turnover, gross_ret, net = [], [], [], []
        for t in range(start, end):
            w, rk = weights[t]
            prev = weights[t - 1][0] or zeros
            ics.append(pearson(rk, self._label_ranks[t]) if rk else None)
            to = math.fsum(abs(a - b) for a, b in zip(w or zeros, prev))
            g = math.fsum(a * r for a, r in zip(w, self._labels[t])) if w else 0.0
            turnover.append(to); gross_ret.append(g); net.append(g - self.fee * to)
        valid = [x for x in ics if x is not None]
        n = len(turnover)
        mean_ic = math.fsum(valid) / len(valid) if valid else None
        out = dict(mean_ic=mean_ic, ic_tstat=_tstat(valid), mean_turnover=math.fsum(turnover) / n,
                   mean_gross=math.fsum(gross_ret) / n, mean_net=math.fsum(net) / n, net_tstat=_tstat(net),
                   sum_net=math.fsum(net),
                   intervals=n, valid_ic_intervals=len(valid), abstentions=sum(weights[t][0] is None for t in range(start, end)))
        if detail:
            out['ic_series'], out['net_series'] = ics, net
        return out

    def fingerprint(self, node, start, end):
        """Hash of the signal's cross-sectional ranks on every interval: equal hash = same
        positions and the same rank IC, so the two expressions are equivalent for this search."""
        self._check(start, end)
        rows, h = self.signal(node), hashlib.sha256()
        for t in range(start, end):
            s = rows[t - self.delay] if t - self.delay >= 0 else None
            h.update((repr(ranks(s)) if s is not None else 'None').encode() + b'|')
        return h.hexdigest()

    def correlation(self, a, b, start, end):
        """Mean cross-sectional rank correlation of two signals over the split; None if never defined."""
        self._check(start, end)
        ra, rb, vals = self.signal(a), self.signal(b), []
        for t in range(start, end):
            x, y = (r[t - self.delay] if t - self.delay >= 0 else None for r in (ra, rb))
            if x is not None and y is not None:
                c = spearman(x, y)
                if c is not None:
                    vals.append(c)
        return math.fsum(vals) / len(vals) if vals else None


def _tstat(xs):
    """Mean over its standard error (sample sd); None below two values or with zero spread."""
    if len(xs) < 2:
        return None
    m = math.fsum(xs) / len(xs)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m / (sd / math.sqrt(len(xs))) if sd > 0 else None


def score(metrics, node, turnover_penalty, complexity_penalty):
    """Fitness: mean rank IC minus a turnover penalty and a size (complexity) penalty."""
    if metrics['mean_ic'] is None:
        return None
    return metrics['mean_ic'] - turnover_penalty * metrics['mean_turnover'] - complexity_penalty * size(node)
