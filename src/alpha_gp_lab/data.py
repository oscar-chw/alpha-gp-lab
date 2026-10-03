"""Data adapters: a SYNTHETIC multi-regime generator and a daily OHLCV CSV-directory loader.

Both produce a ``Panel``: complete rectangular daily bars, no missing cells. Missing or
malformed input is refused, never filled.
"""
import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
import math
from pathlib import Path
import random

BARS = ('open', 'high', 'low', 'close', 'volume')

# Loadings of next-day returns on two lagged, cross-sectionally z-scored features:
#   reversal  = yesterday's industry-neutral one-day return,
#   momentum  = the five-day return ending yesterday.
# 'adverse' is 'reversal' with the reversal sign flipped: a model that learned reversal loses.
REGIMES = {
    'reversal': {'reversal': -1.0, 'momentum': 0.3},
    'momentum': {'momentum': 1.0},
    'noise': {},
    'adverse': {'reversal': 1.0, 'momentum': 0.3},
}


@dataclass(frozen=True)
class Panel:
    symbols: tuple
    groups: tuple
    dates: tuple
    bars: dict          # field name -> list of rows (one row per date, one value per symbol)
    synthetic: bool
    label: str
    regimes: tuple = None   # per-date regime that generated that date's return (synthetic only)
    returns: list = field(init=False, repr=False, compare=False)
    group_index: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        n, t = len(self.symbols), len(self.dates)
        if n < 2 or len(set(self.symbols)) != n or len(self.groups) != n:
            raise ValueError('need at least two unique symbols with one group each')
        if t < 2 or list(self.dates) != sorted(set(self.dates)):
            raise ValueError('dates must be unique and increasing')
        if set(self.bars) != set(BARS):
            raise ValueError('bars must be exactly ' + ', '.join(BARS))
        for name in BARS:
            rows = self.bars[name]
            if len(rows) != t or any(len(r) != n for r in rows):
                raise ValueError(f'{name}: complete rectangular panel required')
        close = self.bars['close']
        object.__setattr__(self, 'returns', [None] + [[b / a - 1 for a, b in zip(p, q)] for p, q in zip(close, close[1:])])
        object.__setattr__(self, 'group_index', tuple(
            tuple(i for i, g in enumerate(self.groups) if g == name) for name in sorted(set(self.groups))))

    def field(self, name):
        return self.returns if name == 'returns' else self.bars[name]

    def head(self, n):
        """The first ``n`` dates only. Evaluators are built on heads, so later data is unreachable."""
        if not 2 <= n <= len(self.dates):
            raise ValueError('head length outside the panel')
        return Panel(self.symbols, self.groups, self.dates[:n], {k: v[:n] for k, v in self.bars.items()},
                     self.synthetic, self.label, self.regimes[:n] if self.regimes else None)

    def to_json(self):
        return dict(schema_version=1, synthetic=self.synthetic, label=self.label, symbols=list(self.symbols),
                    groups=list(self.groups), dates=list(self.dates), regimes=list(self.regimes) if self.regimes else None,
                    **{k: self.bars[k] for k in BARS})

    @classmethod
    def from_json(cls, obj):
        if not isinstance(obj, dict) or obj.get('schema_version') != 1:
            raise ValueError('unsupported panel schema')
        for name in BARS:
            for row in obj[name]:
                for v in row:
                    if type(v) not in (int, float) or not math.isfinite(v):
                        raise ValueError('panel values must be finite numbers')
        return cls(tuple(obj['symbols']), tuple(obj['groups']), tuple(obj['dates']),
                   {k: [[float(v) for v in row] for row in obj[k]] for k in BARS}, obj['synthetic'] is True,
                   obj['label'], tuple(obj['regimes']) if obj.get('regimes') else None)


def _zscore(row):
    m = math.fsum(row) / len(row)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in row) / len(row))
    return [(x - m) / sd if sd > 0 else 0.0 for x in row]


def synthetic_panel(seed, assets, industries, segments, start_date='2024-01-01', strength=0.0025):
    """SYNTHETIC daily bars whose cross-sectional return predictability follows a regime per segment.

    ``segments`` is a list of ``{"regime": name, "days": n}``. Return into date t+1 loads on
    features computed at t-1, so a delay-1 signal (data through t-1, held t -> t+1) can see it.
    """
    if not 2 <= industries <= assets:
        raise ValueError('need 2..assets industries')
    regimes = []
    for seg in segments:
        if seg.get('regime') not in REGIMES or type(seg.get('days')) is not int or seg['days'] < 1:
            raise ValueError('segment needs a known regime and a positive integer day count')
        regimes += [seg['regime']] * seg['days']
    rng = random.Random(seed)
    groups = [f'SYN_IND{a * industries // assets}' for a in range(assets)]
    members = {g: [a for a in range(assets) if groups[a] == g] for g in sorted(set(groups))}
    base_logv = [rng.gauss(13.0, 0.5) for _ in range(assets)]
    close = [[100.0 * math.exp(rng.gauss(0, 0.3)) for _ in range(assets)]]
    opens, highs, lows, vols, logv = [list(close[0])], [list(close[0])], [list(close[0])], [], list(base_logv)
    vols.append([math.exp(v) for v in logv])
    for t in range(1, len(regimes)):
        prev = close[-1]
        loadings = REGIMES[regimes[t]]
        alpha = [0.0] * assets
        if t >= 3:
            # Return into date t loads on data through t-2: exactly what a delay-1 signal
            # for the interval t-1 -> t may see, and nothing it may not.
            c1, c2 = close[-2], close[-3]
            r1 = [b / a - 1 for a, b in zip(c2, c1)]
            neutral = list(r1)
            for idx in members.values():
                m = math.fsum(r1[i] for i in idx) / len(idx)
                for i in idx: neutral[i] = r1[i] - m
            feats = {'reversal': _zscore(neutral)}
            if t >= 7:
                feats['momentum'] = _zscore([b / a - 1 for a, b in zip(close[-7], c1)])
            for name, beta in loadings.items():
                if name in feats:
                    alpha = [x + strength * beta * z for x, z in zip(alpha, feats[name])]
        market = rng.gauss(0, 0.01)
        ind = {g: rng.gauss(0, 0.008) for g in members}
        ret = [max(-0.5, market + ind[groups[a]] + alpha[a] + rng.gauss(0, 0.015)) for a in range(assets)]
        c = [p * (1 + r) for p, r in zip(prev, ret)]
        o = [p * math.exp(rng.gauss(0, 0.003)) for p in prev]
        highs.append([max(a, b) * (1 + abs(rng.gauss(0, 0.004))) for a, b in zip(o, c)])
        lows.append([min(a, b) * (1 - abs(rng.gauss(0, 0.004))) for a, b in zip(o, c)])
        logv = [b + 0.7 * (v - b) + rng.gauss(0, 0.2) + 8 * abs(r) for v, b, r in zip(logv, base_logv, ret)]
        close.append(c); opens.append(o); vols.append([math.exp(v) for v in logv])
    first = date.fromisoformat(start_date)
    dates = tuple((first + timedelta(days=i)).isoformat() for i in range(len(regimes)))
    return Panel(tuple(f'SYN_{a:03d}' for a in range(assets)), tuple(groups), dates,
                 dict(open=opens, high=highs, low=lows, close=close, volume=vols), True,
                 'SYNTHETIC', tuple(regimes))


def _read_symbol(path):
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or set(reader.fieldnames) != {'date', *BARS}:
            raise ValueError(f'{path.name}: header must be date,open,high,low,close,volume')
        rows = {}
        for line in reader:
            d = line['date']
            if date.fromisoformat(d).isoformat() != d:
                raise ValueError(f'{path.name}: ISO date required, got {d!r}')
            if rows and d <= max(rows):
                raise ValueError(f'{path.name}: dates must be strictly increasing')
            try:
                o, h, l, c, v = (float(line[k]) for k in BARS)
            except (TypeError, ValueError):
                raise ValueError(f'{path.name} {d}: non-numeric bar') from None
            if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0:
                raise ValueError(f'{path.name} {d}: prices must be finite and positive, volume >= 0')
            if h < max(o, c) or l > min(o, c):
                raise ValueError(f'{path.name} {d}: high/low inconsistent with open/close')
            rows[d] = (o, h, l, c, v)
    if not rows:
        raise ValueError(f'{path.name}: no rows')
    return rows


def load_csv_dir(path):
    """Load ``<SYMBOL>.csv`` files (date,open,high,low,close,volume) into a Panel.

    Dates are aligned to the intersection across symbols; how many rows each symbol lost is
    returned so the caller can report it. An optional ``groups.csv`` (symbol,group) assigns
    industries; without it every symbol is in one group, so group ops act cross-sectionally.
    """
    root = Path(path)
    files = sorted(p for p in root.glob('*.csv') if p.name != 'groups.csv')
    if len(files) < 2:
        raise ValueError('need at least two symbol CSV files')
    data = {p.stem: _read_symbol(p) for p in files}
    common = sorted(set.intersection(*(set(rows) for rows in data.values())))
    if len(common) < 2:
        raise ValueError('symbols share fewer than two dates')
    dropped = {s: len(rows) - len(common) for s, rows in data.items()}
    groups = {s: 'ALL' for s in data}
    if (root / 'groups.csv').exists():
        with open(root / 'groups.csv', newline='', encoding='utf-8') as fh:
            mapping = {r['symbol']: r['group'] for r in csv.DictReader(fh)}
        if set(mapping) != set(data):
            raise ValueError('groups.csv must name exactly the loaded symbols')
        groups = mapping
    symbols = sorted(data)
    bars = {k: [[data[s][d][i] for s in symbols] for d in common] for i, k in enumerate(BARS)}
    return Panel(tuple(symbols), tuple(groups[s] for s in symbols), tuple(common), bars, False,
                 'CSV:' + root.name), dropped
