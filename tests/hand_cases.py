"""Hand-calculated cases in exact Fractions. Imports nothing from alpha_gp_lab.

``test_hand_cases.py`` compares the evaluator against these numbers; running this file alone
re-checks the arithmetic by two independent routes and prints it.

Scenario: four assets, dates 0..3, closes
    row 0: [4, 3, 2, 1]      row 1: [1, 2, 3, 4]      row 2: [10, 10, 10, 10]      row 3: [11, 14, 12, 13]
signal expression ``close`` with delay 1; one interval, t = 2 (open at close[2], close at close[3]).
"""
from fractions import Fraction as F
import json


def average_ranks(values):
    # By definition: 1 + (number strictly below) + (number of other equal values) / 2.
    return [1 + sum(v < x for v in values) + F(sum(v == x for v in values) - 1, 2) for x in values]


def pearson(x, y):
    n = len(x)
    mx, my = F(sum(x), n), F(sum(y), n)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy, sxx, syy


def unit_gross_weights(signal):
    r = average_ranks(signal)
    m = F(sum(r), len(r))
    centred = [v - m for v in r]
    gross = sum(abs(v) for v in centred)
    return [v / gross for v in centred]


def cases():
    close = [[4, 3, 2, 1], [1, 2, 3, 4], [10, 10, 10, 10], [11, 14, 12, 13]]
    signal, previous = close[1], close[0]            # delay 1: interval t=2 uses row 1; t=1 used row 0
    labels = [F(b, a) - 1 for a, b in zip(close[2], close[3])]
    assert labels == [F(1, 10), F(4, 10), F(2, 10), F(3, 10)]

    sxy, sxx, syy = pearson(average_ranks(signal), average_ranks(labels))
    assert (sxy, sxx, syy) == (2, 5, 5)
    ic = sxy / sxx                                   # sqrt(sxx * syy) = 5 exactly here
    weights, prev_weights = unit_gross_weights(signal), unit_gross_weights(previous)
    target_change = sum(abs(a - b) for a, b in zip(weights, prev_weights))
    # The book held over interval t=1 has drifted with that interval's returns by close[2]; the
    # rebalance trades from there to the new target (P&L goes to cash, the book stays unit gross).
    prev_labels = [F(b, a) - 1 for a, b in zip(close[1], close[2])]   # interval t=1: close[1] -> close[2]
    held = [w * (1 + r) for w, r in zip(prev_weights, prev_labels)]
    turnover = sum(abs(a - b) for a, b in zip(weights, held))
    gross = sum(w * r for w, r in zip(weights, labels))
    # Second route to the same gross return: buy w/P shares at close[2], sell at close[3].
    shares = [w / p for w, p in zip(weights, close[2])]
    assert sum(q * (b - a) for q, a, b in zip(shares, close[2], close[3])) == gross
    # Second route to the same turnover: shares bought at close[1], against the target's shares, at close[2] prices.
    prev_shares = [w / p for w, p in zip(prev_weights, close[1])]
    assert sum(abs(q - h) * p for q, h, p in zip(shares, prev_shares, close[2])) == turnover
    fee = F(5, 10000)                                # 5 bps per unit of turnover
    net = gross - fee * turnover
    size = 3                                         # e.g. rank(ts_delta(close, 1)) has three nodes
    score = ic - F(2, 100) * turnover - F(1, 1000) * size

    ties = average_ranks([4, 4, 9, 1])
    ts_rank = F(sum(x < 3 for x in [3, 1, 3]) + F(sum(x == 3 for x in [3, 1, 3]) - 1, 2), 3 - 1)
    decay = F(sum((i + 1) * x for i, x in enumerate([1, 2, 3])), 1 + 2 + 3)
    group_rows = [1, 3, 10, 20]                      # groups A, A, B, B: subtract each group's mean
    members = [group_rows[:2]] * 2 + [group_rows[2:]] * 2
    neutral = [x - F(sum(g), len(g)) for x, g in zip(group_rows, members)]

    out = dict(labels=labels, ic=ic, weights=weights, prev_weights=prev_weights, held=held, turnover=turnover,
               gross=gross, net=net, score=score, ties=ties, ts_rank=ts_rank, decay=decay, neutral=neutral)
    assert ic == F(2, 5) and target_change == 2 and gross == F(1, 20)
    assert held == [F(15, 4), F(5, 8), F(-5, 12), F(-15, 16)] and turnover == F(323, 48) and net == F(4477, 96000)
    assert weights == [F(-3, 8), F(-1, 8), F(1, 8), F(3, 8)] and score == F(3149, 12000)
    assert ties == [F(5, 2), F(5, 2), 4, 1] and ts_rank == F(3, 4) and decay == F(7, 3)
    assert neutral == [-1, 1, -5, 5]
    return out


if __name__ == '__main__':
    print(json.dumps({k: [str(x) for x in v] if isinstance(v, list) else str(v) for k, v in cases().items()}, indent=2))
