"""The evaluator must reproduce the exact-Fraction hand calculations in hand_cases.py."""
import unittest

import helpers  # noqa: F401  (puts src on the path)
from alpha_gp_lab.data import Panel
from alpha_gp_lab.evaluate import Evaluator, ranks, score
from alpha_gp_lab.grammar import parse
from hand_cases import cases

HAND = cases()


def tiny_panel(close, groups=('A', 'A', 'B', 'B')):
    bars = {k: [list(map(float, r)) for r in close] for k in ('open', 'high', 'low', 'close')}
    bars['volume'] = [[1.0] * len(close[0]) for _ in close]
    return Panel(tuple(f'SYN_{i}' for i in range(len(close[0]))), groups,
                 tuple(f'2024-01-0{i + 1}' for i in range(len(close))), bars, True, 'SYNTHETIC')


class HandCases(unittest.TestCase):
    def setUp(self):
        self.panel = tiny_panel([[4, 3, 2, 1], [1, 2, 3, 4], [10, 10, 10, 10], [11, 14, 12, 13]])
        self.ev = Evaluator(self.panel, delay=1, fee_bps=5)

    def test_interval_metrics_match_fractions(self):
        m = self.ev.metrics(parse('close'), 2, 3, detail=True)
        self.assertEqual(m['intervals'], 1)
        self.assertAlmostEqual(m['mean_ic'], float(HAND['ic']), places=12)
        self.assertAlmostEqual(m['mean_turnover'], float(HAND['turnover']), places=12)
        self.assertAlmostEqual(m['mean_gross'], float(HAND['gross']), places=12)
        self.assertAlmostEqual(m['mean_net'], float(HAND['net']), places=12)

    def test_fitness_penalties_match_fractions(self):
        tree = parse('rank(ts_delta(close, 1))')
        metrics = dict(mean_ic=float(HAND['ic']), mean_turnover=float(HAND['turnover']))
        self.assertAlmostEqual(score(metrics, tree, 0.02, 0.001), float(HAND['score']), places=12)

    def test_average_tie_ranks(self):
        self.assertEqual(ranks([4, 4, 9, 1]), [float(x) for x in HAND['ties']])

    def test_time_series_and_group_operators(self):
        panel = tiny_panel([[3, 1, 10, 1], [1, 2, 3, 20], [3, 3, 1, 1]])
        ev = Evaluator(panel, delay=1, fee_bps=0)
        self.assertAlmostEqual(ev.signal(parse('ts_rank(close, 3)'))[2][0], float(HAND['ts_rank']))
        decay = ev.signal(parse('ts_decay_linear(close, 3)'))
        self.assertIsNone(decay[1])
        self.assertAlmostEqual(decay[2][1], float(HAND['decay']))
        neutral = tiny_panel([[1, 3, 10, 20], [1, 3, 10, 20]])
        row = Evaluator(neutral, 1, 0).signal(parse('group_neutralize(close, industry)'))[0]
        self.assertEqual(row, [float(x) for x in HAND['neutral']])


if __name__ == '__main__':
    unittest.main()
