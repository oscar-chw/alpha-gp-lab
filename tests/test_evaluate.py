import random
import unittest

from helpers import panel_for, perturbed, small_config
from alpha_gp_lab.evaluate import Evaluator, score
from alpha_gp_lab.gp import random_tree
from alpha_gp_lab.grammar import parse


class Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel_for(small_config())

    def test_no_operator_looks_ahead(self):
        """Changing bars from date k on must leave every signal row before k untouched."""
        k = 60
        base, moved = Evaluator(self.panel, 1, 0), Evaluator(perturbed(self.panel, k), 1, 0)
        rng = random.Random(3)
        exprs = ['ts_delta(close, 3)', 'ts_delay(volume, 2)', 'ts_corr(close, volume, 5)', 'ts_rank(returns, 4)',
                 'ts_decay_linear(open, 3)', 'ts_min(low, 3)', 'ts_max(high, 3)', 'ts_std(returns, 5)']
        trees = [parse(e) for e in exprs] + [random_tree(rng, rng.randint(2, 5), [1, 2, 3, 5]) for _ in range(150)]
        for tree in trees:
            a, b = base.signal(tree)[:k], moved.signal(tree)[:k]
            self.assertEqual(a, b, str(tree))

    def test_evaluator_refuses_spans_outside_its_view(self):
        view = Evaluator(self.panel.head(71), 1, 5)
        view.metrics(parse('-returns'), 20, 70)
        with self.assertRaises(ValueError):
            view.metrics(parse('-returns'), 20, 71)   # the label at t=70 needs close[71]: not in the view

    def test_constant_signal_abstains(self):
        m = Evaluator(self.panel, 1, 5).metrics(parse('sign(abs(close))'), 20, 70)
        self.assertIsNone(m['mean_ic'])
        self.assertEqual((m['abstentions'], m['mean_turnover'], m['mean_net']), (50, 0.0, 0.0))
        self.assertIsNone(score(m, parse('close'), 0.02, 0.001))

    def test_protected_division_and_delay(self):
        ev = Evaluator(self.panel, 2, 0)
        self.assertTrue(all(x == 0.0 for x in ev.signal(parse('(close / (close - close))'))[5]))
        # delay 2: the interval at t reads the signal row t-2, so a lag-0 signal of returns
        # equals a delay-1 evaluation of ts_delay(returns, 1).
        a = ev.metrics(parse('returns'), 20, 70)
        b = Evaluator(self.panel, 1, 0).metrics(parse('ts_delay(returns, 1)'), 20, 70)
        self.assertEqual(a['mean_ic'], b['mean_ic'])

    def test_penalties_lower_the_score(self):
        m = dict(mean_ic=0.1, mean_turnover=1.0)
        small, big = parse('close'), parse('rank(ts_mean(close, 5))')
        self.assertGreater(score(m, small, 0.02, 0.001), score(m, big, 0.02, 0.001))
        self.assertGreater(score(m, small, 0.0, 0.0), score(m, small, 0.02, 0.0))


if __name__ == '__main__':
    unittest.main()
