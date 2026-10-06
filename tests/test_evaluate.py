import math
import random
import statistics
import unittest

from helpers import panel_for, perturbed, small_config
from alpha_gp_lab.data import Panel
from alpha_gp_lab.evaluate import Evaluator, score
from alpha_gp_lab.gp import random_tree
from alpha_gp_lab.grammar import Node, parse


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

    def test_turnover_pays_for_the_drift_back_to_target(self):
        """A ranking that never changes still trades: each day's returns move the book off its target.

        Charging only |w_t - w_{t-1}| priced this at zero and overstated net; the trade from the
        drifted book w_{t-1}(1 + r_{t-1}) back to w_t must be paid."""
        rng = random.Random(11)
        close = [[100.0] * 4]
        for _ in range(9):
            close.append([c * (1 + rng.uniform(-0.1, 0.1)) for c in close[-1]])
        bars = {k: [list(r) for r in close] for k in ('open', 'high', 'low', 'close')}
        bars['volume'] = [[1.0, 2.0, 3.0, 4.0] for _ in close]   # fixed ranking every day
        panel = Panel(('A', 'B', 'C', 'D'), ('g',) * 4, tuple(f'2024-01-{d:02d}' for d in range(1, 11)), bars, True, 'SYNTHETIC')
        m = Evaluator(panel, 1, 10).metrics(parse('volume'), 3, 8, detail=True)
        w = [-0.375, -0.125, 0.125, 0.375]
        want = [math.fsum(abs(a * (close[t][i] / close[t - 1][i] - 1)) for i, a in enumerate(w)) for t in range(3, 8)]
        self.assertGreater(min(want), 0.0)
        self.assertAlmostEqual(m['mean_turnover'], statistics.fmean(want), places=12)
        gross = [math.fsum(a * (close[t + 1][i] / close[t][i] - 1) for i, a in enumerate(w)) for t in range(3, 8)]
        for got, g, to in zip(m['net_series'], gross, want):
            self.assertAlmostEqual(got, g - 0.001 * to, places=12)

    def test_overflow_and_non_finite_signals_are_degenerate_not_fatal(self):
        """volume ** 32 is finite but (x - m) ** 2 overflows inside zscore, which raised OverflowError out of
        the whole search; volume ** 64 - volume ** 64 is nan everywhere, which ranks arbitrarily."""
        def power(k):
            t = parse('volume')
            for _ in range(k):
                t = Node('mul', (t, t))
            return t
        ev = Evaluator(self.panel, 1, 5)
        for tree in (Node('zscore', (power(5),)), Node('ts_corr', (power(5), parse('close')), window=5),
                     Node('sub', (power(6), power(6))), Node('rank', (power(6),))):
            with self.subTest(tree=str(tree)[:40]):
                m = ev.metrics(tree, 20, 70)
                self.assertIsNone(m['mean_ic'])
                self.assertEqual(m['abstentions'], 50)
                self.assertIsNone(score(m, tree, 0.02, 0.001))
        self.assertIsNotNone(ev.metrics(power(2), 20, 70)['mean_ic'])   # volume ** 4 is finite: still scored

    def test_t_statistics_and_net_sum_match_the_statistics_module(self):
        m = Evaluator(self.panel, 1, 5).metrics(parse('-returns'), 20, 70, detail=True)
        net, ic = m['net_series'], [x for x in m['ic_series'] if x is not None]
        self.assertEqual(len(net), 50)
        self.assertAlmostEqual(m['mean_net'], statistics.fmean(net), places=15)
        self.assertAlmostEqual(m['sum_net'], math.fsum(net), places=15)
        self.assertAlmostEqual(m['net_tstat'], statistics.fmean(net) / (statistics.stdev(net) / math.sqrt(len(net))), places=9)
        self.assertAlmostEqual(m['ic_tstat'], statistics.fmean(ic) / (statistics.stdev(ic) / math.sqrt(len(ic))), places=9)


if __name__ == '__main__':
    unittest.main()
