"""Timing of the POST-HOC diagnostics' controls in scripts/diagnose_binance.py.

The frozen-ranking, low-beta and neutralisation controls must read only data their stated window
allows. Each input is recomputed on a panel perturbed from its cutoff on (must not move) and from
one row earlier (must move, so the check can see a one-row slide). A separate test pins the windows
``main`` passes in. Four lookahead mutants of these windows once left the whole suite green.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from helpers import ROOT, panel_for, perturbed, small_config
from alpha_gp_lab.config import folds
from alpha_gp_lab.evaluate import Evaluator
from alpha_gp_lab.grammar import parse
import test_analysis

spec = importlib.util.spec_from_file_location('diagnose_binance', ROOT / 'scripts' / 'diagnose_binance.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def series(panel):
    close, volume = panel.field('close'), panel.field('volume')
    labels = [[b / a - 1 for a, b in zip(p, q)] for p, q in zip(close, close[1:])]
    return close, volume, labels, [diag.mean(r) for r in labels]


class ControlWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel_for(small_config())
        cls.n = len(cls.panel.symbols)

    def moved(self, compute, cutoff):
        """(unchanged when bars from ``cutoff`` on move, changed when bars from ``cutoff - 1`` on move)."""
        base = compute(self.panel)
        return compute(perturbed(self.panel, cutoff)) == base, compute(perturbed(self.panel, cutoff - 1)) == base

    def test_frozen_ranking_reads_only_validation_signal_rows(self):
        vs, ve = 71, 95
        for delay in (1, 2):
            def compute(panel):
                rows = Evaluator(panel, delay, 0).signal(parse('ts_mean(returns, 3)'))
                return diag.frozen_ranking(rows, vs, ve, delay, self.n)
            with self.subTest(delay=delay):
                self.assertEqual(self.moved(compute, ve - delay), (True, False))

    def test_low_beta_reads_only_train(self):
        ts, te = 20, 70

        def compute(panel):
            _, _, labels, market = series(panel)
            return diag.train_betas(labels, market, ts, te, self.n)
        self.assertEqual(self.moved(compute, te + 1), (True, False))

    def test_trailing_beta_stops_where_the_signal_does(self):
        for delay in (1, 2):
            t = 100

            def compute(panel):
                _, _, labels, market = series(panel)
                return diag.trailing_beta(labels, market, t, delay, 30, self.n)
            with self.subTest(delay=delay):
                self.assertEqual(self.moved(compute, t - delay + 1), (True, False))

    def test_size_stops_where_the_signal_does(self):
        for delay in (1, 2):
            t = 100

            def compute(panel):
                close, volume, _, _ = series(panel)
                return diag.log_dollar_volume(close, volume, t, delay, 5, self.n)
            with self.subTest(delay=delay):
                self.assertEqual(self.moved(compute, t - delay + 1), (True, False))


class MainPassesTheStatedWindows(unittest.TestCase):
    """The functions above are only as good as the spans ``main`` hands them."""

    def test_main_uses_validation_train_and_trailing_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            proc, _ = test_analysis.analyse(tmp, seeds=2)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            config = json.loads((d / 'main.json').read_text())
            plan = dict(name='t', main_config='main.json', main_result='main_result.json', analysis_result='out.json',
                        market_state_signals=['gp_pick'], beta_window=20, size_window=5, newey_west_lags=3,
                        bootstrap=dict(block=5, reps=20, seed=1, level=0.9), test_window_looks={'a': 1},
                        neutralised_references={})
            (d / 'diag.json').write_text(json.dumps(plan))
            calls = {name: [] for name in ('frozen_ranking', 'train_betas', 'trailing_beta', 'log_dollar_volume')}

            def spy(name):
                real = getattr(diag, name)

                def wrapper(*args):
                    calls[name].append(args)
                    return real(*args)
                return mock.patch.object(diag, name, wrapper)
            with spy('frozen_ranking'), spy('train_betas'), spy('trailing_beta'), spy('log_dollar_volume'):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(diag.main(['--config', str(d / 'diag.json'), '--out', str(d / 'diag_out.json')]), 0)
        dates = panel_for(config).dates
        (ts, te), (vs, ve), (xs, xe) = (folds(config, dates)[0][k] for k in ('train', 'validation', 'test'))
        delay = config['evaluation']['delay']
        self.assertEqual([c[1:4] for c in calls['frozen_ranking']], [(vs, ve, delay)])
        self.assertEqual([c[2:4] for c in calls['train_betas']], [(ts, te)])
        self.assertEqual([c[2:5] for c in calls['trailing_beta']], [(t, delay, 20) for t in range(xs, xe)])
        self.assertEqual([c[2:5] for c in calls['log_dollar_volume']], [(t, delay, 5) for t in range(xs, xe)])


if __name__ == '__main__':
    unittest.main()
