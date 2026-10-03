"""The analysis and diagnostics scripts end to end, in subprocesses, on a small SYNTHETIC config."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, small_config

ENV = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')


def sh(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=ENV, cwd=ROOT)


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        config = small_config()
        config['llm']['use_seeds'] = False
        config['baselines'] = {'reversal_1d': '-returns'}
        (d / 'main.json').write_text(json.dumps(config))
        run = sh('-m', 'alpha_gp_lab', 'run', '--config', str(d / 'main.json'), '--out', str(d / 'run'))
        assert run.returncode == 0, run.stderr
        (d / 'main_result.json').write_text(run.stdout)
        cls.main = json.loads(run.stdout)['folds'][0]
        cls.plan = dict(name='t', main_config='main.json', main_result='main_result.json', primary_seed=config['seed'],
                        seeds=[config['seed'], config['seed'] + 1, config['seed'] + 2], random_search_budget=20,
                        control={'range_5d': '-ts_mean(((close - low) / close), 5)'},
                        interpretation_references={'low_vol': '-ts_std(returns, 5)'},
                        bootstrap=dict(block=5, reps=300, seed=1, level=0.9), newey_west_lags=[0, 3],
                        bonferroni_counts=['validation_candidates', 'unique_expressions'])
        (d / 'plan.json').write_text(json.dumps(cls.plan))
        cls.proc = sh('scripts/analyze_binance.py', '--config', str(d / 'plan.json'), '--out', str(d / 'out.json'), '--workers', '2')
        cls.out = json.loads((d / 'out.json').read_text()) if cls.proc.returncode == 0 else None

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_and_describes_the_committed_pick(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertEqual(self.out['table']['gp_pick']['expression'], self.main['selected'])
        self.assertEqual(self.out['table']['gp_pick']['test']['mean_ic'], self.main['test']['mean_ic'])
        self.assertEqual(self.out['table']['reversal_1d']['test']['mean_net'],
                         self.main['baselines']['reversal_1d']['test']['mean_net'])

    def test_trial_counts_and_seed_rows(self):
        self.assertEqual(self.out['trials']['gp'], dict(train=self.main['occurrences'], distinct=self.main['unique_expressions'],
                                                        validation=self.main['validation_candidates']))
        self.assertEqual(self.out['trials']['random_search']['train'], 20)
        self.assertEqual([(r['kind'], r['seed']) for r in self.out['per_seed']],
                         [('gp', s) for s in self.plan['seeds']] + [('random', s) for s in self.plan['seeds']])
        self.assertEqual(self.out['seeds']['gp']['seeds'], 3)

    def test_inference_is_consistent(self):
        days = self.main['test']['intervals']
        self.assertEqual(len(self.out['test_dates']), days)
        for name, inf in self.out['inference'].items():
            self.assertEqual(len(self.out['series'][name]['net']), days)
            for series in ('ic', 'net'):
                b = inf[series]['bootstrap']
                self.assertLessEqual(b['low'], inf[series]['mean'])
                self.assertGreaterEqual(b['high'], inf[series]['mean'])
        sig = self.out['significance']
        m = self.main['validation_candidates']
        self.assertEqual(sig['bonferroni_iid']['validation_candidates'], dict(m=m, p=min(1.0, sig['p_iid'] * m)))
        self.assertGreaterEqual(sig['newey_west']['3']['bonferroni']['unique_expressions']['p'],
                                sig['newey_west']['3']['p'])

    def test_a_different_committed_pick_is_refused(self):
        d = Path(self.tmp.name)
        bad = json.loads((d / 'main_result.json').read_text())
        bad['folds'][0]['selected'] = 'returns'
        (d / 'bad_result.json').write_text(json.dumps(bad))
        (d / 'bad_plan.json').write_text(json.dumps(dict(self.plan, main_result='bad_result.json')))
        proc = sh('scripts/analyze_binance.py', '--config', str(d / 'bad_plan.json'), '--out', str(d / 'bad.json'))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('does not reproduce', proc.stderr)
        self.assertFalse((d / 'bad.json').exists())


class Diagnostics(unittest.TestCase):
    """The post-hoc diagnostics read the analysis output; run them on the same SYNTHETIC inputs."""

    @classmethod
    def setUpClass(cls):
        Analysis.setUpClass()
        cls.a = Analysis
        d = Path(cls.a.tmp.name)
        cls.plan = dict(name='t', main_config='main.json', main_result='main_result.json', analysis_result='out.json',
                        market_state_signals=['gp_pick', 'range_5d'], beta_window=20, size_window=5, newey_west_lags=3,
                        bootstrap=dict(block=5, reps=200, seed=1, level=0.9), test_window_looks={'a': 3, 'b': 4})
        (d / 'diag.json').write_text(json.dumps(cls.plan))
        cls.proc = sh('scripts/diagnose_binance.py', '--config', str(d / 'diag.json'), '--out', str(d / 'diag_out.json'))
        cls.out = json.loads((d / 'diag_out.json').read_text()) if cls.proc.returncode == 0 else None

    @classmethod
    def tearDownClass(cls):
        cls.a.tearDownClass()

    def test_reproduces_the_pick_and_decomposes_it(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        main = self.a.main
        self.assertAlmostEqual(self.out['pick_ic']['mean'], main['test']['mean_ic'], places=12)
        self.assertAlmostEqual(self.out['pick_net']['mean'], main['test']['mean_net'], places=12)
        self.assertAlmostEqual(self.out['pick_gross']['mean'], main['test']['mean_gross'], places=12)
        ms = self.out['market_state']['gp_pick']
        self.assertEqual(ms['up_days'] + ms['down_days'], main['test']['valid_ic_intervals'])
        weighted = (ms['mean_ic_up'] * ms['up_days'] + ms['mean_ic_down'] * ms['down_days']) / (ms['up_days'] + ms['down_days'])
        self.assertAlmostEqual(weighted, main['test']['mean_ic'], places=12)

    def test_legs_add_up_to_the_gross(self):
        coins = self.out['legs']['all']
        self.assertAlmostEqual(sum(c['contribution_to_mean_gross'] for c in coins), self.out['pick_gross']['mean'], places=12)
        self.assertAlmostEqual(sum(c['mean_weight'] for c in coins), 0.0, places=12)
        worst = min(coins, key=lambda c: c['contribution_to_mean_gross'])['symbol']
        self.assertEqual(self.out['without_largest_negative_contributor']['excluded'], worst)

    def test_family_count_and_break_even(self):
        looks = self.out['test_window_looks']
        self.assertEqual(looks['total'], 7)
        self.assertAlmostEqual(self.out['short_borrow_break_even_per_year'], self.a.main['test']['mean_net'] * 2 * 365, places=12)
        for k in ('beta', 'size', 'beta_and_size'):
            self.assertEqual(self.out['neutralised_ic'][k]['ic']['days'], self.a.main['test']['intervals'])

    def test_a_different_committed_gross_is_refused(self):
        d = Path(self.a.tmp.name)
        bad = json.loads((d / 'main_result.json').read_text())
        bad['folds'][0]['test']['mean_gross'] *= 1.01
        (d / 'bad_main.json').write_text(json.dumps(bad))
        (d / 'bad_diag.json').write_text(json.dumps(dict(self.plan, main_result='bad_main.json')))
        proc = sh('scripts/diagnose_binance.py', '--config', str(d / 'bad_diag.json'), '--out', str(d / 'x.json'))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('does not reproduce', proc.stderr)


if __name__ == '__main__':
    unittest.main()
