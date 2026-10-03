"""Date-based folds, fixed baselines, the pinned-universe check and the live-LLM subprocess seam."""
import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, inputs, perturbed, small_config
from alpha_gp_lab import llm_seed
from alpha_gp_lab.config import folds, validate
from alpha_gp_lab.data import check_universe
from alpha_gp_lab.evaluate import Evaluator
from alpha_gp_lab.gp import compute_report
from alpha_gp_lab.grammar import parse

SAMPLE = ROOT / 'fixtures' / 'ohlcv_sample'


def report_for(config):
    with tempfile.TemporaryDirectory() as tmp:
        _, panel, llm = inputs(config, tmp)
    return compute_report(config, panel, llm), panel


class DateFolds(unittest.TestCase):
    def setUp(self):
        self.config = small_config()   # synthetic dates start 2024-01-01, one per day, 120 days
        self.dates = tuple(f'2024-{m:02d}-{d:02d}' for m, n in ((1, 31), (2, 29), (3, 31), (4, 30)) for d in range(1, n + 1))[:120]

    def test_date_pairs_resolve_to_the_dates_inside_them(self):
        self.config['splits'] = {'train': ['2023-06-01', '2024-01-31'], 'validation': ['2024-02-01', '2024-02-29'],
                                 'test': ['2024-03-01', '2024-12-31']}
        validate(self.config)
        (fold,) = folds(self.config, self.dates)
        self.assertEqual(fold, {'train': [0, 30], 'validation': [31, 59], 'test': [60, 119]})

    def test_a_list_of_folds_runs_as_walk_forward_with_dates_reported(self):
        self.config['splits'] = [
            {'train': ['2024-01-01', '2024-02-15'], 'validation': ['2024-02-16', '2024-03-05'], 'test': ['2024-03-06', '2024-03-25']},
            {'train': ['2024-01-20', '2024-03-05'], 'validation': ['2024-03-06', '2024-03-25'], 'test': ['2024-03-26', '2024-04-29']}]
        report, _ = report_for(self.config)
        self.assertEqual(report['mode'], 'walk_forward')
        self.assertEqual([f['split_dates']['test'] for f in report['folds']],
                         [['2024-03-06', '2024-03-25'], ['2024-03-26', '2024-04-29']])

    def test_bad_date_folds_are_refused(self):
        bad = [{'train': ['2024-01-01', '2024-02-01'], 'validation': ['2024-01-15', '2024-02-20'], 'test': ['2024-03-01', '2024-03-20']},
               {'train': ['2024-01-01', '2024-02-01'], 'validation': [40, 50], 'test': [60, 70]},
               {'train': ['2024-01-01', '2024-2-1'], 'validation': ['2024-02-02', '2024-02-20'], 'test': ['2024-03-01', '2024-03-20']}]
        for split in bad:
            config = copy.deepcopy(self.config)
            config['splits'] = split
            with self.assertRaises(ValueError, msg=split):
                validate(config)
        self.config['splits'] = {'train': ['2024-01-01', '2024-02-01'], 'validation': ['2024-02-02', '2024-02-20'],
                                 'test': ['2024-06-01', '2024-06-20']}
        with self.assertRaises(ValueError):
            folds(validate(self.config), self.dates)


class Baselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.config['baselines'] = {'reversal_1d': '-returns', 'momentum_20d': 'close / ts_delay(close, 20)'}
        cls.report, cls.panel = report_for(cls.config)
        cls.fold = cls.report['folds'][0]

    def test_baselines_use_the_same_splits_timing_and_costs(self):
        ev = self.config['evaluation']
        for name, expr in self.config['baselines'].items():
            got = self.fold['baselines'][name]
            self.assertEqual(got['expression'], str(parse(expr)))
            for split in ('train', 'validation', 'test'):
                start, end = self.fold['splits'][split]
                want = Evaluator(self.panel.head(end + 1), ev['delay'], ev['fee_bps']).metrics(parse(expr), start, end)
                self.assertEqual(got[split], want, (name, split))
        self.assertNotEqual(self.fold['baselines']['reversal_1d']['test'], self.fold['baselines']['momentum_20d']['test'])

    def test_baselines_are_controls_not_candidates(self):
        plain = copy.deepcopy(self.config)
        del plain['baselines']
        other = report_for(plain)[0]['folds'][0]
        self.assertEqual(other['baselines'], {})
        # Node ids hash the config, so compare what the search did, not the ids.
        for key in ('selected_expression', 'validation', 'test', 'counts', 'generations'):
            self.assertEqual(other[key], self.fold[key], key)
        self.assertEqual([n['expression'] for n in other['nodes']], [n['expression'] for n in self.fold['nodes']])

    def test_test_data_reaches_only_the_test_scores(self):
        other = compute_report(self.config, perturbed(self.panel, self.fold['splits']['test'][0]),
                               self.report_llm())['folds'][0]
        for name in self.config['baselines']:
            self.assertEqual(other['baselines'][name]['validation'], self.fold['baselines'][name]['validation'])
            self.assertNotEqual(other['baselines'][name]['test'], self.fold['baselines'][name]['test'])

    def report_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            return inputs(self.config, tmp)[2]

    def test_bad_baseline_is_refused(self):
        config = copy.deepcopy(self.config)
        config['baselines'] = {'vwap': 'rank(vwap)'}
        with self.assertRaises(ValueError):
            validate(config)


class Universe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.data = self.tmp / 'data'
        self.data.mkdir()
        for name in ('SYN_AAA.csv', 'SYN_CCC.csv'):
            shutil.copy(SAMPLE / name, self.data / name)
        rows = (SAMPLE / 'SYN_AAA.csv').read_text().splitlines()[1:]
        self.universe = dict(name='u', symbols=['SYN_AAA', 'SYN_CCC'], first_date=rows[0].split(',')[0],
                             last_date=rows[-1].split(',')[0], rows_per_symbol=len(rows),
                             sha256={n: hashlib.sha256((SAMPLE / n).read_bytes()).hexdigest()
                                     for n in ('SYN_AAA.csv', 'SYN_CCC.csv')})

    def test_clean_directory_passes(self):
        self.assertEqual(check_universe(self.data, self.universe), [])

    def test_every_kind_of_drift_is_reported(self):
        with (self.data / 'SYN_AAA.csv').open('a') as fh:
            fh.write('\n')
        (self.data / 'SYN_CCC.csv').unlink()
        shutil.copy(SAMPLE / 'SYN_BBB.csv', self.data / 'SYN_BBB.csv')
        self.assertEqual(check_universe(self.data, self.universe),
                         ['missing SYN_CCC.csv', 'unexpected SYN_BBB.csv (the loader would read it)',
                          'SYN_AAA.csv: SHA-256 mismatch'])
        self.assertEqual(check_universe(self.tmp / 'absent', self.universe), [f'data directory {self.tmp / "absent"} not found'])

    def test_wrong_metadata_is_reported(self):
        self.universe['rows_per_symbol'] += 1
        self.assertEqual(len(check_universe(self.data, self.universe)), 2)

    def test_cli_exit_codes(self):
        path = self.tmp / 'universe.json'
        path.write_text(json.dumps(self.universe))
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        cmd = [sys.executable, '-m', 'alpha_gp_lab', 'verify-data', '--universe', str(path), '--data']
        ok = subprocess.run(cmd + [str(self.data)], capture_output=True, text=True, env=env)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(json.loads(ok.stdout)['ok'])
        bad = subprocess.run(cmd + [str(self.tmp / 'absent')], capture_output=True, text=True, env=env)
        self.assertEqual(bad.returncode, 1)
        self.assertFalse(json.loads(bad.stdout)['ok'])

    def test_pinned_binance_universe_is_consistent(self):
        u = json.loads((ROOT / 'fixtures' / 'binance_universe.json').read_text())
        self.assertEqual(len(u['symbols']), 34)
        self.assertEqual(set(u['sha256']), {s + '.csv' for s in u['symbols']})
        self.assertNotIn('KNCUSDT', u['symbols'])
        self.assertIn('survivor', u['survivorship_bias'])


FAKE_CLAUDE = '''#!{python}
import json, os, sys
if sys.argv[1:] == ['--version']:
    print('9.9.9 (Fake)'); sys.exit(0)
log = {{'args': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd_files': os.listdir('.')}}
open(os.environ['FAKE_CLAUDE_LOG'], 'w').write(json.dumps(log))
if os.environ.get('FAKE_CLAUDE_FAIL'):
    print(json.dumps({{'type': 'result', 'is_error': True, 'result': 'Not logged in', 'modelUsage': {{}}}})); sys.exit(1)
print(json.dumps({{'type': 'result', 'is_error': False, 'result': 'rank(close)\\n', 'modelUsage': {{'fake-model-1': {{}}}}}}))
'''


class ClaudeCli(unittest.TestCase):
    def call(self, **env):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / 'claude'
            exe.write_text(FAKE_CLAUDE.format(python=sys.executable))
            exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
            log = Path(tmp) / 'log.json'
            old = dict(os.environ)
            os.environ.update(PATH=tmp + os.pathsep + old['PATH'], FAKE_CLAUDE_LOG=str(log), **env)
            try:
                return llm_seed.claude_cli('PROMPT TEXT'), json.loads(log.read_text())
            finally:
                os.environ.clear()
                os.environ.update(old)

    def test_prompt_on_stdin_no_tools_empty_directory(self):
        (text, via), seen = self.call()
        self.assertEqual((text, via), ('rank(close)\n', 'claude -p (9.9.9 (Fake)), model fake-model-1'))
        self.assertEqual(seen['stdin'], 'PROMPT TEXT')
        self.assertEqual(seen['cwd_files'], [])
        self.assertEqual(seen['args'][seen['args'].index('--tools') + 1], '')
        self.assertNotIn('PROMPT TEXT', seen['args'])

    def test_a_failed_call_raises_with_the_cli_message(self):
        with self.assertRaisesRegex(RuntimeError, 'exit 1.*Not logged in'):
            self.call(FAKE_CLAUDE_FAIL='1')


if __name__ == '__main__':
    unittest.main()
