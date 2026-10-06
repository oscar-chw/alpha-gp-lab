"""Date-based folds, fixed baselines, the pinned-universe check and the live-LLM HTTP seam."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
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
        self.config['splits'] = {'train': ['2024-01-01', '2024-01-31'], 'validation': ['2024-02-01', '2024-02-29'],
                                 'test': ['2024-03-01', '2024-04-29']}
        validate(self.config)
        (fold,) = folds(self.config, self.dates)
        self.assertEqual(fold, {'train': [0, 30], 'validation': [31, 59], 'test': [60, 119]})
        # A span edge on a date the panel lacks resolves to the nearest panel date inside the span.
        gappy = tuple(d for d in self.dates if d not in ('2024-01-31', '2024-02-01'))
        (fold,) = folds(self.config, gappy)
        self.assertEqual(fold, {'train': [0, 29], 'validation': [30, 57], 'test': [58, 117]})

    def test_date_spans_outside_the_panel_are_refused_not_clamped(self):
        """A test end after the last date used to resolve to the last date: a shorter test window under the same name."""
        for name, span in (('train', ['2023-06-01', '2024-01-31']), ('test', ['2024-03-01', '2029-12-31']),
                           ('test', ['2024-03-01', '2024-04-30'])):
            config = copy.deepcopy(self.config)
            config['splits'] = {'train': ['2024-01-01', '2024-01-31'], 'validation': ['2024-02-01', '2024-02-29'],
                                'test': ['2024-03-01', '2024-04-29'], name: span}
            validate(config)
            with self.assertRaisesRegex(ValueError, f'{name} dates .* extend outside the panel', msg=span):
                folds(config, self.dates)

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


class _Handler(BaseHTTPRequestHandler):
    """A loopback stand-in for OpenRouter: each path selects one behaviour. Nothing leaves the machine."""
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        type(self).seen.append(dict(path=self.path, headers=dict(self.headers), body=body))
        if self.path == '/slow':
            time.sleep(1)
        status, payload, extra = {
            '/ok': (200, b'{"ok": true}', {}),
            '/limited': (429, b'{"error": {"message": "slow down"}}', {}),
            '/moved': (302, b'', {'Location': '/elsewhere'}),
            '/huge': (200, b'x' * (llm_seed.MAX_BODY_BYTES + 100), {}),
            '/slow': (200, b'late', {}),
        }[self.path]
        self.send_response(status)
        for k, v in extra.items():
            self.send_header(k, v)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        type(self).seen.append(dict(path=self.path, headers=dict(self.headers), body=b''))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class OpenRouterHttp(unittest.TestCase):
    """``llm_seed.http_post`` against a real socket on 127.0.0.1: the seam the unit fakes stand in for."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.seen.clear()

    def post(self, path, timeout=5):
        return llm_seed.http_post(self.base + path, {'Authorization': 'Bearer k', 'Content-Type': 'application/json'},
                                  b'{"q": 1}', timeout)

    def test_post_sends_headers_and_body_and_returns_status_and_body(self):
        self.assertEqual(self.post('/ok'), (200, b'{"ok": true}'))
        self.assertEqual(_Handler.seen[0]['body'], b'{"q": 1}')
        self.assertEqual(_Handler.seen[0]['headers']['Authorization'], 'Bearer k')

    def test_error_status_is_returned_not_raised(self):
        self.assertEqual(self.post('/limited'), (429, b'{"error": {"message": "slow down"}}'))

    def test_a_redirect_is_not_followed_so_the_key_goes_nowhere_else(self):
        self.assertEqual(self.post('/moved')[0], 302)
        self.assertEqual([s['path'] for s in _Handler.seen], ['/moved'])

    def test_the_body_read_is_capped(self):
        status, body = self.post('/huge')
        self.assertEqual((status, len(body)), (200, llm_seed.MAX_BODY_BYTES + 1))

    def test_the_timeout_applies(self):
        with self.assertRaises((TimeoutError, urllib.error.URLError)):
            self.post('/slow', timeout=0.2)


if __name__ == '__main__':
    unittest.main()
