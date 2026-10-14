"""The small public comparison runs real searches and keeps evidence reproducible."""
import hashlib
import io
import json
import runpy
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import ROOT, small_config
from test_analysis import sh


def config_file(root, reject=False):
    config = small_config()
    config['llm']['use_seeds'] = False
    config['baselines'] = {'reversal_1d': '-returns'}
    if reject:
        config['selection'].update(min_ic=1.0, max_turnover=0.0)
    path = root / 'input.json'
    path.write_text(json.dumps(config))
    return path


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class ComparisonExample(unittest.TestCase):
    def test_actual_search_inputs_costs_and_seed_control_are_inspectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = config_file(root)
            outputs = []
            for name in ('first', 'second'):
                out = root / name
                run = sh('scripts/example_comparison.py', '--config', str(config), '--out', str(out))
                self.assertEqual(run.returncode, 0, run.stderr)
                outputs.append(json.loads((out / 'comparison.json').read_text()))
                manifest = json.loads((out / 'example_manifest.json').read_text())
                self.assertEqual(manifest['data_label'], 'SYNTHETIC')
                self.assertEqual(manifest['input_sha256'], hashlib.sha256(config.read_bytes()).hexdigest())
                self.assertEqual(manifest['delay'], 1)
                self.assertEqual(manifest['fee_bps_per_side'], 5)
                self.assertEqual(manifest['candidate_budget_per_search'], 36)
                self.assertEqual(manifest['seeds'], [20261003, 20261004])
                observed = hashes(out)
                self.assertEqual(manifest['files'], {k: v for k, v in observed.items() if k != 'example_manifest.json'})
                self.assertEqual(sh('-m', 'alpha_gp_lab', 'verify', str(out / 'primary')).returncode, 0)
                self.assertIn('not evidence of a market edge', (out / 'report.md').read_text())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(len(outputs[0]['per_seed']), 4)
            self.assertEqual(outputs[0]['fee_bps_per_side'], 5)
            old = hashes(root / 'first')
            collision = sh('scripts/example_comparison.py', '--config', str(config), '--out', str(root / 'first'))
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn('fresh output directory', collision.stderr)
            self.assertEqual(hashes(root / 'first'), old)

    def test_strict_rule_reports_rejection_and_no_gp_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = config_file(root, reject=True)
            run = sh('scripts/example_comparison.py', '--config', str(config), '--out', str(root / 'out'))
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root / 'out' / 'comparison.json').read_text())
            self.assertEqual(report['trials']['test'], 0)
            self.assertEqual(report['seeds']['gp']['no_qualifying_candidate'], 2)
            self.assertIn('NO_QUALIFYING_CANDIDATE', (root / 'out' / 'report.md').read_text())

    def test_example_refuses_live_seeds_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = config_file(root)
            config = json.loads(path.read_text())
            config['llm']['use_seeds'] = True
            path.write_text(json.dumps(config))
            run = sh('scripts/example_comparison.py', '--config', str(path), '--out', str(root / 'out'))
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('llm.use_seeds=false', run.stderr)
            self.assertFalse((root / 'out').exists())

    def test_retry_budget_is_refused_before_data_or_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = config_file(root)
            config = json.loads(path.read_text())
            config['gp'].update(population=4, generations=32, hall_of_fame=4, max_attempts=1000)
            path.write_text(json.dumps(config))
            main = runpy.run_path(str(ROOT / 'scripts' / 'example_comparison.py'))['main']
            load = Mock(side_effect=AssertionError('data/search started before resource admission'))
            error = io.StringIO()
            with patch.dict(main.__globals__, load_inputs=load), redirect_stderr(error):
                with self.assertRaises(SystemExit) as stopped:
                    main(['--config', str(path), '--out', str(root / 'out')])
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn('max_attempts<=10', error.getvalue())
            load.assert_not_called()
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
