"""The command line, run as a real subprocess on a small SYNTHETIC config."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, small_config, write_replay


def cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    return subprocess.run([sys.executable, '-m', 'alpha_gp_lab', *args], capture_output=True, text=True, env=env, cwd=ROOT)


class CommandLine(unittest.TestCase):
    def test_run_then_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = small_config()
            write_replay(tmp, config)
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            done = cli('run', '--config', str(path), '--out', str(Path(tmp) / 'out'))
            self.assertEqual(done.returncode, 0, done.stderr)
            printed = json.loads(done.stdout)
            self.assertEqual(printed['data'], 'SYNTHETIC')
            self.assertEqual(printed['llm']['source'], 'HAND-WRITTEN FIXTURE (not real LLM output)')
            self.assertEqual(printed['folds'][0]['status'], 'SELECTED')
            again = cli('verify', str(Path(tmp) / 'out'))
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(json.loads(again.stdout), printed)

    def test_ablation_needs_no_replay_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = small_config()
            config['llm'].update(use_seeds=False, replay='absent.json')
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            done = cli('run', '--config', str(path), '--out', str(Path(tmp) / 'out'))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)['llm'], dict(used_as_seeds=False, accepted=0, rejected=0,
                                                                   source='none: use_seeds is false, no LLM output read'))
            self.assertEqual(cli('verify', str(Path(tmp) / 'out')).returncode, 0)
            config['llm']['use_seeds'] = True
            path.write_text(json.dumps(config))
            seeded = cli('run', '--config', str(path), '--out', str(Path(tmp) / 'seeded'))
            self.assertNotEqual(seeded.returncode, 0)
            self.assertIn('no replay entry', seeded.stderr)

    def test_bad_config_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = small_config()
            config['provider'] = 'unexpected'
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            done = cli('run', '--config', str(path), '--out', str(Path(tmp) / 'out'))
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('unknown', done.stderr)
            self.assertFalse((Path(tmp) / 'out').exists())


if __name__ == '__main__':
    unittest.main()
