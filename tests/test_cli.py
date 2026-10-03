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
