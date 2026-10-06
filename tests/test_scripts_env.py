"""check.sh and demo.sh run the interpreter the caller names: $PYTHON, else $PORTFOLIO_VENV/bin/python.

check.sh once ignored PORTFOLIO_VENV and ran whatever python3.11 was on PATH, so a gate run "in the
shared venv" tested a different interpreter. Each script is run here with a fake interpreter that
records the call and exits 7, so the first Python step stops the script (no recursion into the suite).
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from helpers import ROOT


def fake_python(folder, name):
    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "$@" >> "{folder}/{name.replace("/", "_")}.log"\nexit 7\n')
    path.chmod(0o755)
    return path


def run(script, **env):
    """Run a script with decoy python3.11 / python3 first on PATH (they exit 9), so a script that ignores the
    variables fails fast here instead of running the real suite, which would run this test again."""
    base = {k: v for k, v in os.environ.items() if k not in ('PYTHON', 'PORTFOLIO_VENV')}
    with tempfile.TemporaryDirectory() as decoys:
        for name in ('python3.11', 'python3'):
            (Path(decoys) / name).write_text('#!/bin/sh\nexit 9\n')
            (Path(decoys) / name).chmod(0o755)
        base['PATH'] = decoys + os.pathsep + '/usr/bin' + os.pathsep + '/bin'
        return subprocess.run(['bash', f'scripts/{script}'], cwd=ROOT, env=dict(base, **env), capture_output=True, text=True)


class Interpreter(unittest.TestCase):
    def test_portfolio_venv_is_used(self):
        for script in ('check.sh', 'demo.sh'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as venv:
                fake_python(venv, 'bin/python')
                proc = run(script, PORTFOLIO_VENV=venv)
                self.assertEqual(proc.returncode, 7, proc.stderr)
                self.assertTrue((Path(venv) / 'bin_python.log').read_text().strip())

    def test_python_overrides_portfolio_venv(self):
        with tempfile.TemporaryDirectory() as venv, tempfile.TemporaryDirectory() as other:
            fake_python(venv, 'bin/python')
            chosen = fake_python(other, 'py')
            proc = run('check.sh', PORTFOLIO_VENV=venv, PYTHON=str(chosen))
            self.assertEqual(proc.returncode, 7, proc.stderr)
            self.assertTrue((Path(other) / 'py.log').exists())
            self.assertFalse((Path(venv) / 'bin_python.log').exists())

    def test_without_either_variable_python3_11_on_path_is_used(self):
        self.assertEqual(run('demo.sh').returncode, 9)

    def test_a_named_venv_without_python_stops_loudly(self):
        with tempfile.TemporaryDirectory() as empty:
            proc = run('check.sh', PORTFOLIO_VENV=empty)
        self.assertEqual(proc.returncode, 3)
        self.assertIn('no Python interpreter', proc.stderr)


if __name__ == '__main__':
    unittest.main()
