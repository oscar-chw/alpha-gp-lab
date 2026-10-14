"""Reserved result names are refused before a small comparison does any work."""
import io
import json
import runpy
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import ROOT
from test_analysis import sh
from test_example_comparison import config_file


class ComparisonNames(unittest.TestCase):
    def test_control_collision_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = config_file(root)
            config = json.loads(path.read_text())
            config['baselines']['range_5d'] = '-returns'
            path.write_text(json.dumps(config))
            run = sh('scripts/example_comparison.py', '--config', str(path), '--out', str(root / 'out'))
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn('range_5d', run.stderr)
            self.assertIn('reserved', run.stderr)
            self.assertFalse((root / 'out').exists())

    def test_all_reserved_labels_are_rejected_before_loading_inputs(self):
        for name in ('range_5d', 'gp_pick', 'random_search_pick'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = config_file(root)
                config = json.loads(path.read_text())
                config['baselines'][name] = '-returns'
                path.write_text(json.dumps(config))
                main = runpy.run_path(str(ROOT / 'scripts' / 'example_comparison.py'))['main']
                load = Mock(side_effect=AssertionError('input generation started before name admission'))
                error = io.StringIO()
                with patch.dict(main.__globals__, load_inputs=load), redirect_stderr(error):
                    with self.assertRaises(SystemExit) as stopped:
                        main(['--config', str(path), '--out', str(root / 'out')])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(name, error.getvalue())
                self.assertIn('reserved', error.getvalue())
                load.assert_not_called()
                self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
