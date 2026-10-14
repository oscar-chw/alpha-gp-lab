"""The canonical comparison labels what its seed uncertainty and budget can establish."""
import json
import tempfile
import unittest
from pathlib import Path

from test_analysis import sh
from test_example_comparison import config_file


class InferenceScope(unittest.TestCase):
    def test_conditional_seed_uncertainty_is_labelled_without_changing_the_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = config_file(root)
            out = root / 'out'
            run = sh('scripts/example_comparison.py', '--config', str(path), '--out', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            raw = json.loads((out / 'comparison.json').read_text())
            canonical = json.loads((out / 'audited_comparison.json').read_text())
            self.assertTrue(canonical['budget_audit']['eligible'])
            self.assertIn('comparison_scope', canonical)
            scope = canonical['comparison_scope']
            self.assertEqual(scope['standard_error'], 'optimizer variability across selected search seeds')
            self.assertEqual(set(scope['conditioned_on']), {'fixed_panel', 'fixed_splits', 'validation_selection'})
            self.assertFalse(scope['market_or_future_regime_uncertainty'])
            self.assertFalse(scope['equal_compute'])
            self.assertEqual(json.loads(run.stdout)['comparison_scope'], scope)
            self.assertEqual(canonical['seeds']['gp_minus_random'], raw['seeds']['gp_minus_random'])
            self.assertEqual(canonical['unadjusted_gp_minus_random'], raw['seeds']['gp_minus_random'])
            text = (out / 'report.md').read_text()
            self.assertIn('optimizer variability', text)
            self.assertIn('fixed panel', text)
            self.assertIn('not uncertainty across markets or future regimes', text)


if __name__ == '__main__':
    unittest.main()
