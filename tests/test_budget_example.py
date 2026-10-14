"""Actual underfilled searches must not inherit an equal-budget winner label."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import small_config
from test_analysis import sh


class BudgetExample(unittest.TestCase):
    def test_real_shortfall_withholds_verdict_and_retains_raw_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = small_config()
            config['llm']['use_seeds'] = False
            config['gp']['max_attempts'] = 1
            path = root / 'input.json'
            path.write_text(json.dumps(config))
            out = root / 'out'
            result = sh('scripts/example_comparison.py', '--config', str(path), '--out', str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / 'budget_audit.json').is_file(), 'comparison omitted realized-budget admission')
            audit = json.loads((out / 'budget_audit.json').read_text())
            raw = json.loads((out / 'comparison.json').read_text())
            printed = json.loads(result.stdout)
            self.assertFalse(audit['eligible'])
            self.assertEqual(audit['nominal_per_search'], 36)
            self.assertEqual([(r['gp'], r['random']) for r in audit['per_seed']], [(29, 24), (34, 24)])
            self.assertEqual(audit['unadjusted_comparison'], raw['seeds']['gp_minus_random'])
            for metric in ('mean_ic', 'mean_net'):
                self.assertIsNone(printed['comparison'][metric]['gp_beats_random'])
                self.assertEqual(printed['comparison'][metric]['difference'], raw['seeds']['gp_minus_random'][metric]['difference'])
            text = (out / 'report.md').read_text()
            self.assertIn('WITHHELD', text)
            self.assertIn('not equal compute', text)
            manifest = json.loads((out / 'example_manifest.json').read_text())
            self.assertEqual(manifest['files']['budget_audit.json'], hashlib.sha256((out / 'budget_audit.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
