"""A canonical machine-readable result must carry the effective budget decision."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import small_config
from test_analysis import sh


class AuditedResult(unittest.TestCase):
    def test_manifest_and_cli_point_to_a_full_audited_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = small_config()
            config['llm']['use_seeds'] = False
            config['gp']['max_attempts'] = 1
            path = root / 'input.json'
            path.write_text(json.dumps(config))
            out = root / 'out'
            run = sh('scripts/example_comparison.py', '--config', str(path), '--out', str(out))
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = json.loads((out / 'example_manifest.json').read_text())
            self.assertIn('canonical_result', manifest)
            result = json.loads((out / manifest['canonical_result']).read_text())
            raw = json.loads((out / 'comparison.json').read_text())
            self.assertEqual(json.loads(run.stdout)['canonical_result'], manifest['canonical_result'])
            self.assertFalse(result['budget_audit']['eligible'])
            self.assertEqual(result['unadjusted_gp_minus_random'], raw['seeds']['gp_minus_random'])
            for metric in ('mean_ic', 'mean_net'):
                self.assertIsNone(result['seeds']['gp_minus_random'][metric]['gp_beats_random'])
            self.assertEqual(result['series'], raw['series'])
            self.assertEqual(result['per_seed'], raw['per_seed'])
            self.assertEqual(result['table'], raw['table'])
            self.assertIn(manifest['canonical_result'], manifest['files'])


if __name__ == '__main__':
    unittest.main()
