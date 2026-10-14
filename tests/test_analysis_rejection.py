"""A rejected primary is a research outcome, not a missing dictionary to index."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import small_config
from test_analysis import sh


def prepare(folder):
    config = small_config()
    config['llm']['use_seeds'] = False
    config['selection'].update(min_ic=1.0, max_turnover=0.0)
    config['baselines'] = {'reversal_1d': '-returns'}
    (folder / 'main.json').write_text(json.dumps(config))
    main = sh('-m', 'alpha_gp_lab', 'run', '--config', str(folder / 'main.json'), '--out', str(folder / 'run'))
    assert main.returncode == 0, main.stderr
    assert json.loads(main.stdout)['folds'][0]['status'] == 'NO_QUALIFYING_CANDIDATE'
    (folder / 'main_result.json').write_text(main.stdout)
    plan = dict(name='rejected-primary', main_config='main.json', main_result='main_result.json',
                primary_seed=config['seed'], seeds=[config['seed'], config['seed'] + 1],
                random_search_budget=36, control={}, interpretation_references={},
                bootstrap=dict(block=5, reps=50, seed=1, level=0.9), newey_west_lags=[3],
                bonferroni_counts=['validation_candidates'])
    (folder / 'plan.json').write_text(json.dumps(plan))


def analyse(folder):
    return sh('scripts/analyze_binance.py', '--config', str(folder / 'plan.json'),
              '--out', str(folder / 'analysis.json'), '--workers', '1')


class RejectedPrimary(unittest.TestCase):
    def test_rejection_retains_controls_without_inventing_a_pick_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            prepare(folder)
            process = analyse(folder)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((folder / 'analysis.json').read_text())
        self.assertEqual(report['primary_selection']['gp'], 'NO_QUALIFYING_CANDIDATE')
        self.assertEqual(report['primary_selection']['random_search'], 'NO_QUALIFYING_CANDIDATE')
        self.assertEqual(report['trials']['test'], 0)
        self.assertEqual(set(report['table']), {'reversal_1d'})
        self.assertEqual(set(report['series']), {'reversal_1d'})
        self.assertEqual(set(report['inference']), {'reversal_1d'})
        self.assertEqual(report['pick_minus'], {})
        self.assertEqual(report['interpretation'], {})
        self.assertIsNone(report['significance']['p_iid'])
        self.assertIn('no qualifying', report['significance']['unavailable'])
        for kind in ('gp', 'random_search'):
            self.assertEqual(report['seeds'][kind]['no_qualifying_candidate'], 2)
            self.assertIsNone(report['seeds'][kind]['mean_net'])
        for metric in ('mean_ic', 'mean_net'):
            self.assertIsNone(report['seeds']['gp_minus_random'][metric]['gp_beats_random'])
        self.assertTrue(all(r['test'] is None for r in report['per_seed']))

    def test_rejected_run_still_refuses_a_different_committed_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            prepare(folder)
            path = folder / 'main_result.json'
            changed = json.loads(path.read_text())
            changed['folds'][0]['selected'] = 'returns'
            path.write_text(json.dumps(changed))
            process = analyse(folder)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('does not reproduce', process.stderr)
            self.assertFalse((folder / 'analysis.json').exists())


if __name__ == '__main__':
    unittest.main()
