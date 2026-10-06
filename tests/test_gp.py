import copy
import random
import tempfile
import unittest

from helpers import inputs, panel_for, perturbed, small_config
from alpha_gp_lab.evaluate import Evaluator
from alpha_gp_lab.gp import _choose, compute_report, crossover, mutate, random_search
from alpha_gp_lab.grammar import canonical, coin_units, depth, parse, positions, size


def inputs_in_tmp(config):
    with tempfile.TemporaryDirectory() as tmp:
        return inputs(config, tmp)


def run(config, response=None):
    with tempfile.TemporaryDirectory() as tmp:
        args = (config, tmp) if response is None else (config, tmp, response)
        _, panel, llm = inputs(*args)
    return compute_report(config, panel, llm), panel, llm


class SplitRoles(unittest.TestCase):
    """Train picks parents, validation picks the winner, test scores only the final pick."""

    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.report, cls.panel, cls.llm = run(cls.config)
        cls.fold = cls.report['folds'][0]

    def test_test_data_cannot_change_search_or_selection(self):
        other = compute_report(self.config, perturbed(self.panel, self.config['splits']['test'][0]), self.llm)['folds'][0]
        self.assertEqual(self.fold['nodes'], other['nodes'])
        self.assertEqual(self.fold['selected_id'], other['selected_id'])
        self.assertEqual([r for r in self.fold['results'] if r['split'] != 'test'],
                         [r for r in other['results'] if r['split'] != 'test'])
        self.assertNotEqual([r for r in self.fold['results'] if r['split'] == 'test'],
                            [r for r in other['results'] if r['split'] == 'test'])

    def test_validation_data_cannot_change_the_search(self):
        other = compute_report(self.config, perturbed(self.panel, self.config['splits']['validation'][0]), self.llm)['folds'][0]
        self.assertEqual(self.fold['nodes'], other['nodes'])
        self.assertEqual([r for r in self.fold['results'] if r['split'] == 'train'],
                         [r for r in other['results'] if r['split'] == 'train'])
        self.assertNotEqual([r for r in self.fold['results'] if r['split'] == 'validation'],
                            [r for r in other['results'] if r['split'] == 'validation'])

    def test_only_the_final_pick_is_tested(self):
        tested = [r for r in self.fold['results'] if r['split'] == 'test']
        self.assertEqual(self.fold['status'], 'SELECTED')
        self.assertEqual([r['node'] for r in tested], [self.fold['selected_id']])
        self.assertIn(self.fold['selected_origin']['operation'].split(':')[0],
                      ('llm_seed', 'random_init', 'crossover', 'mutation'))
        validated = {r['node'] for r in self.fold['results'] if r['split'] == 'validation'}
        self.assertEqual(validated, set(self.fold['hall_of_fame']))

    def test_no_qualifier_means_no_test(self):
        config = copy.deepcopy(self.config)
        config['selection']['min_ic'] = 0.99
        fold = run(config)[0]['folds'][0]
        self.assertEqual((fold['status'], fold['selected_id'], fold['test']), ('NO_QUALIFYING_CANDIDATE', None, None))
        self.assertFalse(any(r['split'] == 'test' for r in fold['results']))


class Search(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.report = run(cls.config)[0]
        cls.fold = cls.report['folds'][0]

    def test_seeded_determinism(self):
        self.assertEqual(run(self.config)[0], self.report)
        other = copy.deepcopy(self.config)
        other['seed'] += 1
        self.assertNotEqual(run(other)[0]['folds'][0]['nodes'], self.fold['nodes'])

    def test_budgets_and_limits(self):
        gp = self.config['gp']
        gens = self.fold['generations']
        self.assertEqual([g['generation'] for g in gens], list(range(gp['generations'])))
        self.assertTrue(all(g['size'] <= gp['population'] for g in gens))
        for n in self.fold['nodes']:
            tree = parse(n['expression'])
            self.assertLessEqual(depth(tree), gp['max_depth'])
            self.assertLessEqual(size(tree), gp['max_nodes'])

    def test_elitism_never_loses_the_best(self):
        best = [g['best_train_score'] for g in self.fold['generations']]
        self.assertEqual(best, sorted(best))

    def test_lineage_points_to_the_previous_generation(self):
        by_id = {n['id']: n for n in self.fold['nodes']}
        for n in self.fold['nodes']:
            if n['generation'] == 0:
                self.assertEqual(n['parents'], [])
                self.assertIn(n['operation'], ('llm_seed', 'random_init'))
            else:
                self.assertTrue(n['parents'])
                self.assertTrue(all(by_id[p]['generation'] == n['generation'] - 1 for p in n['parents']))
        seeds = [n['expression'] for n in self.fold['nodes'] if n['operation'] == 'llm_seed']
        self.assertEqual(seeds, self.report['llm']['accepted'])

    def test_no_duplicates_or_equivalents_within_a_generation(self):
        fps = {r['node']: r['fingerprint'] for r in self.fold['results'] if r['split'] == 'train'}
        for g in range(self.config['gp']['generations']):
            nodes = [n for n in self.fold['nodes'] if n['generation'] == g]
            self.assertEqual(len({canonical(parse(n['expression'])) for n in nodes}), len(nodes))
            self.assertEqual(len({fps[n['id']] for n in nodes}), len(nodes))

    def test_equivalence_filter_rejects_monotone_rewrites(self):
        config = small_config()
        config['gp'].update(population=2, generations=1, tournament=2, elitism=1, hall_of_fame=2)
        _, panel, llm = inputs_in_tmp(config)
        llm = dict(llm, accepted=['returns', 'rank(returns)', 'zscore(returns)', '(returns + close)', '(close + returns)'])
        fold = compute_report(config, panel, llm)['folds'][0]
        self.assertEqual([n['expression'] for n in fold['nodes']], ['returns', '(returns + close)'])
        self.assertEqual((fold['counts']['rejected_equivalent'], fold['counts']['rejected_duplicate']), (2, 1))

    def test_correlation_filter(self):
        config = copy.deepcopy(self.config)
        config['gp']['hall_of_fame'] = 12
        config['selection'].update(select_k=4, max_corr=0.5, min_ic=-1)
        report, panel, _ = run(config)
        fold = report['folds'][0]
        vs, ve = config['splits']['validation']
        ev = Evaluator(panel.head(ve + 1), 1, 5)
        picked = [parse(s['expression']) for s in fold['shortlist']]
        for i, a in enumerate(picked):
            for b in picked[:i]:
                self.assertLessEqual(abs(ev.correlation(a, b, vs, ve)), 0.5)
        self.assertTrue(fold['correlation_rejected'])
        self.assertTrue(all(r['max_abs_corr_to_selected'] > 0.5 for r in fold['correlation_rejected']))
        self.assertEqual(fold['selected_expression'], fold['shortlist'][0]['expression'])

        # An already-selected alpha equal to this run's pick must push the pick elsewhere.
        config['selection']['existing_alphas'] = [fold['selected_expression']]
        other = run(config)[0]['folds'][0]
        self.assertNotEqual(other['selected_expression'], fold['selected_expression'])
        self.assertEqual(other['correlation_rejected'][0]['expression'], fold['selected_expression'])
        self.assertLessEqual(abs(ev.correlation(parse(other['selected_expression']), picked[0], vs, ve)), 0.5)

    def test_variation_operators_keep_trees_valid_and_parents_intact(self):
        rng = random.Random(9)
        a = parse('group_rank(winsorize(ts_delta(open, 2), std=4), industry)')   # leaves: open only
        b = parse('(ts_corr(volume, returns, 5) - rank(returns))')               # leaves: volume, returns
        before = (str(a), str(b))
        for _ in range(100):
            child = crossover(a, b, rng)
            self.assertEqual(parse(str(child)), child)
            self.assertTrue(any(t.op in ('volume', 'returns') for _, t in positions(child)))
            mutant, kind = mutate(a, rng, [2, 3, 5])
            self.assertEqual(parse(str(mutant)), mutant)
            self.assertIn(kind, ('subtree', 'point', 'window', 'hoist', 'industry'))
        self.assertEqual((str(a), str(b)), before)

    def test_ablation_without_llm_seeds(self):
        config = copy.deepcopy(self.config)
        config['llm']['use_seeds'] = False
        report = run(config)[0]
        self.assertFalse(report['llm']['used_as_seeds'])
        self.assertFalse(any(n['operation'] == 'llm_seed' for n in report['folds'][0]['nodes']))

    def test_an_overflowing_candidate_is_rejected_and_the_run_completes(self):
        """max_nodes accepts up to 128, so a seed or baseline can overflow a float; that rejects one expression."""
        boom = 'volume'
        for _ in range(5):
            boom = f'({boom} * {boom})'
        boom = f'zscore({boom})'   # volume ** 32 is finite, but (x - m) ** 2 inside zscore overflows
        config = copy.deepcopy(self.config)
        config['gp'].update(max_nodes=128, max_depth=10)
        config['baselines'] = {'overflow': boom}
        report, _, llm = run(config, response=boom + '\n-returns\n')
        fold = report['folds'][0]
        self.assertIn(str(parse(boom)), llm['accepted'])
        self.assertEqual(fold['status'], 'SELECTED')
        self.assertIsNone(fold['baselines']['overflow']['test']['mean_ic'])
        seed = next(n for n in fold['nodes'] if n['operation'] == 'llm_seed' and n['expression'] == '-returns')
        self.assertEqual(seed['generation'], 0)
        self.assertNotIn(str(parse(boom)), {n['expression'] for n in fold['nodes']})
        self.assertGreaterEqual(fold['counts']['rejected_degenerate'], 1)

    def test_walk_forward_folds_roll_and_stay_disjoint(self):
        config = copy.deepcopy(self.config)
        del config['splits']
        config['walk_forward'] = dict(warmup=10, train=40, validation=15, test=15, step=15)
        report = run(config)[0]
        self.assertEqual(report['mode'], 'walk_forward')
        spans = [f['splits'] for f in report['folds']]
        self.assertGreaterEqual(len(spans), 2)
        for s in spans:
            self.assertLess(s['train'][1], s['validation'][0])
            self.assertLess(s['validation'][1], s['test'][0])
        for a, b in zip(spans, spans[1:]):
            self.assertLessEqual(a['test'][1], b['test'][0])
        self.assertEqual(report['summary']['folds'], len(spans))


class RandomSearch(unittest.TestCase):
    """The equal-budget control: same generator, filter and validation rule as the GP, no breeding."""

    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.panel = panel_for(cls.config)
        cls.split = cls.config['splits']
        cls.out = random_search(cls.panel, cls.split, cls.config, 30, 1)

    def test_budget_is_distinct_admitted_signals(self):
        self.assertEqual(self.out['counts']['occurrences'], 30)
        hall = self.out['hall_of_fame']
        self.assertEqual(len(hall), self.config['gp']['hall_of_fame'])
        self.assertEqual(len({canonical(parse(e)) for e in hall}), len(hall))
        self.assertEqual(self.out['validation_candidates'], len(hall))
        # repeats are refused across the whole sample, not per batch: 30 draws here repeat some signals
        self.assertGreater(self.out['counts']['rejected_duplicate'] + self.out['counts']['rejected_equivalent'], 0)

    def test_pick_follows_the_validation_rule(self):
        self.assertEqual(self.out['status'], 'SELECTED')
        self.assertEqual(self.out['selected'], self.out['shortlist'][0])
        self.assertIn(self.out['selected'], self.out['hall_of_fame'])
        self.assertGreaterEqual(self.out['validation']['mean_ic'], self.config['selection']['min_ic'])
        self.assertLessEqual(self.out['validation']['mean_turnover'], self.config['selection']['max_turnover'])

    def test_seeded(self):
        self.assertEqual(random_search(self.panel, self.split, self.config, 30, 1), self.out)
        self.assertNotEqual(random_search(self.panel, self.split, self.config, 30, 2)['hall_of_fame'],
                            self.out['hall_of_fame'])

    def test_test_data_cannot_change_the_pick(self):
        other = random_search(perturbed(self.panel, self.split['test'][0]), self.split, self.config, 30, 1)
        self.assertEqual((other['selected'], other['validation']), (self.out['selected'], self.out['validation']))
        self.assertNotEqual(other['test'], self.out['test'])

    def test_no_qualifier_means_no_test(self):
        config = copy.deepcopy(self.config)
        config['selection']['min_ic'] = 0.99
        out = random_search(self.panel, self.split, config, 30, 1)
        self.assertEqual((out['status'], out['selected'], out['test']), ('NO_QUALIFYING_CANDIDATE', None, None))


class OptInRules(unittest.TestCase):
    """unit_check and min_coverage are off unless a config asks for them, so the committed runs replay."""

    def test_unit_check_admits_only_unit_free_candidates(self):
        config = small_config()
        config['gp']['unit_check'] = True
        fold = run(config)[0]['folds'][0]
        self.assertTrue(all(coin_units(parse(n['expression'])) == 0 for n in fold['nodes']))
        self.assertGreater(fold['counts']['rejected_units'], 0)
        self.assertNotIn('rejected_units', run(small_config())[0]['folds'][0]['counts'])

    def test_min_coverage_refuses_a_candidate_defined_on_few_validation_days(self):
        config = small_config()
        panel = panel_for(config)
        vs, ve = config['splits']['validation']
        val = Evaluator(panel.head(ve + 1), 1, 5)
        hall = [dict(record=dict(id=e, expression=e), tree=parse(e), score=0.0)
                for e in ('ts_sum(ts_mean(returns, 60), 30)', '-returns')]   # the first is warm until late validation
        sel = dict(config['selection'], min_ic=-1, max_turnover=2, max_corr=1, select_k=5)
        pen = (0.0, 0.0)
        sparse = val.metrics(hall[0]['tree'], vs, ve)
        self.assertLess(sparse['valid_ic_intervals'], 0.5 * sparse['intervals'])
        self.assertEqual(len(_choose(hall, val, vs, ve, sel, pen, [])[0]), 2)
        kept = _choose(hall, val, vs, ve, dict(sel, min_coverage=0.5), pen, [])[0]
        self.assertEqual([p['record']['expression'] for _, p, _ in kept], ['-returns'])


if __name__ == '__main__':
    unittest.main()
