"""Independent hand cases for completed, short and malformed comparison budgets."""
import copy
import unittest

from alpha_gp_lab.budget import audit_budget


def report(counts=((1, 10, 10),), winner=True):
    rows = []
    for seed, gp, random in counts:
        for kind, count in (('gp', gp), ('random', random)):
            rows.append(dict(kind=kind, seed=seed, trials={'train': count}, status='SELECTED', test={'mean_net': 0.1}))
    metric = dict(difference=0.12, standard_error=0.03, gp_beats_random=winner)
    return dict(per_seed=rows, seeds={'gp_minus_random': {'mean_ic': metric, 'mean_net': dict(metric)}})


class BudgetAudit(unittest.TestCase):
    def test_complete_protocol_retains_the_original_decision_and_accounting_basis(self):
        raw = report(((3, 10, 10), (9, 10, 10)))
        audit = audit_budget(raw, [3, 9], 10)
        self.assertTrue(audit['eligible'])
        self.assertEqual(audit['comparison'], raw['seeds']['gp_minus_random'])
        self.assertEqual(audit['reasons'], [])
        self.assertIn('elites', audit['budget_basis'])
        self.assertEqual([row['seed'] for row in audit['per_seed']], [3, 9])

    def test_unequal_completed_work_cannot_be_reported_as_gp_superiority(self):
        raw = report(((1, 10, 7),))
        original = copy.deepcopy(raw)
        audit = audit_budget(raw, [1], 10)
        self.assertFalse(audit['eligible'])
        self.assertEqual((audit['per_seed'][0]['gp'], audit['per_seed'][0]['random']), (10, 7))
        for metric in ('mean_ic', 'mean_net'):
            self.assertIsNone(audit['comparison'][metric]['gp_beats_random'])
            self.assertIn('budget', audit['comparison'][metric]['undecided'])
            self.assertEqual(audit['comparison'][metric]['difference'], 0.12)
            self.assertEqual(audit['comparison'][metric]['standard_error'], 0.03)
            self.assertTrue(audit['unadjusted_comparison'][metric]['gp_beats_random'])
        self.assertEqual(raw, original)

    def test_equal_but_incomplete_counts_do_not_complete_a_declared_protocol(self):
        audit = audit_budget(report(((1, 7, 7),)), [1], 10)
        self.assertFalse(audit['eligible'])
        self.assertIsNone(audit['comparison']['mean_ic']['gp_beats_random'])

    def test_one_underfilled_seed_is_not_silently_removed_from_the_study(self):
        audit = audit_budget(report(((1, 10, 10), (2, 9, 10))), [1, 2], 10)
        self.assertFalse(audit['eligible'])
        self.assertEqual(len(audit['per_seed']), 2)
        self.assertEqual(audit['reasons'][0]['seed'], 2)

    def test_unavailable_inference_stays_unavailable_even_with_complete_budgets(self):
        raw = report(winner=None)
        for row in raw['per_seed']:
            row.update(status='NO_QUALIFYING_CANDIDATE', test=None)
        for metric in raw['seeds']['gp_minus_random'].values():
            metric.update(difference=None, standard_error=None, undecided='too few picks: GP 0, random search 0')
        audit = audit_budget(raw, [1], 10)
        self.assertTrue(audit['eligible'])
        self.assertEqual(audit['comparison'], raw['seeds']['gp_minus_random'])

    def test_missing_duplicate_and_unexpected_rows_fail(self):
        for alteration in ('missing', 'duplicate', 'seed', 'kind'):
            raw = report()
            if alteration == 'missing':
                raw['per_seed'].pop()
            elif alteration == 'duplicate':
                raw['per_seed'].append(copy.deepcopy(raw['per_seed'][0]))
            elif alteration == 'seed':
                raw['per_seed'][0]['seed'] = 99
            else:
                raw['per_seed'][0]['kind'] = 'other'
            with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, 'row|seed|kind'):
                audit_budget(raw, [1], 10)

    def test_invalid_counts_and_protocol_fail(self):
        for count in (-1, 11, True, 2.5, None):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'train'):
                audit_budget(report(((1, count, 10),)), [1], 10)
        for seeds, budget in (([], 10), ([1, 1], 10), ([True], 10), ([1], 0), ([1], True)):
            with self.subTest(seeds=seeds, budget=budget), self.assertRaisesRegex(ValueError, 'seed|budget'):
                audit_budget(report(), seeds, budget)


if __name__ == '__main__':
    unittest.main()
