import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import helpers  # noqa: F401
from alpha_gp_lab.gp import random_tree
from alpha_gp_lab.grammar import canonical, depth, industry_variants, parse, size


class Grammar(unittest.TestCase):
    def test_round_trip(self):
        for expr in ['close', '-returns', '--close', '(close + volume)', '(close - -open)',
                     'ts_corr(close, volume, 10)', 'winsorize(zscore(returns), std=4)',
                     'winsorize(returns, std=2.75)', 'group_neutralize(-returns, industry)',
                     '(ts_rank(volume, 10) * -returns)', 'ts_decay_linear(log(volume), 5)']:
            with self.subTest(expr=expr):
                self.assertEqual(str(parse(expr)), expr)

    def test_random_trees_round_trip_through_the_parser(self):
        rng = random.Random(5)
        for _ in range(300):
            tree = random_tree(rng, rng.randint(1, 6), [1, 2, 5, 20], full=rng.random() < 0.5)
            self.assertEqual(parse(str(tree)), tree)

    def test_refuses_everything_outside_the_grammar(self):
        bad = ['vwap', 'rank(vwap)', 'ts_mean(close, 0)', 'ts_mean(close, 61)', 'ts_mean(close, True)',
               'ts_std(close, 1)', 'ts_corr(close, open, 1)', 'rank(close, 2)', 'close + 1', '+close',
               'winsorize(close, std=0)', 'winsorize(close, std=float("nan"))', 'group_rank(close, sector)',
               'close ** 2', 'close.real', '[close]', 'lambda: close', '', 'x' * 513,
               'rank(' * 11 + 'close' + ')' * 11]
        for expr in bad:
            with self.subTest(expr=expr[:40]), self.assertRaises(ValueError):
                parse(expr)

    def test_parsing_never_executes_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'executed'
            with self.assertRaises(ValueError):
                parse(f"__import__('pathlib').Path({str(marker)!r}).touch()")
            self.assertFalse(marker.exists())

    def test_canonical_form_catches_syntactic_duplicates(self):
        self.assertEqual(canonical(parse('(close + volume)')), canonical(parse('(volume + close)')))
        self.assertEqual(canonical(parse('--close')), canonical(parse('close')))
        self.assertNotEqual(canonical(parse('(close - volume)')), canonical(parse('(volume - close)')))
        self.assertNotEqual(canonical(parse('ts_mean(close, 5)')), canonical(parse('ts_mean(close, 10)')))

    def test_industry_templates(self):
        seed = parse('ts_delta(close, 1)')
        self.assertEqual([str(v) for v in industry_variants(seed)], [
            'group_rank(ts_delta(close, 1), industry)', 'group_zscore(ts_delta(close, 1), industry)',
            'group_neutralize(ts_delta(close, 1), industry)',
            'group_neutralize(zscore(ts_delta(close, 1)), industry)',
            'group_rank(winsorize(ts_delta(close, 1), std=4), industry)'])

    def test_trees_are_immutable_and_measured(self):
        tree = parse('(rank(close) + ts_mean(volume, 5))')
        self.assertEqual((depth(tree), size(tree)), (3, 5))
        with self.assertRaises(FrozenInstanceError):
            tree.op = 'sub'


if __name__ == '__main__':
    unittest.main()
