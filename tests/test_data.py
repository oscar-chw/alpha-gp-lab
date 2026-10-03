import shutil
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT
from alpha_gp_lab.config import canonical
from alpha_gp_lab.data import Panel, load_csv_dir, synthetic_panel
from alpha_gp_lab.evaluate import Evaluator
from alpha_gp_lab.grammar import parse

SAMPLE = ROOT / 'fixtures' / 'ohlcv_sample'


class Synthetic(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        segs = [{'regime': 'noise', 'days': 30}]
        a, b = synthetic_panel(1, 6, 2, segs), synthetic_panel(1, 6, 2, segs)
        self.assertEqual(a, b)
        self.assertNotEqual(a.bars['close'], synthetic_panel(2, 6, 2, segs).bars['close'])
        self.assertEqual(a.label, 'SYNTHETIC')
        self.assertTrue(all(s.startswith('SYN_') for s in a.symbols))

    def test_regimes_shape_predictability(self):
        """The planted effect has the sign its regime says; 'adverse' flips reversal."""
        days = 400
        reversal = parse('group_neutralize(-returns, industry)')
        momentum = parse('ts_sum(returns, 5)')
        ic = {}
        for regime in ('reversal', 'adverse', 'momentum', 'noise'):
            panel = synthetic_panel(7, 30, 3, [{'regime': regime, 'days': days}])
            ev = Evaluator(panel, 1, 0)
            ic[regime] = (ev.metrics(reversal, 10, days - 1)['mean_ic'], ev.metrics(momentum, 10, days - 1)['mean_ic'])
        self.assertGreater(ic['reversal'][0], 0.1)
        self.assertLess(ic['adverse'][0], -0.1)
        self.assertGreater(ic['momentum'][1], 0.1)
        self.assertLess(abs(ic['noise'][0]), 0.05)
        self.assertLess(abs(ic['noise'][1]), 0.05)

    def test_json_round_trip_and_head(self):
        panel = synthetic_panel(3, 5, 2, [{'regime': 'reversal', 'days': 12}])
        again = Panel.from_json(panel.to_json())
        self.assertEqual(canonical(again.to_json()), canonical(panel.to_json()))
        head = panel.head(5)
        self.assertEqual((len(head.dates), head.bars['close'], head.regimes), (5, panel.bars['close'][:5], panel.regimes[:5]))
        with self.assertRaises(ValueError):
            synthetic_panel(3, 5, 2, [{'regime': 'bull', 'days': 12}])


class CsvLoader(unittest.TestCase):
    def test_loads_aligned_panel_with_groups(self):
        panel, dropped = load_csv_dir(SAMPLE)
        self.assertEqual(panel.symbols, ('SYN_AAA', 'SYN_BBB', 'SYN_CCC'))
        self.assertEqual(panel.groups, ('SYN_TECH', 'SYN_TECH', 'SYN_ENERGY'))
        self.assertEqual(panel.dates, ('2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'))
        self.assertEqual(dropped, {'SYN_AAA': 1, 'SYN_BBB': 0, 'SYN_CCC': 1})
        self.assertEqual(panel.bars['close'][0], [10.2, 50.5, 3.05])   # column order in the file does not matter
        self.assertFalse(panel.synthetic)

    def _broken(self, name, text):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        for f in SAMPLE.iterdir():
            shutil.copy(f, tmp / f.name)
        (tmp / name).write_text(text)
        return tmp

    def test_refuses_malformed_input(self):
        head = 'date,open,high,low,close,volume\n'
        good = '2024-01-01,10,11,9,10,5\n'
        cases = {
            'header': ('SYN_AAA.csv', 'date,open,high,low,close\n2024-01-01,1,1,1,1\n'),
            'high below close': ('SYN_AAA.csv', head + '2024-01-01,10,10.5,9,11,5\n'),
            'non-finite': ('SYN_AAA.csv', head + '2024-01-01,nan,11,9,10,5\n'),
            'non-numeric': ('SYN_AAA.csv', head + '2024-01-01,ten,11,9,10,5\n'),
            'unordered dates': ('SYN_AAA.csv', head + '2024-01-02,10,11,9,10,5\n' + good),
            'bad date': ('SYN_AAA.csv', head + '01/02/2024,10,11,9,10,5\n'),
            'negative volume': ('SYN_AAA.csv', head + '2024-01-01,10,11,9,10,-5\n'),
            'groups mismatch': ('groups.csv', 'symbol,group\nSYN_AAA,X\n'),
        }
        for label, (name, text) in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                load_csv_dir(self._broken(name, text))


if __name__ == '__main__':
    unittest.main()
