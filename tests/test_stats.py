"""Newey-West t, the circular block bootstrap, p-values and the Bonferroni bound, against hand values."""
import math
import random
import unittest

import helpers  # noqa: F401  (puts src on the path)
from alpha_gp_lab.stats import block_bootstrap_ci, bonferroni, newey_west_tstat, two_sided_p


class NeweyWest(unittest.TestCase):
    def test_lag_zero_is_the_population_variance_t(self):
        # mean 2.5, population variance 1.25, n 4: t = 2.5 / sqrt(1.25 / 4) = sqrt(20)
        self.assertAlmostEqual(newey_west_tstat([1, 2, 3, 4], 0), math.sqrt(20), places=12)

    def test_one_lag_by_hand(self):
        # deviations -1.5, 0.5, -0.5, 1.5: gamma0 = 1.25, gamma1 = -1.75 / 4, Bartlett weight 1/2,
        # long-run variance 13/16, so t^2 = 6.25 / (13/64) = 400/13
        self.assertAlmostEqual(newey_west_tstat([1, 3, 2, 4], 1), math.sqrt(400 / 13), places=12)

    def test_positive_autocorrelation_shrinks_the_t(self):
        rng = random.Random(3)
        x, xs = 0.0, []
        for _ in range(600):
            x = 0.9 * x + rng.gauss(0, 1)
            xs.append(x + 0.3)
        self.assertLess(abs(newey_west_tstat(xs, 10)), 0.5 * abs(newey_west_tstat(xs, 0)))

    def test_constant_series_has_no_t(self):
        self.assertIsNone(newey_west_tstat([0.2] * 10, 3))


class Bootstrap(unittest.TestCase):
    def test_constant_series_gives_a_point_interval(self):
        ci = block_bootstrap_ci([0.5] * 37, 5, 200, 1)
        self.assertEqual((ci['low'], ci['high'], ci['share_at_or_below_zero']), (0.5, 0.5, 0.0))

    def test_blocks_widen_the_interval_for_persistent_series(self):
        xs = ([1.0] * 25 + [-1.0] * 25) * 6   # long runs: day-by-day resampling understates the spread
        narrow, wide = block_bootstrap_ci(xs, 1, 2000, 7), block_bootstrap_ci(xs, 20, 2000, 7)
        self.assertGreater(wide['high'] - wide['low'], 2 * (narrow['high'] - narrow['low']))

    def test_interval_brackets_the_mean_and_is_seeded(self):
        rng = random.Random(5)
        xs = [rng.gauss(0.1, 1) for _ in range(300)]
        ci = block_bootstrap_ci(xs, 20, 1000, 11)
        self.assertLess(ci['low'], sum(xs) / len(xs))
        self.assertGreater(ci['high'], sum(xs) / len(xs))
        self.assertEqual(ci, block_bootstrap_ci(xs, 20, 1000, 11))
        self.assertNotEqual(ci, block_bootstrap_ci(xs, 20, 1000, 12))


class PValues(unittest.TestCase):
    def test_normal_two_sided(self):
        self.assertAlmostEqual(two_sided_p(1.959963984540054), 0.05, places=12)
        self.assertEqual(two_sided_p(-2.5), two_sided_p(2.5))

    def test_bonferroni_scales_and_caps(self):
        self.assertAlmostEqual(bonferroni(0.01, 16), 0.16)
        self.assertEqual(bonferroni(0.2, 16), 1.0)


if __name__ == '__main__':
    unittest.main()
