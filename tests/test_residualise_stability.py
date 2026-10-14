"""Independent exact OLS projections survive changes of regressor origin and units."""
import unittest

from alpha_gp_lab.stats import residualise


class ResidualiseStability(unittest.TestCase):
    def assert_projection(self, y, columns, expected):
        actual = residualise(y, columns)
        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, delta=1e-12)

    def test_regressor_translation_preserves_the_exact_projection(self):
        # Exact slope9/10, intercept-1/10; an intercept absorbs any representable shift.
        y, x = [0, 1, 1, 3], [0, 1, 2, 3]
        for offset in (0, 1e6, -1e6, 2**40):
            with self.subTest(offset=offset):
                self.assert_projection(y, [[v + offset for v in x]], [.1, .2, -.7, .4])

    def test_regressor_units_preserve_the_exact_projection(self):
        for scale in (1e-100, 1e-6, 1e6, 1e100, -1e6):
            with self.subTest(scale=scale):
                self.assert_projection([0, 1, 1, 3], [[v * scale for v in (0, 1, 2, 3)]], [.1, .2, -.7, .4])

    def test_multiple_covariates_preserve_their_column_space(self):
        x, z, y = [0, 1, 2, 3], [0, 1, 0, 1], [0, 1, 1, 3]
        # w=(1,-1,-1,1) is orthogonal to1,x,z; y'w=1 and w'w=4.
        variants = ([x, z], [z, x],
                    [[a + 2 * b for a, b in zip(x, z)], [2 * a - b for a, b in zip(x, z)]],
                    [[v * 2**-20 + 2**20 for v in x], [v * 2**20 - 2**20 for v in z]])
        for i, columns in enumerate(variants):
            with self.subTest(basis=i):
                self.assert_projection(y, columns, [.25, -.25, -.25, .25])

    def test_row_order_preserves_the_projection(self):
        order = [2, 0, 3, 1]
        y, x, expected = [0, 1, 1, 3], [0, 1, 2, 3], [.1, .2, -.7, .4]
        self.assert_projection([y[i] for i in order], [[x[i] + 1e6 for i in order]],
                               [expected[i] for i in order])

    def test_intercept_only_and_target_origin(self):
        self.assert_projection([0, 1, 1, 3], [], [-1.25, -.25, -.25, 1.75])
        self.assert_projection([v + 2**20 for v in (0, 1, 1, 3)], [[0, 1, 2, 3]], [.1, .2, -.7, .4])

    def test_translation_preserves_a_nonbinary_mean(self):
        # w=(2,-3,1) spans the residual space; y'w=1 and w'w=14.
        for origin in (0, 1e12):
            with self.subTest(origin=origin):
                self.assert_projection([origin + v for v in (0, 1, 4)],
                                       [[origin + v for v in (0, 1, 3)]], [1/7, -3/14, 1/14])

    def test_extreme_finite_units_are_preconditioned_before_arithmetic(self):
        cases = (([0, 1, 2, 3], [[0, 1e308, 1e308, 1e308]], [0, -1, 0, 1]),
                 ([0, 1, 2, 3], [[-1e308, -1e308, 1e308, 1e308]], [-.5, .5, -.5, .5]),
                 ([0, 1, 2, 3, 4], [[0, -1e308, -1e308, 1e308, 1e308]], [-2, 0, 1, 0, 1]),
                 ([0, 1, 2, 3], [[0, 5e-324, 5e-324, 5e-324]], [0, -1, 0, 1]))
        for i, (y, columns, expected) in enumerate(cases):
            with self.subTest(case=i):
                self.assert_projection(y, columns, expected)

    def test_singular_or_unresolved_designs_remain_refused(self):
        x, y = [0, 1, 2, 3], [0, 1, 1, 3]
        variants = ([[1]*4], [x, [2*v for v in x]],
                    [x, [v + 2**-30 * w for v, w in zip(x, (0, 1, 0, 1))]],
                    [[v + 1e20 for v in x]])
        for i, columns in enumerate(variants):
            with self.subTest(design=i):
                self.assertIsNone(residualise(y, columns))


if __name__ == '__main__':
    unittest.main()
