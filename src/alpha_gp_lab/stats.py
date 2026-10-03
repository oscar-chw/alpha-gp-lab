"""Inference on daily series of IC or net return: autocorrelation-robust t-statistics, a block
bootstrap and a Bonferroni bound. The evaluator's own ``ic_tstat`` treats days as independent;
these do not, because a smoothed signal's daily IC is autocorrelated and the naive t overstates it.
"""
import math
import random


def newey_west_tstat(xs, lags):
    """Mean over its Newey-West (Bartlett kernel) standard error with ``lags`` lags.
    Autocovariances use the 1/n normalisation, so ``lags=0`` is the mean over sqrt(population var / n)."""
    n = len(xs)
    m = math.fsum(xs) / n
    d = [x - m for x in xs]
    lrv = math.fsum(v * v for v in d) / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = math.fsum(d[i] * d[i - lag] for i in range(lag, n)) / n
        lrv += 2 * (1 - lag / (lags + 1)) * gamma
    return m / math.sqrt(lrv / n) if lrv > 0 else None


def two_sided_p(t):
    """Two-sided p-value of a t-statistic under the normal approximation (n is in the hundreds here)."""
    return math.erfc(abs(t) / math.sqrt(2))


def bonferroni(p, m):
    """Family-wise bound for one tested hypothesis chosen from ``m``. With a single hypothesis
    actually tested, Holm's step-down procedure gives the same number."""
    return min(1.0, p * m)


def block_bootstrap_ci(xs, block, reps, seed, level=0.95):
    """Circular block bootstrap of the mean: each resample joins blocks of ``block`` consecutive
    days, starting at uniform random days and wrapping at the end, until it has len(xs) days.
    Two series of the same length resampled with the same seed use the same days, so their
    intervals are joint. Returns the percentile interval and the share of resampled means <= 0."""
    n, rng, means = len(xs), random.Random(seed), []
    for _ in range(reps):
        total, k = 0.0, 0
        while k < n:
            start = rng.randrange(n)
            for j in range(min(block, n - k)):
                total += xs[(start + j) % n]
            k += block
        means.append(total / n)
    means.sort()
    lo = means[int(math.floor((1 - level) / 2 * reps))]
    hi = means[int(math.ceil((1 + level) / 2 * reps)) - 1]
    return dict(low=lo, high=hi, share_at_or_below_zero=sum(x <= 0 for x in means) / reps)
