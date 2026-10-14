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


def residualise(y, columns):
    """Residuals of an OLS fit of ``y`` on an intercept and ``columns`` (lists the length of y),
    by centered/scaled Householder QR; None for a numerically rank-deficient design.
    Unit-norm columns retain the 1e-12 squared independent-component rank floor."""
    n, p = len(y), len(columns)
    if n <= p:
        return None

    def centered(xs):
        # Subtract an anchor first: an arbitrary large origin must not erase local variation.
        shifted = [x - xs[0] for x in xs]
        scale = max(abs(x) for x in shifted)
        if not math.isfinite(scale):
            # Opposite finite extremes can overflow the subtraction; normalize first only then.
            scale = max(abs(x) for x in xs)
            shifted = [x / scale - xs[0] / scale for x in xs]
        elif scale:
            shifted = [x / scale for x in shifted]
        else:
            return [0.0] * n, 1.0
        mean = math.fsum(shifted) / n
        return [x - mean for x in shifted], scale

    a = []
    for column in columns:
        c, _ = centered(column)
        norm = math.hypot(*c)
        if norm == 0:
            return None
        a.append([x / norm for x in c])
    residual, response_scale = centered(y)
    reflectors = []

    def reflect(xs, start, v):
        dot = 2 * math.fsum(h * xs[start + i] for i, h in enumerate(v))
        for i, h in enumerate(v):
            xs[start + i] -= h * dot

    for j in range(p):
        pivot = max(range(j, p), key=lambda q: math.hypot(*a[q][j:]))
        a[j], a[pivot] = a[pivot], a[j]
        v = a[j][j:]
        norm = math.hypot(*v)
        if norm * norm < 1e-12:
            return None
        v[0] += math.copysign(norm, v[0])   # opposite-sign reflector avoids cancellation
        length = math.hypot(*v)
        v = [h / length for h in v]
        for c in a[j:]:
            reflect(c, j, v)
        reflect(residual, j, v)
        reflectors.append((j, v))
    # Project directly; recovering a large intercept and subtracting fitted values loses bits.
    residual[:p] = [0.0] * p
    for j, v in reversed(reflectors):
        reflect(residual, j, v)
    return [x * response_scale for x in residual]
