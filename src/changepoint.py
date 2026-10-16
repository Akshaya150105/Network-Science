import math

import numpy as np
from scipy.optimize import minimize_scalar

from src.mle import DELTA_BOUNDS


def _fit(counts):
    """
    Maximised log-likelihood and MLE of delta from degree counts.

    counts[d] = number of steps that picked a node of degree d. Grouping by
    degree makes each fit cost O(distinct degrees) instead of O(steps).
    The sum(log(k - 1)) term is left out: it does not depend on delta and
    cancels in the likelihood ratio.
    """
    degrees = np.nonzero(counts)[0]
    c = counts[degrees]
    m = c.sum()
    if m == 0:
        return 0.0, float("nan")

    def neg_ll(delta):
        return -(np.dot(c, np.log(degrees + delta)) - m * np.log(2 + delta))

    result = minimize_scalar(
        neg_ll, bounds=DELTA_BOUNDS, method="bounded", options={"xatol": 1e-6}
    )
    return -float(result.fun), float(result.x)


def scan_curve(picks, trim=0.1, step=1):
    """
    Likelihood-ratio statistic for every split m (Paper B, Eq. 20).

    stat(m) = 2 * (ell_{k<=m}(d1_hat) + ell_{k>m}(d2_hat) - ell_all(d_hat))
    for m in [floor(trim * n), floor((1 - trim) * n)], taking every `step`-th m.

    Returns arrays (m_values, stat_values).
    """
    arr = np.asarray(picks, dtype=int).reshape(-1, 2)
    degrees, steps = arr[:, 0], arr[:, 1]
    n = int(steps.max())

    # cum[j] = degree counts of the first j picks; the split at m uses j = m - 2.
    onehot = np.zeros((len(degrees), degrees.max() + 1), dtype=np.int32)
    onehot[np.arange(len(degrees)), degrees] = 1
    cum = np.vstack([np.zeros((1, onehot.shape[1]), dtype=np.int32), np.cumsum(onehot, axis=0)])

    lo = max(int(np.floor(trim * n)), 3)
    hi = min(int(np.floor((1 - trim) * n)), n - 1)
    m_values = np.arange(lo, hi + 1, step)

    ell_all, _ = _fit(cum[-1])
    stats = np.empty(len(m_values))
    for i, m in enumerate(m_values):
        left, _ = _fit(cum[m - 2])
        right, _ = _fit(cum[-1] - cum[m - 2])
        stats[i] = 2 * (left + right - ell_all)

    # Numerical noise from the optimiser can give tiny negative values.
    return m_values, np.maximum(stats, 0.0)


def scan(picks, trim=0.1, step=1):
    """Return (stat, tau_hat): the largest statistic over splits and where it occurs."""
    m_values, stats = scan_curve(picks, trim=trim, step=step)
    best = int(np.argmax(stats))
    return float(stats[best]), int(m_values[best])


def _degree_count_frames(picks):
    """
    Replay the degree sequence and return, for each node count m, the vector of
    degree counts N[d] = number of nodes with degree d when the graph has m nodes.

    Seed graph G(2) has two nodes of degree 2 (the contract's [2, 2] seed). Each
    pick (d, k) moves one node from degree d to d + 1 and adds a new degree-1 node,
    so the graph has k nodes afterwards. Returns (frames, n) where frames[m] is the
    count vector at m nodes, for m = 2 .. n.
    """
    arr = np.asarray(picks, dtype=int).reshape(-1, 2)
    degrees, steps = arr[:, 0], arr[:, 1]
    n = int(steps.max())
    width = int(max(degrees.max() + 2, 3))

    counts = np.zeros(width, dtype=float)
    counts[2] = 2.0  # seed [2, 2]
    frames = {2: counts.copy()}
    for d, k in zip(degrees, steps):
        counts[d] -= 1
        counts[d + 1] += 1
        counts[1] += 1
        frames[int(k)] = counts.copy()
    return frames, n


def that_nonparametric(picks):
    """
    Bhamidi, Jin & Nobel non-parametric changepoint estimator (Paper B, Eq. 18).

        T_hat = inf { t >= 1/h_n :
                      sum_{i>=0} 2^-i |N_{i+1}(nt)/nt - N_{i+1}(n/h_n)/(n/h_n)| > 1/b_n }

    with the paper's defaults h_n = log(log n) and b_n = n^(1/log log n). It tracks the
    R-infinity distance between the current empirical degree distribution and an early
    reference one, flagging a changepoint at the first time the distance exceeds 1/b_n.
    Unlike `scan`, it gives only a location, no hypothesis test. Returns the estimated
    changepoint as a fraction t_hat in (0, 1), or nan if the distance never crosses.
    """
    frames, n = _degree_count_frames(picks)
    h = math.log(math.log(n))
    b = n ** (1.0 / h)
    threshold = 1.0 / b
    m0 = int(np.floor(n / h))
    if m0 < 2:
        m0 = 2
    m0 = min(m0, n)

    ref = frames[m0]
    weights = 0.5 ** np.arange(len(ref) - 1)  # 2^-i for degrees d = i + 1 >= 1

    for m in range(m0 + 1, n + 1):
        cur = frames[m]
        width = min(len(cur), len(ref))
        # degrees d = 1 .. width-1 (index 0 is unused: no degree-0 nodes)
        diff = np.abs(cur[1:width] / m - ref[1:width] / m0)
        dist = float(np.dot(weights[: width - 1], diff))
        if dist > threshold:
            return m / n
    return float("nan")
