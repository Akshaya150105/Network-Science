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
