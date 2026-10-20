import math

import numpy as np
from scipy.optimize import minimize_scalar

from src.mle import DELTA_BOUNDS


def _fit_reference(counts):
    """
    REFERENCE (slow) solver, kept for tests: maximised log-likelihood and MLE of delta from degree counts
    for ONE count vector, by bounded Brent search.

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


BISECT_ITERS = 48  # bisection steps: the bracket (-0.99, 50) shrinks to about 1e-13


def _fit_many(counts):
    """
    Maximised log-likelihood and MLE of delta for EVERY ROW of a degree-count matrix at once.

    counts[r, d] = number of steps in window r that picked a node of degree d (column 0 unused).
    Same objective as `_fit_reference`: sum_d c_d log(d + delta) - m log(2 + delta), delta in DELTA_BOUNDS,
    the delta-free term sum(log(k - 1)) left out. Instead of one Brent search per row, the score
    u(delta) = sum_d c_d / (d + delta) - m / (2 + delta) is bisected for all rows together:
    if u(lower bound) <= 0 or u(upper bound) >= 0 the maximum sits on that bound, otherwise it is the
    sign change of u inside the bracket. The best of {root, both bounds} is returned, which also guards
    against a second local maximum. Returns (loglik, delta_hat), each of shape (rows,).
    """
    c = np.atleast_2d(counts)[:, 1:].astype(float)       # drop the unused degree-0 column
    d = np.arange(1, c.shape[1] + 1, dtype=float)
    m = c.sum(axis=1)
    lo_b, hi_b = DELTA_BOUNDS

    def score(delta):
        return (c / (d[None, :] + delta[:, None])).sum(axis=1) - m / (2.0 + delta)

    def loglik(delta):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (c * np.log(d[None, :] + delta[:, None])).sum(axis=1) - m * np.log(2.0 + delta)

    rows = len(m)
    lo = np.full(rows, lo_b)
    hi = np.full(rows, hi_b)
    u_lo, u_hi = score(lo), score(hi)
    a, b = lo.copy(), hi.copy()                           # invariant while bracketed: u(a) > 0 > u(b)
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (a + b)
        positive = score(mid) > 0
        a = np.where(positive, mid, a)
        b = np.where(positive, b, mid)
    root = 0.5 * (a + b)

    candidates = np.stack([root, lo, hi])                 # (3, rows)
    lls = np.stack([loglik(x) for x in candidates])
    # the root is only a candidate when the bracket really had a sign change
    bracketed = (u_lo > 0) & (u_hi < 0)
    lls[0] = np.where(bracketed, lls[0], -np.inf)
    best = np.argmax(lls, axis=0)
    idx = np.arange(rows)
    return lls[best, idx], candidates[best, idx]


def _fit(counts):
    """Maximised log-likelihood and MLE of delta for one degree-count vector (see `_fit_many`)."""
    ll, delta = _fit_many(np.asarray(counts)[None, :])
    return float(ll[0]), float(delta[0])


def scan_curve(picks, trim=0.1, step=1):
    """
    Likelihood-ratio statistic for every split m (Paper B, Eq. 22 in the IEEE version, Eq. 20 in arXiv v3).

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
    left_counts = cum[m_values - 2]                       # steps 3..m
    ell_left, _ = _fit_many(left_counts)
    ell_right, _ = _fit_many(cum[-1][None, :] - left_counts)  # steps m+1..n
    stats = 2 * (ell_left + ell_right - ell_all)

    # Numerical noise from the optimiser can give tiny negative values.
    return m_values, np.maximum(stats, 0.0)


def window_mle_curve(picks, window=100, step=10):
    """
    Rolling-window estimate of delta along one run: the MLE from each block of `window` consecutive picks.

    A diagnostic, not a test: it shows drift, a fading effect, or a second change that a single-changepoint
    test cannot describe. Window j covers picks j*step .. j*step + window - 1 (steps k = 3 + j*step onward).
    Returns a dict of arrays:
      k_center    centre step of each window
      delta_hat   window MLE (same solver and bounds as `scan`)
      se          sqrt(1 / observed information) at delta_hat; NaN when delta_hat sits on a search bound
      at_bound    True where delta_hat sits on a bound (-0.99 or 50), which signals mismatch, not a real value
    The standard error is only a rough guide: windows are short and overlap when step < window.
    """
    arr = np.asarray(picks, dtype=int).reshape(-1, 2)
    degrees = arr[:, 0]
    n_picks = len(degrees)
    if not 3 <= window <= n_picks:
        raise ValueError(f"window must be between 3 and the number of picks ({n_picks}), got {window}")
    onehot = np.zeros((n_picks, degrees.max() + 1), dtype=np.int32)
    onehot[np.arange(n_picks), degrees] = 1
    cum = np.vstack([np.zeros((1, onehot.shape[1]), dtype=np.int32), np.cumsum(onehot, axis=0)])

    starts = np.arange(0, n_picks - window + 1, step)
    counts = cum[starts + window] - cum[starts]
    _, delta = _fit_many(counts)

    c = counts[:, 1:].astype(float)
    d = np.arange(1, c.shape[1] + 1, dtype=float)
    m = c.sum(axis=1)
    info = (c / (d[None, :] + delta[:, None]) ** 2).sum(axis=1) - m / (2.0 + delta) ** 2  # = -u'(delta_hat)
    lo, hi = DELTA_BOUNDS
    at_b = (delta <= lo + 0.01) | (delta >= hi - 0.01)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where((info > 0) & ~at_b, 1.0 / np.sqrt(info), np.nan)
    return {"k_center": 3 + starts + (window - 1) / 2.0, "delta_hat": delta, "se": se, "at_bound": at_b}


TIE_TOL = 1e-9


def scan(picks, trim=0.1, step=1):
    """
    Return (stat, tau_hat): the largest statistic over splits and where it occurs.

    A pick of a degree-2 node adds log(2 + delta) - log(2 + delta) = 0 to the log-likelihood for every
    delta, so runs of such picks leave the statistic exactly flat and the maximiser is not unique.
    Plain argmax then breaks the tie with floating-point noise (about 1e-13), which differs between
    machines. To make tau_hat reproducible we take the LEFTMOST split whose statistic is within
    TIE_TOL of the maximum.
    """
    m_values, stats = scan_curve(picks, trim=trim, step=step)
    best = int(np.flatnonzero(stats >= stats.max() - TIE_TOL)[0])
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


def that_reference_fraction(n):
    """
    Fraction of the run, 1 / h_n with h_n = log(log(n)), at which T_hat takes its reference
    degree distribution (and starts scanning).

    T_hat can only see a changepoint that happens AFTER this point. With the paper's default
    h_n this is 0.547 at n = 500, 0.517 at n = 1000 and 0.493 at n = 2000, so for a changepoint
    at n/2 the estimator is unusable at n = 500 (the reference is already post-change) and only
    just usable at n = 2000.
    """
    return 1.0 / math.log(math.log(n))


def that_nonparametric(picks):
    """
    Non-parametric changepoint estimator of Banerjee, Bhamidi & Carmichael (2023), as given in
    Paper B Eq. 20 (IEEE version; Eq. 18 in arXiv v3), Paper B's reference [11] (arXiv v3: [2]); NOT
    Bhamidi, Jin & Nobel (2018) (reference [12]), which tracks only the proportion of leaves.
    Paper B notes that this estimator was designed for out-degree PA but stays valid for linear PA on
    total degree, with the delta parameters differing by 1, which is how it is used here.

        T_hat = inf { t >= 1/h_n :
                      sum_{i>=0} 2^-i |N_{i+1}(nt)/nt - N_{i+1}(n/h_n)/(n/h_n)| > 1/b_n }

    with the paper's defaults h_n = log(log n) and b_n = n^(1/log log n). It tracks the
    R-infinity distance between the current empirical degree distribution and an early
    reference one, flagging a changepoint at the first time the distance exceeds 1/b_n.
    Unlike `scan`, it gives only a location, no hypothesis test. Returns the estimated
    changepoint as a fraction t_hat in (0, 1), or nan if the distance never crosses
    (the infimum of an empty set is +infinity; callers must decide how to score that).

    Limitation: only changepoints after `that_reference_fraction(n)` can be detected.
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
