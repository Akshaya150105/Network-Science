import numpy as np
from scipy.optimize import minimize_scalar

DELTA_BOUNDS = (-0.99, 50.0)


def _arrays(picks):
    """Split picks [(chosen_degree, k), ...] into degree and step arrays."""
    arr = np.asarray(picks, dtype=float).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def loglik(delta, picks):
    """
    Log-likelihood of delta (Paper B, Eq. 10-13, two-node seed).

    sum(log(d + delta)) - sum(log(k - 1)) - m * log(2 + delta)
    """
    d, k = _arrays(picks)
    m = len(d)
    return np.sum(np.log(d + delta)) - np.sum(np.log(k - 1)) - m * np.log(2 + delta)


def score(delta, picks):
    """Derivative of loglik with respect to delta: sum(1/(d + delta)) - m/(2 + delta)."""
    d, _ = _arrays(picks)
    return np.sum(1.0 / (d + delta)) - len(d) / (2 + delta)


def mle(picks):
    """Maximum likelihood estimate of delta, searched within DELTA_BOUNDS."""
    if len(picks) == 0:
        raise ValueError("mle needs at least one pick")
    result = minimize_scalar(
        lambda delta: -loglik(delta, picks),
        bounds=DELTA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(result.x)


def _window(picks, lo, hi):
    """Picks with lo < k <= hi."""
    return [(d, k) for d, k in picks if lo < k <= hi]


def loglik_window(delta, picks, lo, hi):
    """Log-likelihood using only the steps with lo < k <= hi."""
    return loglik(delta, _window(picks, lo, hi))


def mle_window(picks, lo, hi):
    """MLE of delta using only the steps with lo < k <= hi."""
    return mle(_window(picks, lo, hi))
