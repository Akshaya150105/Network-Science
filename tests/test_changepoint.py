import numpy as np
import pytest

from src.changepoint import scan, scan_curve
from src.mle import loglik_window, mle, mle_window
from src.pa_sim import simulate_pa, simulate_pa_changepoint


def _slow_stat(picks, m, n):
    """Direct statistic from mle.py, used to check the fast implementation."""
    d1 = mle_window(picks, 2, m)
    d2 = mle_window(picks, m, n)
    d_all = mle(picks)
    ell = loglik_window(d1, picks, 2, m) + loglik_window(d2, picks, m, n)
    return 2 * (ell - loglik_window(d_all, picks, 2, n))


def test_fast_curve_matches_direct_statistic():
    n = 300
    picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(0))
    m_values, stats = scan_curve(picks, step=40)
    for m, s in zip(m_values, stats):
        assert s == pytest.approx(_slow_stat(picks, int(m), n), abs=1e-3)


def test_curve_grid_and_scan_agree():
    n = 500
    picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(1))
    m_values, stats = scan_curve(picks, trim=0.1, step=5)
    assert m_values[0] == 50 and m_values[-1] <= 450
    assert np.all(stats >= 0)
    stat, tau_hat = scan(picks, trim=0.1, step=5)
    assert stat == stats.max()
    assert tau_hat == m_values[np.argmax(stats)]


def test_detects_changepoint_location():
    n = 2000
    picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(2))
    stat, tau_hat = scan(picks, step=10)
    assert abs(tau_hat - n // 2) <= 0.1 * n


def test_no_change_gives_smaller_stat():
    # Single draws are noisy (the null statistic has a heavy right tail), so compare averages.
    n = 2000
    changed = [
        scan(simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(s)), step=20)[0]
        for s in range(10)
    ]
    null = [scan(simulate_pa(n, 1.0, np.random.default_rng(s)), step=20)[0] for s in range(10)]
    assert np.mean(changed) > 2 * np.mean(null)
