import numpy as np
import pytest

from src.changepoint import (_fit, _fit_many, _fit_reference, scan, scan_curve, that_nonparametric,
                             that_reference_fraction, window_mle_curve)
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
    # leftmost split within the tie tolerance of the maximum
    assert tau_hat == m_values[np.flatnonzero(stats >= stats.max() - 1e-9)[0]]


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


def test_that_nonparametric_locates_strong_change():
    # On a strong shift the estimator lands near the true changepoint fraction 0.5.
    n = 2000
    ests = np.array(
        [that_nonparametric(simulate_pa_changepoint(n, 0.5, 5.0, n // 2, np.random.default_rng(s)))
         for s in range(10)]
    )
    assert np.isnan(ests).sum() <= 2
    assert abs(np.nanmean(ests) - 0.5) < 0.1


def test_likelihood_beats_nonparametric_on_pa_data():
    # Paper B, Table V (arXiv v3: Table 2): the likelihood estimator has smaller location error than T_hat
    # when the PA model is correctly specified.
    n = 2000
    t_err, T_err = [], []
    for s in range(10):
        picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(s))
        t_err.append(abs(scan(picks, step=5)[1] / n - 0.5))
        est = that_nonparametric(picks)
        if not np.isnan(est):
            T_err.append(abs(est - 0.5))
    assert np.mean(t_err) < np.mean(T_err)


def test_that_nonparametric_returns_fraction_or_nan():
    est = that_nonparametric(simulate_pa_changepoint(500, 0.5, 3.0, 250, np.random.default_rng(0)))
    assert np.isnan(est) or 0.0 < est < 1.0


def test_degree_two_picks_leave_statistic_flat_and_scan_takes_leftmost():
    # A pick of a degree-2 node contributes log(2+delta) - log(2+delta) = 0, so moving such a pick
    # across the split cannot change the statistic. scan must then return the leftmost tied split.
    n = 500
    picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(40089))
    assert [d for d, k in picks if k in (250, 251)] == [2, 2]
    m_values, stats = scan_curve(picks, trim=0.1, step=1)
    at = {int(m): s for m, s in zip(m_values, stats)}
    assert at[249] == pytest.approx(at[250], abs=1e-9) == pytest.approx(at[251], abs=1e-9)
    stat, tau_hat = scan(picks, trim=0.1, step=1)
    assert tau_hat == 249  # leftmost of the plateau 249..251, not whichever float noise favours


def test_that_reference_fraction_makes_n500_unusable_for_midpoint_change():
    # Paper B default h_n = log log n: the reference time is 1/h_n of the run.
    assert that_reference_fraction(500) == pytest.approx(0.547, abs=1e-3)
    assert that_reference_fraction(500) > 0.5          # reference is after tau = n/2: T_hat cannot see it
    assert that_reference_fraction(2000) == pytest.approx(0.493, abs=1e-3)
    assert that_reference_fraction(2000) < 0.5


def test_vectorised_solver_matches_reference_brent_solver():
    # _fit_many bisects the score for all windows at once; _fit_reference is the original per-window Brent search.
    rng = np.random.default_rng(0)
    for _ in range(60):
        width = int(rng.integers(3, 40))
        counts = rng.integers(0, 12, size=width)
        counts[0] = 0
        if counts.sum() == 0:
            counts[1] = 1
        ll_new, _ = _fit(counts)
        ll_ref, _ = _fit_reference(counts)
        # Never worse than the Brent search; where they differ (maximum on a bound, where Brent stops about
        # 1e-6 short of it and the likelihood is steep) the vectorised solver is the more accurate one.
        assert ll_new >= ll_ref - 1e-9
        assert ll_new - ll_ref < 1e-3


def test_full_grid_curve_matches_direct_statistic_at_every_split():
    n = 200
    picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(5))
    m_values, stats = scan_curve(picks, trim=0.1, step=1)
    for m, s in list(zip(m_values, stats))[::7]:
        assert s == pytest.approx(_slow_stat(picks, int(m), n), abs=1e-5)


def test_fit_many_handles_bound_cases():
    leaf_only = np.array([[0, 40]])             # every pick a leaf -> attractiveness of hubs is nil -> upper bound
    hub_only = np.array([[0, 0, 0, 0, 0, 1]])   # a single pick of a degree-5 node -> lower bound
    assert _fit_many(leaf_only)[1][0] == pytest.approx(50.0, abs=1e-6)
    assert _fit_many(hub_only)[1][0] == pytest.approx(-0.99, abs=1e-6)


def test_window_mle_curve_matches_window_mle_from_mle_module():
    from src.mle import mle_window

    picks = simulate_pa_changepoint(400, 0.5, 3.0, 200, np.random.default_rng(3))
    out = window_mle_curve(picks, window=100, step=25)
    for kc, dh in zip(out["k_center"], out["delta_hat"]):
        lo = int(kc - 49.5) - 1          # window covers steps k = kc-49.5 .. kc+49.5
        hi = lo + 100
        assert dh == pytest.approx(mle_window(picks, lo, hi), abs=1e-4)


def test_window_mle_curve_shows_the_shift_and_reports_bounds_and_se():
    n = 1000
    picks = simulate_pa_changepoint(n, 0.5, 4.0, n // 2, np.random.default_rng(11))
    out = window_mle_curve(picks, window=200, step=50)
    first = out["delta_hat"][out["k_center"] < 400]
    second = out["delta_hat"][out["k_center"] > 700]
    assert first.mean() < 1.5 < second.mean()
    assert np.all(out["se"][~out["at_bound"]] > 0)
    leaf_only = [(1, k) for k in range(3, 120)]            # every pick a leaf -> upper bound
    flat = window_mle_curve(leaf_only, window=50, step=20)
    assert flat["at_bound"].all() and np.isnan(flat["se"]).all()


def test_window_mle_curve_rejects_bad_window():
    picks = simulate_pa_changepoint(100, 1.0, 1.0, 100, np.random.default_rng(0))
    with pytest.raises(ValueError):
        window_mle_curve(picks, window=500)
