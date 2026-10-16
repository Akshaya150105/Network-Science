import numpy as np
import pytest

from src.mle import loglik, loglik_window, mle, mle_window, score
from src.pa_sim import simulate_pa, simulate_pa_changepoint


def test_loglik_hand_checked():
    # log3 - log6 + log4 - log9 = -1.5041
    assert loglik(1, [(2, 3), (3, 4)]) == pytest.approx(-1.5041, abs=1e-3)


def test_score_near_zero_at_mle():
    picks = simulate_pa(2000, 1.0, np.random.default_rng(0))
    delta_hat = mle(picks)
    assert abs(score(delta_hat, picks)) < 1e-3
    # Score is positive to the left of the MLE and negative to the right.
    assert score(delta_hat - 0.5, picks) > 0
    assert score(delta_hat + 0.5, picks) < 0


def test_score_matches_numerical_derivative():
    picks = simulate_pa(300, 2.0, np.random.default_rng(1))
    h = 1e-6
    numeric = (loglik(1.5 + h, picks) - loglik(1.5 - h, picks)) / (2 * h)
    assert score(1.5, picks) == pytest.approx(numeric, rel=1e-4)


def test_mle_roughly_recovers_delta():
    picks = simulate_pa(2000, 1.0, np.random.default_rng(2))
    assert abs(mle(picks) - 1.0) < 0.3


def test_window_over_full_range_equals_full():
    picks = simulate_pa(100, 1.0, np.random.default_rng(3))
    assert loglik_window(1.0, picks, 2, 100) == pytest.approx(loglik(1.0, picks))
    assert mle_window(picks, 2, 100) == pytest.approx(mle(picks))


def test_window_loglik_is_additive():
    picks = simulate_pa(100, 1.0, np.random.default_rng(4))
    split = loglik_window(0.8, picks, 2, 50) + loglik_window(0.8, picks, 50, 100)
    assert split == pytest.approx(loglik(0.8, picks))


def test_window_uses_lo_exclusive_hi_inclusive():
    picks = [(2, 3), (3, 4), (1, 5)]
    assert loglik_window(1.0, picks, 3, 4) == pytest.approx(loglik(1.0, [(3, 4)]))


def test_window_mle_tracks_regime():
    n = 2000
    picks = simulate_pa_changepoint(n, 0.5, 5.0, n // 2, np.random.default_rng(5))
    d1 = mle_window(picks, 2, n // 2)
    d2 = mle_window(picks, n // 2, n)
    assert d2 > d1
