import numpy as np

from src.pa_sim import attachment_probabilities, simulate_pa, simulate_pa_changepoint


def test_probabilities_step3():
    # delta = 1, degrees [2, 2], k = 3: each node has probability 1/2.
    p = attachment_probabilities([2, 2], 1.0)
    assert np.allclose(p, [1 / 2, 1 / 2])


def test_probabilities_step4():
    # After node 1 is picked, degrees [3, 2, 1], k = 4: 4/9, 3/9, 2/9.
    p = attachment_probabilities([3, 2, 1], 1.0)
    assert np.allclose(p, [4 / 9, 3 / 9, 2 / 9])


def test_probabilities_match_paper_b_denominator():
    # (D_v + delta) / ((2 + delta)(k - 1)) with k - 1 = number of nodes.
    degrees, delta = [3, 2, 1], 0.7
    p = attachment_probabilities(degrees, delta)
    k = len(degrees) + 1
    expected = (np.array(degrees) + delta) / ((2 + delta) * (k - 1))
    assert np.allclose(p, expected)


def test_length_is_n_minus_2():
    rng = np.random.default_rng(0)
    assert len(simulate_pa(100, 1.0, rng)) == 98
    assert len(simulate_pa_changepoint(100, 0.5, 3.0, 50, rng)) == 98


def test_picks_format_and_steps():
    picks = simulate_pa(50, 1.0, np.random.default_rng(1))
    assert [k for _, k in picks] == list(range(3, 51))
    assert picks[0][0] == 2  # first pick is one of the seed nodes, degree 2
    assert all(d >= 1 for d, _ in picks)


def test_same_seed_same_picks():
    a = simulate_pa_changepoint(200, 0.5, 3.0, 100, np.random.default_rng(7))
    b = simulate_pa_changepoint(200, 0.5, 3.0, 100, np.random.default_rng(7))
    assert a == b


def test_no_changepoint_matches_simulate_pa():
    a = simulate_pa(200, 1.5, np.random.default_rng(3))
    b = simulate_pa_changepoint(200, 1.5, 1.5, 100, np.random.default_rng(3))
    assert a == b


def test_changepoint_regimes_differ():
    # A small delta favours hubs strongly; a large delta is close to uniform.
    # Compare the mean degree of the chosen node in the second half.
    n = 2000
    low = simulate_pa_changepoint(n, 0.5, 0.5, n // 2, np.random.default_rng(0))
    shifted = simulate_pa_changepoint(n, 0.5, 20.0, n // 2, np.random.default_rng(0))
    mean_low = np.mean([d for d, k in low if k > n // 2])
    mean_shifted = np.mean([d for d, k in shifted if k > n // 2])
    assert mean_low > mean_shifted
