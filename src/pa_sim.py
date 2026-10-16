import numpy as np


def attachment_probabilities(degrees, delta):
    """
    Calculate preferential-attachment probabilities.

    P(v) = (degree[v] + delta) / sum(degree[u] + delta)

    With the seed [2, 2] the sum equals (2 + delta) * (k - 1) at step k
    (Paper B, Eq. 10), because total degree is 2(k - 1).
    """
    degrees = np.asarray(degrees, dtype=float)

    weights = degrees + delta
    probabilities = weights / weights.sum()

    return probabilities


def simulate_pa_changepoint(n, delta1, delta2, tau, rng):
    """
    Simulate an affine preferential-attachment network with one changepoint.

    Seed graph: nodes 1 and 2 with degrees [2, 2] (two parallel edges).
    Each new node k = 3..n attaches to exactly one existing node.
    Steps k <= tau use delta1, steps k > tau use delta2; the graph carries over.

    Returns:
        picks: list of (chosen_degree, k) for k = 3..n, where chosen_degree
        is the chosen node's degree BEFORE the new node attaches. Length n - 2.
    """
    degrees = [2, 2]
    picks = []

    for k in range(3, n + 1):
        delta = delta1 if k <= tau else delta2
        probabilities = attachment_probabilities(degrees, delta)

        chosen_index = rng.choice(len(degrees), p=probabilities)

        picks.append((degrees[chosen_index], k))

        # The chosen existing node gains one degree.
        degrees[chosen_index] += 1

        # New node has degree 1.
        degrees.append(1)

    return picks


def simulate_pa(n, delta, rng):
    """Simulate a PA network with a single, constant delta (no changepoint)."""
    return simulate_pa_changepoint(n, delta, delta, n, rng)
