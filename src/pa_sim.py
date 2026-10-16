import numpy as np


def attachment_probabilities(degrees, delta):
    """
    Calculate preferential-attachment probabilities.

    P(v) = (degree[v] + delta) /
           sum(degree[u] + delta)
    """
    degrees = np.asarray(degrees, dtype=float)

    weights = degrees + delta
    probabilities = weights / weights.sum()

    return probabilities


def simulate_pa(n, delta, rng):
    """
    Simulate an affine preferential-attachment network.

    Starts with nodes 1 and 2 connected.
    Each new node attaches to exactly one existing node.

    Returns:
        picks: list of (chosen_degree, t)
    """
    degrees = [1, 1]
    picks = []

    for t in range(3, n + 1):
        probabilities = attachment_probabilities(degrees, delta)

        chosen_index = rng.choice(
            len(degrees),
            p=probabilities
        )

        chosen_degree = degrees[chosen_index]

        picks.append((chosen_degree, t))

        # The chosen existing node gains one degree.
        degrees[chosen_index] += 1

        # New node has degree 1.
        degrees.append(1)

    return picks