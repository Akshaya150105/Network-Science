# Method (statistics)

*Equation, theorem and table numbers follow the published IEEE version of Paper B (IEEE TNSE, vol. 13, 2026, pp. 344-359; author's accepted manuscript), with the arXiv v3 number (arXiv:2206.01076v3) in brackets where it differs. The full mapping is in `docs/project_documentation.md`.*

## 1. Model

We use the undirected affine preferential-attachment model of Paper B, Section III-A [arXiv: Section 2.1]. The seed graph G(2) has two nodes joined by two parallel edges, so both start with degree 2. At each step k = 3, ..., n a new node attaches to exactly one existing node v with probability

    P(v) = (D_v(k-1) + delta) / ((2 + delta)(k - 1)),        delta > -1        (Paper B, Eq. 1)

The denominator is the sum of the weights, because the total degree at step k is 2(k-1). Larger delta flattens the attachment probabilities (closer to uniform); delta near -1 concentrates attachment on hubs. With a changepoint at tau, steps k <= tau use delta1 and steps k > tau use delta2, and the graph carries over (Paper B, Section IV: PA(t*; delta1, delta2) [arXiv: Section 3]).

Each run is stored as `picks = [(chosen_degree, k)]` for k = 3..n, where `chosen_degree` is the chosen node's degree *before* the new node attaches. This is all the likelihood needs. In the LLM runs every existing node is shown to the model, so the candidate set is never a subset and the likelihood is exact.

## 2. Likelihood and estimator

For the steps in a window (lo, hi], the log-likelihood of delta is (Paper B, the display below Eq. 10 [arXiv: Eq. 11])

    l(delta) = sum log(d_k + delta) - sum log(k - 1) - m log(2 + delta)

with m the number of steps in the window and d_k the chosen node's degree. The middle term does not depend on delta and cancels in every likelihood ratio. The score is sum 1/(d_k + delta) - m/(2 + delta) (Eq. 11 [arXiv: Eq. 12]). The maximum-likelihood estimate is found by bounded scalar optimisation over (-0.99, 50); the score is about zero at the optimum (checked in `tests/test_mle.py`).

Two properties of this likelihood matter for interpretation:

* A pick of a degree-2 node contributes log(2 + delta) - log(2 + delta) = 0 for every delta. Such picks carry no information about delta.
* If every pick lands on the hub (or every pick lands on a leaf) the likelihood is monotone in delta and the estimate runs to a search bound (-0.99 or 50). We report the number of runs where an estimate sits on a bound; it signals a model mismatch, not a real estimate.

Paper B proves the windowed MLE stays consistent after a changepoint despite long-range dependence (Theorem 4.3 [arXiv: 3.3]), which is what allows delta2 to be estimated from the post-change window alone.

## 3. Changepoint test

For a split m in [floor(gamma n), floor((1 - gamma) n)] with trim gamma = 0.1, the likelihood ratio (Eq. 22 [arXiv: Eq. 20]) compares one delta for all steps with separate delta's either side of m. We compute the statistic

    stat(m) = 2 * ( l_{k<=m}(delta1_hat) + l_{k>m}(delta2_hat) - l_{all}(delta_hat) )

The test statistic is the maximum over m and the changepoint estimate tau_hat is the maximising m (Theorem 4.5, Eq. 24 [arXiv: Theorem 3.5, Eq. 22]: tau_hat / n converges to the true location). Under the null of no changepoint, Theorem 4.2 (Eq. 23 [arXiv: Theorem 3.2, Eq. 21]) gives the limit

    sup_m stat(m)  ->  sup_{t in [gamma, 1-gamma]} B(t)^2 / (t (1 - t)),        B = Brownian bridge.

For speed, the implementation (`src/changepoint.py`) groups the picks by degree and uses cumulative degree counts, so each split costs O(distinct degrees). All splits are solved together by bisecting the score, which replaces one Brent search per split; it agrees with the Brent solver (never worse, and more accurate where the maximum sits on a bound) and is 20 to 90 times faster, so every split is scanned at every n, as in Paper B. Both are checked in `tests/test_changepoint.py`.

**Ties.** Because degree-2 picks leave the statistic exactly flat, the maximiser is not always unique. Plain argmax then resolves the tie by floating-point noise (about 1e-13), which differs between machines. We therefore take the leftmost split within 1e-9 of the maximum, so tau_hat is reproducible. The test statistic itself is unaffected.

## 4. Calibration

The asymptotic 5% critical value of the Brownian-bridge supremum on [0.1, 0.9] was simulated with 100,000 replicates on a 10,000-point grid and equals 9.22. Paper B reports that the asymptotics are accurate at 50,000 edges; our runs have 500 to 2,000, so we calibrate by Monte Carlo as well. We simulate 5,000 no-change networks (delta = 1) at each n, scanning every split, take the 95th percentile of the statistic, and then estimate the Type I error on a fresh set of 500 networks. The grid step is stored next to each value in `results/critical_values.json` so the two cannot be separated. The three Monte Carlo values (9.19, 9.20 and 9.02; bootstrap standard error 0.12 to 0.17) are all within 1.2 standard errors of the asymptotic 9.22, so Paper B's own critical value would have served at these sizes too; we keep the Monte Carlo values because they are calibrated at the exact n and grid. (An earlier version scanned every 5th split at n = 1000 and 2000, which lowered the maximum and gave 8.85 and 8.76; that was a speed shortcut and has been removed.)

| n | grid step | MC critical value | asymptotic | Type I (MC value) | Type I (asymptotic value) |
|---|---|---|---|---|---|
| 500 | 1 | 9.193 | 9.223 | 0.048 | 0.048 |
| 1000 | 1 | 9.201 | 9.223 | 0.044 | 0.044 |
| 2000 | 1 | 9.017 | 9.223 | 0.032 | 0.028 |

All Type I errors with the Monte Carlo value lie in the target range 0.02 to 0.09 (standard error about 0.01 with 500 networks). The calibration is at delta = 1; Controls, section 2 checks it at delta = -0.5, 0, 1 and 2.

## 5. Reporting conventions

* **Location error.** |tau_hat - tau| / n. Paper B (Fig. 1, same in both versions) averages it only over runs that rejected the null; we report both that and the average over all runs, because at n = 500 power is low and the two differ.
* **Delta estimates.** The window MLEs either side of tau_hat are selection-biased (tau_hat is chosen to make the two sides differ) and heavy-tailed, so we report medians and the number of bound hits next to the means, and we also report the estimates at the *known* tau, which the LLM experiments allow.
* **Stub runs.** Runs from the stub chooser or test fixtures are labelled `source = stub` and never enter a final table.

## 6. Non-parametric competitor

As in Paper B Section IV-D [arXiv: Section 3.4] we compare tau_hat with the estimator T_hat of Banerjee, Bhamidi and Carmichael (2023) (Paper B Eq. 20, their reference [11]; arXiv: Eq. 18, [2]), which tracks the distance between the current empirical degree distribution and an early reference one. T_hat needs a reference time n / h_n with h_n = log log n, so it can only detect changes after 0.547 n (n = 500), 0.517 n (n = 1000) or 0.493 n (n = 2000). It is therefore unusable at n = 500 for a change at n/2 and is run only at n = 2000.
