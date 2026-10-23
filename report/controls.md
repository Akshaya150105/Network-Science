# Controls: the classical preferential-attachment benchmark

Before the test is applied to LLM-grown networks it is checked on networks simulated from the model it assumes, where the truth is known. All numbers below are from `results/` (100 runs per cell unless stated; seeds 40000 + r).

## 1. The estimator recovers delta

For n = 2000 and 100 networks per setting, the MLE is close to the truth, with a small positive finite-sample bias (`results/mle_validation.csv`).

| true delta | mean estimate | std | bias |
|---|---|---|---|
| 0.5 | 0.519 | 0.113 | +0.019 |
| 1.0 | 1.025 | 0.147 | +0.025 |
| 3.0 | 3.084 | 0.458 | +0.084 |

The spread of the estimates grows with delta, as expected: for large delta the attachment is close to uniform and each pick carries less information about delta.

## 2. The test has the right size

At the Monte Carlo critical values (every split scanned) the Type I error is 0.048 (n = 500), 0.044 (n = 1000) and 0.032 (n = 2000) on 500 fresh networks; see Method, section 4. Using the asymptotic value 9.22 instead gives 0.048, 0.044 and 0.028.

The calibration is at delta = 1, so it was also checked at the four delta values of Paper B's Table IV on fresh networks (`results/type1_across_delta.csv`; 1,000 networks at n = 500, 500 at the other sizes; Monte Carlo critical value; standard error 0.006 to 0.011):

| delta | n = 500 | n = 1000 | n = 2000 | Paper B (50,000 edges) |
|---|---|---|---|---|
| -0.5 | 0.060 | 0.050 | 0.048 | 0.050 |
| 0 | 0.041 | 0.038 | 0.038 | 0.042 |
| 1 | 0.056 | 0.064 | 0.030 | 0.046 |
| 2 | 0.047 | 0.054 | 0.042 | 0.052 |

All values lie between 0.03 and 0.064. The 0.030 at n = 2000, delta = 1 (and the 0.032 above) looked low, so we checked it with 1,800 further networks (`results/type1_n2000_extra_check.csv`): 0.056 with the Monte Carlo value and 0.050 with the asymptotic one, and about 0.047 pooled over all 2,800 networks, so the low readings were chance. The test therefore has about the right size at these run lengths, for negative as well as positive delta.

## 3. Detection and location accuracy grow with n

Shift delta 0.5 -> 3 at tau = n/2 (`results/classical_cp.csv`). Location error is |tau_hat - tau| / n.

| n | rejection rate (SE) | location error, all runs | location error, rejected runs only | median delta1_hat / delta2_hat at tau_hat | median delta1_hat / delta2_hat at the true tau | runs with an estimate on a bound |
|---|---|---|---|---|---|---|
| 500 | 0.64 (0.048) | 0.088 | 0.065 | 0.41 / 3.16 | 0.57 / 2.89 | 2 |
| 1000 | 0.98 (0.014) | 0.041 | 0.041 | 0.44 / 3.23 | 0.49 / 3.01 | 0 |
| 2000 | 1.00 (0.000) | 0.016 | 0.016 | 0.47 / 3.07 | 0.49 / 3.03 | 0 |

Power approaches one and the location error falls as n grows, as the consistency results (Theorem 4.5 [arXiv: 3.5]) predict; the 100,000-edge numbers of Paper B are not reachable at this size, only the trend. At n = 500 the *mean* of delta2_hat at tau_hat is 4.50 against a median of 3.16: two runs hit the upper search bound, which is why we report medians.

## 4. How large a shift is detectable at n = 500 (the LLM run length)

delta1 = 1 and delta2 varied, change at tau = 250 (`results/power_curve_n500.csv`). The delta2 = 1 row has no changepoint and is the Type I error (0.08 here with 100 runs, SE 0.027, consistent with 0.05).

| delta2 | 1 (no change) | 1.25 | 1.5 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| rejection rate | 0.08 | 0.12 | 0.11 | 0.15 | 0.28 | 0.62 |
| SE | 0.027 | 0.032 | 0.031 | 0.036 | 0.045 | 0.049 |
| location error, rejected runs | n/a | 0.226 | 0.256 | 0.173 | 0.124 | 0.089 |
| median delta2_hat at the true tau | 0.96 | 1.21 | 1.41 | 1.84 | 2.90 | 4.65 |

**Consequence for the LLM experiments.** With 500 nodes a change from delta = 1 to delta = 3 is detected only 28% of the time, and a change to delta = 5 about 62% of the time. A failure to reject on a single LLM run at n = 500 therefore does not show that the behaviour did not change; only a large shift is reliably detectable, and the delta estimates at the known changepoint are more informative than the yes/no rejection.

## 5. Comparison with the non-parametric estimator (n = 2000, delta1 = 0.5)

Mean absolute error of the changepoint location, as a fraction of n (`results/nonparam_compare.csv`). T_hat never crossing its threshold counts as no detection; we show the error over crossing runs only and with non-crossing runs scored as T_hat = 1.

| true t* | delta2 | likelihood t_hat | T_hat (crossing runs) | T_hat (non-crossing scored 1) | runs with no crossing |
|---|---|---|---|---|---|
| 0.5 | 1.5 | 0.074 | 0.130 | 0.156 | 7 / 100 |
| 0.5 | 3 | 0.022 | 0.074 | 0.074 | 0 |
| 0.5 | 5 | 0.010 | 0.052 | 0.052 | 0 |
| 0.6 | 1.5 | 0.073 | 0.141 | 0.169 | 11 / 100 |
| 0.6 | 3 | 0.020 | 0.084 | 0.087 | 1 |
| 0.6 | 5 | 0.011 | 0.067 | 0.067 | 0 |

The likelihood estimator has the smaller error in every cell, in line with Paper B's finding (Table V [arXiv: Table 2]) that a correctly specified likelihood beats the non-parametric estimator on PA data. At t* = 0.5 the reference time of T_hat (0.493 n) sits just before the change, which handicaps it; t* = 0.6 is the like-for-like comparison with the paper.

## 6. A rolling-window diagnostic

The test asks one question (is there one changepoint?). To see what a run looks like in between, `experiments/08_rolling_delta.py` plots the MLE of delta from each block of 100 consecutive picks (`figures/fig_rolling_delta_classical.png`; 20 runs per panel, n = 500). It is a diagnostic, not a test.

* With a shock (delta 0.5 -> 3 at step 250) the median curve over runs moves from about 0.46 before the change to about 2.50 after it; the median of a 100-pick window underestimates 3 because the windows near the change straddle it.
* With no shock (delta = 1) the median stays between 0.84 and 1.09.
* A single window estimate is noisy: the between-run standard deviation of a window estimate is about 0.94 in the shock panel and 0.66 in the no-shock panel. A single run's curve therefore says little, and the plot is only useful as an overlay of many runs with their median.

The same script reads real or fixture logs (`--prefix`, `--logs-dir`). A no-shock control whose median drifts, or whose runs scatter more than the classical panel, would point to non-stationary behaviour, and a shock run whose curve rises and falls back would point to a fading effect. Either case is the signal that the multiple-changepoint methods of Paper B (Section V) could be worth building.

## 7. Figures

`figures/fig_null_hist.png` (null distribution and critical values), `figures/fig_power_curve_n500.png`, `figures/fig_n_sweep.png`, `figures/fig_scan_curve_example.png`, `figures/fig_rolling_delta_classical.png`.
