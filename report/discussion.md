# Discussion: how to read the results

*Statistics-side discussion. Everything marked **[FILL AFTER MERGE]** needs the real LLM runs (Akshaya's logs and result files); no LLM number has been invented here. Paste the numbers from `results/comparison_table.csv` and `report/tables_statistics.md`, which `python -m src.analysis_stats` regenerates.*

## 1. What the classical benchmark says about the test

On networks that follow the model exactly, the test has the intended size (Type I error 0.03 to 0.064 across delta = -0.5 to 2 and n = 500 to 2000) and its power and location accuracy improve with n (Controls, sections 2 and 3). At the LLM run length n = 500 it is only moderately powerful: a shift from delta = 1 to 3 is detected 28% of the time and to 5 about 62% of the time (Controls, section 4). Every LLM result must be read against that curve, not against an ideal detector.

## 2. A decision guide for each LLM condition

The test assumes the network is exactly PA(delta) before and after the shift. The control runs (no shock) tell us whether that assumption is tolerable, so they are read first.

| Step | Question | What to compare |
|---|---|---|
| 1 | Is the test calibrated on LLM data? | No-shock control rejection rate with its binomial SE, against 0.05 (classical value at n = 500: 0.048). |
| 2 | Is a shock detected? | Persona and temperature rejection rates, against the classical power at a comparable effect size. |
| 3 | Is it located correctly? | Location error over rejected runs, against the classical 0.065 at n = 500 for 0.5 -> 3 (0.12 to 0.26 for smaller shifts). |
| 4 | Does delta move the way the intervention predicts? | delta2_hat - delta1_hat at the *known* tau (sign, size, how many of the runs), not at tau_hat. |

**Step 1 first.** A no-shock rejection rate well above 0.05 means the LLM network is not exactly PA(delta) (for example it has persistent hubs, position effects, or non-stationary behaviour within a single setting). In that case a rejection in a shock run is not by itself evidence that the intervention was detected, and every other conclusion carries that caveat. **[FILL AFTER MERGE: control rejection rate = ___ +- ___ over ___ runs; calibrated / not calibrated.]**

## 3. Reading each outcome

* **LLM changepoints are detected like classical PA** (rejection rate and location error comparable to the classical row, control near 0.05, delta moving in the predicted direction): LLM-grown networks behave, under this test, like classical preferential attachment. That is evidence that a PA model is an adequate description of the degree-based part of the agents' choices, not proof that they are PA processes. **[FILL: persona-shift rejection ___, location error ___, median delta1_hat ___ -> delta2_hat ___.]**
* **Detection is worse than classical, or delta is biased:** the agents add variance or dependence that the PA likelihood does not model (for example position bias in the candidate list, or random fallbacks after parse failures). Report the parse-failure fraction of each run, which is stored in the results (`parse_fail_frac`); fallbacks are uniform random choices and push delta up. **[FILL.]**
* **Temperature alone moves delta (delta2 > delta1 after the shift from low to high temperature):** this matches Paper A's observation that higher temperature gives a larger power-law exponent (gamma = 3 + delta in the PA model), and would suggest temperature acts as a tuning parameter for preferential-attachment strength. The direction is reported even when it is not significant, and the sign test and Wilcoxon test on delta2 - delta1 at the known tau are given in `results/temperature_shift_summary.csv`. **[FILL: delta2 > delta1 in ___ / ___ runs; sign-test p = ___; rejection rate ___.]**
* **No temperature effect, with a calibrated control:** still a valid result. Given the power curve, say how large a shift the design could have detected (Controls, section 4) rather than claiming the absence of an effect.

## 4. Cautions that apply to every comparison

* **Low power at n = 500.** A non-rejection on one run is weak evidence. Prefer the pooled delta estimates at the known tau and the fraction of runs moving in the predicted direction.
* **Selection bias in delta1_hat and delta2_hat at tau_hat.** They are measured at the split that maximises the statistic, so they overstate the difference; use the known-tau estimates and medians for conclusions.
* **Bound hits.** A delta_hat on the bound -0.99 or 50 means the likelihood has no interior optimum (all picks on hubs, or none). Count them and treat them as a model-mismatch signal. **[FILL: ___ of ___ runs.]**
* **Critical value provenance.** Confirm that the `crit_source` column of every LLM results file is `mc`; an `asymptotic` value is not calibrated at n = 500.
* **Look at the runs, not only the test.** Overlay the rolling-window delta_hat of all runs of a condition (`experiments/08_rolling_delta.py --prefix ...`). Compare the no-shock control with the classical no-shock panel (median flat near the true value, between-run sd about 0.7 for a 100-pick window); extra drift or scatter is non-stationarity. Compare a shock condition with the classical shock panel; a rise that falls back is a fading effect that a single changepoint describes badly. **[FILL AFTER MERGE: what the real overlays show.]**
* **Misspecification looks like change.** Paper B (published version, Section IV-D) warns that when attachment is not PA(delta), for example near-uniform, the fitted delta can drift as the network grows and mimic a changepoint. Treat a rejection as a shift in the fitted delta, and rely on the no-shock control to say how often that happens.
* **Few replicates.** Ten runs per condition give wide intervals (a rejection rate of 0.5 has SE 0.16).
