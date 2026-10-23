# plan_valincia.md: Statistics side (Valincia)

You are helping **Valincia**, a data science student, build HER half of a two-person network science project. Her teammate **Akshaya** builds the LLM side in a separate session and repository branch. Read this whole file first, then do **one step at a time**.

## 0. Working rules (follow strictly)

1. Do ONE step, run its acceptance test, then STOP and report: what you built, the test output, and any problems. Wait for the user to say "next" before starting the next step.
2. Never skip a step or start a later step early, except where a step below says to push something early.
3. Only work on the steps in this file and only edit files that Valincia owns (see the ownership table in the contract). Never edit Akshaya's files.
4. Keep code minimal and readable. Plain functions, no classes unless needed. Short comments only where the math is not obvious.
5. Fixed random seeds everywhere (`np.random.default_rng(seed)`).
6. If a test fails, fix it before reporting. If you cannot fix it, say so plainly instead of loosening the test.
7. Do not invent formulas from the papers. All equations you need are in the contract below. If a step needs an equation that is not written here (for example the optional T-hat statistic), ask the user to paste it.
8. Explain results with a small concrete number example when helpful.

## 1. Project context

**Title:** Changepoint Detection in Preferential Attachment Networks Formed by LLM Agents.

**Idea:** Grow a network where each new node asks an LLM which existing node to connect to. Halfway through, deliberately change something (persona prompt, temperature, or model). Then apply a likelihood-based changepoint test to see whether the change is detected and whether the estimated attachment parameter shifts as expected. Compare with a classical simulated PA network.

**Papers:** Paper A: Papachristou & Yuan (2024), *Network Formation and Dynamics Among Multi-LLMs*. Paper B: Cirkovic, Wang & Zhang (2025), *Likelihood-based Inference for Random Networks with Changepoints*.

**Your half:** everything that ANALYSES a network: the classical simulator, the likelihood and MLE, the changepoint test, its calibration, and the classical control experiment. Later (merge phase) you run the test on the temperature-shift data that Akshaya generates.

## Shared contract (IDENTICAL in both plan files; do not change it without telling the other member)

This is the only thing the two halves have in common. If you need to change anything here, stop and tell the user so they can tell the teammate.

### Model and likelihood (Paper B, Eq. 10-13, with a two-node seed)
- Affine preferential attachment, one edge per new node. **Seed graph G(2): nodes 1 and 2 with degrees `[2, 2]`** (two parallel edges, total degree 4, no self-loops). New nodes k = 3..n.
- Node k picks existing node v with probability `(D_v(k-1) + delta) / ((2 + delta) * (k - 1))`. The denominator equals the sum of degrees `2(k-1)` plus `delta * (k-1)`.
- `picks` = list of `(chosen_degree, k)` for k = 3..n, where `chosen_degree = D_v(k-1)` is the chosen node's degree BEFORE the new node attaches. Length is n-2.
- `loglik(delta, picks) = sum(log(d + delta)) - sum(log(k - 1)) - m * log(2 + delta)` with `m = len(picks)`. Score: `sum(1/(d + delta)) - m/(2 + delta)`. The `sum(log(k-1))` term does not depend on delta, so it cancels in every likelihood ratio.
- Changepoint at `tau`: steps `k <= tau` use `delta1`, steps `k > tau` use `delta2`. The graph carries over (seed graph for the second regime is G(tau)). Default `tau = n // 2`.
- Test statistic (Paper B Eq. 20): for a split `m` in `[floor(gamma*n), floor((1-gamma)*n)]`, `stat(m) = 2 * (ell_{k<=m}(d1_hat) + ell_{k>m}(d2_hat) - ell_all(d_hat))`. Overall `stat = max_m stat(m)`, `tau_hat = argmax_m` (Eq. 22 is the consistency result for `tau_hat / n`).
- Null distribution (Paper B **Theorem 3.2, Eq. 21**): `sup stat -> sup_{t in [gamma, 1-gamma]} B(t)^2 / (t(1-t))` where B is a Brownian bridge.

### Hand-checked numbers (use these as unit tests)
- delta = 1, degrees `[2, 2]`, step k = 3: each node has probability 1/2.
- After node 1 is picked, degrees `[3, 2, 1]`, step k = 4: probabilities 4/9, 3/9, 2/9.
- `loglik(1, [(2, 3), (3, 4)])` = log3 - log6 + log4 - log9 = **-1.5041** (tolerance 1e-3).

### Constants (frozen)
- gamma (trim) = 0.1. alpha = 0.05. delta search bounds (-0.99, 50). LLM run length n = 500 (classical runs also at 1000 and 2000). True changepoint `tau = n // 2`.
- Seeds: `np.random.default_rng(seed)`. Replicate r of any experiment uses `seed = base_seed + r`.

### Function signatures (frozen)
- `src/pa_sim.py`: `simulate_pa(n, delta, rng)`, `simulate_pa_changepoint(n, delta1, delta2, tau, rng)`, both return `picks`.
- `src/mle.py`: `loglik(delta, picks)`, `mle(picks)`, `loglik_window(delta, picks, lo, hi)` and `mle_window(picks, lo, hi)` (use the steps with `lo < k <= hi`).
- `src/changepoint.py`: `scan(picks, trim=0.1, step=1)` returns `(stat, tau_hat)`; `scan_curve(picks, trim=0.1, step=1)` returns arrays `(m_values, stat_values)`.
- `src/llm_growth.py`: `grow_network(n, prompt_schedule, temperature_schedule, model_schedule, run_id, seed, chooser="llm")`, `load_picks(run_id)` returns `picks` (all rows, including `parse_ok=false` rows), `load_meta(run_id)`.

### Log format (written by the LLM side, read by the statistics side)
- `logs/{run_id}.jsonl`: one line per step with `run_id, t, candidate_degrees, chosen_idx, chosen_degree, temperature, model, prompt_id, raw_response, parse_ok`. Candidates are ALL existing nodes (never a subset), so the likelihood above is exact.
- `logs/{run_id}.meta.json`: `{run_id, condition, n, seed, tau_true (null if no shock), model_schedule, prompt_schedule, temperature_schedule, chooser}`.
- `run_id = "{condition}_n{n}_s{seed}"`. Conditions: `baseline`, `persona`, `temp`, `tempctrl`, `modelswap`. Stub runs get the prefix `stub_` (for example `stub_persona_n500_s0`).
- Never edit a log by hand and never overwrite one; use a new run_id.

### Results CSV schema (one row per run; every experiment writes this)
`run_id, condition, source, n, tau_true, stat, crit, rejected, tau_hat, tau_err_frac, delta1_hat, delta2_hat, delta_full_hat`
- `source` is `classical`, `llm` or `stub`. `tau_err_frac = |tau_hat - tau_true| / n` (empty when `tau_true` is empty). `rejected` is 1 if `stat > crit`.

### Critical values file (written by the statistics side, read by the LLM side)
`results/critical_values.json`: `{"asymptotic": {"0.05": x}, "n500": {"0.05": x, "source": "asymptotic"|"mc", "n_sims": N}, "n2000": {...}}`. The first version uses `"source": "asymptotic"`; the final version overwrites it with `"source": "mc"`. Always read the `source` field and report it.

### Repo ownership (avoids merge conflicts)
| Owner | Files |
|---|---|
| Valincia (statistics side) | `src/pa_sim.py`, `src/mle.py`, `src/changepoint.py`, `src/analysis_stats.py`, `experiments/01_*`, `02_*`, `03_*`, `06_temperature_shift_analyze.py`, `tests/test_pa_sim.py`, `test_mle.py`, `test_changepoint.py`, `results/critical_values.json`, `results/classical_cp.csv`, `results/mle_validation.csv`, `results/power_curve_n500.csv`, `results/temperature_shift.csv` |
| Akshaya (LLM side) | `src/prompts.py`, `src/llm_growth.py`, `src/analysis_llm.py`, `experiments/04_*`, `05_*`, `06_temperature_shift_generate.py`, `06_temperature_control_analyze.py`, `tests/test_prompts.py`, `test_llm_growth.py`, `logs/`, `results/persona_shift.csv`, `results/temperature_control.csv` |
| Shared (append-only or agreed edits) | `requirements.txt`, `README.md`, `report/` (each writes only their own section files) |

Work on your own branch (`valincia/stats`, `akshaya/llm`) and merge into `main` by pull request. Do not edit files the other person owns; if you need a change there, tell the user.

## 2. Your steps (parallel phase, 60 points)

The step numbers match the original project plan so the two of you can talk about the same step.

### Step 1 / setup (shared groundwork, 0 points)
- You create the repo skeleton: folders `src tests experiments logs results figures report`, `requirements.txt` (numpy scipy pandas matplotlib networkx; Akshaya will append her LLM client and `powerlaw`), and a short `README.md`. API keys are environment variables only.
- **Acceptance:** `python -c "import numpy, scipy, pandas, matplotlib, networkx"` runs.

### Step 2 (finish): classical simulator, `src/pa_sim.py` (4 points)
- `pa_sim.py` already has `attachment_probabilities` and `simulate_pa`. Change the seed degrees from `[1, 1]` to `[2, 2]` (see the contract).
- Add `simulate_pa_changepoint(n, delta1, delta2, tau, rng)`: delta1 for k <= tau, delta2 after, degrees carried over.
- **Tests (`tests/test_pa_sim.py`):** the hand-checked probabilities in the contract (1/2, 1/2 and 4/9, 3/9, 2/9). `len(picks) == n - 2`.

### Step 3: likelihood and MLE, `src/mle.py` (12 points)
- `loglik`, `mle` (bounded scalar minimisation of the negative log-likelihood over (-0.99, 50) with `scipy.optimize.minimize_scalar`), `loglik_window`, `mle_window`, exactly as in the contract. The closed-form score can be used for a Newton step or to check that the score is about 0 at the MLE.
- Do NOT write `loglik_subset`: we always show all nodes, so it is not needed.
- **Tests (`tests/test_mle.py`):** `loglik(1, [(2, 3), (3, 4)])` is within 1e-3 of -1.5041. The score at `mle(picks)` is near 0 on a simulated network.
- **PUSH EARLY:** as soon as this passes, commit and push `mle.py`. Akshaya's baseline check (her Step 10) and the merge phase use it. Tell the user so they can tell her.

### Step 4: validate the MLE, `experiments/01_validate_mle.py` (6 points)
- For delta in {0.5, 1, 3}: simulate n = 2000 networks, 100 repeats each, compute the MLE each time. Save `results/mle_validation.csv` with mean, std and bias per delta.
- **Acceptance:** the mean of delta_hat is within about 0.1 (about 10% relative) of the true delta for each case. If not, debug the likelihood before continuing.

### Step 5: changepoint test, `src/changepoint.py` (13 points)
- `scan(picks, trim=0.1, step=1)` returns `(stat, tau_hat)` using the statistic in the contract; `scan_curve` returns the whole curve for plots. A coarse grid (`step=5` or `10`) is fine for speed at n = 2000; use a finer grid at n = 500.
- Use `loglik_window`/`mle_window`. The `sum(log(k-1))` term cancels, so you only need the sums of `log(d + delta)` and the step counts on each side.
- **Sanity test:** simulate delta1 = 0.5, delta2 = 3 at tau = n/2 with n = 2000. `tau_hat` should be within about 10% of n of the truth, and a no-change network should give a visibly smaller `stat`.

### Step 6: null calibration, `experiments/02_null_calibration.py` (12 points)
- **(a) Asymptotic critical value.** Simulate `sup_{t in [0.1, 0.9]} B(t)^2 / (t(1-t))` (Theorem 3.2, Eq. 21) with a fine grid (at least 10,000 points) and at least 20,000 replicates. Report the 95th percentile. A rough check I ran gave about 8.9, so treat roughly 9 as a sanity check, not as the answer.
- **(b) STUB, do this immediately after (a):** write `results/critical_values.json` with `"source": "asymptotic"` for `n500` and `n2000`, commit and push it. Akshaya's persona analysis reads this file, so this unblocks her before (c) finishes.
- **(c) Monte Carlo critical values.** Simulate no-change networks (delta = 1) at n = 500 and at n = 2000 (at least 500 simulations each; use 1000 for n = 500 if it is cheap). Take the 95th percentile of `stat`. Overwrite `critical_values.json` with `"source": "mc"` and the number of simulations.
- **(d) Type I error.** Use the Monte Carlo critical value on a fresh set of 200 no-change networks per n. Also report the Type I error using the asymptotic critical value, to show the finite-sample gap.
- **Acceptance:** Type I error with the Monte Carlo critical value is between 0.02 and 0.09.
- Do not expect to reproduce Paper B's Table 1 exactly: they use 50,000 edges, we use 500 to 2000.

### Step 7: classical control experiment, `experiments/03_classical_cp.py` (13 points)
- For n in {500, 1000, 2000}: delta 0.5 to 3 at tau = n/2, 100 repeats, using the Monte Carlo critical value for that n. Write `results/classical_cp.csv` in the contract schema.
- Also run a power curve at n = 500 (the LLM run length): delta1 = 1 and delta2 in {1, 1.25, 1.5, 2, 3, 5}, 100 repeats each, saved to `results/power_curve_n500.csv`. This tells us how large a shift is detectable at n = 500, which matters for interpreting the LLM results.
- Record rejection rate (power), mean absolute error of `tau_hat` as a fraction of n, and mean delta1_hat and delta2_hat, with standard errors.
- **Acceptance:** power is high at n = 2000 and the MAE falls as n grows. This is the correctness gate for everything that follows. Paper B's Table 2 numbers (100,000 edges) are not reachable; only the trend is.

## 3. Merge phase (20 points; start only when Akshaya's logs exist)

### M1: temperature-shift analysis, `experiments/06_temperature_shift_analyze.py` (6 points)
- For each run `temp_n500_s*` use `load_picks`, `scan`, and the `n500` critical value from `results/critical_values.json` (check `source == "mc"`). Write `results/temperature_shift.csv` in the contract schema. Expectation from Paper A: higher temperature gives a larger power-law exponent, which should show up as delta2 > delta1. Report the direction even if it is not significant.
- Watch for delta_hat hitting the bounds (-0.99 or 50): report how many runs do, since that signals a model mismatch rather than a real estimate.

### M2: classical table and classical-vs-LLM comparison, `src/analysis_stats.py` (4 points)
- Build the classical table (rejection rate, MAE of `tau_hat`, mean delta1_hat and delta2_hat, standard errors) and a comparison table that reads Akshaya's `persona_shift.csv`, your `temperature_shift.csv`, and her `temperature_control.csv`.

### M3: statistics figures (4 points)
- Null histogram with the critical value marked, power curves (n = 500 and the n sweep), and an example `scan_curve` for a classical run. Save to `figures/`.

### M4: report sections (6 points)
- Draft in `report/`: Method (statistics), Controls, Discussion (how to read the results; see section 5), Future Work, References. Limitations must mention small n compared to Paper B, a single LLM, and few replicates (Akshaya writes the data-side limitations; coordinate through the user).

## 4. Dependencies on Akshaya and stubs

| What | Direction | Needed when | Stub or fallback |
|---|---|---|---|
| Real LLM logs (`temp_n500_s*`) and `load_picks` | You need from Akshaya | Merge phase only (M1) | Until they exist, test your analysis scripts on **stub logs** (`stub_persona_*`, `stub_temp_*`) that Akshaya's `stub_pa` chooser produces, or on your own fixture (below). Her stub gives each regime a known fake delta, so your script should recover a changepoint near n/2. Stub results are labelled `source = stub` and must never go into the final tables. |
| Fixture logs in the contract format | Needed by you from about day 1 | Writing the analysis scripts | If Akshaya's fixtures are not pushed yet, create your own `tests/make_fixture.py` (about 20 lines) that runs `simulate_pa_changepoint` and writes a `.jsonl` plus `.meta.json` in the contract format, with a `fixture_` prefix. |
| `mle.py` | Akshaya needs it from you | Her Step 10 baseline check and the merge phase | Push it right after Step 3 (see above). |
| `results/critical_values.json` | Akshaya needs it from you | Her M1 persona analysis | Push the asymptotic stub right after Step 6(a). Overwrite with the Monte Carlo version after Step 6(c) and tell the user so Akshaya re-runs with the final value. |
| Results CSV schema | Both | Always | Frozen in the contract. |

You do NOT depend on Akshaya for Steps 2 to 7 at all.

## 5. How to interpret results (for the report)
- LLM changepoints detected like classical PA: LLM networks behave close to classical PA under this test.
- Detection worse or delta biased: LLM agents add variance or dependence beyond the classical model.
- Temperature alone shifts delta: temperature acts as a PA tuning parameter (novel).
- No temperature effect but the control is calibrated: still a valid, reportable result.
- The test assumes the network is exactly PA(delta). A rejection on LLM data with no shock (the control runs) means model misspecification, not a detected shock.

## 6. Report format after each step
Reply with exactly: **1. Done** (one or two sentences). **2. Test output** (the numbers). **3. Issues / decisions** (anything the user must know or decide, including anything that changes the contract). **4. Next:** the name of the next step, then wait.
