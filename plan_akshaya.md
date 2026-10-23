# plan_akshaya.md: LLM side (Akshaya)

You are helping **Akshaya**, a data science student, build HER half of a two-person network science project. Her teammate **Valincia** builds the statistics side in a separate session and repository branch. Read this whole file first, then do **one step at a time**.

## 0. Working rules (follow strictly)

1. Do ONE step, run its acceptance test, then STOP and report: what you built, the test output, and any problems. Wait for the user to say "next" before starting the next step.
2. Never skip a step or start a later step early, except where a step below says to push something early.
3. Only work on the steps in this file and only edit files that Akshaya owns (see the ownership table in the contract). Never edit Valincia's files, including `mle.py` and `changepoint.py`. Do not write your own copy of the MLE.
4. Keep code minimal and readable. Plain functions, no classes unless needed. Short comments only where the logic is not obvious.
5. Fixed random seeds everywhere (`np.random.default_rng(seed)`). API keys in environment variables only, never in code.
6. Never edit files in `logs/` by hand. Never overwrite a log; use a new run_id.
7. If a test fails, fix it before reporting. If you cannot fix it, say so plainly instead of loosening the test.
8. Do not invent formulas from the papers. If a step needs an equation that is not written in the contract, ask the user to paste it.
9. Explain results with a small concrete number example when helpful.

## 1. Project context

**Title:** Changepoint Detection in Preferential Attachment Networks Formed by LLM Agents.

**Idea:** Grow a network where each new node asks an LLM which existing node to connect to. Halfway through, deliberately change something (persona prompt, temperature, or model). Then apply a likelihood-based changepoint test to see whether the change is detected and whether the estimated attachment parameter shifts as expected. Compare with a classical simulated PA network.

**Papers:** Paper A: Papachristou & Yuan (2024), *Network Formation and Dynamics Among Multi-LLMs* (LLMs show preferential attachment, triadic closure, homophily; its degree-only prompt is Algorithm SI.2). Paper B: Cirkovic, Wang & Zhang (2025), *Likelihood-based Inference for Random Networks with Changepoints*.

**Your half:** everything that PRODUCES a network: the prompts, the LLM growth engine, the baseline and Paper A validation, and the generation of the persona-shift and temperature-shift data. Later (merge phase) you run the changepoint test on the persona-shift data and the no-shock control.

**Decisions to confirm with the user BEFORE Step 9:**
1. Which LLM provider and model, and the budget. The runs need roughly 15,000 calls in total, with prompts that list up to 500 node degrees. Use a small, cheap model and keep the cache on.
2. The temperature range the provider allows. The plan uses 0.2 then 1.2. If the provider caps temperature at 1.0, use 0.2 then 1.0 and tell the user so Valincia is told (it changes the experiment description, not her code).

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

The step numbers match the original project plan so the two of you can talk about the same step. The repo skeleton (folders, `requirements.txt`, README) is created by Valincia; pull it before you start and append to `requirements.txt` only (your LLM client, `powerlaw`).

### Step 8: prompts, `src/prompts.py` (5 points)
Three prompts, each a plain string template, plus `build_prompt(prompt_id, candidate_degrees)`:
- `baseline`: "You join a social network. Choose exactly one existing member to connect to. Reply with only the number."
- `popular`: baseline plus "Prefer members who already have many connections."
- `newer`: baseline plus "Prefer newer members with few connections."

Each shows a numbered list of ALL candidates with their current degree, in shuffled order (the shuffle is done by the caller, so the function stays pure).
- **Test (`tests/test_prompts.py`):** the prompt contains every candidate number and degree exactly once, and the three prompt ids produce different instruction text.

### Step 9: LLM growth engine, `src/llm_growth.py` (20 points)
`grow_network(n, prompt_schedule, temperature_schedule, model_schedule, run_id, seed, chooser="llm")`:
- Seed graph degrees `[2, 2]` (see the contract). For each new node t = 3..n: candidates are ALL existing nodes (never a subset), shuffled, labelled 1..m anonymously; call the LLM; parse the integer.
- On parse failure retry up to 3 times, then fall back to a uniform random choice and set `parse_ok = false`.
- Write one JSON line per step to `logs/{run_id}.jsonl` and a `logs/{run_id}.meta.json`, in the exact fields of the contract. Cache responses by prompt text + temperature + model so reruns are free. Add simple rate limiting and a cost counter.
- Schedules are functions of t, so one engine runs every experiment. Example: persona shift is `lambda t: "popular" if t <= tau else "newer"` with `tau = n // 2`.
- `load_picks(run_id)` returns `[(chosen_degree, t), ...]` for ALL rows (including `parse_ok=false`); `load_meta(run_id)` returns the meta dict. Valincia's code depends on these two functions, so keep the signatures exactly.
- **STUB, build this first inside this step:** `chooser="stub_pa"` makes no API call. It samples the choice from the affine PA rule with a fake delta chosen from the schedule values: `popular` = 0.5, `newer` = 6.0, `baseline` = 1.0 if temperature <= 0.7 else 2.5. It goes through the same code path as the real chooser (it returns the chosen label as a string so the parser is exercised) and writes `model = "stub"` and a `stub_` run_id prefix. These numbers are fake and exist only to test the pipeline.
- **HANDOFF, commit and push as soon as the stub works (target: first day):** two small fixture runs, `stub_persona_n200_s0` and `stub_temp_n200_s0`, plus `load_picks`/`load_meta`. Tell the user so Valincia can start her analysis scripts.
- **Tests (`tests/test_llm_growth.py`, all with the stub, no API cost):** log has n-2 lines; `chosen_degree` equals the degree of the chosen candidate; the replayed degrees equal the real degree sequence; `load_picks` has length n-2; rerun with the same seed gives the same log; a forced parse failure falls back and sets `parse_ok=false`.
- **Acceptance:** a 50-node real pilot (`experiments/04_llm_pilot.py`) runs and reports the parse-failure rate, average seconds per call, and the histogram of chosen index (to check position bias).

### Step 10: baseline run and Paper A validation (8 + 9 points)
- **Baseline (8):** one run `baseline_n500_s0` of 500 nodes, `baseline` prompt, fixed temperature 0.7 (`experiments/04_baseline_run.py`). Acceptance: parse failures under 2% and no strong position bias.
- **Paper A validation (9), in `src/analysis_llm.py` and `experiments/04_paper_a_validation.py`:**
  - log-log degree distribution of the final network;
  - top-k connection probability: at each step, is the chosen node among the k highest-degree candidates, compared with the random null k/m (the style of Paper A's Figure 1);
  - power-law exponent gamma_hat (use the `powerlaw` package) and a K-S comparison against a Barabasi-Albert graph of the same size (`networkx.barabasi_albert_graph(n, 1)`).
  - Acceptance: heavy-tailed degrees. Paper A found that degree-only prompts at low temperature give star-like networks; if that happens, report it rather than hide it.
- **Deferred check:** "delta_hat roughly stable over time" needs Valincia's `mle.py`. If it is already on `main`, add a rolling-window delta_hat plot; if not, leave it for the merge phase (M3). Do not write your own MLE.

### Step 11 (generation only): persona shift, `experiments/05_persona_shift.py` (9 points)
- Same model and temperature (0.7). Prompt `popular` for t <= 250, `newer` after. n = 500, 10 replicates, `persona_n500_s0` .. `s9`, `tau_true = 250` in the meta file. About 5,000 calls; support `--stub` for a free dry run and `--resume` so an interrupted campaign continues.
- Check each run: parse failures under 2%, and the fraction of steps choosing a top-degree candidate should differ between the two halves (a quick sanity check that the persona actually changed the behaviour).
- **Acceptance:** all 10 logs exist and pass the checks. **HANDOFF:** tell the user when done so Valincia and you can start the merge phase.

### Step 12 (generation only): temperature shift and the no-shock control, `experiments/06_temperature_shift_generate.py` (9 points)
- `baseline` prompt. Shift runs `temp_n500_s0..s9`: temperature 0.2 for t <= 250, 1.2 after (or 1.0 if the provider caps it; see section 1). Control runs `tempctrl_n500_s0..s9`: constant temperature 0.7, `tau_true = null`. About 10,000 calls. Support `--stub` and `--resume`.
- **Acceptance:** all 20 logs exist and pass the same checks. **HANDOFF:** tell the user when done; Valincia analyses the shift runs, you analyse the control.

### Step 13 (optional, about 8 points): model swap
- Only if the user asks. Model X for t <= 250, model Y after, `modelswap_n500_s*`. Same analysis as the others.

## 3. Merge phase (20 points; start only when Valincia has pushed `mle.py`, `changepoint.py` and the Monte Carlo `critical_values.json`)

### M1: persona-shift analysis, `experiments/05_persona_shift_analyze.py` (6 points)
- For each `persona_n500_s*` use `load_picks`, Valincia's `scan`, and the `n500` critical value from `results/critical_values.json`. Check that its `source` is `"mc"`; if it is `"asymptotic"`, say so in the output because that value is not calibrated at n = 500. Write `results/persona_shift.csv` in the contract schema. Expectation: delta2_hat > delta1_hat and a rejection in most runs. Report how many runs hit the delta bounds.

### M2: no-shock control, `experiments/06_temperature_control_analyze.py` (4 points)
- Apply the test to `tempctrl_n500_s*` and `baseline_n500_s0`. The rejection rate is the false-positive rate on LLM data; report it with a binomial standard error. Write `results/temperature_control.csv`. A rate well above 0.05 means the LLM network is not exactly PA(delta), and every other LLM result must be read with that in mind.

### M3: LLM figures, `src/analysis_llm.py` (4 points)
- Log-log degree distributions per condition, and delta_hat over time (rolling windows) with the true and detected changepoints marked. Save to `figures/`.

### M4: report sections (6 points)
- Draft in `report/`: Introduction, Background (Paper A and Paper B), Method (data generation), Results for persona and control, Limitations (small n compared to Paper B, all candidates shown so no subset effect, a single LLM, few replicates, any temperature cap). Valincia writes the statistics sections.

## 4. Dependencies on Valincia and stubs

| What | Direction | Needed when | Stub or fallback |
|---|---|---|---|
| `mle.py` (`mle_window`) | You need from Valincia | Step 10's delta_hat stability plot, and the whole merge phase | Do the stability plot at merge time (M3) if it is not pushed. No stub MLE; do not duplicate her code. |
| `changepoint.py` (`scan`) | You need from Valincia | Merge phase (M1, M2) | Test your analysis scripts on `stub_*` logs once it arrives; until then there is nothing to wait on in Steps 8 to 12. |
| `results/critical_values.json` | You need from Valincia | M1 and M2 | She pushes an **asymptotic placeholder** right after her Step 6(a) and the final Monte Carlo version after Step 6(c). Re-run M1 and M2 with the final one. |
| `stub_pa` chooser, `load_picks`, fixture logs | Valincia needs from you | Her analysis scripts, from about day 1 | You build them first in Step 9 and push the two fixture runs. |
| Real logs (`persona_*`, `temp_*`, `tempctrl_*`) | Valincia needs from you (`temp_*`) | Her merge phase (M1) | Tell the user as soon as each batch is complete. Stub logs are labelled `source = stub` and must never go into the final tables. |
| Contract (seed `[2, 2]`, log format, CSV schema) | Both | Always | Frozen. Your logs must replay to degrees that start at `[2, 2]`, or her likelihood is wrong. |

You do NOT depend on Valincia for Steps 8 to 12 (apart from the optional stability plot).

## 5. How to interpret results (for the report)
- LLM changepoints detected like classical PA: LLM networks behave close to classical PA under this test.
- Detection worse or delta biased: LLM agents add variance or dependence beyond the classical model.
- Temperature alone shifts delta: temperature acts as a PA tuning parameter (novel).
- No temperature effect but the control is calibrated: still a valid, reportable result.
- A "prefer newer" persona pushes delta toward large values, and a low-temperature degree-only prompt pushes it toward -1 (star-like). Expect some estimates at the bounds.

## 6. Report format after each step
Reply with exactly: **1. Done** (one or two sentences). **2. Test output** (the numbers). **3. Issues / decisions** (anything the user must know or decide, including anything that changes the contract). **4. Next:** the name of the next step, then wait.
