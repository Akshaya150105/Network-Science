# Project Documentation
## Changepoint Detection in Preferential Attachment Networks Formed by LLM Agents

**Student:** Achu | **Program:** MSc Data Science, PSG College of Technology
**Course:** 20XD92/20XD98 Network Science

---

## 1. Project Title

**Primary title:** Changepoint Detection in Preferential Attachment Networks Formed by LLM Agents

**Alternate titles (pick based on faculty preference):**
- Likelihood-Based Changepoint Detection in LLM-Agent-Formed Preferential Attachment Networks
- From Network Formation to Network Change: Applying Rigorous Changepoint Detection to LLM-Agent Social Networks
- Do LLM Agents Remember Their Personality? Changepoint Detection in AI-Formed Social Networks

---

## 2. One-Paragraph Summary

Large language model (LLM) agents have recently been shown to reproduce classical network-formation laws — preferential attachment, triadic closure, homophily — when asked to decide who to connect to in a simulated social network. Separately, a rigorous statistical framework now exists to formally detect *when* a preferential-attachment network's growth behavior shifts mid-evolution (a "changepoint"), with proven mathematical guarantees. No existing work combines the two. This project builds LLM-agent-grown networks, deliberately introduces a controlled "shock" partway through their growth (changing the LLM's model, persona, or sampling temperature), and applies the rigorous changepoint-detection framework to test whether the shock is correctly detected — and whether the recovered shift in the attachment parameter matches what the intervention predicts. The result either validates that LLM-agent networks are statistically indistinguishable from classical mathematical preferential-attachment processes, or reveals a measurable gap between the two — both are genuine, citable findings.

---

## 3. The Two Papers Being Extended

### Paper A — The mechanism being tested (data-generating process)

**Citation:** Papachristou, M. & Yuan, Y. (2024). *Network Formation and Dynamics Among Multi-LLMs.* arXiv:2402.10659. Published version: PNAS Nexus, 4(12), pgaf317 (2025).

**What it does:**
- Builds a framework where multiple LLM agents (GPT, Claude, Llama families) are asked, one at a time, to choose who to "befriend" from a set of candidate profiles.
- Benchmarks the resulting network-formation behavior against real human decisions, across three settings: friendship networks, telecommunication networks, and employment/organizational networks.
- Finds that LLM agents consistently reproduce three micro-level principles:
  - **Preferential attachment** — agents favor connecting to already well-connected nodes.
  - **Triadic closure** — agents favor connecting to friends-of-friends.
  - **Homophily** — agents favor connecting to similar agents.
- Also finds macro-level properties emerge naturally: community structure and the small-world phenomenon (consistent with the Watts-Strogatz model covered in your course).
- Critically, the *relative weight* LLMs give to these three principles is context-dependent: LLMs favor homophily in friendship settings but favor heterophily (connecting to different others) in organizational settings — mirroring real patterns of human social mobility.

**What it does NOT do (the gap you are closing):**
- It only evaluates network formation as a static, aggregate snapshot — degree distribution shape, clustering coefficients, homophily correlation, measured once the network is fully grown.
- It never tests whether the attachment behavior is *stable over time* during growth, or whether changing something about the LLM mid-growth (model, persona, temperature) produces a detectable shift in that behavior.

### Paper B — The statistical tool being borrowed (the changepoint test)

**Citation:** Cirkovic, D., Wang, T., & Zhang, X. (2025). *Likelihood-based Inference for Random Networks with Changepoints.* IEEE Transactions on Network Science and Engineering. DOI: 10.1109/TNSE.2025.3583550.

**What it does:**
- Works with the classical, undirected, affine preferential attachment model: at each time step, a new node joins and attaches to an existing node `v` with probability proportional to `(degree(v) + δ)`, where δ is the "offset parameter" controlling how strongly the network favors high-degree nodes.
- Proposes a **likelihood-based hypothesis test** to answer two questions at once: (1) did the network's δ parameter change at some point during its growth (a changepoint), and (2) if so, exactly where?
- Formally proves:
  - The maximum likelihood estimator (MLE) of δ remains consistent even under long-range dependence (i.e., even though early network growth influences everything downstream, the MLE computed on recent edges alone still correctly estimates the current δ).
  - The test statistic (a likelihood ratio, `-2 log Λ`) converges to a known distribution (the supremum of a scaled Brownian bridge process) under the null hypothesis of "no changepoint" — meaning you can compute a rigorous p-value, not just an ad hoc threshold.
  - The changepoint location estimator is consistent (converges to the true changepoint location as the network grows).
- Extends the single-changepoint test to **multiple changepoints** via two methods: a sliding-window local-maximum method (SaRa-style) and a faster score-statistic method suited for binary segmentation.
- Validates all of this extensively: Type I error simulations (500 trials per setting), statistical power curves, mean-absolute-error tables for changepoint location accuracy — and applies it to a real dataset, the Mathematics Genealogy Project (94,858 mathematician advisor-advisee records spanning 1950–2024), correctly detecting real historical disruptions (Vietnam-War-era PhD trends in 1978, oversaturation concerns around 1995, and a 2014 shift) without being told where to look.

**What it does NOT do (the gap you are closing):**
- Every experiment in the paper uses either purely synthetic, mathematically-generated PA networks, or one real human-generated dataset (mathematician advisor records).
- It has never been applied to networks generated by an AI decision-making process (LLM agents) rather than a human process or a closed-form probability formula.
- The authors explicitly flag extending their framework to new data-generating processes and models (e.g., with vertex death, or multiple-attachment mechanisms) as future work — this project is exactly that kind of extension, applied to a data-generating process (LLM agents) they never considered.

---

## 4. The Core Research Question

> If you deliberately change something about a population of LLM agents partway through a network's growth (the underlying model, its persona/instructions, or its sampling temperature), does Paper B's rigorous statistical test correctly detect that a changepoint occurred — and does the recovered shift in the attachment parameter δ match what the intervention would predict?

This is a legitimate two-paper synthesis: Paper A supplies the novel data-generating process; Paper B supplies the rigorous analysis tool. Nobody has connected them.

---

## 5. Exact Build Plan — What You Are Going to Do

### Component 1 — LLM-agent network growth engine (based on Paper A)
- Recreate the core mechanism: at each time step, a new LLM-driven "agent" joins and picks exactly one existing node to connect to, following Paper A's Algorithm 1 prompt structure (profile-based selection).
- Simplify the profile to just the current degree sequence (since the project's focus is degree-based attachment behavior, not full social-profile matching like Paper A's richer settings).
- Grow the network for enough time steps to produce a statistically meaningful edge count (Paper B's own experiments use 50,000–100,000 edges for reliable asymptotics; you will scale down thoughtfully for LLM-call cost and explicitly note this as a finite-sample limitation in your write-up).

### Component 2 — A controlled "shock" partway through growth (the novel intervention)
Choose one intervention type (recommended order below):
1. **Persona/prompt shift** (do this first — easiest to validate): first half of growth, the agent is instructed to "prefer well-connected agents"; second half, instructed to "prefer newer, less-connected agents." This should force an obvious, large δ shift — useful for confirming your whole pipeline works correctly before testing anything subtle.
2. **Temperature shift** (the genuinely novel, uncertain-outcome experiment): same model and same prompt throughout, only the LLM's sampling temperature changes partway through. Tests whether "creativity" alone measurably changes preferential-attachment strength — a finding nobody has shown.
3. **Model swap** (optional stretch goal): grow the first half with one LLM, swap to a different LLM for the second half.

### Component 3 — Apply Paper B's statistical machinery, unmodified
- Compute the MLE of δ at each stage using Paper B's Equations 10–14.
- Run Paper B's single-changepoint likelihood ratio test (Equation 22, Theorem 4.2) to test whether a changepoint occurred, and estimate its location.
- Compare the detected changepoint location against the *true* location (known exactly, since you control when the intervention happens) — this mirrors Paper B's own Table 5 (mean absolute error of changepoint location).
- Optional stretch: also run Paper B's non-parametric competitor (Bhamidi, Jin & Nobel's degree-distribution-based estimator) side-by-side, exactly as Paper B's Section 4.4 does, to see which method wins on LLM-generated data the way the likelihood method wins on their synthetic data.

### Component 4 — Classical-PA control group (your baseline)
- Run the identical changepoint-detection pipeline on a purely synthetic, mathematically-generated PA network with a known, hand-set δ₁ → δ₂ shift (directly reproducing Paper B's own Table 4/5 experiment).
- This establishes whether LLM-agent networks behave "the same" as textbook PA networks under changepoint detection, or whether LLM decision-making produces a systematically different signature (noisier degree growth, slower-to-detect shifts, biased δ estimates, etc.).

---

## 6. Step-by-Step Timeline

| Week | Task |
|---|---|
| 1–2 | Implement the LLM-agent network growth engine. Validate: does the resulting degree distribution follow a power law, consistent with Paper A's own validation? |
| 3 | Implement Paper B's likelihood estimator (δ̂, solving their score equation, Eq. 11). Validate against their own synthetic PA(δ) simulation — your Type I error rate should land near their reported ~0.05. |
| 4 | Run the control experiment: classical synthetic PA network with a known changepoint. Confirm you can reproduce their reported MAE numbers before touching anything new — this is your correctness sanity check. |
| 5–6 | Run the LLM-agent network with the persona-shift intervention. Apply the changepoint test. Record: detection accuracy, estimated δ before/after, comparison to expected values. |
| 7 | Run the temperature-shift intervention (the genuinely uncertain, most novel experiment). |
| 8 | Write up comparison tables mirroring Paper B's Table 5/6/7 structure (mean absolute error, rejection rates, statistical power) across three conditions: classical synthetic PA, LLM-agent persona-shift, LLM-agent temperature-shift. |

---

## 7. What the Results Section Will Look Like (every outcome is publishable)

- **If the test detects the changepoint accurately in LLM data, matching classical PA behavior:** Conclusion — "LLM-agent-formed networks are statistically indistinguishable from classical preferential-attachment processes under rigorous changepoint testing." A strong, rigorous validation of Paper A's core claim, using a far more demanding statistical tool than Paper A's own evaluation.
- **If detection is worse or biased in LLM data** (delayed detection, noisier δ estimates): Conclusion — "LLM agents introduce additional variance/dependence beyond what the classical PA model captures." A genuine, citable limitation for the growing "LLMs as agent-based social simulators" literature.
- **If the temperature shift alone produces a detectable δ change:** A clean, standalone, novel finding — "LLM sampling temperature functions like a preferential-attachment tuning parameter." Nobody has shown this.

---

## 8. Explicitly Stated Gaps You Are Citing as Your Novelty Justification

- **From Paper B's conclusion:** the authors explicitly state their model only allows new nodes to attach to exactly one existing node, and explicitly flag extending their framework to new data-generating processes and models (e.g., with vertex death, Bayesian treatments for latent-variable models) as future work. This project is exactly that kind of extension — applied to a data-generating process (LLM agents) they never considered.
- **From Paper A:** the authors validate LLM-network-formation principles only against static, aggregate statistics (final degree distribution, clustering, homophily correlation) and never test the *temporal stability* of the attachment parameter during growth. Your changepoint analysis is a temporal-robustness test their paper never attempted.

---

## 9. Reference List (for your paper's related-work section)

1. Papachristou, M. & Yuan, Y. (2024/2025). *Network Formation and Dynamics Among Multi-LLMs.* arXiv:2402.10659; PNAS Nexus, 4(12), pgaf317.
2. Cirkovic, D., Wang, T., & Zhang, X. (2025). *Likelihood-based Inference for Random Networks with Changepoints.* IEEE Transactions on Network Science and Engineering. DOI: 10.1109/TNSE.2025.3583550.
3. Barabási, A.-L. & Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science, 286(5439), 509–512. (Foundational BA model — your network-generation baseline.)
4. Bhamidi, S., Jin, J., & Nobel, A. (2018). *Change point detection in network models: Preferential attachment and long range dependence.* The Annals of Applied Probability, 28(1), 35–78. (The non-parametric competitor method Paper B compares against — useful as a second baseline in your own experiments.)
