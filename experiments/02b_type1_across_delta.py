"""
Type I error of the changepoint test across null values of delta (Paper B, IEEE Table IV; arXiv v3 Table 1).

Paper B simulates 500 PA(delta) networks of 50,000 edges for delta in {-0.5, 0, 1, 2} and finds rejection
rates 0.050, 0.042, 0.046, 0.052. The critical values here were calibrated at delta = 1 only, so this checks that
they also hold at the other delta values (including the negative delta that Paper B reaches), at our run lengths.

    python experiments/02b_type1_across_delta.py        ->  results/type1_across_delta.csv

Needs results/critical_values.json with source "mc". Uses the Monte Carlo value and, for comparison, the asymptotic
value (Theorem 4.2), with the same grid step the value was calibrated on. Fresh seeds, never used in calibration.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.changepoint import scan  # noqa: E402
from src.pa_sim import simulate_pa  # noqa: E402

DELTAS = [-0.5, 0.0, 1.0, 2.0]
PAPER_B_TABLE_IV = {-0.5: 0.050, 0.0: 0.042, 1.0: 0.046, 2.0: 0.052}   # 500 networks, 50,000 edges
SIMS = {500: 1000, 1000: 500, 2000: 500}
BASE_SEED = 60000      # calibration used 20000+ and 30000+; the classical control 40000+; T-hat comparison 50000+
TRIM = 0.1


def one(args):
    n, delta, seed, step = args
    return scan(simulate_pa(n, delta, np.random.default_rng(seed)), trim=TRIM, step=step)[0]


if __name__ == "__main__":
    cv = json.loads((ROOT / "results" / "critical_values.json").read_text())
    asym = cv["asymptotic"]["0.05"]
    rows = []
    with Pool() as pool:
        for n, sims in SIMS.items():
            entry = cv[f"n{n}"]
            assert entry["source"] == "mc", f"n{n} critical value is not Monte Carlo"
            crit, step = entry["0.05"], entry["step"]
            for k, delta in enumerate(DELTAS):
                seeds = [BASE_SEED + 10000 * k + r for r in range(sims)]
                stats = np.array(pool.map(one, [(n, delta, s, step) for s in seeds], chunksize=10))
                t1, t1a = float(np.mean(stats > crit)), float(np.mean(stats > asym))
                rows.append({"n": n, "delta": delta, "sims": sims, "step": step, "crit_mc": crit,
                             "type1_mc": t1, "type1_mc_se": float(np.sqrt(t1 * (1 - t1) / sims)),
                             "crit_asym": asym, "type1_asym": t1a,
                             "paper_b_table_iv_50000_edges": PAPER_B_TABLE_IV[delta]})
                print(f"n={n} delta={delta:+.1f}: Type I (MC value {crit:.2f}) = {t1:.3f} "
                      f"+- {rows[-1]['type1_mc_se']:.3f}; (asymptotic {asym:.2f}) = {t1a:.3f}", flush=True)
    out = ROOT / "results" / "type1_across_delta.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
