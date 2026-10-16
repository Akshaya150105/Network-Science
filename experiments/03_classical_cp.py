"""
Step 7: classical control experiment (known changepoint on simulated PA networks).

(a) n in {500, 1000, 2000}: delta 0.5 -> 3 at tau = n // 2, 100 repeats  -> results/classical_cp.csv
(b) power curve at n = 500: delta1 = 1, delta2 in {1, 1.25, 1.5, 2, 3, 5} -> results/power_curve_n500.csv

Needs results/critical_values.json with source "mc" for n500, n1000 and n2000.
delta1_hat / delta2_hat are the window MLEs on each side of the estimated tau_hat.
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
from src.mle import mle, mle_window  # noqa: E402
from src.pa_sim import simulate_pa_changepoint  # noqa: E402

REPEATS = 100
BASE_SEED = 40000
SWEEP_N = [500, 1000, 2000]
SWEEP_DELTAS = (0.5, 3.0)
POWER_N = 500
POWER_DELTA1 = 1.0
POWER_DELTA2 = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def one_run(args):
    n, delta1, delta2, seed, crit, step = args
    tau_true = n // 2
    picks = simulate_pa_changepoint(n, delta1, delta2, tau_true, np.random.default_rng(seed))
    stat, tau_hat = scan(picks, trim=0.1, step=step)
    return {
        "n": n,
        "tau_true": tau_true,
        "stat": stat,
        "crit": crit,
        "rejected": int(stat > crit),
        "tau_hat": tau_hat,
        "tau_err_frac": abs(tau_hat - tau_true) / n,
        "delta1_hat": mle_window(picks, 2, tau_hat),
        "delta2_hat": mle_window(picks, tau_hat, n),
        "delta_full_hat": mle(picks),
    }


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def summarise(df):
    """Rejection rate, MAE of tau_hat (fraction of n) and mean window estimates, with standard errors."""
    p = df["rejected"].mean()
    return {
        "power": p,
        "power_se": float(np.sqrt(p * (1 - p) / len(df))),
        "mae_tau_frac": df["tau_err_frac"].mean(),
        "mae_tau_frac_se": se(df["tau_err_frac"]),
        "mean_delta1_hat": df["delta1_hat"].mean(),
        "delta1_hat_se": se(df["delta1_hat"]),
        "mean_delta2_hat": df["delta2_hat"].mean(),
        "delta2_hat_se": se(df["delta2_hat"]),
    }


if __name__ == "__main__":
    crit_values = json.loads((ROOT / "results" / "critical_values.json").read_text())
    for n in SWEEP_N + [POWER_N]:
        entry = crit_values[f"n{n}"]
        assert entry["source"] == "mc", f"n{n} critical value is not Monte Carlo"

    # (a) n sweep
    rows, sweep_summary = [], []
    with Pool() as pool:
        for n in SWEEP_N:
            entry = crit_values[f"n{n}"]
            crit, step = entry["0.05"], entry["step"]
            jobs = [(n, *SWEEP_DELTAS, BASE_SEED + r, crit, step) for r in range(REPEATS)]
            df = pd.DataFrame(pool.map(one_run, jobs, chunksize=5))
            for r, rec in enumerate(df.to_dict("records")):
                rows.append({"run_id": f"classical_n{n}_s{BASE_SEED + r}", "condition": "classical_cp",
                             "source": "classical", **rec})
            sweep_summary.append({"n": n, **summarise(df)})

        pd.DataFrame(rows)[
            ["run_id", "condition", "source", "n", "tau_true", "stat", "crit", "rejected", "tau_hat",
             "tau_err_frac", "delta1_hat", "delta2_hat", "delta_full_hat"]
        ].to_csv(ROOT / "results" / "classical_cp.csv", index=False)

        # (b) power curve at n = 500
        entry = crit_values[f"n{POWER_N}"]
        crit, step = entry["0.05"], entry["step"]
        power_rows = []
        for delta2 in POWER_DELTA2:
            jobs = [(POWER_N, POWER_DELTA1, delta2, BASE_SEED + r, crit, step) for r in range(REPEATS)]
            df = pd.DataFrame(pool.map(one_run, jobs, chunksize=5))
            power_rows.append({"n": POWER_N, "delta1": POWER_DELTA1, "delta2": delta2, "repeats": REPEATS,
                               **summarise(df)})
        pd.DataFrame(power_rows).to_csv(ROOT / "results" / "power_curve_n500.csv", index=False)

    pd.set_option("display.width", 200)
    print("n sweep, delta 0.5 -> 3 at tau = n/2")
    print(pd.DataFrame(sweep_summary).round(3).to_string(index=False))
    print("\npower curve, n = 500, delta1 = 1 (delta2 = 1 is the Type I error)")
    print(pd.DataFrame(power_rows).drop(columns=["n", "delta1", "repeats"]).round(3).to_string(index=False))
