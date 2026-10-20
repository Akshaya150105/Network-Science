"""
Step 7: classical control experiment (known changepoint on simulated PA networks).

(a) n in {500, 1000, 2000}: delta 0.5 -> 3 at tau = n // 2, 100 repeats  -> results/classical_cp.csv
(b) power curve at n = 500: delta1 = 1, delta2 in {1, 1.25, 1.5, 2, 3, 5} -> results/power_curve_n500.csv

Needs results/critical_values.json with source "mc" for n500, n1000 and n2000.

delta1_hat / delta2_hat are the window MLEs on each side of the estimated tau_hat (contract columns).
Because tau_hat is the split that maximises the statistic, these are selection-biased and heavy
tailed, so every summary also reports medians, the number of runs that hit a delta bound, and the
same estimates computed at the KNOWN tau (`*_true_tau`), which is what the LLM runs allow.

Location error follows Paper B Fig. 1: `mae_tau_frac_rejected` averages |tau_hat - tau| / n only over
runs that rejected the null; `mae_tau_frac` averages over all runs. Rows with delta2 == delta1 have no
changepoint, so their location error is left blank and `row_type` is `type1_error`.
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
from src.mle import at_bound, mle, mle_window  # noqa: E402
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
    d1_hat = mle_window(picks, 2, tau_hat)
    d2_hat = mle_window(picks, tau_hat, n)
    d1_true = mle_window(picks, 2, tau_true)
    d2_true = mle_window(picks, tau_true, n)
    return {
        "n": n,
        "tau_true": tau_true,
        "stat": stat,
        "crit": crit,
        "rejected": int(stat > crit),
        "tau_hat": tau_hat,
        "tau_err_frac": abs(tau_hat - tau_true) / n,
        "delta1_hat": d1_hat,
        "delta2_hat": d2_hat,
        "delta_full_hat": mle(picks),
        # extras (appended after the contract columns)
        "delta1_hat_true_tau": d1_true,
        "delta2_hat_true_tau": d2_true,
        "delta1_at_bound": int(at_bound(d1_hat)),
        "delta2_at_bound": int(at_bound(d2_hat)),
        "delta1_at_bound_true_tau": int(at_bound(d1_true)),
        "delta2_at_bound_true_tau": int(at_bound(d2_true)),
    }


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def summarise(df, has_changepoint=True):
    """
    Rejection rate, location error and delta estimates, each with a standard error where meaningful.

    has_changepoint=False (delta2 == delta1): the rejection rate is the Type I error and the location
    error columns are NaN, because tau_hat has no true value to be compared with.
    """
    p = df["rejected"].mean()
    rej = df[df["rejected"] == 1]
    out = {
        "row_type": "power" if has_changepoint else "type1_error",
        "power": p,
        "power_se": float(np.sqrt(p * (1 - p) / len(df))),
        "n_rejected": int(len(rej)),
    }
    if has_changepoint:
        out.update(
            mae_tau_frac=df["tau_err_frac"].mean(),  # all runs
            mae_tau_frac_se=se(df["tau_err_frac"]),
            mae_tau_frac_rejected=rej["tau_err_frac"].mean() if len(rej) else np.nan,  # Paper B Fig. 1
            mae_tau_frac_rejected_se=se(rej["tau_err_frac"]) if len(rej) > 1 else np.nan,
        )
    else:
        out.update(mae_tau_frac=np.nan, mae_tau_frac_se=np.nan,
                   mae_tau_frac_rejected=np.nan, mae_tau_frac_rejected_se=np.nan)
    for tag, suffix in (("", ""), ("_true_tau", "_true_tau")):
        for j in (1, 2):
            col = f"delta{j}_hat{suffix}"
            out[f"mean_delta{j}_hat{suffix}"] = df[col].mean()
            out[f"delta{j}_hat{suffix}_se"] = se(df[col])
            out[f"median_delta{j}_hat{suffix}"] = df[col].median()
            out[f"n_delta{j}_at_bound{suffix}"] = int(df[f"delta{j}_at_bound{suffix}"].sum())
    return out


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
             "tau_err_frac", "delta1_hat", "delta2_hat", "delta_full_hat",
             "delta1_hat_true_tau", "delta2_hat_true_tau", "delta1_at_bound", "delta2_at_bound",
             "delta1_at_bound_true_tau", "delta2_at_bound_true_tau"]
        ].to_csv(ROOT / "results" / "classical_cp.csv", index=False)

        # (b) power curve at n = 500
        entry = crit_values[f"n{POWER_N}"]
        crit, step = entry["0.05"], entry["step"]
        power_rows = []
        for delta2 in POWER_DELTA2:
            jobs = [(POWER_N, POWER_DELTA1, delta2, BASE_SEED + r, crit, step) for r in range(REPEATS)]
            df = pd.DataFrame(pool.map(one_run, jobs, chunksize=5))
            power_rows.append({"n": POWER_N, "delta1": POWER_DELTA1, "delta2": delta2, "repeats": REPEATS,
                               **summarise(df, has_changepoint=(delta2 != POWER_DELTA1))})
        pd.DataFrame(power_rows).to_csv(ROOT / "results" / "power_curve_n500.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["n", "power", "power_se", "mae_tau_frac", "mae_tau_frac_rejected", "mean_delta1_hat",
            "median_delta1_hat", "mean_delta2_hat", "median_delta2_hat", "median_delta1_hat_true_tau",
            "median_delta2_hat_true_tau", "n_delta2_at_bound"]
    print("n sweep, delta 0.5 -> 3 at tau = n/2")
    print(pd.DataFrame(sweep_summary)[cols].round(3).to_string(index=False))
    print("\npower curve, n = 500, delta1 = 1 (row_type type1_error = no changepoint, so no location error)")
    pcols = ["delta2", "row_type"] + [c for c in cols if c != "n"]
    print(pd.DataFrame(power_rows)[pcols].round(3).to_string(index=False))
