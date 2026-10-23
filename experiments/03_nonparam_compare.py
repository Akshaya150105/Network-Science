"""
Optional extra (Paper B, Section IV-D, Table V in the IEEE version; Section 3.4, Table 2 in arXiv v3): compare the likelihood-ratio
changepoint estimator t_hat (our `scan`) against the non-parametric estimator T_hat of
Banerjee, Bhamidi & Carmichael (2023) (`that_nonparametric`, Paper B Eq. 20, reference [11]; arXiv v3: Eq. 18, [2])
on classical PA data with a known changepoint. Scaled down to our n (Paper B uses 100,000 edges).

    results/nonparam_compare.csv

Design choices, stated so the table is not over-read:

* Only n = 2000 is run. With the paper's default h_n = log log n, T_hat takes its reference degree
  distribution at the fraction 1 / h_n of the run (0.547 at n = 500, 0.493 at n = 2000), so at
  n = 500 the reference is already AFTER a changepoint at n/2 and T_hat cannot work there.
* The changepoint is run at t* = 0.5 (the project's tau = n/2) AND at t* = 0.6 (Paper B's Table V; Table 2 in arXiv v3).
  At t* = 0.5 the reference time (0.493) sits only just before the changepoint, which handicaps T_hat;
  t* = 0.6 is the like-for-like comparison with the paper.
* T_hat is an infimum; if the distance never crosses the threshold the infimum is +infinity (no
  detection). Averaging only the runs that did cross flatters T_hat, so two numbers are reported:
  `mae_T_hat_crossed` (crossing runs only) and `mae_T_hat_all`, which scores a non-crossing run as
  T_hat = 1 (the largest possible location, i.e. the worst case on [0, 1]).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.changepoint import scan, that_nonparametric, that_reference_fraction  # noqa: E402
from src.pa_sim import simulate_pa_changepoint  # noqa: E402

N = 2000
STEP = 1  # must match the grid used for the n2000 critical value (every split)
REPEATS = 100
DELTA1 = 0.5
DELTA2 = [1.5, 3.0, 5.0]
TRUE_T = [0.5, 0.6]
BASE_SEED = 50000


def one_run(args):
    true_t, delta2, seed = args
    tau = int(round(true_t * N))
    picks = simulate_pa_changepoint(N, DELTA1, delta2, tau, np.random.default_rng(seed))
    t_hat = scan(picks, trim=0.1, step=STEP)[1] / N
    T_hat = that_nonparametric(picks)
    return {
        "true_t": true_t,
        "delta2": delta2,
        "t_hat_err": abs(t_hat - true_t),
        "T_hat": T_hat,
        "T_hat_err": np.nan if np.isnan(T_hat) else abs(T_hat - true_t),
        "T_hat_err_capped": abs((1.0 if np.isnan(T_hat) else T_hat) - true_t),
    }


def se(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


if __name__ == "__main__":
    ref = that_reference_fraction(N)
    print(f"T_hat reference fraction 1/h_n at n={N}: {ref:.3f} "
          f"(changepoint must come after it; n=500 would be {that_reference_fraction(500):.3f})")
    rows = []
    with Pool() as pool:
        for true_t in TRUE_T:
            for delta2 in DELTA2:
                jobs = [(true_t, delta2, BASE_SEED + r) for r in range(REPEATS)]
                df = pd.DataFrame(pool.map(one_run, jobs, chunksize=5))
                crossed = df["T_hat_err"].dropna()
                rows.append({
                    "n": N, "true_t": true_t, "delta1": DELTA1, "delta2": delta2, "repeats": REPEATS,
                    "mae_t_hat": df["t_hat_err"].mean(), "mae_t_hat_se": se(df["t_hat_err"]),
                    "mae_T_hat_crossed": crossed.mean(), "mae_T_hat_crossed_se": se(crossed),
                    "mae_T_hat_all": df["T_hat_err_capped"].mean(), "mae_T_hat_all_se": se(df["T_hat_err_capped"]),
                    "T_hat_no_cross": int(df["T_hat"].isna().sum()),
                    "ref_fraction": ref, "ref_before_changepoint": bool(ref < true_t),
                })
                r = rows[-1]
                print(f"t*={true_t} delta2={delta2}: MAE t_hat={r['mae_t_hat']:.4f}, "
                      f"T_hat crossed-only={r['mae_T_hat_crossed']:.4f}, all={r['mae_T_hat_all']:.4f} "
                      f"(no-cross {r['T_hat_no_cross']}/{REPEATS})")

    out = ROOT / "results" / "nonparam_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")
