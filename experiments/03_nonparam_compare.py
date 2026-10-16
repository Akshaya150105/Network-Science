"""
Optional extra (Paper B, Section 3.4, Tables 2-3): compare the likelihood-ratio
changepoint estimator t_hat (our `scan`) against the non-parametric Bhamidi-Jin-Nobel
estimator T_hat (`that_nonparametric`, Eq. 18) on classical PA data with a known
changepoint. Mirrors Table 2 (mean absolute error of the changepoint location),
scaled down to our n. Paper B finds t_hat wins when the PA model is correct.

    results/nonparam_compare.csv
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.changepoint import scan, that_nonparametric  # noqa: E402
from src.pa_sim import simulate_pa_changepoint  # noqa: E402

N = 2000
STEP = 5  # must match the grid used for the n2000 critical value
REPEATS = 100
DELTA1 = 0.5
DELTA2 = [1.5, 3.0, 5.0]
TRUE_T = 0.5
BASE_SEED = 50000


def one_run(args):
    delta2, seed = args
    picks = simulate_pa_changepoint(N, DELTA1, delta2, N // 2, np.random.default_rng(seed))
    t_hat = scan(picks, trim=0.1, step=STEP)[1] / N
    T_hat = that_nonparametric(picks)
    return {"delta2": delta2, "t_hat_err": abs(t_hat - TRUE_T), "T_hat": T_hat,
            "T_hat_err": (np.nan if np.isnan(T_hat) else abs(T_hat - TRUE_T))}


def se(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


if __name__ == "__main__":
    rows = []
    with Pool() as pool:
        for delta2 in DELTA2:
            jobs = [(delta2, BASE_SEED + r) for r in range(REPEATS)]
            df = pd.DataFrame(pool.map(one_run, jobs, chunksize=5))
            T_err = df["T_hat_err"].dropna()
            rows.append({
                "n": N, "delta1": DELTA1, "delta2": delta2, "repeats": REPEATS,
                "mae_t_hat": df["t_hat_err"].mean(), "mae_t_hat_se": se(df["t_hat_err"]),
                "mae_T_hat": T_err.mean(), "mae_T_hat_se": se(T_err),
                "T_hat_no_cross": int(df["T_hat"].isna().sum()),
            })
            print(f"delta2={delta2}: MAE t_hat={rows[-1]['mae_t_hat']:.4f}, "
                  f"MAE T_hat={rows[-1]['mae_T_hat']:.4f} "
                  f"(T_hat no-cross {rows[-1]['T_hat_no_cross']}/{REPEATS})")

    pd.DataFrame(rows).to_csv(ROOT / "results" / "nonparam_compare.csv", index=False)
    print(f"saved {ROOT / 'results' / 'nonparam_compare.csv'}")
