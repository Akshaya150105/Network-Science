"""
Step 6: null calibration of the changepoint statistic.

    python experiments/02_null_calibration.py asym   # (a)+(b) asymptotic value, write critical_values.json
    python experiments/02_null_calibration.py mc     # (c)+(d) Monte Carlo values and Type I error

Run `asym` first: it writes the placeholder that Akshaya's analysis reads.
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

CRIT_PATH = ROOT / "results" / "critical_values.json"
ALPHA = 0.05
TRIM = 0.1
NULL_DELTA = 1.0

# Grid step used for the scan at each n. The same step must be used when this
# critical value is applied (experiments 03, 05, 06). Step 1 = every split, as in Paper B (IEEE Eq. 22).
# (Earlier versions used step 5 at n = 1000 and 2000 for speed, which lowers the maximum slightly; the
# vectorised scan in src/changepoint.py makes the full grid cheap.)
CONFIG = {
    "n500": {"n": 500, "step": 1, "mc_sims": 5000},
    "n1000": {"n": 1000, "step": 1, "mc_sims": 5000},  # added for the Step 7 n sweep
    "n2000": {"n": 2000, "step": 1, "mc_sims": 5000},
}
MC_SEED = 20000
FRESH_SEED = 30000
FRESH_SIMS = 500


def asymptotic_critical_value(grid=10_000, reps=100_000, batch=500, seed=0):
    """95th percentile of sup_{t in [0.1, 0.9]} B(t)^2 / (t (1 - t)) (Theorem 4.2, Eq. 23 in the IEEE version; Theorem 3.2, Eq. 21 in arXiv v3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, grid + 1) / grid
    keep = (t >= TRIM) & (t <= 1 - TRIM)
    weight = t[keep] * (1 - t[keep])
    sups = []
    for _ in range(reps // batch):
        w = np.cumsum(rng.standard_normal((batch, grid)), axis=1) / np.sqrt(grid)
        bridge = w - t * w[:, -1:]  # Brownian bridge B(t) = W(t) - t W(1)
        sups.append(np.max(bridge[:, keep] ** 2 / weight, axis=1))
    sups = np.concatenate(sups)
    return float(np.quantile(sups, 1 - ALPHA)), reps


def write_asymptotic_placeholder():
    crit, reps = asymptotic_critical_value()
    print(f"asymptotic 95th percentile = {crit:.3f} ({reps} replicates)")
    payload = {"asymptotic": {str(ALPHA): crit}}
    for key, cfg in CONFIG.items():
        payload[key] = {str(ALPHA): crit, "source": "asymptotic", "n_sims": reps, "step": cfg["step"]}
    CRIT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {CRIT_PATH}")


def _null_stat(args):
    n, step, seed = args
    return scan(simulate_pa(n, NULL_DELTA, np.random.default_rng(seed)), trim=TRIM, step=step)[0]


def null_stats(n, step, sims, base_seed, pool):
    return np.array(pool.map(_null_stat, [(n, step, base_seed + r) for r in range(sims)], chunksize=10))


def run_monte_carlo():
    payload = json.loads(CRIT_PATH.read_text())
    asym = payload["asymptotic"][str(ALPHA)]
    rows, summary = [], []
    with Pool() as pool:
        for key, cfg in CONFIG.items():
            n, step = cfg["n"], cfg["step"]
            calib = null_stats(n, step, cfg["mc_sims"], MC_SEED, pool)
            crit = float(np.quantile(calib, 1 - ALPHA))
            payload[key] = {str(ALPHA): crit, "source": "mc", "n_sims": len(calib), "step": step}

            fresh = null_stats(n, step, FRESH_SIMS, FRESH_SEED, pool)
            type1_mc = float(np.mean(fresh > crit))
            type1_asym = float(np.mean(fresh > asym))
            summary.append(
                {"n": n, "step": step, "mc_sims": len(calib), "crit_mc": crit, "crit_asym": asym,
                 "fresh_sims": FRESH_SIMS, "type1_mc": type1_mc, "type1_asym": type1_asym}
            )
            rows += [{"n": n, "set": "calibration", "stat": s} for s in calib]
            rows += [{"n": n, "set": "fresh", "stat": s} for s in fresh]
            print(f"n={n}: MC crit={crit:.3f} (asym {asym:.3f}); Type I MC={type1_mc:.3f}, asym={type1_asym:.3f}")

    CRIT_PATH.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "null_stats.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "results" / "null_calibration.csv", index=False)
    print(f"wrote {CRIT_PATH}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "asym":
        write_asymptotic_placeholder()
    elif stage == "mc":
        run_monte_carlo()
    else:
        sys.exit("usage: 02_null_calibration.py asym|mc")
