"""
M3: statistics figures (Valincia). Writes PNGs to figures/ from the saved results and one regenerated example run.

    python experiments/07_stats_figures.py

  fig_null_hist.png            null distribution of the statistic with the Monte Carlo and asymptotic critical values
  fig_power_curve_n500.png     power curve at n = 500 (delta2 = 1 is the Type I error) and location error
  fig_n_sweep.png              power and location error against n for the delta 0.5 -> 3 shift
  fig_scan_curve_example.png   the scan curve of one classical run, with the true and estimated changepoint
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.changepoint import scan_curve  # noqa: E402
from src.pa_sim import simulate_pa_changepoint  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
CRIT = json.loads((RES / "critical_values.json").read_text())
ASYM = CRIT["asymptotic"]["0.05"]
BLUE, RED, GREY = "#1f77b4", "#d62728", "#555555"


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", FIG / name)


def fig_null_hist():
    stats = pd.read_csv(RES / "null_stats.csv")
    calib = pd.read_csv(RES / "null_calibration.csv").set_index("n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, n in zip(axes, (500, 1000, 2000)):
        s = stats[(stats["n"] == n) & (stats["set"] == "calibration")]["stat"]
        ax.hist(s, bins=60, range=(0, 20), density=True, color=BLUE, alpha=0.6)
        mc = CRIT[f"n{n}"]["0.05"]
        ax.axvline(mc, color=RED, lw=2, label=f"Monte Carlo 5% = {mc:.2f}")
        ax.axvline(ASYM, color=GREY, ls="--", lw=1.5, label=f"asymptotic 5% = {ASYM:.2f}")
        ax.set_title(f"n = {n} ({len(s)} null runs, step {CRIT[f'n{n}']['step']})\n"
                     f"Type I error with MC value: {calib.loc[n, 'type1_mc']:.3f}", fontsize=10)
        ax.set_xlabel("sup of likelihood-ratio statistic")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Null distribution of the changepoint statistic (no-change PA networks, delta = 1)")
    save(fig, "fig_null_hist.png")


def fig_power_curve():
    pc = pd.read_csv(RES / "power_curve_n500.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    null = pc[pc["row_type"] == "type1_error"]
    alt = pc[pc["row_type"] == "power"]
    a.errorbar(pc["delta2"], pc["power"], yerr=pc["power_se"], marker="o", color=BLUE, capsize=3)
    a.scatter(null["delta2"], null["power"], s=120, facecolors="none", edgecolors=RED, zorder=3,
              label="delta2 = delta1 = 1: Type I error")
    a.axhline(0.05, color=GREY, ls="--", lw=1, label="alpha = 0.05")
    a.set_xlabel("delta2 (delta1 = 1, change at n/2)")
    a.set_ylabel("rejection rate")
    a.set_ylim(0, 1)
    a.set_title(f"Power at n = 500 ({int(pc['repeats'].iloc[0])} runs per point, +-1 SE)")
    a.legend(fontsize=8)
    b.errorbar(alt["delta2"], alt["mae_tau_frac"], yerr=alt["mae_tau_frac_se"], marker="o", color=BLUE,
               capsize=3, label="all runs")
    b.errorbar(alt["delta2"], alt["mae_tau_frac_rejected"], yerr=alt["mae_tau_frac_rejected_se"], marker="s",
               color=RED, capsize=3, label="rejected runs only (Paper B Fig. 1)")
    b.set_xlabel("delta2")
    b.set_ylabel("mean |tau_hat - tau| / n")
    b.set_title("Changepoint location error at n = 500")
    b.legend(fontsize=8)
    save(fig, "fig_power_curve_n500.png")


def fig_n_sweep():
    cp = pd.read_csv(RES / "classical_cp.csv")
    rows = []
    for n, g in cp.groupby("n"):
        rej = g[g["rejected"] == 1]
        p = g["rejected"].mean()
        rows.append({"n": n, "power": p, "se": np.sqrt(p * (1 - p) / len(g)),
                     "mae": g["tau_err_frac"].mean(), "mae_se": g["tau_err_frac"].std(ddof=1) / np.sqrt(len(g)),
                     "mae_rej": rej["tau_err_frac"].mean(),
                     "mae_rej_se": rej["tau_err_frac"].std(ddof=1) / np.sqrt(len(rej))})
    d = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.errorbar(d["n"], d["power"], yerr=d["se"], marker="o", color=BLUE, capsize=3)
    a.set_xscale("log")
    a.minorticks_off()
    a.set_xticks(d["n"], [str(n) for n in d["n"]])
    a.set_ylim(0, 1.05)
    a.set_xlabel("n (number of nodes)")
    a.set_ylabel("rejection rate")
    a.set_title("Power, delta 0.5 -> 3 at n/2 (100 runs each)")
    b.errorbar(d["n"], d["mae"], yerr=d["mae_se"], marker="o", color=BLUE, capsize=3, label="all runs")
    b.errorbar(d["n"], d["mae_rej"], yerr=d["mae_rej_se"], marker="s", color=RED, capsize=3, label="rejected runs only")
    b.set_xscale("log")
    b.minorticks_off()
    b.set_xticks(d["n"], [str(n) for n in d["n"]])
    b.set_xlabel("n (number of nodes)")
    b.set_ylabel("mean |tau_hat - tau| / n")
    b.set_title("Location error falls as n grows")
    b.legend(fontsize=8)
    save(fig, "fig_n_sweep.png")


def fig_scan_curve():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, n in zip(axes, (500, 2000)):
        entry = CRIT[f"n{n}"]
        picks = simulate_pa_changepoint(n, 0.5, 3.0, n // 2, np.random.default_rng(40000))  # first run of Step 7
        m, s = scan_curve(picks, trim=0.1, step=entry["step"])
        tau_hat = int(m[np.flatnonzero(s >= s.max() - 1e-9)[0]])
        ax.plot(m, s, color=BLUE, lw=1.4)
        ax.axhline(entry["0.05"], color=GREY, ls="--", label=f"5% critical value {entry['0.05']:.2f}")
        ax.axvline(n // 2, color="green", lw=1.5, label=f"true tau = {n // 2}")
        ax.axvline(tau_hat, color=RED, lw=1.5, ls=":", label=f"estimated tau_hat = {tau_hat}")
        ax.set_xlabel("split m")
        ax.set_ylabel("-2 log Lambda_m")
        ax.set_title(f"Classical PA, n = {n}, delta 0.5 -> 3 (seed 40000)")
        ax.legend(fontsize=8)
    save(fig, "fig_scan_curve_example.png")


if __name__ == "__main__":
    fig_null_hist()
    fig_power_curve()
    fig_n_sweep()
    fig_scan_curve()
