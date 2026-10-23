"""
Rolling-window delta_hat plot: a cheap diagnostic of drift, a fading effect, or a second change.

    python experiments/08_rolling_delta.py                       # classical demo (no logs needed)
    python experiments/08_rolling_delta.py --prefix temp_n500_s  # real LLM runs from logs/ (Akshaya's)

Each point is the MLE of delta from `window` consecutive picks (default 100 of the 498 picks at n = 500), plotted at
the window centre. Windows are short, so single runs are noisy; the plot overlays all runs and draws their median.
Values above Y_MAX are drawn at the top edge, and windows whose estimate sits on a search bound (-0.99 or 50, a
model-mismatch signal) are counted in the title rather than trusted.

It is not a test. Use it to look at runs the single-changepoint test has rejected or failed to reject:
  * a no-shock control whose median curve drifts, or whose runs scatter more than the classical demo, points to
    non-stationary behaviour (the reason a no-shock rejection rate above 5% would matter);
  * a shock run whose curve rises and then falls back suggests a fading effect, which one changepoint describes badly;
  * a second step suggests the multiple-changepoint methods of Paper B Section V could be worth building.

Stub or fixture runs (prefix stub_/fixture_) are labelled source = stub and written to *_stub files only.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis_stats as A  # noqa: E402
from src.changepoint import window_mle_curve  # noqa: E402
from src.pa_sim import simulate_pa_changepoint  # noqa: E402

FIG = ROOT / "figures"
Y_MIN, Y_MAX = -1.0, 8.0
BLUE, RED, GREY = "#1f77b4", "#d62728", "#555555"


def draw_panel(ax, curves, title, truth=None, tau=None):
    """curves: list of dicts from window_mle_curve (same k_center grid). truth: [(k_from, k_to, delta)]."""
    for c in curves:
        ax.plot(c["k_center"], np.clip(c["delta_hat"], Y_MIN, Y_MAX), color=BLUE, alpha=0.22, lw=1)
    k = curves[0]["k_center"]
    if all(len(c["k_center"]) == len(k) for c in curves):
        med = np.median(np.vstack([c["delta_hat"] for c in curves]), axis=0)
        ax.plot(k, np.clip(med, Y_MIN, Y_MAX), color=RED, lw=2.4, label="median over runs")
    for k0, k1, d in truth or []:
        ax.hlines(d, k0, k1, color="black", lw=1.6, ls="--", label="true delta" if k0 == truth[0][0] else None)
    if tau is not None:
        ax.axvline(tau, color="green", lw=1.3, label=f"true changepoint, step {tau}")
    n_bound = int(sum(c["at_bound"].sum() for c in curves))
    n_all = int(sum(len(c["at_bound"]) for c in curves))
    ax.set_ylim(Y_MIN, Y_MAX + 0.3)
    ax.set_xlabel("step k (centre of window)")
    ax.set_ylabel("window MLE of delta")
    ax.set_title(f"{title}\n{len(curves)} runs; {n_bound}/{n_all} windows on a bound; values above {Y_MAX:g} clipped",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper left")


def classical_demo(n, window, step, runs):
    tau = n // 2
    shock = [window_mle_curve(simulate_pa_changepoint(n, 0.5, 3.0, tau, np.random.default_rng(70000 + r)), window, step)
             for r in range(runs)]
    null = [window_mle_curve(simulate_pa_changepoint(n, 1.0, 1.0, n, np.random.default_rng(71000 + r)), window, step)
            for r in range(runs)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    draw_panel(a, shock, f"Classical PA with a shock: delta 0.5 -> 3 at step {tau} (n = {n})",
               truth=[(3, tau, 0.5), (tau, n, 3.0)], tau=tau)
    draw_panel(b, null, f"Classical PA, no shock: delta = 1 throughout (n = {n})", truth=[(3, n, 1.0)])
    fig.suptitle(f"Rolling-window estimate of delta, window = {window} picks", y=1.02)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "fig_rolling_delta_classical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # numbers that go with the picture
    k = shock[0]["k_center"]
    first, last = k < tau - window // 2, k > tau + window // 2
    for name, cs in (("shock", shock), ("no shock", null)):
        m = np.vstack([c["delta_hat"] for c in cs])
        print(f"{name:9s}: median delta_hat before the change {np.median(m[:, first]):.2f}, after {np.median(m[:, last]):.2f}; "
              f"between-run sd of a window estimate: {np.median(m.std(axis=0, ddof=1)):.2f}")
    print("saved", out)


def real_runs(prefix, logs_dir, window, step):
    run_ids = A.discover_runs(prefix, logs_dir)
    if not run_ids:
        raise SystemExit(f"No logs matching {prefix}*.meta.json in {logs_dir or A.LOGS_DIR}.")
    load_picks, load_meta = A.get_loaders(logs_dir)
    curves, rows, taus, stub = [], [], set(), False
    for rid in run_ids:
        meta = load_meta(rid)
        stub |= A.run_source(rid, meta) == "stub"
        c = window_mle_curve(load_picks(rid), window, step)
        curves.append(c)
        if meta.get("tau_true") is not None:
            taus.add(int(meta["tau_true"]))
        rows.append(pd.DataFrame({"run_id": rid, "k_center": c["k_center"], "delta_hat": c["delta_hat"],
                                  "se": c["se"], "at_bound": c["at_bound"]}))
    tau = taus.pop() if len(taus) == 1 else None
    tag = prefix.rstrip("_").replace("/", "_") + ("_stub" if stub else "")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    draw_panel(ax, curves, f"{prefix}*: rolling delta_hat, window = {window}" + ("  [STUB, not for results]" if stub else ""),
               tau=tau)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / f"fig_rolling_delta_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    csv = A.RESULTS_DIR / f"rolling_delta_{tag}.csv"
    pd.concat(rows).to_csv(csv, index=False)
    print(f"{len(run_ids)} runs -> {out} and {csv}" + ("  (stub: keep out of the final report)" if stub else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=None, help="run_id prefix of real/fixture runs; omit for the classical demo")
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--n", type=int, default=500, help="demo only")
    ap.add_argument("--runs", type=int, default=20, help="demo only")
    a = ap.parse_args()
    if a.prefix:
        real_runs(a.prefix, a.logs_dir, a.window, a.step)
    else:
        classical_demo(a.n, a.window, a.step, a.runs)
