"""Checks, Paper A validation, and the merge-phase analysis for the LLM side."""
import csv
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import llm_growth as lg

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CSV_COLUMNS = ["run_id", "condition", "source", "n", "tau_true", "stat", "crit", "rejected",
               "tau_hat", "tau_err_frac", "delta1_hat", "delta2_hat", "delta_full_hat"]


# ---------- per-run checks (no statistics-side code needed) ----------

def replay_degrees(rows):
    """Replay the log as a multiset of degrees; fail loudly if any step is inconsistent.
    Returns the final degree sequence (starts from the contract seed [2, 2])."""
    degs = list(lg.SEED_DEGREES)
    for r in rows:
        if r["t"] != len(degs) + 1:
            raise ValueError(f"step {r['t']} out of order")
        if sorted(r["candidate_degrees"]) != sorted(degs):
            raise ValueError(f"step {r['t']}: candidate degrees do not match the replayed network")
        d = r["chosen_degree"]
        if r["candidate_degrees"][r["chosen_idx"]] != d:
            raise ValueError(f"step {r['t']}: chosen_degree does not match chosen_idx")
        degs.remove(d)
        degs.append(d + 1)
        degs.append(1)
    return degs


def top_k_rates(rows, ks=(1, 5, 10)):
    """For each k: share of steps whose pick is among the k highest-degree candidates (ties included)
    vs the share expected from a uniformly random pick. Only parse_ok rows (model behaviour)."""
    out = {}
    for k in ks:
        hit, null, used = 0.0, 0.0, 0
        for r in rows:
            d = np.array(r["candidate_degrees"])
            if not r["parse_ok"] or len(d) <= k:
                continue
            thr = np.sort(d)[-k]
            hit += r["chosen_degree"] >= thr
            null += (d >= thr).sum() / len(d)
            used += 1
        out[k] = (hit / used, null / used) if used else (float("nan"), float("nan"))
    return out


def position_bias(rows):
    """Share of picks on the first / last listed member vs the 1/m expected by chance."""
    ok = [r for r in rows if r["parse_ok"]]
    if not ok:
        return float("nan"), float("nan"), float("nan")
    m = np.array([len(r["candidate_degrees"]) for r in ok])
    idx = np.array([r["chosen_idx"] for r in ok])
    return float((idx == 0).mean()), float((idx == m - 1).mean()), float((1 / m).mean())


def check_run(run_id):
    """Quality checks used by every experiment. Returns a dict and prints one summary."""
    rows = lg.read_rows(run_id)
    meta = lg.load_meta(run_id)
    tau = meta["tau_true"]
    c = {"run_id": run_id, "rows": len(rows), "complete": len(rows) == meta["n"] - 2}
    replay_degrees(rows)  # raises if inconsistent
    c["parse_fail_rate"] = float(np.mean([not r["parse_ok"] for r in rows]))
    c["first_share"], c["last_share"], c["chance_share"] = position_bias(rows)
    halves = [("all", rows)] if tau is None else [("first", [r for r in rows if r["t"] <= tau]),
                                                  ("second", [r for r in rows if r["t"] > tau])]
    for name, part in halves:
        hit, null = top_k_rates(part, ks=(5,))[5]
        c[f"top5_{name}"], c[f"top5_null_{name}"] = hit, null
    print(f"{run_id}: rows={c['rows']} parse_fail={c['parse_fail_rate']:.1%} "
          f"first/last pick share={c['first_share']:.3f}/{c['last_share']:.3f} (chance {c['chance_share']:.3f}) "
          + " ".join(f"top5_{n}={c[f'top5_{n}']:.2f} (null {c[f'top5_null_{n}']:.2f})" for n, _ in halves))
    if c["parse_fail_rate"] > 0.02:
        print("  WARNING: parse failures above 2%")
    return c


# ---------- plots ----------

def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _ccdf(degs):
    x = np.sort(np.array(degs))
    return x, 1.0 - np.arange(len(x)) / len(x)


def ccdf_overlay(run_ids, labels, out_path, title="Final degree distribution (CCDF)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for rid, lab, col in zip(run_ids, labels, [BLUE, ORANGE]):
        x, y = _ccdf(replay_degrees(lg.read_rows(rid)))
        ax.step(x, y, where="post", color=col, linewidth=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "degree", "P(degree >= x)", title)
    ax.legend(frameon=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def paper_a_validation(run_id, fig_dir=FIG_DIR):
    """Paper A style checks on one run: top-k vs random null, power-law exponent, K-S vs Barabasi-Albert."""
    import networkx as nx
    import powerlaw
    from scipy.stats import ks_2samp

    rows = lg.read_rows(run_id)
    degs = replay_degrees(rows)
    ba = [d for _, d in nx.barabasi_albert_graph(len(degs), 1, seed=0).degree()]
    res = {"run_id": run_id, "n": len(degs), "max_degree": max(degs)}
    for name, d in (("llm", degs), ("ba", ba)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # powerlaw is noisy about optimiser bounds
                fit = powerlaw.Fit(d, discrete=True, verbose=False)
                res[f"alpha_{name}"], res[f"xmin_{name}"] = float(fit.power_law.alpha), float(fit.power_law.xmin)
        except Exception as e:  # e.g. star-like networks have nothing to fit
            res[f"alpha_{name}"], res[f"xmin_{name}"] = float("nan"), float("nan")
            print(f"  power-law fit failed for {name}: {e}")
    ks = ks_2samp(degs, ba)
    res["ks_vs_ba"], res["ks_p"] = float(ks.statistic), float(ks.pvalue)
    res["top_k"] = top_k_rates(rows)
    res["position"] = position_bias(rows)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d, lab, col in ((degs, "LLM network", BLUE), (ba, "Barabasi-Albert (same size)", ORANGE)):
        x, y = _ccdf(d)
        ax.step(x, y, where="post", color=col, linewidth=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "degree", "P(degree >= x)", f"{run_id}: final degree distribution")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / f"{run_id}_degree_ccdf.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ks_list = list(res["top_k"])
    xs = np.arange(len(ks_list))
    ax.bar(xs - 0.2, [res["top_k"][k][0] for k in ks_list], 0.4, color=BLUE, label="LLM picks")
    ax.bar(xs + 0.2, [res["top_k"][k][1] for k in ks_list], 0.4, color=ORANGE, label="random picks")
    ax.set_xticks(xs, [f"top {k}" for k in ks_list])
    _style(ax, "candidate rank by degree", "share of picks", f"{run_id}: picks among highest-degree candidates")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / f"{run_id}_topk.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return res


# ---------- merge phase (needs the statistics side's src/mle.py and src/changepoint.py) ----------

def analyze_runs(run_ids, out_csv, crit_key="n500"):
    """Apply the changepoint test to each run and write the contract CSV. Prints a summary."""
    from src.changepoint import scan
    from src.mle import mle, mle_window

    info = json.loads((ROOT / "results" / "critical_values.json").read_text())[crit_key]
    crit = info["0.05"]
    print(f"critical value {crit:.3f} (source: {info.get('source')}, n_sims: {info.get('n_sims')})")
    if info.get("source") != "mc":
        print("WARNING: critical value is the asymptotic placeholder, not calibrated at this n. Re-run later.")
    out = []
    for rid in run_ids:
        picks, meta = lg.load_picks(rid), lg.load_meta(rid)
        n, tau = meta["n"], meta["tau_true"]
        if len(picks) != n - 2:
            print(f"skipping {rid}: incomplete ({len(picks)}/{n - 2} rows)")
            continue
        stat, tau_hat = scan(picks, trim=0.1, step=1)
        out.append({"run_id": rid, "condition": meta["condition"],
                    "source": "stub" if rid.startswith("stub_") else "llm", "n": n,
                    "tau_true": "" if tau is None else tau, "stat": stat, "crit": crit,
                    "rejected": int(stat > crit), "tau_hat": tau_hat,
                    "tau_err_frac": "" if tau is None else abs(tau_hat - tau) / n,
                    "delta1_hat": mle_window(picks, 0, tau_hat), "delta2_hat": mle_window(picks, tau_hat, n),
                    "delta_full_hat": mle(picks)})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(out)
    summarize(out)
    print("wrote", out_csv)
    return out


def summarize(rows):
    N = len(rows)
    if not N:
        print("no complete runs found")
        return
    p = sum(r["rejected"] for r in rows) / N
    print(f"runs: {N}   rejection rate: {p:.2f} (binomial SE {math.sqrt(p * (1 - p) / N):.2f})")
    print(f"delta2_hat > delta1_hat in {sum(r['delta2_hat'] > r['delta1_hat'] for r in rows)}/{N} runs")
    bound = sum(1 for r in rows for k in ("delta1_hat", "delta2_hat", "delta_full_hat")
                if r[k] <= -0.98 or r[k] >= 49.9)
    print(f"delta estimates at a search bound (-0.99 or 50): {bound} of {3 * N}")
    errs = [r["tau_err_frac"] for r in rows if r["tau_err_frac"] != ""]
    if errs:
        print(f"mean |tau_hat - tau_true| / n: {np.mean(errs):.3f}")


def rolling_delta_plot(run_id, out_path, window=100, step=25):
    """delta_hat over time in rolling windows, with true and detected changepoints marked."""
    from src.mle import mle_window

    picks, meta = lg.load_picks(run_id), lg.load_meta(run_id)
    n = meta["n"]
    los = list(range(2, n - window + 1, step))
    ys = [mle_window(picks, lo, lo + window) for lo in los]
    xs = [lo + window / 2 for lo in los]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, ys, color=BLUE, linewidth=1.5, label=f"delta_hat ({window}-step windows)")
    if meta["tau_true"] is not None:
        ax.axvline(meta["tau_true"], color=INK, linestyle="--", linewidth=1, label="true changepoint")
    try:
        from src.changepoint import scan
        ax.axvline(scan(picks, trim=0.1, step=1)[1], color=ORANGE, linewidth=1.5, label="detected changepoint")
    except ImportError:
        pass
    _style(ax, "step t", "delta_hat", f"{run_id}: attachment parameter over time")
    ax.legend(frameon=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
