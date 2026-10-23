"""
Statistics-side analysis helpers for the merge phase (Valincia).

    M1  analyze_picks / direction_summary   used by experiments/06_temperature_shift_analyze.py
    M2  classical_table / comparison_table  run as  `python -m src.analysis_stats`

Reading logs: `get_loaders()` returns Akshaya's `load_picks` / `load_meta` from src/llm_growth.py when they
exist, and otherwise a reader that implements the same contract (logs/{run_id}.jsonl and .meta.json).
Both return ALL rows, including parse_ok=false rows (those are uniform-random fallbacks and are kept on
purpose; the parse-failure fraction is reported next to every run).

Stub and fixture runs (run_id prefix `stub_` / `fixture_`, or meta chooser != "llm") are labelled
source = "stub" and are excluded from the final tables.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.changepoint import scan
from src.mle import at_bound, mle, mle_window

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
ALPHA = 0.05
TRIM = 0.1

CONTRACT_COLUMNS = [
    "run_id", "condition", "source", "n", "tau_true", "stat", "crit", "rejected", "tau_hat",
    "tau_err_frac", "delta1_hat", "delta2_hat", "delta_full_hat",
]
# Appended after the contract columns; readers that only know the contract schema can ignore them.
EXTRA_COLUMNS = [
    "crit_source", "crit_step", "delta1_hat_true_tau", "delta2_hat_true_tau", "parse_fail_frac",
]


# ----------------------------------------------------------------------------- reading logs
def read_meta(run_id, logs_dir=None):
    """logs/{run_id}.meta.json as a dict (log contract)."""
    path = Path(logs_dir or LOGS_DIR) / f"{run_id}.meta.json"
    return json.loads(path.read_text())


def _read_rows(run_id, logs_dir=None):
    path = Path(logs_dir or LOGS_DIR) / f"{run_id}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["t"])


def read_picks(run_id, logs_dir=None):
    """[(chosen_degree, t), ...] for ALL rows of the log, in step order (log contract)."""
    return [(int(r["chosen_degree"]), int(r["t"])) for r in _read_rows(run_id, logs_dir)]


def parse_fail_fraction(run_id, logs_dir=None):
    """Share of steps with parse_ok = false (those steps are uniform-random fallbacks)."""
    rows = _read_rows(run_id, logs_dir)
    return float(np.mean([not r["parse_ok"] for r in rows])) if rows else float("nan")


def get_loaders(logs_dir=None):
    """(load_picks, load_meta): Akshaya's functions if present and logs_dir is the default, else ours."""
    if logs_dir is None:
        try:
            from src.llm_growth import load_meta, load_picks  # noqa: WPS433

            return load_picks, load_meta
        except ImportError:
            pass
    return (lambda run_id: read_picks(run_id, logs_dir)), (lambda run_id: read_meta(run_id, logs_dir))


def discover_runs(prefix, logs_dir=None):
    """Sorted run_ids whose meta file starts with `prefix` (e.g. 'temp_n500_s'), ordered by seed."""
    metas = Path(logs_dir or LOGS_DIR).glob(f"{prefix}*.meta.json")
    ids = [p.name[: -len(".meta.json")] for p in metas]

    def seed(run_id):
        tail = run_id[len(prefix):]
        return (0, int(tail)) if tail.isdigit() else (1, tail)

    return sorted(ids, key=seed)


def run_source(run_id, meta):
    """'stub' for stub/fixture runs, otherwise 'llm'."""
    if run_id.startswith(("stub_", "fixture_")) or meta.get("chooser", "llm") != "llm":
        return "stub"
    return "llm"


# ----------------------------------------------------------------------------- critical values
def load_critical_value(n, path=None, allow_asymptotic=False):
    """
    (crit, step, source) for sample size n from results/critical_values.json.

    Refuses a non-Monte-Carlo value unless allow_asymptotic=True: the asymptotic value is not calibrated
    at n = 500, so using it silently would misstate the Type I error. The grid step used at calibration is
    returned and MUST be used when scanning, otherwise the value is not comparable.
    """
    path = Path(path or RESULTS_DIR / "critical_values.json")
    entry = json.loads(path.read_text())[f"n{n}"]
    source = entry.get("source", "unknown")
    if source != "mc" and not allow_asymptotic:
        raise ValueError(
            f"critical value for n={n} has source={source!r}, not 'mc'. Run "
            f"`python experiments/02_null_calibration.py mc` or pass allow_asymptotic=True and say so."
        )
    return float(entry[str(ALPHA)]), int(entry.get("step", 1)), source


# ----------------------------------------------------------------------------- one run (M1)
def analyze_picks(run_id, picks, meta, crit, step, crit_source, source, parse_fail=float("nan")):
    """
    One results row (contract schema + EXTRA_COLUMNS) for one run.

    delta1_hat / delta2_hat are the window MLEs either side of the ESTIMATED tau_hat (contract columns,
    selection-biased). delta*_hat_true_tau use the KNOWN tau_true from the meta file and are the better
    estimates of the regimes; they are NaN when the run has no shock (tau_true is null).
    """
    n = int(meta["n"])
    if len(picks) != n - 2:
        raise ValueError(f"{run_id}: expected {n - 2} picks, got {len(picks)}")
    tau_true = meta.get("tau_true")
    stat, tau_hat = scan(picks, trim=TRIM, step=step)
    nan = float("nan")
    has_tau = tau_true is not None
    return {
        "run_id": run_id,
        "condition": meta.get("condition", ""),
        "source": source,
        "n": n,
        "tau_true": int(tau_true) if has_tau else nan,
        "stat": stat,
        "crit": crit,
        "rejected": int(stat > crit),
        "tau_hat": tau_hat,
        "tau_err_frac": abs(tau_hat - int(tau_true)) / n if has_tau else nan,
        "delta1_hat": mle_window(picks, 2, tau_hat),
        "delta2_hat": mle_window(picks, tau_hat, n),
        "delta_full_hat": mle(picks),
        "crit_source": crit_source,
        "crit_step": step,
        "delta1_hat_true_tau": mle_window(picks, 2, int(tau_true)) if has_tau else nan,
        "delta2_hat_true_tau": mle_window(picks, int(tau_true), n) if has_tau else nan,
        "parse_fail_frac": parse_fail,
    }


def _bound_flag(series):
    return series.map(lambda v: bool(at_bound(v)) if pd.notna(v) else False)


# ----------------------------------------------------------------------------- summaries
def binom_se(p, n):
    return float(math.sqrt(p * (1 - p) / n)) if n > 0 else float("nan")


def _se(x):
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def direction_summary(df):
    """
    Did delta go up after the shock? Counts and tests on delta2 - delta1, from the KNOWN-tau estimates
    when present (else the tau_hat estimates). Reported even when not significant.

    sign_test_p: two-sided exact binomial test of 'delta2 > delta1' against 1/2 (ties dropped).
    wilcoxon_p: two-sided Wilcoxon signed-rank test (NaN when fewer than 6 non-zero differences).
    """
    from scipy import stats

    if "delta1_hat_true_tau" in df and df["delta1_hat_true_tau"].notna().any():
        d1, d2, basis = df["delta1_hat_true_tau"], df["delta2_hat_true_tau"], "known tau"
    else:
        d1, d2, basis = df["delta1_hat"], df["delta2_hat"], "estimated tau_hat"
    diff = (d2 - d1).dropna()
    up = int((diff > 0).sum())
    down = int((diff < 0).sum())
    sign_p = float(stats.binomtest(up, up + down, 0.5).pvalue) if up + down > 0 else float("nan")
    try:
        wil_p = float(stats.wilcoxon(diff).pvalue) if (diff != 0).sum() >= 6 else float("nan")
    except ValueError:
        wil_p = float("nan")
    return {
        "basis": basis, "n_runs": int(len(diff)), "n_delta2_gt_delta1": up, "n_delta2_lt_delta1": down,
        "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
        "median_diff": float(diff.median()) if len(diff) else float("nan"),
        "sign_test_p": sign_p, "wilcoxon_p": wil_p,
    }


def summarise_group(df, label, note=""):
    """One comparison-table row for a set of runs in the contract schema (any source)."""
    n_runs = len(df)
    rej = df["rejected"].mean() if n_runs else float("nan")
    shock = df["tau_true"].notna() if "tau_true" in df else pd.Series(False, index=df.index)
    err = df.loc[shock, "tau_err_frac"] if "tau_err_frac" in df else pd.Series(dtype=float)
    err_rej = df.loc[shock & (df["rejected"] == 1), "tau_err_frac"] if "tau_err_frac" in df else err
    at_any_bound = (_bound_flag(df["delta1_hat"]) | _bound_flag(df["delta2_hat"])) if n_runs else pd.Series(dtype=bool)
    row = {
        "label": label,
        "n_runs": n_runs,
        "rejection_rate": rej,
        "rejection_se": binom_se(rej, n_runs) if n_runs else float("nan"),
        "mae_tau_frac_all": float(err.mean()) if len(err) else float("nan"),
        "mae_tau_frac_rejected": float(err_rej.mean()) if len(err_rej) else float("nan"),
        "median_delta1_hat": float(df["delta1_hat"].median()) if n_runs else float("nan"),
        "median_delta2_hat": float(df["delta2_hat"].median()) if n_runs else float("nan"),
    }
    has_true = "delta1_hat_true_tau" in df and df["delta1_hat_true_tau"].notna().any()
    row["median_delta1_hat_true_tau"] = float(df["delta1_hat_true_tau"].median()) if has_true else float("nan")
    row["median_delta2_hat_true_tau"] = float(df["delta2_hat_true_tau"].median()) if has_true else float("nan")
    if has_true:
        d = (df["delta2_hat_true_tau"] - df["delta1_hat_true_tau"]).dropna()
        row["frac_delta2_gt_delta1"] = float((d > 0).mean()) if len(d) else float("nan")
    else:
        d = df["delta2_hat"] - df["delta1_hat"]
        row["frac_delta2_gt_delta1"] = float((d > 0).mean()) if n_runs else float("nan")
    row["n_runs_at_bound"] = int(at_any_bound.sum()) if n_runs else 0
    row["crit_source"] = ",".join(sorted(set(df["crit_source"].dropna()))) if "crit_source" in df else ""
    row["note"] = note
    return row


# ----------------------------------------------------------------------------- M2: tables
def classical_table(cp):
    """Per-n summary of the classical control (rejection, location error, delta estimates, bound hits)."""
    rows = []
    for n, g in cp.groupby("n"):
        rej = g["rejected"].mean()
        rejected = g[g["rejected"] == 1]
        row = {
            "n": int(n), "runs": len(g), "tau_true": int(g["tau_true"].iloc[0]),
            "rejection_rate": rej, "rejection_se": binom_se(rej, len(g)),
            "mae_tau_frac_all": g["tau_err_frac"].mean(), "mae_tau_frac_all_se": _se(g["tau_err_frac"]),
            "mae_tau_frac_rejected": rejected["tau_err_frac"].mean() if len(rejected) else float("nan"),
            "mae_tau_frac_rejected_se": _se(rejected["tau_err_frac"]),
            "mean_delta1_hat": g["delta1_hat"].mean(), "delta1_hat_se": _se(g["delta1_hat"]),
            "median_delta1_hat": g["delta1_hat"].median(),
            "mean_delta2_hat": g["delta2_hat"].mean(), "delta2_hat_se": _se(g["delta2_hat"]),
            "median_delta2_hat": g["delta2_hat"].median(),
            "n_runs_at_bound": int((_bound_flag(g["delta1_hat"]) | _bound_flag(g["delta2_hat"])).sum()),
        }
        if "delta1_hat_true_tau" in g:
            row.update(
                median_delta1_hat_true_tau=g["delta1_hat_true_tau"].median(),
                median_delta2_hat_true_tau=g["delta2_hat_true_tau"].median(),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(results_dir=None):
    """
    Classical vs LLM comparison. Reads classical_cp.csv, power_curve_n500.csv, null_calibration.csv and
    Akshaya's persona_shift.csv / temperature_control.csv plus our temperature_shift.csv. Missing files give
    a 'not available yet' row, and stub rows are excluded (their count is noted).
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    rows = []

    cp_path = results_dir / "classical_cp.csv"
    if cp_path.exists():
        cp = pd.read_csv(cp_path)
        g = cp[cp["n"] == 500].copy()
        g["crit_source"] = "mc"
        rows.append(summarise_group(g, "classical PA, n=500, delta 0.5->3 (known shift)", "100 simulated runs"))

    nc_path = results_dir / "null_calibration.csv"
    if nc_path.exists():
        nc = pd.read_csv(nc_path)
        r = nc[nc["n"] == 500].iloc[0]
        rows.append({"label": "classical PA, n=500, NO shock (Type I error)", "n_runs": int(r["fresh_sims"]),
                     "rejection_rate": r["type1_mc"], "rejection_se": binom_se(r["type1_mc"], int(r["fresh_sims"])),
                     "crit_source": "mc", "note": "fresh simulations, MC critical value"})

    for fname, label in (
        ("persona_shift.csv", "LLM persona shift (popular -> newer)"),
        ("temperature_shift.csv", "LLM temperature shift"),
        ("temperature_control.csv", "LLM no-shock control"),
    ):
        path = results_dir / fname
        if not path.exists():
            rows.append({"label": label, "note": f"not available yet ({fname} missing)"})
            continue
        df = pd.read_csv(path)
        n_stub = int((df["source"] == "stub").sum()) if "source" in df else 0
        df = df[df["source"] != "stub"] if "source" in df else df
        if df.empty:
            rows.append({"label": label, "note": f"{fname} has only stub rows ({n_stub}); excluded"})
            continue
        for cond, g in df.groupby("condition"):
            note = f"{n_stub} stub rows excluded" if n_stub else ""
            tag = f"{label} [{cond}]" if df["condition"].nunique() > 1 else label
            rows.append(summarise_group(g, tag, note))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- markdown
def to_markdown(df, floatfmt="{:.3f}"):
    """Minimal GitHub-markdown table (avoids needing the optional `tabulate` package). Count columns
    (`n`, `runs`, `n_*`) print as integers even when a NaN row made pandas store them as floats."""
    count_cols = {c for c in df.columns if c in ("n", "runs") or str(c).startswith("n_")}

    def cell(v, col):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return str(int(round(v))) if col in count_cols else floatfmt.format(v)
        return "" if v is None else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v, c) for v, c in zip(row, df.columns)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    cp = pd.read_csv(RESULTS_DIR / "classical_cp.csv")
    classical = classical_table(cp)
    classical.to_csv(RESULTS_DIR / "classical_table.csv", index=False)

    power = pd.read_csv(RESULTS_DIR / "power_curve_n500.csv")
    comparison = comparison_table()
    comparison.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    cv = json.loads((RESULTS_DIR / "critical_values.json").read_text())
    crit_lines = [f"- n={k[1:]}: {v['0.05']:.3f} (source `{v['source']}`, grid step {v.get('step', 1)})"
                  for k, v in cv.items() if k != "asymptotic"]
    md = [
        "# Statistics tables (generated by `python -m src.analysis_stats`; do not edit by hand)\n",
        f"Asymptotic 5% critical value (Theorem 4.2, trim 0.1): {cv['asymptotic']['0.05']:.3f}. "
        "Critical values used:\n" + "\n".join(crit_lines) + "\n",
        "## Classical control (delta 0.5 -> 3 at tau = n/2, 100 runs each)\n",
        to_markdown(classical[["n", "runs", "rejection_rate", "rejection_se", "mae_tau_frac_all",
                               "mae_tau_frac_rejected", "median_delta1_hat", "median_delta2_hat",
                               "median_delta1_hat_true_tau", "median_delta2_hat_true_tau",
                               "n_runs_at_bound"]]) + "\n",
        "## Power curve at n = 500 (delta1 = 1; the delta2 = 1 row is the Type I error)\n",
        to_markdown(power[["delta2", "row_type", "power", "power_se", "mae_tau_frac",
                           "mae_tau_frac_rejected", "median_delta1_hat_true_tau",
                           "median_delta2_hat_true_tau"]]) + "\n",
        "## Classical vs LLM\n",
        to_markdown(comparison) + "\n",
    ]
    (REPORT_DIR / "tables_statistics.md").write_text("\n".join(md))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(classical.round(3).to_string(index=False))
    print()
    print(comparison.round(3).to_string(index=False))
    print(f"\nwrote {RESULTS_DIR / 'classical_table.csv'}, {RESULTS_DIR / 'comparison_table.csv'}, "
          f"{REPORT_DIR / 'tables_statistics.md'}")


if __name__ == "__main__":
    main()
