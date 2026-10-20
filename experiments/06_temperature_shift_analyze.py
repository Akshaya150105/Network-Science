"""
M1: temperature-shift analysis (Valincia). Reads Akshaya's runs temp_n500_s* and writes
results/temperature_shift.csv in the contract schema.

    python experiments/06_temperature_shift_analyze.py

For each run: load_picks -> scan (with the n500 critical value AND the grid step it was calibrated with)
-> window MLEs. The critical value must have source "mc" (an asymptotic value is not calibrated at n = 500;
pass --allow-asymptotic to override, and the output then records crit_source = asymptotic).

Expectation from Paper A: higher temperature -> larger power-law exponent (gamma = 3 + delta), so delta2 > delta1
after the shift from low to high temperature. The direction is reported even if it is not significant.
A delta_hat sitting on a search bound (-0.99 or 50) means model mismatch, not a real estimate, so those runs
are counted.

Stub or fixture runs (--prefix stub_temp_n500_s ...) are labelled source = stub and written to
results/temperature_shift_stub.csv, never to the final file.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis_stats as A  # noqa: E402
from src.mle import at_bound  # noqa: E402

FINAL = A.RESULTS_DIR / "temperature_shift.csv"
STUB = A.RESULTS_DIR / "temperature_shift_stub.csv"


def analyze_prefix(prefix, logs_dir=None, allow_asymptotic=False, crit_path=None):
    """Analyse every run_id starting with `prefix`; return the results DataFrame."""
    run_ids = A.discover_runs(prefix, logs_dir)
    if not run_ids:
        raise SystemExit(
            f"No logs matching {prefix}*.meta.json in {logs_dir or A.LOGS_DIR}. "
            "The real runs come from Akshaya's Step 12 (experiments/06_temperature_shift_generate.py)."
        )
    load_picks, load_meta = A.get_loaders(logs_dir)
    rows = []
    for run_id in run_ids:
        meta = load_meta(run_id)
        crit, step, crit_source = A.load_critical_value(meta["n"], crit_path, allow_asymptotic)
        rows.append(A.analyze_picks(
            run_id, load_picks(run_id), meta, crit, step, crit_source, A.run_source(run_id, meta),
            parse_fail=A.parse_fail_fraction(run_id, logs_dir),
        ))
    return pd.DataFrame(rows)[A.CONTRACT_COLUMNS + A.EXTRA_COLUMNS]


def report(df):
    n = len(df)
    rej = df["rejected"].mean()
    print(f"\n{n} runs; critical value {df['crit'].iloc[0]:.3f} (source {df['crit_source'].iloc[0]}, "
          f"grid step {df['crit_step'].iloc[0]})")
    print(f"rejection rate {rej:.2f} +- {A.binom_se(rej, n):.2f} (binomial SE)")
    shock = df["tau_true"].notna()
    if shock.any():
        e = df.loc[shock, "tau_err_frac"]
        er = df.loc[shock & (df["rejected"] == 1), "tau_err_frac"]
        print(f"|tau_hat - tau|/n: mean {e.mean():.3f} over all runs; "
              f"{er.mean() if len(er) else float('nan'):.3f} over the {len(er)} rejected runs")
    d = A.direction_summary(df)
    print(f"direction of delta2 - delta1 ({d['basis']}): delta2 > delta1 in {d['n_delta2_gt_delta1']}/{d['n_runs']} runs, "
          f"mean {d['mean_diff']:+.3f}, median {d['median_diff']:+.3f}; "
          f"sign test p = {d['sign_test_p']:.3f}, Wilcoxon p = {d['wilcoxon_p']:.3f}")
    bound = df["delta1_hat"].map(at_bound) | df["delta2_hat"].map(at_bound)
    print(f"runs with a delta_hat on a search bound: {int(bound.sum())}/{n}")
    print(f"parse-failure fraction: mean {df['parse_fail_frac'].mean():.3f}, max {df['parse_fail_frac'].max():.3f} "
          "(failed parses are random fallbacks and dilute any effect)")
    print("\nPaper A expectation: delta2 > delta1 (higher temperature -> larger gamma). "
          "Report the direction above even if the test is not significant.")
    return d, int(bound.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="temp_n500_s", help="run_id prefix to analyse")
    ap.add_argument("--logs-dir", default=None, help="folder with the logs (default: logs/)")
    ap.add_argument("--out", default=None, help="output CSV (default results/temperature_shift.csv)")
    ap.add_argument("--allow-asymptotic", action="store_true")
    args = ap.parse_args()

    df = analyze_prefix(args.prefix, args.logs_dir, args.allow_asymptotic)
    out = Path(args.out) if args.out else FINAL
    if (df["source"] == "stub").any():
        if out == FINAL:
            out = STUB
        print(f"WARNING: stub/fixture runs found. They are labelled source=stub and written to {out}, "
              "which must NOT be used in the final tables.")
    pd.set_option("display.width", 220)
    print(df[["run_id", "source", "stat", "crit", "rejected", "tau_hat", "tau_err_frac", "delta1_hat",
              "delta2_hat", "delta1_hat_true_tau", "delta2_hat_true_tau"]].round(3).to_string(index=False))
    d, n_bound = report(df)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    summary = {"n_runs": len(df), "rejection_rate": df["rejected"].mean(), "crit": df["crit"].iloc[0],
               "crit_source": df["crit_source"].iloc[0], "n_runs_at_bound": n_bound, **d}
    pd.DataFrame([summary]).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"\nwrote {out} and {out.stem}_summary.csv")
