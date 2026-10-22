"""Merge phase M2: false-positive rate of the changepoint test on LLM runs with NO shock.

Applies the test to tempctrl_n500_s* and baseline_n500_s0. Their rejection rate is the false-positive
rate on LLM data. A rate well above 0.05 means the LLM network is not exactly PA(delta), and every
other LLM result has to be read with that in mind.

    python experiments/06_temperature_control_analyze.py            # writes results/temperature_control.csv
    python experiments/06_temperature_control_analyze.py --stub     # writes results/stub_temperature_control.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis_llm as an
from src import llm_growth as lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    prefix = "stub_" if a.stub else ""
    names = [p.name[:-len(".jsonl")] for p in lg.LOG_DIR.glob(f"{prefix}tempctrl_n{a.n}_s*.jsonl")]
    names += [p.name[:-len(".jsonl")] for p in lg.LOG_DIR.glob(f"{prefix}baseline_n{a.n}_s0.jsonl")]
    run_ids = sorted(names)
    if not run_ids:
        sys.exit("no control logs found; run experiments/06_temperature_shift_generate.py first")
    an.analyze_runs(run_ids, an.ROOT / "results" / f"{prefix}temperature_control.csv", crit_key=f"n{a.n}")
    if a.figures:
        out = an.FIG_DIR / f"{run_ids[0]}_rolling_delta.png"
        an.rolling_delta_plot(run_ids[0], out)
        print("wrote", out)


if __name__ == "__main__":
    main()
