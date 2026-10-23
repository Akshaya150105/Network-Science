"""Merge phase M1: changepoint test on the persona-shift runs.

Needs Valincia's src/mle.py, src/changepoint.py and results/critical_values.json (pull her branch first).

    python experiments/05_persona_shift_analyze.py              # writes results/persona_shift.csv
    python experiments/05_persona_shift_analyze.py --stub       # writes results/stub_persona_shift.csv
    python experiments/05_persona_shift_analyze.py --figures    # also saves a delta_hat-over-time plot
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
    run_ids = sorted(p.name[:-len(".jsonl")] for p in lg.LOG_DIR.glob(f"{prefix}persona_n{a.n}_s*.jsonl"))
    if not run_ids:
        sys.exit("no persona logs found; run experiments/05_persona_shift.py first")
    an.analyze_runs(run_ids, an.ROOT / "results" / f"{prefix}persona_shift.csv", crit_key=f"n{a.n}")
    if a.figures:
        out = an.FIG_DIR / f"{run_ids[0]}_rolling_delta.png"
        an.rolling_delta_plot(run_ids[0], out)
        print("wrote", out)


if __name__ == "__main__":
    main()
