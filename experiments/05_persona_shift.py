"""Persona shift (Step 11): 'popular' prompt for t <= n//2, 'newer' prompt after. 10 replicates.

    python experiments/05_persona_shift.py --stub            # free dry run (writes stub_* logs)
    python experiments/05_persona_shift.py                   # real run (first time)
    python experiments/05_persona_shift.py --resume          # continue after an interruption

Runs: persona_n500_s0 .. s9. Existing logs are never overwritten: without --resume the script
refuses to touch an existing run.
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
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.5)
    a = ap.parse_args()

    tau = a.n // 2
    model = "stub" if a.stub else lg.default_model()
    prefix = "stub_" if a.stub else ""
    specs = [dict(n=a.n, prompt_schedule=lambda t, tau=tau: "popular" if t <= tau else "newer",
                  temperature_schedule=lambda t: a.temp, model_schedule=lambda t: model,
                  run_id=f"{prefix}persona_n{a.n}_s{a.base_seed + r}", seed=a.base_seed + r)
             for r in range(a.reps)]
    lg.run_campaign(specs, a.resume, "stub_pa" if a.stub else "llm")

    print("\nChecks (top5 share should differ between the two halves):")
    for s in specs:
        an.check_run(s["run_id"])


if __name__ == "__main__":
    main()
