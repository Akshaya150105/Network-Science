"""Temperature shift (Step 12): baseline prompt, temperature LOW for t <= n//2, HIGH after.
Also the no-shock control: constant temperature. 10 replicates each.

    python experiments/06_temperature_shift_generate.py --stub
    python experiments/06_temperature_shift_generate.py
    python experiments/06_temperature_shift_generate.py --resume
    python experiments/06_temperature_shift_generate.py --high 1.0     # if your provider caps temperature at 1.0

Runs: temp_n500_s0..s9 (shift) and tempctrl_n500_s0..s9 (control).
Defaults follow Paper A's temperatures: 0.5 -> 1.5, control at 1.0 (the middle value).
If you change --low/--high, tell Valincia: the experiment description changes, her code does not.
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
    ap.add_argument("--low", type=float, default=0.5)
    ap.add_argument("--high", type=float, default=1.5)
    ap.add_argument("--control", type=float, default=1.0)
    a = ap.parse_args()

    tau = a.n // 2
    model = "stub" if a.stub else lg.default_model()
    prefix = "stub_" if a.stub else ""
    common = dict(n=a.n, prompt_schedule=lambda t: "baseline", model_schedule=lambda t: model)
    specs = []
    for r in range(a.reps):
        s = a.base_seed + r
        specs.append(dict(common, temperature_schedule=lambda t, tau=tau: a.low if t <= tau else a.high,
                          run_id=f"{prefix}temp_n{a.n}_s{s}", seed=s))
    for r in range(a.reps):
        s = a.base_seed + r
        specs.append(dict(common, temperature_schedule=lambda t: a.control,
                          run_id=f"{prefix}tempctrl_n{a.n}_s{s}", seed=s))
    lg.run_campaign(specs, a.resume, "stub_pa" if a.stub else "llm")

    print("\nChecks:")
    for s in specs:
        an.check_run(s["run_id"])


if __name__ == "__main__":
    main()
