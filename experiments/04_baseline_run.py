"""Baseline run (Step 10): one 500-node network, baseline prompt, constant temperature.

    python experiments/04_baseline_run.py            # real model (set LLM_* variables first)
    python experiments/04_baseline_run.py --stub     # free dry run

Paper A's main results use temperature 0.5, so that is the default.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis_llm as an
from src import llm_growth as lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()

    rid = ("stub_" if a.stub else "") + f"baseline_llama32_n{a.n}_s{a.seed}"
    model = "stub" if a.stub else lg.default_model()
    lg.grow_network(a.n, lambda t: "baseline", lambda t: a.temp, lambda t: model, rid, a.seed,
                    "stub_pa" if a.stub else "llm")
    print(lg.stats_line())
    an.check_run(rid)


if __name__ == "__main__":
    main()
