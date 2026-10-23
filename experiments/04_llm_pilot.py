"""Pilot runs. Run from the repo root:

    python experiments/04_llm_pilot.py                 # 50-node pilot
    python experiments/04_llm_pilot.py --sweep         # 3 x 200-node runs at temperatures 0.5, 1.0, 1.5
    python experiments/04_llm_pilot.py --stub          # free dry run, no model needed

Prints parse-failure rate, seconds per call, position bias, how often the pick is a top-degree
candidate, and an estimate of how long the full campaign will take.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis_llm as an
from src import llm_growth as lg
from src.prompts import build_prompt

CAMPAIGN_RUNS = 31  # 10 persona + 10 temp + 10 control + 1 baseline, each 500 nodes


def one_run(tag, n, temp, seed, stub):
    rid = ("stub_" if stub else "") + f"{tag}_n{n}_s{seed}"
    model = "stub" if stub else lg.default_model()
    lg.grow_network(n, lambda t: "baseline", lambda t: temp, lambda t: model, rid, seed,
                    "stub_pa" if stub else "llm")
    return an.check_run(rid)


def estimate_campaign_hours():
    s = lg.STATS
    if not s["calls"] or not s["seconds"]:
        return None
    chars_per_sec = s["prompt_chars"] / s["seconds"]
    one_run_chars = sum(len(build_prompt("baseline", [3] * (t - 1))) for t in range(3, 501))
    return CAMPAIGN_RUNS * one_run_chars / chars_per_sec / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="use a new seed for a fresh pilot")
    a = ap.parse_args()

    if a.sweep:
        for tag, temp in (("pilott05", 0.5), ("pilott10", 1.0), ("pilott15", 1.5)):
            print(f"--- constant temperature {temp}")
            one_run(tag, 200, temp, a.seed, a.stub)
        print("\nIf top5 share is about the same at 0.5 and 1.5, temperature barely moves the model's "
              "attachment behaviour and the temperature-shift experiment is unlikely to show anything.")
    else:
        one_run("pilot", 50, 0.5, a.seed, a.stub)

    s = lg.STATS
    if s["calls"]:
        print(f"\nnew model calls: {s['calls']}   average seconds per call: {s['seconds'] / s['calls']:.2f}")
    print(lg.stats_line())
    hours = estimate_campaign_hours()
    if hours:
        print(f"Rough estimate for the full campaign ({CAMPAIGN_RUNS} runs of 500 nodes): {hours:.1f} hours "
              "(scaled from this pilot's speed; prompts get longer as the network grows).")


if __name__ == "__main__":
    main()
