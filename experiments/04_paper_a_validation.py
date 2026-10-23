"""Paper A style validation of one finished run (Step 10).

    python experiments/04_paper_a_validation.py                   # baseline_n500_s0
    python experiments/04_paper_a_validation.py --run-id stub_baseline_n500_s0

Saves figures/<run_id>_degree_ccdf.png and figures/<run_id>_topk.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis_llm as an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="baseline_n500_s0")
    a = ap.parse_args()

    res = an.paper_a_validation(a.run_id)
    print(f"\n{a.run_id}: n={res['n']}  max degree={res['max_degree']}")
    print(f"power-law exponent: LLM {res['alpha_llm']:.2f} (xmin {res['xmin_llm']:.0f})   "
          f"BA {res['alpha_ba']:.2f} (xmin {res['xmin_ba']:.0f})")
    print(f"K-S distance between degree distributions (LLM vs BA): {res['ks_vs_ba']:.3f} (p={res['ks_p']:.3f})")
    print("share of picks among the k highest-degree candidates (LLM vs random pick):")
    for k, (hit, null) in res["top_k"].items():
        print(f"  top {k:>2}: {hit:.2f} vs {null:.2f}")
    first, last, chance = res["position"]
    print(f"position bias: first listed {first:.3f}, last listed {last:.3f}, chance {chance:.3f}")
    if res["max_degree"] > 0.5 * res["n"]:
        print("NOTE: one member has degree above n/2, so over half of all new nodes attached to the same "
              "member: a star-like network. Paper A also reports star-like networks for degree-only "
              "prompts. Report it as a finding.")


if __name__ == "__main__":
    main()
