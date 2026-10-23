"""
Write fake runs in the log contract format (logs/{run_id}.jsonl and .meta.json) from the classical simulator.

Used only to test the analysis scripts before real LLM logs exist. Run ids get the `fixture_` prefix and the
meta file says chooser = "fixture", so the analysis labels them source = "stub" and keeps them out of the final
tables. By default nothing is written into logs/; pass --out explicitly if you really want that.

    python tests/make_fixture.py --out /tmp/fixture_logs --condition temp --runs 10 --delta1 0.5 --delta2 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def write_fixture(logs_dir, condition="temp", n=500, seed=0, delta1=0.5, delta2=3.0, tau=None,
                  shock=True, parse_fail_every=0):
    """Write one fixture run and return its run_id. shock=False gives a constant-delta (no-shock) run."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tau = n // 2 if tau is None else tau
    run_id = f"fixture_{condition}_n{n}_s{seed}"
    rng = np.random.default_rng(seed)
    degrees = [2, 2]  # seed graph G(2), as in the contract
    lines = []
    for t in range(3, n + 1):
        delta = delta1 if (t <= tau or not shock) else delta2
        weights = np.asarray(degrees, float) + delta
        order = rng.permutation(len(degrees))                 # candidates shown in shuffled order
        shown = [degrees[i] for i in order]
        idx = int(rng.choice(len(shown), p=weights[order] / weights[order].sum()))
        chosen_node = int(order[idx])
        parse_ok = not (parse_fail_every and t % parse_fail_every == 0)
        lines.append(json.dumps({
            "run_id": run_id, "t": t, "candidate_degrees": shown, "chosen_idx": idx,
            "chosen_degree": degrees[chosen_node], "temperature": 0.2 if (t <= tau or not shock) else 1.2,
            "model": "fixture", "prompt_id": "baseline", "raw_response": str(idx + 1), "parse_ok": parse_ok,
        }))
        degrees[chosen_node] += 1
        degrees.append(1)
    (logs_dir / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n")
    meta = {"run_id": run_id, "condition": condition, "n": n, "seed": seed,
            "tau_true": tau if shock else None, "model_schedule": "fixture", "prompt_schedule": "baseline",
            "temperature_schedule": "0.2 then 1.2" if shock else "constant", "chooser": "fixture"}
    (logs_dir / f"{run_id}.meta.json").write_text(json.dumps(meta, indent=2))
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", default="temp")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--delta1", type=float, default=0.5)
    ap.add_argument("--delta2", type=float, default=3.0)
    ap.add_argument("--no-shock", action="store_true")
    a = ap.parse_args()
    for s in range(a.runs):
        print(write_fixture(a.out, a.condition, a.n, s, a.delta1, a.delta2, shock=not a.no_shock))
