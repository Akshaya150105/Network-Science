"""Step 4: check that the MLE recovers the true delta on classical PA networks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mle import mle  # noqa: E402
from src.pa_sim import simulate_pa  # noqa: E402

N = 2000
REPEATS = 100
DELTAS = [0.5, 1.0, 3.0]
BASE_SEED = 1000

rows = []
for delta in DELTAS:
    estimates = np.array(
        [mle(simulate_pa(N, delta, np.random.default_rng(BASE_SEED + r))) for r in range(REPEATS)]
    )
    mean = estimates.mean()
    rows.append(
        {
            "delta": delta,
            "n": N,
            "repeats": REPEATS,
            "mean": mean,
            "std": estimates.std(ddof=1),
            "bias": mean - delta,
            "rel_bias": (mean - delta) / delta,
        }
    )
    print(f"delta={delta}: mean={mean:.4f} std={rows[-1]['std']:.4f} bias={rows[-1]['bias']:+.4f}")

out = Path(__file__).resolve().parent.parent / "results" / "mle_validation.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"saved {out}")
