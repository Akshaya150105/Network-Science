"""Guards on the saved calibration, so a stale or coarse-grid critical value cannot slip back in."""
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"


def test_critical_values_are_monte_carlo_on_the_full_grid():
    cv = json.loads((RES / "critical_values.json").read_text())
    for key in ("n500", "n1000", "n2000"):
        assert cv[key]["source"] == "mc"
        assert cv[key]["step"] == 1            # every split, as in Paper B
        assert 8.5 < cv[key]["0.05"] < 9.8     # near the asymptotic 9.22 (bootstrap SE 0.12 to 0.17)


def test_type_one_error_is_near_nominal_across_delta():
    t = pd.read_csv(RES / "type1_across_delta.csv")
    assert set(t["delta"]) == {-0.5, 0.0, 1.0, 2.0}
    assert t["type1_mc"].between(0.02, 0.09).all()
    assert (t["step"] == 1).all()
