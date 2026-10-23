import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "classical_cp", Path(__file__).resolve().parent.parent / "experiments" / "03_classical_cp.py"
)
classical_cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical_cp)


def _frame():
    # 4 runs: two rejected (errors 0.02, 0.04), two not rejected (errors 0.30, 0.34)
    return pd.DataFrame({
        "rejected": [1, 1, 0, 0],
        "tau_err_frac": [0.02, 0.04, 0.30, 0.34],
        "delta1_hat": [0.4, 0.5, 49.99999, 0.6],
        "delta2_hat": [3.0, 3.2, 2.0, 3.1],
        "delta1_hat_true_tau": [0.45, 0.5, 0.55, 0.5],
        "delta2_hat_true_tau": [3.0, 3.1, 2.9, 3.0],
        "delta1_at_bound": [0, 0, 1, 0],
        "delta2_at_bound": [0, 0, 0, 0],
        "delta1_at_bound_true_tau": [0, 0, 0, 0],
        "delta2_at_bound_true_tau": [0, 0, 0, 0],
    })


def test_mae_reported_over_all_runs_and_over_rejected_runs():
    out = classical_cp.summarise(_frame())
    assert out["row_type"] == "power"
    assert out["power"] == 0.5 and out["n_rejected"] == 2
    assert out["mae_tau_frac"] == pytest.approx(np.mean([0.02, 0.04, 0.30, 0.34]))
    assert out["mae_tau_frac_rejected"] == pytest.approx(0.03)  # Paper B Fig. 1 convention


def test_median_and_bound_counts_are_robust_to_a_boundary_estimate():
    out = classical_cp.summarise(_frame())
    assert out["n_delta1_at_bound"] == 1 and out["n_delta2_at_bound"] == 0
    assert out["mean_delta1_hat"] > 10          # one bound hit drags the mean...
    assert out["median_delta1_hat"] < 1.0       # ...but not the median
    assert out["median_delta2_hat_true_tau"] == pytest.approx(3.0)


def test_no_changepoint_row_has_blank_location_error():
    out = classical_cp.summarise(_frame(), has_changepoint=False)
    assert out["row_type"] == "type1_error"
    assert np.isnan(out["mae_tau_frac"]) and np.isnan(out["mae_tau_frac_rejected"])
    assert out["power"] == 0.5  # still the rejection rate, i.e. the Type I error
