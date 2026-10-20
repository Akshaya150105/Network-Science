import importlib.util
from pathlib import Path

import pytest

from src import analysis_stats as A
from tests.make_fixture import write_fixture

_spec = importlib.util.spec_from_file_location(
    "rolling_delta", Path(__file__).resolve().parent.parent / "experiments" / "08_rolling_delta.py"
)
rolling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rolling)


def test_fixture_runs_are_written_to_stub_files_only(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    for s in range(3):
        write_fixture(logs, "temp", n=300, seed=s, delta1=0.5, delta2=3.0)
    monkeypatch.setattr(rolling, "FIG", tmp_path / "figs")
    monkeypatch.setattr(A, "RESULTS_DIR", tmp_path / "res")
    (tmp_path / "res").mkdir()
    rolling.real_runs("fixture_temp_n300_s", logs, window=80, step=20)
    assert (tmp_path / "figs" / "fig_rolling_delta_fixture_temp_n300_s_stub.png").exists()
    csv = tmp_path / "res" / "rolling_delta_fixture_temp_n300_s_stub.csv"
    assert csv.exists()
    import pandas as pd
    df = pd.read_csv(csv)
    assert set(df["run_id"]) == {f"fixture_temp_n300_s{s}" for s in range(3)}
    assert {"k_center", "delta_hat", "se", "at_bound"} <= set(df.columns)


def test_missing_logs_give_a_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="No logs matching"):
        rolling.real_runs("temp_n500_s", tmp_path, window=100, step=10)
