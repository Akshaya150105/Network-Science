import json

import numpy as np
import pandas as pd
import pytest

from src import analysis_stats as A
from tests.make_fixture import write_fixture


@pytest.fixture()
def logs(tmp_path):
    for s in (0, 1, 10, 2):
        write_fixture(tmp_path, "temp", n=300, seed=s, delta1=0.5, delta2=3.0)
    write_fixture(tmp_path, "tempctrl", n=300, seed=0, shock=False, delta1=1.0)
    write_fixture(tmp_path, "temp", n=120, seed=99, parse_fail_every=10)
    return tmp_path


def test_read_picks_matches_contract(logs):
    picks = A.read_picks("fixture_temp_n300_s0", logs)
    assert len(picks) == 298 and [k for _, k in picks] == list(range(3, 301))
    assert picks[0][0] == 2


def test_parse_failed_rows_are_kept_and_counted(logs):
    run = "fixture_temp_n120_s99"
    assert len(A.read_picks(run, logs)) == 118          # ALL rows, including parse_ok = false
    assert A.parse_fail_fraction(run, logs) == pytest.approx(12 / 118)


def test_discover_runs_sorts_by_numeric_seed(logs):
    ids = A.discover_runs("fixture_temp_n300_s", logs)
    assert [i.split("_s")[-1] for i in ids] == ["0", "1", "2", "10"]


def test_run_source_labels_stub_and_fixture_runs():
    assert A.run_source("fixture_temp_n500_s0", {"chooser": "fixture"}) == "stub"
    assert A.run_source("stub_temp_n500_s0", {"chooser": "stub_pa"}) == "stub"
    assert A.run_source("temp_n500_s0", {"chooser": "llm"}) == "llm"
    assert A.run_source("temp_n500_s0", {"chooser": "stub_pa"}) == "stub"


def test_critical_value_refuses_asymptotic_unless_allowed(tmp_path):
    path = tmp_path / "cv.json"
    path.write_text(json.dumps({"n500": {"0.05": 9.0, "source": "asymptotic", "step": 1}}))
    with pytest.raises(ValueError, match="not 'mc'"):
        A.load_critical_value(500, path)
    assert A.load_critical_value(500, path, allow_asymptotic=True) == (9.0, 1, "asymptotic")
    path.write_text(json.dumps({"n500": {"0.05": 9.2, "source": "mc", "step": 5}}))
    assert A.load_critical_value(500, path) == (9.2, 5, "mc")      # step travels with the value


def test_analyze_picks_row_has_contract_columns_and_true_tau_estimates(logs):
    run = "fixture_temp_n300_s0"
    meta = A.read_meta(run, logs)
    row = A.analyze_picks(run, A.read_picks(run, logs), meta, crit=9.19, step=1, crit_source="mc", source="stub")
    assert list(row)[: len(A.CONTRACT_COLUMNS)] == A.CONTRACT_COLUMNS
    assert row["tau_true"] == 150 and row["rejected"] == int(row["stat"] > 9.19)
    assert row["tau_err_frac"] == pytest.approx(abs(row["tau_hat"] - 150) / 300)
    assert row["delta2_hat_true_tau"] > row["delta1_hat_true_tau"]       # fixture shifts 0.5 -> 3


def test_no_shock_run_has_blank_tau_fields(logs):
    run = "fixture_tempctrl_n300_s0"
    meta = A.read_meta(run, logs)
    assert meta["tau_true"] is None
    row = A.analyze_picks(run, A.read_picks(run, logs), meta, 9.19, 1, "mc", "stub")
    assert np.isnan(row["tau_true"]) and np.isnan(row["tau_err_frac"]) and np.isnan(row["delta1_hat_true_tau"])


def test_wrong_pick_count_is_rejected(logs):
    run = "fixture_temp_n300_s0"
    with pytest.raises(ValueError, match="expected 298 picks"):
        A.analyze_picks(run, A.read_picks(run, logs)[:-1], A.read_meta(run, logs), 9.19, 1, "mc", "stub")


def test_direction_summary_counts_and_tests():
    df = pd.DataFrame({"delta1_hat_true_tau": [0.5] * 8, "delta2_hat_true_tau": [1, 2, 3, 2, 1.5, 4, 2.5, 0.4],
                       "delta1_hat": [0] * 8, "delta2_hat": [0] * 8})
    d = A.direction_summary(df)
    assert d["basis"] == "known tau" and (d["n_delta2_gt_delta1"], d["n_delta2_lt_delta1"]) == (7, 1)
    assert d["sign_test_p"] == pytest.approx(0.0703125)             # exact binomial, 7 of 8 -> 2 * 9/256
    no_true = df.drop(columns=["delta1_hat_true_tau", "delta2_hat_true_tau"])
    assert A.direction_summary(no_true)["basis"] == "estimated tau_hat"


def test_comparison_table_excludes_stub_rows_and_flags_missing_files(tmp_path):
    cols = A.CONTRACT_COLUMNS
    real = {"run_id": "persona_n500_s0", "condition": "persona", "source": "llm", "n": 500, "tau_true": 250,
            "stat": 12.0, "crit": 9.19, "rejected": 1, "tau_hat": 240, "tau_err_frac": 0.02,
            "delta1_hat": 0.8, "delta2_hat": 4.0, "delta_full_hat": 1.5}
    stub = {**real, "run_id": "stub_persona_n500_s0", "source": "stub", "delta2_hat": 99.0}
    pd.DataFrame([real, stub])[cols].to_csv(tmp_path / "persona_shift.csv", index=False)
    table = A.comparison_table(tmp_path)
    persona = table[table["label"].str.contains("persona")].iloc[0]
    assert persona["n_runs"] == 1 and persona["median_delta2_hat"] == 4.0   # stub row (99) not counted
    assert "1 stub rows excluded" in persona["note"]
    missing = table[table["label"].str.contains("temperature shift")].iloc[0]
    assert "not available yet" in missing["note"]


def test_to_markdown_renders_header_and_blank_nan():
    md = A.to_markdown(pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}))
    assert md.splitlines()[0] == "| a | b |" and md.splitlines()[3] == "|  | y |"


def test_to_markdown_prints_count_columns_as_integers():
    md = A.to_markdown(pd.DataFrame({"n_runs": [100.0, np.nan], "rate": [0.5, 1.0]}))
    assert md.splitlines()[2] == "| 100 | 0.500 |" and md.splitlines()[3] == "|  | 1.000 |"
