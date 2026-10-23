"""All tests are free: the stub chooser, plus a fake local HTTP server for the real chooser."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis_llm as an
from src import llm_growth as lg

CONTRACT_FIELDS = {"run_id", "t", "candidate_degrees", "chosen_idx", "chosen_degree", "temperature",
                   "model", "prompt_id", "raw_response", "parse_ok"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(lg, "CACHE_PATH", tmp_path / "cache" / "responses.jsonl")
    monkeypatch.setattr(lg, "_cache", None)
    for k in lg.STATS:
        lg.STATS[k] = 0
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")


def persona(tau):
    return lambda t: "popular" if t <= tau else "newer"


def grow_stub(run_id, n=60, seed=3, tau=30):
    return lg.grow_network(n, persona(tau), lambda t: 0.5, lambda t: "stub", run_id, seed, "stub_pa")


# ---------- stub chooser ----------

def test_log_shape_and_meta():
    picks = grow_stub("stub_persona_n60_s3")
    rows = lg.read_rows("stub_persona_n60_s3")
    assert len(rows) == 58 and len(picks) == 58 == len(lg.load_picks("stub_persona_n60_s3"))
    for r in rows:
        assert set(r) == CONTRACT_FIELDS
        assert r["candidate_degrees"][r["chosen_idx"]] == r["chosen_degree"]
        assert len(r["candidate_degrees"]) == r["t"] - 1
        assert r["model"] == "stub" and r["parse_ok"] is True
    meta = lg.load_meta("stub_persona_n60_s3")
    assert meta["condition"] == "persona" and meta["n"] == 60 and meta["seed"] == 3
    assert meta["tau_true"] == 30 and meta["chooser"] == "stub_pa"


def test_no_shock_run_has_null_tau():
    lg.grow_network(30, lambda t: "baseline", lambda t: 1.0, lambda t: "stub", "stub_tempctrl_n30_s0", 0, "stub_pa")
    assert lg.load_meta("stub_tempctrl_n30_s0")["tau_true"] is None


def test_replayed_degrees_equal_real_degrees():
    n, seed = 60, 3
    grow_stub("stub_persona_n60_s3", n, seed)
    rows = lg.read_rows("stub_persona_n60_s3")
    # independent node-level replay using the same shuffles
    deg = [2, 2]
    for r in rows:
        perm = lg._rng(seed, r["t"], 0).permutation(r["t"] - 1)
        assert [deg[v] for v in perm] == r["candidate_degrees"]
        deg[int(perm[r["chosen_idx"]])] += 1
        deg.append(1)
    assert sorted(an.replay_degrees(rows)) == sorted(deg)
    assert sum(deg) == 2 * n  # seed total 4, plus 2 per new node


def test_same_seed_same_log():
    grow_stub("stub_persona_n60_s3")
    grow_stub("stub_persona2_n60_s3")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "run_id"} for r in rows]
    assert strip(lg.read_rows("stub_persona_n60_s3")) == strip(lg.read_rows("stub_persona2_n60_s3"))


def test_forced_parse_failure_falls_back(monkeypatch):
    monkeypatch.setattr(lg, "stub_response", lambda *a, **k: "I cannot decide")
    grow_stub("stub_persona_n20_s1", n=20, seed=1, tau=10)
    rows = lg.read_rows("stub_persona_n20_s1")
    assert all(r["parse_ok"] is False for r in rows)
    assert all(0 <= r["chosen_idx"] < len(r["candidate_degrees"]) for r in rows)
    assert lg.STATS["parse_fail"] == 18 and lg.STATS["retries"] == 18 * lg.MAX_TRIES
    an.replay_degrees(rows)  # fallback picks still give a consistent network


def test_resume_continues_an_interrupted_run():
    grow_stub("stub_persona_n50_s7", n=50, seed=7, tau=25)
    full = lg.LOG_DIR / "stub_persona_n50_s7.jsonl"
    reference = full.read_text()
    lines = reference.splitlines()
    full.write_text("\n".join(lines[:20]) + "\n")  # pretend the run stopped after 20 steps
    grow_stub("stub_persona_n50_s7", n=50, seed=7, tau=25)
    assert full.read_text() == reference


def test_stub_prefix_is_enforced():
    with pytest.raises(ValueError):
        lg.grow_network(10, lambda t: "baseline", lambda t: 0.5, lambda t: "x", "persona_n10_s0", 0, "stub_pa")
    with pytest.raises(ValueError):
        lg.grow_network(10, lambda t: "baseline", lambda t: 0.5, lambda t: "x", "stub_persona_n10_s0", 0, "llm")


def test_campaign_refuses_to_overwrite_without_resume():
    spec = dict(n=10, prompt_schedule=lambda t: "baseline", temperature_schedule=lambda t: 0.5,
                model_schedule=lambda t: "stub", run_id="stub_persona_n10_s0", seed=0)
    lg.run_campaign([spec], resume=False, chooser="stub_pa")
    with pytest.raises(SystemExit):
        lg.run_campaign([spec], resume=False, chooser="stub_pa")
    lg.run_campaign([spec], resume=True, chooser="stub_pa")  # fine


def test_check_run_and_validation_helpers():
    grow_stub("stub_persona_n60_s3")
    c = an.check_run("stub_persona_n60_s3")
    assert c["complete"] and c["parse_fail_rate"] == 0
    rows = lg.read_rows("stub_persona_n60_s3")
    first, last, chance = an.position_bias(rows)
    assert 0 <= first <= 1 and 0 < chance < 1
    for k, (hit, null) in an.top_k_rates(rows).items():
        assert 0 <= hit <= 1 and 0 < null <= 1


# ---------- parser ----------

@pytest.mark.parametrize("text,expected", [
    ('[{"name": "member_7", "reason": "most connected"}]', 6),
    ('{"name": "member_3", "reason": "x"}', 2),
    ('```json\n[{"name": "member_12", "reason": "ok"}]\n```', 11),
    ('<think>member_2 looks good</think>[{"name": "member_9", "reason": "x"}]', 8),
    ("I pick member_4 because it is popular", 3),
    ('[{"name": "member_999", "reason": "x"}]', None),
    ("no idea", None),
    ("", None),
])
def test_parse_choice(text, expected):
    assert lg.parse_choice(text, 20) == expected


# ---------- real chooser against a fake OpenAI-compatible server ----------

class FakeLLM(BaseHTTPRequestHandler):
    mode = "ok"       # ok | garbage | rate_limit_once | bad_model
    calls = 0
    limited = False

    def do_POST(self):
        cls = type(self)
        cls.calls += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if cls.mode == "bad_model":
            return self._send(400, {"error": "model not found"})
        if cls.mode == "rate_limit_once" and not cls.limited:
            cls.limited = True
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        text = "I do not know" if cls.mode == "garbage" else '[{"name": "member_1", "reason": "first"}]'
        self._send(200, {"choices": [{"message": {"content": text}}],
                         "usage": {"prompt_tokens": len(body["messages"][0]["content"]) // 3, "completion_tokens": 12}})

    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_llm(monkeypatch):
    FakeLLM.mode, FakeLLM.calls, FakeLLM.limited = "ok", 0, False
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
    yield FakeLLM
    server.shutdown()


def grow_real(run_id, n=12, model="fake"):
    return lg.grow_network(n, lambda t: "baseline", lambda t: 0.5, lambda t: model, run_id, 0, "llm")


def test_real_chooser_parses_logs_and_caches(fake_llm):
    grow_real("baseline_n12_s0")
    rows = lg.read_rows("baseline_n12_s0")
    assert len(rows) == 10 and all(r["parse_ok"] and r["chosen_idx"] == 0 for r in rows)
    assert all(r["model"] == "fake" for r in rows)
    assert fake_llm.calls == 10 and lg.STATS["calls"] == 10
    # delete the run but keep the cache: the same run costs no new calls
    (lg.LOG_DIR / "baseline_n12_s0.jsonl").unlink()
    (lg.LOG_DIR / "baseline_n12_s0.meta.json").unlink()
    grow_real("baseline_n12_s0")
    assert fake_llm.calls == 10 and lg.STATS["cached"] == 10


def test_cache_is_not_shared_between_runs(fake_llm):
    """Step 3 sends the identical prompt in every run; each run must still get its own sample."""
    grow_real("baseline_n12_s0")
    grow_real("baseline_n12_s1")
    assert fake_llm.calls == 20


def test_rate_limit_is_retried(fake_llm):
    fake_llm.mode = "rate_limit_once"
    grow_real("baseline_n12_s0")
    assert all(r["parse_ok"] for r in lg.read_rows("baseline_n12_s0"))


def test_garbage_replies_retry_then_fall_back(fake_llm):
    fake_llm.mode = "garbage"
    grow_real("baseline_n12_s0")
    rows = lg.read_rows("baseline_n12_s0")
    assert all(r["parse_ok"] is False for r in rows)
    assert fake_llm.calls == 10 * lg.MAX_TRIES


def test_prompt_too_long_for_context_stops_the_run(fake_llm, monkeypatch):
    monkeypatch.setenv("LLM_NUM_CTX", "300")  # fake server reports len(prompt)//3 tokens
    with pytest.raises(RuntimeError, match="context"):
        grow_real("baseline_n40_s0", n=40)
    rows = lg.read_rows("baseline_n40_s0") if (lg.LOG_DIR / "baseline_n40_s0.jsonl").exists() else []
    assert len(rows) < 38  # stopped early, and nothing from the failing step was logged
    monkeypatch.setenv("LLM_NUM_CTX", "8192")
    grow_real("baseline_n40_s0", n=40)  # resumes and finishes once the context is big enough
    assert len(lg.read_rows("baseline_n40_s0")) == 38


def test_http_error_is_not_hidden_as_a_random_choice(fake_llm):
    fake_llm.mode = "bad_model"
    with pytest.raises(RuntimeError, match="400"):
        grow_real("baseline_n12_s0", model="no-such-model")