"""LLM growth engine: grows a network one node at a time, logs every step.

Contract (shared with the statistics side):
  - seed graph: nodes 1 and 2 with degrees [2, 2]; new nodes t = 3..n
  - at step t the candidates are ALL t-1 existing nodes, shuffled
  - logs/{run_id}.jsonl  : one line per step
  - logs/{run_id}.meta.json
  - load_picks(run_id) -> [(chosen_degree, t), ...] ; load_meta(run_id)
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests

from src.prompts import build_prompt, member_name

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
CACHE_PATH = ROOT / "cache" / "responses.jsonl"

SEED_DEGREES = [2, 2]
MAX_TRIES = 4  # first try + 3 retries, then uniform-random fallback
STUB_DELTA = {"popular": 0.5, "newer": 6.0}  # fake deltas; "baseline" depends on temperature

STATS = {"calls": 0, "cached": 0, "retries": 0, "parse_fail": 0,
         "prompt_tokens": 0, "completion_tokens": 0, "prompt_chars": 0, "seconds": 0.0}
_cache = None
_last_call = 0.0


# ---------- small helpers ----------

def _rng(seed, t, stream):
    """Independent random stream per (seed, step, purpose): 0 shuffle, 1 stub choice, 2 fallback."""
    return np.random.default_rng([seed, t, stream])


def default_model():
    return os.environ.get("LLM_MODEL", "llama3.1-8k")


def stats_line():
    s = STATS
    return (f"calls={s['calls']} cached={s['cached']} retries={s['retries']} "
            f"parse_fail={s['parse_fail']} prompt_tokens={s['prompt_tokens']} "
            f"completion_tokens={s['completion_tokens']}")


# ---------- choosers ----------

def stub_response(prompt_id, degrees, temperature, seed, t):
    """No API call. Samples from the affine PA rule with a FAKE delta (pipeline testing only)."""
    if prompt_id == "baseline":
        delta = 1.0 if temperature <= 0.7 else 2.5
    else:
        delta = STUB_DELTA[prompt_id]
    w = np.array(degrees, dtype=float) + delta
    i = int(_rng(seed, t, 1).choice(len(w), p=w / w.sum()))
    return json.dumps([{"name": member_name(i + 1), "reason": "stub"}])


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_PATH.exists():
            for line in CACHE_PATH.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    _cache[row["k"]] = row["text"]
    return _cache


def _throttle():
    global _last_call
    gap = float(os.environ.get("LLM_MIN_INTERVAL", "0"))
    wait = _last_call + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def ask_llm(prompt, model, temperature, key):
    """One call to an OpenAI-compatible chat endpoint (Ollama, Groq, ...). Cached by key.

    HTTP errors are never turned into random choices: rate limits are retried with a wait,
    anything else raises so bad settings (wrong model name, temperature too high) are visible.
    """
    cache = _load_cache()
    if key in cache:
        STATS["cached"] += 1
        return cache[key]
    base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "ollama")}
    body = {"model": model, "temperature": temperature, "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]}
    wait, err = 5.0, ""
    for _ in range(8):
        _throttle()
        t0 = time.time()
        try:
            r = requests.post(base + "/chat/completions", headers=headers, json=body, timeout=300)
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code == 200:
                break
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"LLM error {r.status_code}: {r.text[:300]}")
            err = f"HTTP {r.status_code}"
            try:
                wait = min(float(r.headers.get("Retry-After", wait)), 120.0)
            except ValueError:
                pass
        print(f"  LLM call failed ({err}); waiting {wait:.0f}s", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 120.0)
    else:
        raise RuntimeError("LLM not reachable after retries: " + err)

    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    STATS["calls"] += 1
    STATS["seconds"] += time.time() - t0
    STATS["prompt_chars"] += len(prompt)
    STATS["prompt_tokens"] += usage.get("prompt_tokens", 0)
    STATS["completion_tokens"] += usage.get("completion_tokens", 0)
    pt = usage.get("prompt_tokens")
    num_ctx = int(os.environ.get("LLM_NUM_CTX", "4096"))  # must match the model's num_ctx
    if pt and pt + 250 >= num_ctx:  # leave room for the reply; a silently cut prompt would corrupt the run
        raise RuntimeError(f"prompt is {pt} tokens but the context is {num_ctx}. Recreate the model with a larger "
                           "num_ctx and set LLM_NUM_CTX to the same value (if it is already larger, just set "
                           "LLM_NUM_CTX). Nothing was logged for this step; rerun with --resume.")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps({"k": key, "text": text}) + "\n")
    cache[key] = text
    return text


def parse_choice(text, m):
    """Return the 0-based index of the chosen member, or None if the reply is unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    name = ""
    try:
        start = min(i for i in (text.find("["), text.find("{")) if i >= 0)
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        if isinstance(obj, list):
            obj = obj[0]
        name = str(obj.get("name", ""))
    except Exception:
        pass
    for s in (name, text):  # JSON "name" first, then the first member_N anywhere in the reply
        hit = re.search(r"member_(\d+)", s)
        if hit and 1 <= int(hit.group(1)) <= m:
            return int(hit.group(1)) - 1
    return None


# ---------- engine ----------

def _segments(fn, n):
    segs = []
    for t in range(3, n + 1):
        v = fn(t)
        if segs and segs[-1]["value"] == v:
            segs[-1]["to"] = t
        else:
            segs.append({"from": t, "to": t, "value": v})
    return segs


def _make_meta(n, prompt_schedule, temperature_schedule, model_schedule, run_id, seed, chooser):
    segs = {"prompt_schedule": _segments(prompt_schedule, n),
            "temperature_schedule": _segments(temperature_schedule, n),
            "model_schedule": _segments(model_schedule, n)}
    changes = [s[1]["from"] - 1 for s in segs.values() if len(s) > 1]
    return {"run_id": run_id,
            "condition": run_id.removeprefix("stub_").split("_n")[0],
            "n": n, "seed": seed,
            "tau_true": min(changes) if changes else None,  # last step of regime 1
            **segs, "chooser": chooser}


def _step(t, shown, seed, prompt_id, temperature, model, run_id, chooser):
    m = len(shown)
    prompt = build_prompt(prompt_id, shown)
    idx, text = None, ""
    for attempt in range(MAX_TRIES):
        if chooser == "stub_pa":
            text = stub_response(prompt_id, shown, temperature, seed, t)
        else:
            key = hashlib.sha1(json.dumps([model, temperature, run_id, t, attempt, prompt]).encode()).hexdigest()
            text = ask_llm(prompt, model, temperature, key)
        idx = parse_choice(text, m)
        if idx is not None:
            break
        STATS["retries"] += 1
    ok = idx is not None
    if not ok:
        STATS["parse_fail"] += 1
        idx = int(_rng(seed, t, 2).integers(m))  # uniform fallback, flagged parse_ok=false
    return {"run_id": run_id, "t": t, "candidate_degrees": shown, "chosen_idx": int(idx),
            "chosen_degree": shown[idx], "temperature": temperature,
            "model": "stub" if chooser == "stub_pa" else model,
            "prompt_id": prompt_id, "raw_response": text, "parse_ok": ok}


def grow_network(n, prompt_schedule, temperature_schedule, model_schedule, run_id, seed, chooser="llm"):
    """Grow (or resume) one run. Schedules are functions of t. chosen_idx is 0-based into candidate_degrees."""
    if chooser not in ("llm", "stub_pa"):
        raise ValueError("chooser must be 'llm' or 'stub_pa'")
    if (chooser == "stub_pa") != run_id.startswith("stub_"):
        raise ValueError("stub runs must have a 'stub_' run_id prefix, real runs must not")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path, meta_path = LOG_DIR / f"{run_id}.jsonl", LOG_DIR / f"{run_id}.meta.json"
    meta = _make_meta(n, prompt_schedule, temperature_schedule, model_schedule, run_id, seed, chooser)
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{meta_path.name} exists with different settings; use a new run_id")
    else:
        meta_path.write_text(json.dumps(meta, indent=1))
    done = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()] if log_path.exists() else []
    if len(done) > n - 2:
        raise RuntimeError(f"{log_path.name} has more rows than n-2")

    deg = list(SEED_DEGREES)
    with open(log_path, "a") as f:
        for t in range(3, n + 1):
            perm = _rng(seed, t, 0).permutation(t - 1)  # display order of node ids
            shown = [int(deg[v]) for v in perm]
            if t - 3 < len(done):  # step already logged: replay it (this is how --resume works)
                row = done[t - 3]
                if row["t"] != t or row["candidate_degrees"] != shown:
                    raise RuntimeError("existing log does not match this seed; use a new run_id")
            else:
                row = _step(t, shown, seed, prompt_schedule(t), temperature_schedule(t),
                            model_schedule(t), run_id, chooser)
                f.write(json.dumps(row) + "\n")
                f.flush()
                if t % 50 == 0:
                    print(f"  {run_id}: step {t}/{n}  {stats_line()}", flush=True)
            deg[int(perm[row["chosen_idx"]])] += 1
            deg.append(1)
    return load_picks(run_id)


def run_campaign(specs, resume, chooser="llm"):
    """specs: list of dicts with the grow_network arguments (except chooser)."""
    for i, s in enumerate(specs, start=1):
        path = LOG_DIR / f"{s['run_id']}.jsonl"
        if path.exists() and not resume:
            raise SystemExit(f"{path.name} already exists. Pass --resume to continue/skip it, "
                             "or use a different seed. Logs are never overwritten.")
        print(f"[{i}/{len(specs)}] {s['run_id']}", flush=True)
        grow_network(chooser=chooser, **s)
    print("done.", stats_line())


# ---------- reading logs (the statistics side uses load_picks / load_meta) ----------

def read_rows(run_id):
    return [json.loads(x) for x in (LOG_DIR / f"{run_id}.jsonl").read_text().splitlines() if x.strip()]


def load_picks(run_id):
    """[(chosen_degree, t), ...] for ALL rows, including parse_ok=false rows."""
    return [(r["chosen_degree"], r["t"]) for r in read_rows(run_id)]


def load_meta(run_id):
    return json.loads((LOG_DIR / f"{run_id}.meta.json").read_text())