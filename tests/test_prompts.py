import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import PERSONA, build_prompt, candidate_line


def test_every_candidate_appears_exactly_once():
    degrees = [3, 3, 1, 12, 2]
    for pid in PERSONA:
        lines = build_prompt(pid, degrees).splitlines()
        for label, d in enumerate(degrees, start=1):
            assert lines.count(candidate_line(label, d)) == 1
        assert sum(l.startswith("member_") for l in lines) == len(degrees)


def test_the_three_prompts_differ():
    texts = {pid: build_prompt(pid, [1, 2, 3]) for pid in PERSONA}
    assert len(set(texts.values())) == 3
    assert "many connections" in texts["popular"]
    assert "few connections" in texts["newer"]


def test_unknown_prompt_id_raises():
    with pytest.raises(ValueError):
        build_prompt("nope", [1, 2])
