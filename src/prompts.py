"""Prompt templates, in the structure of Paper A's prompts: Task / Profile / Candidates / Output.

Paper A asks for a JSON list of objects with "name" and "reason", capped at one pick.
We do the same. Candidates show only their degree (our degree-only setting).
"""

# Extra instruction line per prompt id ("" = none).
PERSONA = {
    "baseline": "",
    "popular": "Prefer members who already have many connections.",
    "newer": "Prefer newer members with few connections.",
}


def member_name(label):
    """label is the 1-based number shown to the model."""
    return f"member_{label}"


def candidate_line(label, degree):
    return f"{member_name(label)}: degree {degree}"


def build_prompt(prompt_id, candidate_degrees):
    """candidate_degrees is already in the (shuffled) order to show. Pure function."""
    if prompt_id not in PERSONA:
        raise ValueError(f"unknown prompt id: {prompt_id}")
    lines = [candidate_line(i, d) for i, d in enumerate(candidate_degrees, start=1)]
    parts = ["Task: You are a new member joining a social network. Choose exactly one existing member to connect with."]
    if PERSONA[prompt_id]:
        parts.append(PERSONA[prompt_id])
    parts += [
        "Your profile: you have just joined and have no connections yet.",
        "",
        "Candidate profiles (member and current degree = number of connections):",
        *lines,
        "",
        'Output: a JSON list with exactly one object with the keys "name" and "reason", '
        'for example [{"name": "member_<number>", "reason": "<one short sentence>"}]. '
        "Reply with the JSON only.",
    ]
    return "\n".join(parts)
