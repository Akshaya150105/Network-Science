# Changepoint Detection in Preferential Attachment Networks Formed by LLM Agents

Network Science project (20XD92/20XD98), PSG College of Technology.

- Statistics side (Valincia): `pa_sim.py`, `mle.py`, `changepoint.py`, `analysis_stats.py`, experiments 01-03.
- LLM side (Akshaya): `prompts.py`, `llm_growth.py`, `analysis_llm.py`, experiments 04-06.

See `docs/` for the project documentation. The shared contract (seed graph `[2, 2]`, log format, results CSV schema) is in the plan files.

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

API keys go in environment variables only, never in code.

## Layout

```
src/          library code
tests/        unit tests
experiments/  runnable experiment scripts
logs/         LLM run logs (never edited by hand, never overwritten)
results/      CSV outputs and critical_values.json
figures/      plots
report/       report sections
docs/         project documentation
```
