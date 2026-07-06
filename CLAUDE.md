# CLAUDE.md

Guidance for working in this repo. See `README.md` for setup/run and `spec.md`
for the full design.

## What this is
A LangGraph multi-agent data-analysis demo: a supervisor plans and routes
between EDA, analysis, and reviewer agents that write real Python executed in a
persistent sandbox. Steps are logged to JSONL and rendered live with Rich.

## Hard rules
- **Never let `data/ground_truth.json` into agent context.** It records the
  injected faults and exists only for manual grading. Agents must reach their
  answer from the data alone.
- **Never commit secrets, generated data, or run logs.** `.env`, `data/*.csv`,
  `data/ground_truth.json`, and `runs/` are gitignored — keep it that way.
- **Model policy: `claude-haiku-4-5` only** while developing (cost). Provider is
  the Anthropic API via the project-specific **`HYDRO_ANTHROPIC_API_KEY`** (the
  global `ANTHROPIC_API_KEY` is intentionally not read). `main.py` also loads it
  from `.env`.
- **No local LLM inference on this machine** (M1, 8 GB — Ollama froze it and was
  scrapped). Code execution can stay on Docker/AWS; that's a separate axis from
  the model provider.

## Architecture conventions
- Plain functions over one shared `AgentState` TypedDict — no class hierarchies
  for agents. The supervisor plans on its first turn; there is no separate
  planner node.
- The sandbox namespace persists across `execute()` calls like a notebook. The
  dataset is loaded once at graph-build time as `df` and `outcomes`; agents
  reuse those and must not re-read the CSVs.
- Two sandbox backends behind one `execute(code, clear_context) -> {stdout,
  error, figures}` contract: `LocalDockerSandbox` (default) and
  `AgentCoreSandbox`. The graph is backend-agnostic; keep it that way.
- `ground_truth.json` is the grading key: after a run, compare the final answer
  to it by hand (expected: batch 4, lowest yield, pH drift; batch 2 runner-up).

## Common commands
```bash
.venv/bin/python data/generate_data.py                      # regenerate dataset
docker build -t hydro-sandbox docker/                        # build sandbox image
.venv/bin/python main.py "Which batch had the worst outcome and why?"
.venv/bin/python main.py --sandbox agentcore "..."           # run on AWS AgentCore
```
