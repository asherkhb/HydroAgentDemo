# HydroAgentDemo

LangGraph multi-agent data-analysis system: a supervisor plans and routes
between an EDA agent, an analysis agent, and a reviewer. Agents answer
questions by writing real Python that executes in a sandboxed persistent
kernel — the dataset is loaded once (as `df` and `outcomes`) and reused
across every step. Every step is logged to `runs/run_<ts>/log.jsonl` and
rendered live with Rich.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install langgraph langchain-anthropic langchain-openai rich pandas numpy matplotlib pyyaml requests

# generate the dataset (writes data/hydroponics.csv, batch_outcomes.csv, ground_truth.json)
.venv/bin/python data/generate_data.py

# build the local sandbox image
docker build -t hydro-sandbox docker/

# model provider: Claude Console (Anthropic API) — project-specific env var,
# the global ANTHROPIC_API_KEY is intentionally NOT read.
# Either export it, or put it in a .env file (gitignored; main.py loads it):
#   HYDRO_ANTHROPIC_API_KEY=sk-ant-...
export HYDRO_ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com
```

Dev cost policy: `claude-haiku-4-5` only while developing (set in
`config.yaml`). Haiku supports vision, so the reviewer sees the plots.

## Run

```bash
.venv/bin/python main.py "Which batch had the worst outcome and why?"

# alternate sandbox: AWS AgentCore code interpreter (model provider unchanged)
#   needs: pip install bedrock-agentcore, valid AWS creds, region in config.yaml
.venv/bin/python main.py --sandbox agentcore "..."

# escape hatch: any OpenAI-compatible endpoint (set local.base_url/model in config.yaml)
.venv/bin/python main.py --provider local "..."
```

Config lives in `config.yaml` (sandbox backend, model provider, vision flag
for the reviewer). CLI flags override it.

## Grading

`data/ground_truth.json` records exactly which faults were injected
(batch, sensor, day range, magnitude). **Never let it into agent context** —
after a run, diff the final answer / reviewer conclusions against it by hand.

## Files

- `graph.py` — state TypedDict, node functions, graph wiring
- `sandbox.py` — CodeSandbox protocol: LocalDockerSandbox + AgentCoreSandbox
- `models.py` — get_model(provider) factory (Anthropic API or OpenAI-compatible)
- `logging_utils.py` — JSONL step log + Rich renderer
- `docker/` — sandbox image: FastAPI exec server with persistent namespace
- `data/generate_data.py` — synthetic hydroponics dataset with seeded faults
- `main.py` — CLI entrypoint
