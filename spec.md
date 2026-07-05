# Data Analysis Agent — Project Spec

## Goal
A LangGraph-based multi-agent system that answers analytical questions about a
dataset by **writing and executing real Python code**, not by reasoning in
free text. A supervisor plans and delegates to three subagents. Every step is
logged in a structured, replayable format (short description + code + output)
so it can later be rendered in a UI. Tonight: Rich terminal output. Later: a
proper UI reading the same log.

**Design principle: minimalism.** No class hierarchies, no agent "framework
within the framework," no premature abstraction. Plain functions operating on
one shared state dict. If a class isn't pulling real weight (holding state
across calls, e.g.), don't write it.

---

## Architecture

### Agents (LangGraph nodes)
1. **Supervisor** — on its first turn, reads the question and writes a short
   plan (2-4 sentences, stored in `state["plan"]`). On every turn after,
   decides which subagent runs next (or whether to end) by checking progress
   against that plan, and revises the plan if the reviewer sends something
   back. No separate planner node — for a 3-subagent graph, folding planning
   into the supervisor's first turn avoids an extra hop without losing the
   "think before acting" behavior. Worth splitting into a real planner node
   later only if this grows to many subagents or needs real replanning
   logic, not preemptively.
2. **EDA agent** — profiles the dataset: shape, dtypes, missingness,
   distributions, basic time-series structure. Produces plots.
3. **Analysis agent** — runs the actual statistical/process analysis relevant
   to the question (correlations, control charts, process capability, drift
   detection, regression on yield, etc). Produces plots + numeric results.
4. **Reviewer agent** — looks at the analysis agent's code, results, *and
   plots* (vision input) and critiques validity: wrong test, unaddressed
   confounders, misleading chart, overfit claim, etc. Can send it back to
   Analysis with feedback, or approve.

Routing: `Supervisor -> {EDA, Analysis, Reviewer} -> Supervisor -> ... -> END`

### Shared execution state (single TypedDict)
```python
class AgentState(TypedDict):
    question: str
    dataset_path: str
    namespace: dict          # persistent exec() globals — shared "kernel"
    log: list[dict]          # append-only step log (see schema below)
    next_agent: str          # supervisor's routing decision
    review_feedback: str | None
    done: bool
```

### The code execution primitive — pluggable sandbox
Two backends, one interface. Config (env var or CLI flag) picks which one:

```python
class CodeSandbox(Protocol):
    def execute(self, code: str, clear_context: bool = False) -> dict:
        """Returns {stdout, error, figures: list[path]}"""
```

**`AgentCoreSandbox`** — thin wrapper around `bedrock_agentcore`:
```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

class AgentCoreSandbox:
    def __init__(self, region: str):
        self.client = CodeInterpreter(region)
        self.client.start()

    def execute(self, code: str, clear_context: bool = False) -> dict:
        response = self.client.invoke("executeCode", {
            "language": "python", "code": code, "clearContext": clear_context
        })
        # parse response["stream"] -> text content + any image content items
        ...
```
`clearContext=False` is exactly your persistent-namespace behavior, just
running server-side instead of in-process — the session holds state across
calls, so this needs no extra design work on your end. Confirmed against
AWS's current docs: `StartCodeInterpreterSession` / `InvokeCodeInterpreter`
with `executeCode`, and the result stream includes text output plus
generated visualizations, so plots come back through the same channel as
stdout — no separate file-fetch step needed for the common case.

**`LocalDockerSandbox`** — a minimal FastAPI server *inside* the container
holding a persistent `globals()` dict, hit over HTTP:
```python
# server.py, runs inside the container
namespace = {}
@app.post("/execute")
def execute(req: ExecRequest):
    if req.clear_context:
        namespace.clear()
    # exec(req.code, namespace), capture stdout via redirect_stdout,
    # capture any new matplotlib figure -> save to /output, return base64/path
```
This deliberately mirrors the AgentCore contract (one `execute` call, same
response shape) rather than reaching for `jupyter_client`/ZMQ — it's a
20-line server, not a kernel protocol implementation, and it means
`AgentCoreSandbox` and `LocalDockerSandbox` are truly interchangeable from
the graph's point of view. Mount a volume for figure files, run with
`docker run -p 8765:8765 -v ./runs:/output sandbox-image`.

Either way, `graph.py` only ever calls `sandbox.execute(code)` — it doesn't
know or care which backend is live.

### Log entry schema (one per step, appended to a JSONL file)
```json
{
  "step_id": 4,
  "agent": "analysis",
  "description": "Fit control chart limits for pH by batch and flag violations",
  "code": "...",
  "stdout": "3 out-of-control points found in batch 7",
  "figure_path": "run_2026-07-03/step_04.png",
  "status": "ok",
  "timestamp": "..."
}
```
This one file is the entire "trace" — a future UI just reads it top to
bottom and renders description (collapsed) + code (expandable) + stdout +
image.

### Model backend — Claude Console (Anthropic API), same interface
Since every agent node is a plain function calling a LangChain chat model,
swapping providers is just swapping which chat model class the factory
returns — no wrapper layer needed:

```python
def get_model(provider: str):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", max_tokens=4096)
    else:  # "local" — any OpenAI-compatible endpoint, kept as an escape hatch
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(base_url=LOCAL_BASE_URL, api_key="not-needed", model=LOCAL_MODEL)
```

The primary provider is the **Claude Console / Anthropic API**. The key
(from console.anthropic.com) is exported as the project-specific
`HYDRO_ANTHROPIC_API_KEY` and passed to `ChatAnthropic` explicitly — the
demo deliberately does not read the global `ANTHROPIC_API_KEY`, so its
spend is isolated from anything else on the machine.

**Cost policy for development: `claude-haiku-4-5` only** ($1/M input,
$5/M output — a full graph run is a fraction of a cent). Haiku 4.5 supports
vision, so the reviewer agent can attach the analysis plots as real image
input (`vision: true` in config). Upgrading a single node (e.g. supervisor
or reviewer) to a bigger Claude model later is a one-line config change, but
that's an explicit decision, not a default.

Code execution stays on AWS/Docker as specified below — the model provider
and the sandbox are independent axes.

### Tonight's UI: Rich
Each step prints as a panel: bold agent name + description as the header,
code in a syntax-highlighted collapsible-by-convention block (just print it
under a `code >` label so it's visually secondary), stdout below it, and a
note like `[figure saved: step_04.png]` (Rich can't inline images in most
terminals, so just reference the path — open it separately or `imgcat` if
you're on iTerm2).

---

## Sample dataset: synthetic hydroponics sensor data

Generate synthetic data for **6 batches** of lettuce grown in an NFT
(nutrient film technique) system, each batch ~35 days, sensors logged
hourly:

- `timestamp`, `batch_id`
- `ph` (target ~5.8, slow drift + noise)
- `ec` (electrical conductivity, mS/cm — nutrient strength)
- `water_temp_c`, `air_temp_c`, `humidity_pct`
- `ppfd` (light intensity, umol/m2/s — clear diurnal cycle)
- `co2_ppm`
- `dissolved_oxygen_mg_l`

Plus **per-batch outcomes**: `yield_kg`, `pct_tipburn` (a real lettuce
defect linked to calcium uptake, which is affected by EC/humidity/airflow).

**Seed 2-3 realistic failure modes** so the agents have something to find:
- One batch has a slow pH sensor drift (calibration fault) that correlates
  with a yield dip.
- One batch has an EC spike (dosing pump error) during a specific week that
  correlates with elevated tipburn.
- Add plausible sensor noise and a couple of dropped/missing readings so the
  EDA agent's missingness check isn't trivially empty.

I can generate this dataset for you right now if you want to start with real
data in hand rather than having Claude Code invent it (more reproducible,
and you know exactly where the "planted" signal is, which makes it easy to
sanity-check whether the agents actually found it).

**Ground truth matters here more than usual** — the entire point of this
dataset is to check whether your agents actually find the seeded faults, not
just produce plausible-sounding analysis. Have Claude Code write
`generate_data.py` so that:
- Random seed is fixed (reproducible runs).
- It writes a second file, `data/ground_truth.json`, recording exactly which
  batch, date range, sensor, and fault type were injected (e.g.
  `{"batch_id": 4, "fault": "ph_drift", "start_day": 12, "magnitude": "..."}`).
  This file is for **you**, not the agents — don't let it leak into the
  agent's context, or you're grading an open-book exam.
- After a run, you manually diff the Reviewer agent's conclusions against
  `ground_truth.json` to judge whether the system actually works, rather
  than eyeballing whether the output "sounds right."

---

## Minimal file layout
```
data_analysis_agent/
  graph.py           # state, node functions, graph wiring — the whole system
  sandbox.py          # CodeSandbox protocol + AgentCoreSandbox + LocalDockerSandbox
  models.py           # get_model(provider) factory
  logging_utils.py    # append_log(), Rich renderer
  docker/
    Dockerfile         # local sandbox image
    server.py           # the exec server that runs inside it
  data/
    generate_data.py   # builds hydroponics.csv with seeded failure modes (see below)
    hydroponics.csv     # generated, not hand-written — gitignore or commit, your call
  runs/
    run_<timestamp>/
      log.jsonl
      step_XX.png
  main.py            # CLI entrypoint: question in, trace out
  config.yaml        # sandbox: agentcore|docker, model_provider: anthropic|local
```
Eight files plus a couple dirs — still no `agents/` package or class
hierarchy per subagent. The two swappable pieces (sandbox, model) are each
one file with one factory function, which is the whole abstraction budget
this project needs.

