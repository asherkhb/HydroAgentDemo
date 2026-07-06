"""State, node functions, and graph wiring — the whole system.

Plain functions over one shared TypedDict. The supervisor plans on its
first turn (no separate planner node) and routes on every turn after.
EDA/Analysis write python that runs in the sandbox; the Reviewer critiques
the analysis (with figures as vision input when the model supports it).

Note: the persistent exec() namespace lives inside the sandbox session
(container / AgentCore both hold state across execute() calls), so it is
not carried in graph state.
"""

import base64
import json
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from logging_utils import append_log

MAX_TOTAL_STEPS = 26      # hard cap on log entries before forcing end
MAX_STEPS_PER_VISIT = 3   # code executions per eda/analysis visit
MAX_REVIEW_ROUNDS = 2     # revise cycles before auto-approve

DATA_DESCRIPTION = """\
The sandbox is a persistent notebook kernel. Two pandas DataFrames are ALREADY
loaded and ready to use — do not read any CSV files:
- df       : hourly sensor readings for 6 lettuce batches grown in an NFT
  hydroponics system (~35 days each). Columns: timestamp (parsed datetime),
  batch_id, ph, ec, water_temp_c, air_temp_c, humidity_pct, ppfd, co2_ppm,
  dissolved_oxygen_mg_l.
- outcomes : per-batch outcomes. Columns: batch_id, yield_kg, pct_tipburn.
"""

# Loaded once into the sandbox namespace before any agent runs, so no step
# ever re-reads the CSVs (they persist across execute() calls like a notebook).
BOOTSTRAP_CODE = """\
import pandas as pd, numpy as np
df = pd.read_csv("/data/hydroponics.csv", parse_dates=["timestamp"])
outcomes = pd.read_csv("/data/batch_outcomes.csv")
"""

CODE_RULES = """\
Rules for every code step:
- Reply with exactly: one line `DESCRIPTION: <short imperative summary>` followed by
  ONE ```python code block. Nothing else.
- The sandbox is a persistent notebook: `df` and `outcomes` are already loaded, and
  variables you define persist between steps. Never re-read the CSVs or recompute
  something an earlier step already put in a variable.
- print() every result you want to see; keep printed output small (aggregates, not
  raw dumps).
- At most one matplotlib figure per step; never call plt.show().
- Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn,
  statsmodels. Nothing else is installed.
"""


class AgentState(TypedDict):
    question: str
    dataset_path: str
    plan: str
    log: list          # append-only step log (also persisted to log.jsonl)
    next_agent: str
    review_feedback: str | None
    review_rounds: int
    done: bool
    final_answer: str


# ---------------------------------------------------------------- helpers

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _description(text: str) -> str:
    m = re.search(r"DESCRIPTION:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    first = text.strip().splitlines()[0] if text.strip() else "code step"
    return first[:120]


def _history(log: list, last: int = 10, stdout_chars: int = 350) -> str:
    """Compact step history for prompts."""
    lines = []
    for e in log[-last:]:
        out = (e.get("stdout") or "").strip().replace("\n", " | ")
        if len(out) > stdout_chars:
            out = out[:stdout_chars] + "..."
        status = "" if e.get("status") == "ok" else f" ({e.get('status')})"
        lines.append(f"- step {e['step_id']} [{e['agent']}]{status} {e['description']}"
                     + (f" => {out}" if out else ""))
    return "\n".join(lines) if lines else "(no steps yet)"


def _invoke(model, messages) -> str:
    return _strip_think(model.invoke(messages).content)


# ---------------------------------------------------------------- graph factory

def build_graph(model, sandbox, run_dir: Path, vision: bool = False):
    run_dir = Path(run_dir)
    step_counter = {"n": 0}

    # Preload the data once; every agent step then reuses df/outcomes from the
    # persistent namespace instead of re-reading the CSVs.
    boot = sandbox.execute(BOOTSTRAP_CODE)
    if boot.get("error"):
        raise RuntimeError(f"sandbox data bootstrap failed:\n{boot['error']}")

    def log_step(state: AgentState, **fields) -> dict:
        step_counter["n"] += 1
        entry = {"step_id": step_counter["n"], **fields}
        append_log(run_dir, entry)
        state["log"].append(entry)
        return entry

    # ---------------------------------------------------------- supervisor

    def supervisor(state: AgentState) -> AgentState:
        if not state["plan"]:
            prompt = (
                f"You are the supervisor of a data-analysis team answering:\n"
                f"QUESTION: {state['question']}\n\n{DATA_DESCRIPTION}\n"
                "Your team: 'eda' (profiles the dataset: shape, missingness, "
                "distributions, time-series structure), 'analysis' (statistical/"
                "process analysis targeted at the question), 'reviewer' (validates "
                "the analysis).\n\n"
                "If the question contains a term that could be measured more than one "
                "way (e.g. 'worst', 'best', 'most affected'), state up front in ONE "
                "sentence exactly how you will define it (which metric, and how you "
                "break ties or combine metrics). Every downstream step and the final "
                "answer must use this definition consistently.\n"
                "Write a short plan (2-4 sentences) for answering the question — begin "
                "with that definition sentence — then on the final line output JSON "
                'choosing who acts first: {"next": "eda"} or {"next": "analysis"}'
            )
            text = _invoke(model, [HumanMessage(prompt)])
            decision = _extract_json(text)
            state["plan"] = re.sub(r"\{.*?\}\s*$", "", text, flags=re.DOTALL).strip()
            state["next_agent"] = decision.get("next", "eda")
            if state["next_agent"] not in ("eda", "analysis"):
                state["next_agent"] = "eda"
            log_step(state, agent="supervisor",
                     description="Wrote plan and chose first agent",
                     stdout=f"PLAN: {state['plan']}\nFIRST: {state['next_agent']}",
                     status="ok")
            return state

        # routing turn
        agents_run = {e["agent"] for e in state["log"]}
        over_budget = len(state["log"]) >= MAX_TOTAL_STEPS

        if over_budget:
            choice = "end"
        else:
            prompt = (
                f"You are the supervisor. QUESTION: {state['question']}\n"
                f"PLAN: {state['plan']}\n\nSteps completed so far:\n"
                f"{_history(state['log'])}\n\n"
                + (f"Reviewer feedback pending: {state['review_feedback']}\n"
                   if state["review_feedback"] else "")
                + "Decide the next move. Route to 'analysis' if reviewer feedback is "
                  "pending. Route to 'reviewer' once analysis has produced results "
                  "that haven't been reviewed. Choose 'end' only after the reviewer "
                  "has approved. Reply with JSON only: "
                  '{"next": "eda"|"analysis"|"reviewer"|"end", "reason": "..."}'
            )
            decision = _extract_json(_invoke(model, [HumanMessage(prompt)]))
            choice = decision.get("next", "")

        # guardrails for a small model: sensible fallback + no unreviewed end
        if choice not in ("eda", "analysis", "reviewer", "end"):
            if "eda" not in agents_run:
                choice = "eda"
            elif "analysis" not in agents_run:
                choice = "analysis"
            elif "reviewer" not in agents_run:
                choice = "reviewer"
            else:
                choice = "end"
        if choice == "end" and "analysis" in agents_run and "reviewer" not in agents_run \
                and not over_budget:
            choice = "reviewer"
        if state["review_feedback"] and choice not in ("analysis", "end"):
            choice = "analysis"

        if choice == "end":
            state["final_answer"] = _final_answer(state)
            state["done"] = True
            log_step(state, agent="supervisor", description="Final answer",
                     stdout=state["final_answer"], status="ok")
        else:
            state["next_agent"] = choice
            log_step(state, agent="supervisor",
                     description=f"Routing to {choice}", stdout="", status="ok")
        return state

    def _final_answer(state: AgentState) -> str:
        prompt = (
            f"QUESTION: {state['question']}\n\n"
            f"Your team's work log:\n{_history(state['log'], last=30, stdout_chars=600)}\n\n"
            "Write the final answer to the question in 3-6 sentences, grounded in the "
            "numbers above: name the batch, the outcome metrics, and the sensor-level "
            "root cause with its time window.\n"
            "Before writing, double-check every claim against the numbers in the log — "
            "especially superlatives like 'worst' or 'lowest'. If the reviewer's last "
            "critique flagged factual errors or ambiguity, resolve them in your answer. "
            "Apply the definition of the key term stated in your PLAN's first sentence "
            "consistently, and lead with the batch that definition selects; if a "
            "different batch is worst under an alternate metric, mention it as a "
            "secondary reading rather than switching your lead."
        )
        return _invoke(model, [HumanMessage(prompt)])

    # ---------------------------------------------------------- coder agents

    EDA_ROLE = (
        "You are the EDA agent. Profile the dataset: shape, dtypes, missing values, "
        "per-batch summary statistics, outcome overview, and time-series behavior of "
        "key sensors per batch. Produce at most 2 plots. Focus on surfacing anything "
        "anomalous that later analysis should chase."
    )
    ANALYSIS_ROLE = (
        "You are the analysis agent. Run the statistical/process analysis that "
        "answers the question: compare batch outcomes, then dig into the sensor "
        "time series of the worst batch(es) to find the root cause (drift, spikes, "
        "out-of-range periods). Quantify what you claim (means, deviations from "
        "target, time windows). Produce plots that show the evidence."
    )

    def _coder(state: AgentState, agent: str, role: str) -> AgentState:
        context = (
            f"{role}\n\nQUESTION: {state['question']}\nTEAM PLAN: {state['plan']}\n\n"
            f"{DATA_DESCRIPTION}\n"
            f"Work already done by the team:\n{_history(state['log'])}\n\n"
            + (f"REVIEWER FEEDBACK to address: {state['review_feedback']}\n\n"
               if agent == "analysis" and state["review_feedback"] else "")
            + CODE_RULES
            + "\nWrite your first code step now."
        )
        messages = [SystemMessage("You write python for a persistent sandbox."),
                    HumanMessage(context)]

        executed = 0
        calls = 0
        while executed < MAX_STEPS_PER_VISIT and calls < MAX_STEPS_PER_VISIT * 2 + 2:
            calls += 1
            text = _invoke(model, messages)
            code = _extract_code(text)

            if code is None:
                if "FINDINGS:" in text:
                    findings = text.split("FINDINGS:", 1)[1].strip()
                    log_step(state, agent=agent, description="Findings",
                             stdout=findings, status="ok")
                    break
                messages.append(AIMessage(text))
                messages.append(HumanMessage(
                    "Reply with either DESCRIPTION + one ```python block, or "
                    "`FINDINGS: <summary>` if you are finished."))
                continue

            result = sandbox.execute(code)
            executed += 1
            status = "error" if result["error"] else "ok"
            log_step(state, agent=agent, description=_description(text),
                     code=code, stdout=result["stdout"], error=result["error"],
                     figure_paths=result["figures"], status=status)

            messages.append(AIMessage(text))
            if result["error"]:
                feedback = f"Your code raised an error:\n{result['error'][-1500:]}\nFix it and retry."
            else:
                out = result["stdout"][:2000] or "(no stdout)"
                figs = "".join(f"\n[figure saved: {Path(f).name}]" for f in result["figures"])
                feedback = (
                    f"Execution result:\n{out}{figs}\n\n"
                    f"You have used {executed}/{MAX_STEPS_PER_VISIT} code steps this turn. "
                    "Either write the next code step (DESCRIPTION + ```python block) or, "
                    "if you have what you need, reply with `FINDINGS: <2-4 sentence "
                    "summary of what the numbers show>` and no code."
                )
            messages.append(HumanMessage(feedback))
        else:
            # step budget exhausted -> force a findings summary
            messages.append(HumanMessage(
                "Step budget reached. Reply with `FINDINGS: <2-4 sentence summary "
                "of what your executed steps showed>` and no code."))
            text = _invoke(model, messages)
            findings = text.split("FINDINGS:", 1)[-1].strip()
            log_step(state, agent=agent, description="Findings",
                     stdout=findings, status="ok")

        if agent == "analysis":
            state["review_feedback"] = None  # feedback consumed
        return state

    def eda(state: AgentState) -> AgentState:
        return _coder(state, "eda", EDA_ROLE)

    def analysis(state: AgentState) -> AgentState:
        return _coder(state, "analysis", ANALYSIS_ROLE)

    # ---------------------------------------------------------- reviewer

    def reviewer(state: AgentState) -> AgentState:
        analysis_steps = [e for e in state["log"]
                          if e["agent"] in ("eda", "analysis")
                          and e.get("status") == "ok"]
        detail = []
        for e in analysis_steps[-6:]:
            detail.append(f"--- step {e['step_id']}: {e['description']}\n"
                          f"code:\n{e.get('code', '(none)')}\n"
                          f"output:\n{(e.get('stdout') or '')[:1200]}")
        figures = [f for e in analysis_steps for f in (e.get("figure_paths") or [])]

        prompt_text = (
            f"You are a statistics reviewer. QUESTION: {state['question']}\n\n"
            f"The analysis agent produced:\n\n" + "\n\n".join(detail) +
            "\n\nCritique validity: wrong test, unaddressed confounders, claims not "
            "supported by the numbers, misleading or missing charts, correlation "
            "presented as causation without caveats. Be pragmatic — request revision "
            "only for flaws that would change the answer.\n"
            "Reply with exactly one of:\n"
            "APPROVE: <1-2 sentence rationale>\n"
            "REVISE: <specific, actionable feedback>"
        )
        if vision and figures:
            content = [{"type": "text", "text": prompt_text}]
            for fig in figures[-4:]:
                b64 = base64.b64encode(Path(fig).read_bytes()).decode()
                content.append({"type": "image",
                                "source": {"type": "base64",
                                           "media_type": "image/png",
                                           "data": b64}})
            msg = HumanMessage(content=content)
        else:
            msg = HumanMessage(prompt_text)

        text = _invoke(model, [msg])
        approved = text.upper().startswith("APPROVE") or "REVISE" not in text.upper()
        auto = state["review_rounds"] >= MAX_REVIEW_ROUNDS and not approved
        if auto:
            text += "\n[auto-approved: max review rounds reached]"
            approved = True

        state["review_rounds"] += 1
        state["review_feedback"] = None if approved else \
            text.split("REVISE:", 1)[-1].strip()
        log_step(state, agent="reviewer",
                 description="Auto-approved (review budget exhausted)" if auto
                 else "Approved analysis" if approved else "Requested revision",
                 stdout=text, status="ok")
        return state

    # ---------------------------------------------------------- wiring

    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("eda", eda)
    g.add_node("analysis", analysis)
    g.add_node("reviewer", reviewer)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: "end" if s["done"] else s["next_agent"],
        {"eda": "eda", "analysis": "analysis", "reviewer": "reviewer", "end": END},
    )
    for node in ("eda", "analysis", "reviewer"):
        g.add_edge(node, "supervisor")

    return g.compile()


def initial_state(question: str, dataset_path: str) -> AgentState:
    return AgentState(
        question=question, dataset_path=dataset_path, plan="", log=[],
        next_agent="", review_feedback=None, review_rounds=0,
        done=False, final_answer="",
    )
