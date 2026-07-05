"""Structured step logging: append-only JSONL + live Rich rendering.

One log entry per step:
  {step_id, agent, description, code, stdout, error, figure_paths,
   status, timestamp}
A future UI reads log.jsonl top to bottom; tonight Rich prints each entry
as it happens.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

AGENT_COLORS = {
    "supervisor": "bold magenta",
    "eda": "bold cyan",
    "analysis": "bold green",
    "reviewer": "bold yellow",
}


def append_log(run_dir: Path, entry: dict) -> dict:
    """Stamp, persist, and render one step. Returns the completed entry."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with open(Path(run_dir) / "log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    render_step(entry)
    return entry


def render_step(entry: dict) -> None:
    agent = entry.get("agent", "?")
    style = AGENT_COLORS.get(agent, "bold white")
    parts = []

    if entry.get("code"):
        parts.append(Text("code >", style="dim"))
        parts.append(Syntax(entry["code"].strip(), "python", theme="monokai",
                            line_numbers=False, word_wrap=True))
    if entry.get("stdout", "").strip():
        parts.append(Text("output >", style="dim"))
        stdout = entry["stdout"].strip()
        if len(stdout) > 2500:
            stdout = stdout[:2500] + "\n... [truncated in display, full text in log.jsonl]"
        parts.append(Text(stdout))
    if entry.get("error"):
        parts.append(Text("error >", style="dim red"))
        err = entry["error"].strip().splitlines()
        parts.append(Text("\n".join(err[-12:]), style="red"))
    for fig in entry.get("figure_paths", []) or []:
        parts.append(Text(f"[figure saved: {Path(fig).name}]", style="italic blue"))

    status = entry.get("status", "ok")
    status_txt = f" [{status}]" if status != "ok" else ""
    header = f"step {entry.get('step_id', '?')} · {agent}{status_txt} — {entry.get('description', '')}"
    console.print(Panel(Group(*parts) if parts else Text(""), title=header,
                        title_align="left", border_style=style))
