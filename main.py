"""CLI entrypoint: question in, trace out.

    python main.py "Which batch had the worst outcome and why?"
    python main.py --sandbox agentcore --provider bedrock "..."
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

from graph import build_graph, initial_state
from models import get_model
from sandbox import get_sandbox

console = Console()


def load_dotenv(path=".env"):
    """Load KEY=value lines into the env (existing env vars win)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?",
                    default="Which batch had the worst outcome and why?")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sandbox", choices=["docker", "agentcore"])
    ap.add_argument("--provider", choices=["anthropic", "local"])
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    sandbox_backend = args.sandbox or config.get("sandbox", "docker")
    provider = args.provider or config.get("model_provider", "anthropic")

    run_dir = Path("runs") / f"run_{datetime.now():%Y-%m-%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    console.print(Panel(
        f"[bold]{args.question}[/bold]\n"
        f"sandbox={sandbox_backend}  model={provider}  run_dir={run_dir}",
        title="hydro analysis agent", border_style="bold white"))

    model = get_model(provider, config)
    sandbox_kwargs = {"region": config["agentcore"]["region"]} \
        if sandbox_backend == "agentcore" else {}
    sandbox = get_sandbox(sandbox_backend, run_dir, **sandbox_kwargs)

    try:
        graph = build_graph(model, sandbox, run_dir,
                            vision=config.get("vision", False))
        state = initial_state(args.question, "/data/hydroponics.csv")
        final = graph.invoke(state, config={"recursion_limit": 80})
        console.print(Panel(final["final_answer"] or "(no answer produced)",
                            title="FINAL ANSWER", border_style="bold red"))
        console.print(f"trace: {run_dir}/log.jsonl")
    finally:
        sandbox.close()


if __name__ == "__main__":
    main()
