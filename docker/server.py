"""Minimal exec server that runs inside the sandbox container.

Holds one persistent globals() dict across /execute calls (the shared
"kernel"). Captures stdout and any matplotlib figures created by the code;
figures are saved under /output and returned as container paths.
"""

import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
namespace: dict = {}
fig_counter = 0

OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)


class ExecRequest(BaseModel):
    code: str
    clear_context: bool = False


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/execute")
def execute(req: ExecRequest):
    global fig_counter
    if req.clear_context:
        namespace.clear()
    plt.close("all")

    buf = io.StringIO()
    error = None
    try:
        with redirect_stdout(buf):
            exec(compile(req.code, "<agent-code>", "exec"), namespace)
    except Exception:
        error = traceback.format_exc()

    figures = []
    for num in plt.get_fignums():
        fig_counter += 1
        path = OUTPUT_DIR / f"fig_{fig_counter:03d}.png"
        plt.figure(num).savefig(path, dpi=110, bbox_inches="tight")
        figures.append(str(path))
    plt.close("all")

    stdout = buf.getvalue()
    if len(stdout) > 20000:
        stdout = stdout[:20000] + "\n... [stdout truncated]"
    return {"stdout": stdout, "error": error, "figures": figures}
