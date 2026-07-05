"""Pluggable code-execution sandbox.

Two backends behind one interface:
  LocalDockerSandbox - FastAPI exec server in a local container (default)
  AgentCoreSandbox   - AWS Bedrock AgentCore code interpreter

Both expose:  execute(code, clear_context=False) -> {stdout, error, figures}
where figures is a list of host-side PNG paths.
"""

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Protocol

import requests

PROJECT_ROOT = Path(__file__).parent


class CodeSandbox(Protocol):
    def execute(self, code: str, clear_context: bool = False) -> dict: ...
    def close(self) -> None: ...


class LocalDockerSandbox:
    """Runs the docker/ exec server; data/ mounted read-only at /data,
    the run directory mounted at /output for figures."""

    CONTAINER = "hydro-sandbox-run"
    IMAGE = "hydro-sandbox"

    def __init__(self, run_dir: Path, port: int = 8765):
        self.run_dir = Path(run_dir).resolve()
        self.port = port
        self.url = f"http://localhost:{port}"
        self._start()

    def _start(self):
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER],
            capture_output=True,
        )
        subprocess.run(
            [
                "docker", "run", "-d", "--name", self.CONTAINER,
                "-p", f"{self.port}:8765",
                "-v", f"{PROJECT_ROOT / 'data'}:/data:ro",
                "-v", f"{self.run_dir}:/output",
                self.IMAGE,
            ],
            check=True,
            capture_output=True,
        )
        for _ in range(40):
            try:
                if requests.get(f"{self.url}/health", timeout=1).ok:
                    return
            except requests.ConnectionError:
                time.sleep(0.5)
        raise RuntimeError("sandbox container did not become healthy")

    def execute(self, code: str, clear_context: bool = False) -> dict:
        resp = requests.post(
            f"{self.url}/execute",
            json={"code": code, "clear_context": clear_context},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        # container paths (/output/fig_001.png) -> host paths in run_dir
        result["figures"] = [
            str(self.run_dir / Path(p).name) for p in result["figures"]
        ]
        return result

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.CONTAINER], capture_output=True)


class AgentCoreSandbox:
    """Wrapper around the Bedrock AgentCore code interpreter.

    Presents the same contract as LocalDockerSandbox so the graph is
    backend-agnostic:
      - clearContext=False keeps a persistent Python namespace server-side.
      - The dataset is uploaded once at startup. AgentCore rejects absolute
        upload paths and runs unprivileged (can't create /data), so files
        live in a relative DATA_DIR and we translate the agents' `/data/...`
        references to it in submitted code — emulating the Docker /data mount.
      - Matplotlib figures are NOT returned inline (unlike a notebook kernel),
        so after each step a tiny harvest cell saves any open figures to
        fig_NNN.png in-sandbox; we then download them into run_dir, mirroring
        the Docker server's auto-capture.
    """

    DATA_DIR = "hydro_data"  # relative to the sandbox cwd; stands in for /data

    # Runs after each user step in the SAME persistent context: save any open
    # figures with a monotonic counter, print their names on a sentinel line.
    _HARVEST = (
        "import sys as __sys, json as __json\n"
        "__hydro_figs = []\n"
        "if 'matplotlib.pyplot' in __sys.modules:\n"
        "    import matplotlib.pyplot as __plt\n"
        "    try:\n"
        "        __hydro_fig_counter\n"
        "    except NameError:\n"
        "        __hydro_fig_counter = 0\n"
        "    for __n in __plt.get_fignums():\n"
        "        __hydro_fig_counter += 1\n"
        "        __name = 'fig_%03d.png' % __hydro_fig_counter\n"
        "        __plt.figure(__n).savefig(__name, dpi=110, bbox_inches='tight')\n"
        "        __hydro_figs.append(__name)\n"
        "    __plt.close('all')\n"
        "print('__HYDRO_FIGS__' + __json.dumps(__hydro_figs))\n"
    )

    def __init__(self, run_dir: Path, region: str = "us-east-1"):
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        self.run_dir = Path(run_dir).resolve()
        self.client = CodeInterpreter(region)
        self.client.start()
        self._upload_data()

    def _upload_data(self):
        files = []
        for f in sorted((PROJECT_ROOT / "data").glob("*.csv")):
            files.append({"path": f"{self.DATA_DIR}/{f.name}", "text": f.read_text()})
        if files:
            self.client.invoke("writeFiles", {"content": files})

    def _collect(self, response) -> tuple[str, str | None]:
        parts, error = [], None
        for event in response["stream"]:
            result = event.get("result", {})
            texts = [it.get("text", "") for it in result.get("content", [])
                     if it.get("type") == "text"]
            if result.get("isError"):
                error = "\n".join(texts) or json.dumps(result.get("content"))[:4000]
            else:
                parts.extend(texts)
        stdout = "\n".join(parts)
        if len(stdout) > 20000:
            stdout = stdout[:20000] + "\n... [stdout truncated]"
        return stdout, error

    def execute(self, code: str, clear_context: bool = False) -> dict:
        # emulate the /data mount + force a headless backend, as the Docker
        # server does (matplotlib.use must precede any pyplot import).
        code = code.replace("/data/", f"{self.DATA_DIR}/")
        code = "import matplotlib as _m; _m.use('Agg')\n" + code
        response = self.client.invoke(
            "executeCode",
            {"language": "python", "code": code, "clearContext": clear_context},
        )
        stdout, error = self._collect(response)
        return {"stdout": stdout, "error": error, "figures": self._harvest_figures()}

    def _harvest_figures(self) -> list:
        response = self.client.invoke(
            "executeCode",
            {"language": "python", "code": self._HARVEST, "clearContext": False},
        )
        stdout, _ = self._collect(response)
        names = []
        for line in stdout.splitlines():
            if line.startswith("__HYDRO_FIGS__"):
                names = json.loads(line[len("__HYDRO_FIGS__"):])
        figures = []
        for name in names:
            data = self.client.download_file(name)
            if isinstance(data, str):
                data = base64.b64decode(data)
            path = self.run_dir / name
            path.write_bytes(data)
            figures.append(str(path))
        return figures

    def close(self):
        self.client.stop()


def get_sandbox(backend: str, run_dir: Path, **kwargs) -> CodeSandbox:
    if backend == "docker":
        return LocalDockerSandbox(run_dir, **kwargs)
    if backend == "agentcore":
        return AgentCoreSandbox(run_dir, **kwargs)
    raise ValueError(f"unknown sandbox backend: {backend}")
