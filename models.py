"""Chat model factory: Anthropic API (Claude Console) or a local
OpenAI-compatible endpoint.

Dev cost policy: claude-haiku-4-5 only while developing. Reads the key
from HYDRO_ANTHROPIC_API_KEY — project-specific on purpose, so this demo
never picks up a global ANTHROPIC_API_KEY by accident.
"""

import os


def get_model(provider: str, config: dict):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        ac = config.get("anthropic", {})
        api_key = os.environ.get("HYDRO_ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "HYDRO_ANTHROPIC_API_KEY is not set (project-specific key "
                "from console.anthropic.com)")
        return ChatAnthropic(
            model=ac.get("model", "claude-haiku-4-5"),
            api_key=api_key,
            temperature=0.2,
            max_tokens=4096,
            timeout=300,
        )
    # "local": any OpenAI-compatible endpoint (LM Studio, vLLM, llama.cpp, ...)
    from langchain_openai import ChatOpenAI

    lc = config.get("local", {})
    return ChatOpenAI(
        base_url=lc.get("base_url", "http://localhost:11434/v1"),
        api_key="not-needed",
        model=lc.get("model", ""),
        temperature=0.2,
        max_tokens=4096,
        timeout=300,
    )
