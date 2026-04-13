from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def _build_llm() -> Any:
    """Return the best available LLM: Ollama locally, Claude API as fallback."""
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model="llama3.2",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    except Exception:
        pass

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=api_key,
            max_tokens=4096,
        )

    raise RuntimeError(
        "No LLM available: start Ollama (ollama serve) or set ANTHROPIC_API_KEY"
    )


class BaseAgent(ABC):
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm if llm is not None else _build_llm()

    def _invoke(self, system: str, user: str) -> str:
        """Unified invoke: handles both ChatModel and plain LLM."""
        if hasattr(self.llm, "invoke"):
            # ChatModel path
            try:
                result = self.llm.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
                return result.content if hasattr(result, "content") else str(result)
            except TypeError:
                # Plain LLM doesn't take message lists
                prompt = f"{system}\n\n{user}"
                return self.llm.invoke(prompt)
        return str(self.llm(f"{system}\n\n{user}"))

    @abstractmethod
    def build_chain(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
