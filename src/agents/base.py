from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class _RuleBasedLLM:
    """
    No-op LLM used when neither Ollama nor Claude API is available.
    Returns a minimal JSON stub so agents degrade gracefully.
    """

    def invoke(self, messages: Any) -> "_RuleBasedResponse":
        return _RuleBasedResponse()


class _RuleBasedResponse:
    content = '{"score": 50, "rationale": "Rule-based default — no LLM configured.", "strengths": [], "risks": [], "recommended_action": "watch"}'


def _build_llm() -> Any:
    """Return the best available LLM: Ollama locally, Claude API, then rule-based fallback."""
    # 1. Try Ollama
    try:
        from langchain_ollama import OllamaLLM
        import httpx
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Quick connectivity check before constructing
        httpx.get(f"{base}/api/tags", timeout=2).raise_for_status()
        logger.info("Using Ollama LLM at %s", base)
        return OllamaLLM(model="llama3.2", base_url=base)
    except Exception:
        pass

    # 2. Try Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            logger.info("Using Claude API (haiku)")
            return ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=api_key,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.warning("Claude API init failed: %s", exc)

    # 3. Rule-based fallback — no crash
    logger.warning(
        "No LLM available (Ollama not running, ANTHROPIC_API_KEY not set). "
        "Using rule-based fallback. Set ANTHROPIC_API_KEY or run `ollama serve`."
    )
    return _RuleBasedLLM()


class BaseAgent(ABC):
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm if llm is not None else _build_llm()

    def _invoke(self, system: str, user: str) -> str:
        """Unified invoke: handles ChatModel, plain LLM, and rule-based fallback."""
        if hasattr(self.llm, "invoke"):
            try:
                result = self.llm.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
                return result.content if hasattr(result, "content") else str(result)
            except TypeError:
                result = self.llm.invoke(f"{system}\n\n{user}")
                return result.content if hasattr(result, "content") else str(result)
        return str(self.llm(f"{system}\n\n{user}"))

    @abstractmethod
    def build_chain(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
