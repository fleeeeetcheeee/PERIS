"""LLM adapters for agent roles.

OllamaLLM wraps the existing llm.ollama_client with a per-role temperature and
model; ScriptedLLM replays canned responses so the whole graph is testable
deterministically and offline.
"""

from __future__ import annotations

import logging

import httpx

from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The pipeline cannot run without agent output (no silent HOLD-by-crash)."""


class OllamaLLM:
    """max_tokens caps the answer; think=False disables chain-of-thought on
    thinking-mode models (qwen3 family) — without it a single agent turn takes
    ~10 minutes of hidden reasoning and a 12-call decision takes 2 hours."""

    def __init__(
        self,
        model: str | None,
        temperature: float,
        timeout: float = 300.0,
        max_tokens: int = 600,
    ) -> None:
        self._client = OllamaClient(model=model) if model else OllamaClient()
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._checked = False
        self._supports_think_flag = True

    @property
    def model(self) -> str:
        return self._client.model

    def generate(self, system: str, prompt: str) -> str:
        if not self._checked:
            if not self._client.available():
                raise LLMUnavailableError(
                    "Ollama is not reachable or has no models installed — "
                    "`brew services start ollama` and pull a model"
                )
            self._checked = True
        payload = {
            "model": self._client.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if self._supports_think_flag:
            payload["think"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self._client.base_url}/api/generate", json=payload)
                if resp.status_code == 400 and self._supports_think_flag:
                    # older Ollama / non-thinking model: drop the flag and retry once
                    self._supports_think_flag = False
                    payload.pop("think")
                    resp = client.post(f"{self._client.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama call failed: {exc}") from exc
        if not text:
            raise LLMUnavailableError("Ollama returned an empty response")
        return text


class ScriptedLLM:
    """Deterministic test double: pops responses in order and records prompts."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.model = "scripted"

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)
