"""Ollama client wrapper, reused from the PERIS pattern.

THE LLM NEVER PRODUCES NUMBERS. Every figure in a briefing is computed upstream and
injected into the prompt; the model formats and explains only. If Ollama is down the
briefing falls back to a plain template — prose is optional, numbers are not.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or OLLAMA_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL

    def _installed_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=3) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def available(self) -> bool:
        """Server up and a usable model resolved. If the configured model isn't
        pulled, fall back to the first installed one rather than failing."""
        models = self._installed_models()
        if not models:
            return False
        if self.model not in models and f"{self.model}:latest" not in models:
            logger.info("Ollama model %s not installed — using %s", self.model, models[0])
            self.model = models[0]
        return True

    def generate(self, system: str, prompt: str, timeout: float = 120.0) -> str | None:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "system": system,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3},
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip() or None
        except Exception as exc:
            logger.warning("Ollama unavailable (%s) — briefing uses plain template", exc)
            return None
