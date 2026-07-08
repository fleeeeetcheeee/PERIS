"""Base HTTP client, ported from PERIS src/integrations/base.py (sync variant for batch jobs)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseClient(ABC):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self._last_call = 0.0

    @property
    def rate_limit_delay(self) -> float:
        return 0.0

    def _throttle(self) -> None:
        if self.rate_limit_delay > 0:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self._last_call = time.monotonic()

    def fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        self._throttle()
        url = (
            endpoint
            if endpoint.startswith(("http://", "https://"))
            else f"{self.base_url}/{endpoint.lstrip('/')}"
        )
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url, params=params, headers=self.headers())
            response.raise_for_status()
        return self.parse(response)

    def headers(self) -> dict[str, str]:
        return {}

    @abstractmethod
    def parse(self, response: httpx.Response) -> Any:
        raise NotImplementedError
