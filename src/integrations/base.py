from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseIntegration(ABC):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")

    @property
    def rate_limit_delay(self) -> float:
        return 0.0

    async def fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        if self.rate_limit_delay > 0:
            await asyncio.sleep(self.rate_limit_delay)

        url = endpoint if endpoint.startswith(("http://", "https://")) else f"{self.base_url}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        return self.parse(response)

    @abstractmethod
    def parse(self, response: httpx.Response) -> Any:
        raise NotImplementedError
