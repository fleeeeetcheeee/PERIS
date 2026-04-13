from __future__ import annotations

from typing import Any

import httpx

from .base import BaseIntegration


class OpenCorporatesIntegration(BaseIntegration):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the OpenCorporates integration client."""
        super().__init__(api_key=api_key, base_url=base_url or "https://api.opencorporates.com")

    def parse(self, response: httpx.Response) -> Any:
        """Parse an OpenCorporates HTTP response into structured data."""
        raise NotImplementedError

    async def search_companies(
        self,
        query: str,
        jurisdiction_code: str | None = None,
        per_page: int = 20,
    ) -> Any:
        """Search OpenCorporates companies by name or keyword."""
        raise NotImplementedError

    async def get_company(self, jurisdiction_code: str, company_number: str) -> Any:
        """Fetch a single company record from OpenCorporates."""
        raise NotImplementedError
