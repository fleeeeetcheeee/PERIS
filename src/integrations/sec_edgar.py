from __future__ import annotations

import re
from typing import Any

import httpx

from .base import BaseIntegration


class SecEdgarIntegration(BaseIntegration):
    """Async client for SEC EDGAR full-text search and company facts APIs."""

    BASE_URL = "https://data.sec.gov"
    EFTS_URL = "https://efts.sec.gov"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)
        # SEC requires a descriptive User-Agent per their fair-access policy
        self._headers = {
            "User-Agent": "PERIS Research Tool contact@peris.local",
            "Accept": "application/json",
        }

    @property
    def rate_limit_delay(self) -> float:
        # SEC requests ≤10 req/s; 0.15 s gives comfortable headroom
        return 0.15

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    # ------------------------------------------------------------------
    # Full-text search (EFTS endpoint)
    # ------------------------------------------------------------------

    async def search_filings(
        self,
        query: str,
        form_type: str | None = None,
        date_range: tuple[str, str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search EDGAR full-text index. Returns list of filing dicts."""
        params: dict[str, Any] = {
            "q": f'"{query}"',
            "_source": "file_date,entity_name,file_num,form_type,period_of_report",
            "dateRange": "custom",
            "startdt": date_range[0] if date_range else "2020-01-01",
            "enddt": date_range[1] if date_range else "2099-12-31",
            "hits.hits.total.value": limit,
        }
        if form_type:
            params["forms"] = form_type

        url = f"{self.EFTS_URL}/LATEST/search-index"
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        hits = data.get("hits", {}).get("hits", [])
        return [self._parse_filing_hit(h) for h in hits[:limit]]

    def _parse_filing_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        src = hit.get("_source", {})
        return {
            "entity_name": src.get("entity_name", ""),
            "form_type": src.get("form_type", ""),
            "file_date": src.get("file_date", ""),
            "period_of_report": src.get("period_of_report", ""),
            "accession_number": hit.get("_id", ""),
        }

    # ------------------------------------------------------------------
    # Company filings (submissions endpoint)
    # ------------------------------------------------------------------

    async def get_company_filings(
        self,
        cik: str,
        form_type: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Fetch recent filings for a CIK from the submissions endpoint."""
        cik_padded = str(cik).zfill(10)
        url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        filings: list[dict[str, Any]] = []
        for form, date, acc, doc in zip(forms, dates, accessions, descriptions):
            if form_type and form != form_type:
                continue
            filings.append({
                "cik": cik,
                "entity_name": data.get("name", ""),
                "form_type": form,
                "filing_date": date,
                "accession_number": acc,
                "primary_document": doc,
            })
            if len(filings) >= limit:
                break

        return filings

    # ------------------------------------------------------------------
    # Company facts (XBRL inline data)
    # ------------------------------------------------------------------

    async def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch XBRL company facts for a CIK."""
        cik_padded = str(cik).zfill(10)
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Convenience: parse filing into companies-table dict
    # ------------------------------------------------------------------

    async def filing_to_company_dict(self, cik: str) -> dict[str, Any]:
        """Build a dict suitable for inserting into the companies table."""
        filings = await self.get_company_filings(cik, limit=1)
        entity_name = filings[0]["entity_name"] if filings else f"CIK-{cik}"

        facts: dict[str, Any] = {}
        try:
            raw = await self.get_company_facts(cik)
            us_gaap = raw.get("facts", {}).get("us-gaap", {})

            revenue_series = (
                us_gaap.get("Revenues", {})
                or us_gaap.get("RevenueFromContractWithCustomerExcludingAssessedTax", {})
            )
            if revenue_series:
                units = list(revenue_series.get("units", {}).values())
                if units:
                    latest = sorted(units[0], key=lambda x: x.get("end", ""))[-1]
                    facts["revenue_estimate"] = latest.get("val")

            emp_series = us_gaap.get("NumberOfEmployees", {}) or us_gaap.get(
                "EntityNumberOfEmployees", {}
            )
            if emp_series:
                units = list(emp_series.get("units", {}).values())
                if units:
                    latest = sorted(units[0], key=lambda x: x.get("end", ""))[-1]
                    facts["employee_count"] = int(latest.get("val", 0))
        except Exception:
            pass

        return {
            "name": entity_name,
            "source": "sec_edgar",
            **facts,
        }
