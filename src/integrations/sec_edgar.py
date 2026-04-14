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
        self._headers = {
            "User-Agent": "PERIS Research Tool contact@peris.local",
            "Accept": "application/json",
        }

    @property
    def rate_limit_delay(self) -> float:
        return 0.2

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
        """
        Search EDGAR full-text index.

        IMPORTANT: httpx params-encoding double-encodes quotes, causing 500s.
        Build the URL manually so %22 stays as %22.
        """
        # Encode the query term — wrap in quotes for phrase search
        q_encoded = f"%22{httpx.QueryParams({'q': query})['q']}%22"

        parts = [f"q={q_encoded}"]
        if form_type:
            parts.append(f"forms={form_type}")
        parts.append("dateRange=custom")
        parts.append(f"startdt={date_range[0] if date_range else '2022-01-01'}")
        parts.append(f"enddt={date_range[1] if date_range else '2099-12-31'}")

        url = f"{self.EFTS_URL}/LATEST/search-index?{'&'.join(parts)}"

        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        hits = data.get("hits", {}).get("hits", [])
        return [self._parse_filing_hit(h) for h in hits[:limit]]

    def _parse_filing_hit(self, hit: dict[str, Any]) -> dict[str, Any]:
        """
        Parse a hit from the EFTS response.

        Real response fields (confirmed 2024):
          _source.display_names  -> ["PROGRESS SOFTWARE CORP /MA  (PRGS)  (CIK 0000876167)"]
          _source.ciks           -> ["0000876167"]
          _source.form           -> "10-K"
          _source.file_date      -> "2024-01-26"
          _source.adsh           -> "0000876167-24-000031"
          _source.sics           -> ["7372"]
          _source.biz_locations  -> ["Burlington, MA"]
        """
        src = hit.get("_source", {})

        # Extract entity name from display_names
        display_names = src.get("display_names", [])
        entity_name = ""
        ticker = None
        if display_names:
            raw = display_names[0]
            # Format: "COMPANY NAME  (TICKER)  (CIK 0000...)"
            # Strip CIK part and ticker
            name_part = re.sub(r"\s*\(CIK\s+\d+\)\s*$", "", raw).strip()
            ticker_match = re.search(r"\(([A-Z]{1,5})\)\s*$", name_part)
            if ticker_match:
                ticker = ticker_match.group(1)
                entity_name = name_part[: ticker_match.start()].strip()
            else:
                entity_name = name_part

        ciks = src.get("ciks", [])
        cik = ciks[0].lstrip("0") if ciks else ""

        sics = src.get("sics", [])
        sic = sics[0] if sics else None

        locations = src.get("biz_locations", [])
        location = locations[0] if locations else None

        return {
            "entity_name": entity_name,
            "ticker": ticker,
            "cik": cik,
            "form_type": src.get("form", src.get("root_forms", [""])[0] if src.get("root_forms") else ""),
            "file_date": src.get("file_date", ""),
            "accession_number": src.get("adsh", hit.get("_id", "")),
            "sic": sic,
            "location": location,
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

    async def get_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch XBRL company facts for a CIK."""
        cik_padded = str(cik).zfill(10)
        url = f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

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
