"""SEC EDGAR client, ported from PERIS src/integrations/sec_edgar.py (sync, schema adapted).

Fundamentals are lagged to their filing date — features must join on `filed`, never on
the fiscal period end, to stay point-in-time safe.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from ingestion.base import BaseClient


class EdgarClient(BaseClient):
    BASE_URL = "https://data.sec.gov"

    def __init__(self) -> None:
        super().__init__(base_url=self.BASE_URL)

    @property
    def rate_limit_delay(self) -> float:
        return 0.2

    def headers(self) -> dict[str, str]:
        return {
            "User-Agent": "overnight-desk research fletcherwoojin@gmail.com",
            "Accept": "application/json",
        }

    def parse(self, response: httpx.Response) -> Any:
        return response.json()

    def company_filings(
        self, cik: str, form_type: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        cik_padded = str(cik).zfill(10)
        data = self.fetch(f"/submissions/CIK{cik_padded}.json")
        recent = data.get("filings", {}).get("recent", {})
        filings: list[dict[str, Any]] = []
        for form, date, acc in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            strict=False,
        ):
            if form_type and form != form_type:
                continue
            filings.append(
                {
                    "cik": cik,
                    "entity_name": data.get("name", ""),
                    "form_type": form,
                    "filing_date": date,
                    "accession_number": acc,
                }
            )
            if len(filings) >= limit:
                break
        return filings

    def company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = str(cik).zfill(10)
        return self.fetch(f"/api/xbrl/companyfacts/CIK{cik_padded}.json")

    def concept_series(self, cik: str, tag: str, taxonomy: str = "us-gaap") -> pd.DataFrame:
        """One XBRL concept as DataFrame(period_end, filed, value) — join on `filed`."""
        facts = self.company_facts(cik)
        concept = facts.get("facts", {}).get(taxonomy, {}).get(tag, {})
        rows: list[dict[str, Any]] = []
        for unit_rows in concept.get("units", {}).values():
            for r in unit_rows:
                if "end" in r and "filed" in r and "val" in r:
                    rows.append({"period_end": r["end"], "filed": r["filed"], "value": r["val"]})
        df = pd.DataFrame(rows, columns=["period_end", "filed", "value"])
        if not df.empty:
            df["period_end"] = pd.to_datetime(df["period_end"])
            df["filed"] = pd.to_datetime(df["filed"])
            df = df.sort_values("filed").drop_duplicates(subset=["period_end"], keep="first")
        return df.reset_index(drop=True)
