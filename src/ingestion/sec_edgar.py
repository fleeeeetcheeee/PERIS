from __future__ import annotations

import asyncio
import logging

from src.db.schema import SessionLocal, init_db
from src.db.queries import create_company, create_signal, list_companies
from src.integrations.sec_edgar import SecEdgarIntegration

logger = logging.getLogger(__name__)

# SIC-code → human-readable sector
SIC_SECTOR_MAP = {
    "7372": "Software",
    "7371": "Software",
    "7374": "Technology Services",
    "7379": "Technology Services",
    "3674": "Semiconductors",
    "3672": "Hardware",
    "6021": "Banking",
    "6022": "Banking",
    "6211": "Finance",
    "8000": "Healthcare",
    "8011": "Healthcare",
    "5047": "Healthcare Distribution",
    "2836": "Pharmaceuticals",
    "3841": "Medical Devices",
    "4813": "Telecom",
    "4812": "Telecom",
    "4911": "Utilities",
}

# Search terms mapped to broad sector hint
SEARCH_QUERIES = [
    ("software", "10-K"),
    ("SaaS revenue", "10-K"),
    ("acquisition", "8-K"),
    ("merger agreement", "8-K"),
]


async def _run_ingestion() -> int:
    init_db()
    client = SecEdgarIntegration()
    ingested = 0

    with SessionLocal() as session:
        existing_names = {c.name.lower() for c in list_companies(session, limit=10000)}

        for query, form in SEARCH_QUERIES:
            try:
                filings = await client.search_filings(
                    query,
                    form_type=form,
                    date_range=("2024-01-01", "2026-04-13"),
                    limit=15,
                )
                for filing in filings:
                    name = filing.get("entity_name", "").strip()
                    if not name or name.lower() in existing_names:
                        continue

                    sic = filing.get("sic")
                    sector = SIC_SECTOR_MAP.get(str(sic), None) if sic else None

                    location = filing.get("location", "")
                    country = "US" if location else None

                    company = create_company(
                        session,
                        name=name,
                        sector=sector,
                        country=country,
                        source="sec_edgar",
                    )
                    existing_names.add(name.lower())

                    create_signal(
                        session,
                        company_id=company.id,
                        signal_type="sec_8k" if form == "8-K" else "news",
                        summary=(
                            f"{form} filed {filing.get('file_date', '')} | "
                            f"CIK {filing.get('cik', '')} | query: {query}"
                        ),
                        raw_data=filing,
                        confidence=0.7,
                    )
                    ingested += 1
                    logger.info("Ingested: %s (sector=%s)", name, sector)

            except Exception as exc:
                logger.warning("SEC EDGAR ingestion error for '%s': %s", query, exc)

    return ingested


def ingest_sec_edgar() -> None:
    count = asyncio.run(_run_ingestion())
    logger.info("SEC EDGAR ingestion complete: %d new companies", count)
    print(f"[sec_edgar] Ingested {count} new companies")
