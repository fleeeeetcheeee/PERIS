from __future__ import annotations

import asyncio
import logging

from src.db.schema import SessionLocal, init_db
from src.db.queries import create_company, create_signal, list_companies
from src.integrations.sec_edgar import SecEdgarIntegration

logger = logging.getLogger(__name__)

# M&A-related search terms to ingest
SEARCH_TERMS = ["acquisition", "merger agreement", "definitive agreement", "leveraged buyout"]


async def _run_ingestion() -> int:
    """Async core: search EDGAR, upsert companies and signals."""
    init_db()
    client = SecEdgarIntegration()
    ingested = 0

    with SessionLocal() as session:
        existing_names = {c.name.lower() for c in list_companies(session, limit=10000)}

        for term in SEARCH_TERMS:
            try:
                filings = await client.search_filings(term, form_type="8-K", limit=10)
                for filing in filings:
                    name = filing.get("entity_name", "").strip()
                    if not name or name.lower() in existing_names:
                        continue

                    company = create_company(
                        session,
                        name=name,
                        source="sec_edgar",
                    )
                    existing_names.add(name.lower())

                    create_signal(
                        session,
                        company_id=company.id,
                        signal_type="sec_8k",
                        summary=(
                            f"8-K filed {filing.get('file_date', '')} — "
                            f"search term: {term}"
                        ),
                        raw_data=filing,
                        confidence=0.7,
                    )
                    ingested += 1
            except Exception as exc:
                logger.warning("SEC EDGAR ingestion error for term '%s': %s", term, exc)

    return ingested


def ingest_sec_edgar() -> None:
    """Entry point called by APScheduler."""
    count = asyncio.run(_run_ingestion())
    logger.info("SEC EDGAR ingestion complete: %d new companies", count)
