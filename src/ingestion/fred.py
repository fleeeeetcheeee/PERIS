from __future__ import annotations

import asyncio
import logging
import os

from src.db.schema import SessionLocal, init_db
from src.db.queries import create_company, create_signal, list_companies
from src.integrations.fred import FredIntegration

logger = logging.getLogger(__name__)


async def _run_ingestion() -> None:
    """Async core: fetch FRED macro snapshot, store as signals on a sentinel company."""
    init_db()
    client = FredIntegration(api_key=os.getenv("FRED_API_KEY", ""))
    snapshot = await client.fetch_macro_snapshot()

    with SessionLocal() as session:
        # Find or create a "MACRO_DATA" sentinel company for storing macro signals
        companies = list_companies(session, limit=10000)
        macro_co = next((c for c in companies if c.name == "_MACRO_DATA_"), None)
        if macro_co is None:
            macro_co = create_company(
                session,
                name="_MACRO_DATA_",
                sector="macro",
                source="fred",
            )

        for series_name, obs in snapshot.items():
            if obs.get("value") is None:
                continue
            create_signal(
                session,
                company_id=macro_co.id,
                signal_type="macro",
                summary=f"{series_name}: {obs['value']} (as of {obs.get('date', 'unknown')})",
                raw_data={"series": series_name, **obs},
                confidence=1.0,
            )

    logger.info("FRED macro ingestion complete: %d series stored", len(snapshot))


def ingest_fred_macro() -> None:
    """Entry point called by APScheduler."""
    asyncio.run(_run_ingestion())
