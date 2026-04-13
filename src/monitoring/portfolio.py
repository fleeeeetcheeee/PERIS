from __future__ import annotations

import asyncio
import logging

from src.db.schema import SessionLocal, init_db
from src.db.queries import list_companies, list_signals, list_portfolio_kpis, create_signal
from src.agents.monitoring_agent import MonitoringAgent
from src.integrations.yahoo_finance import YahooFinanceIntegration

logger = logging.getLogger(__name__)


async def _run_monitoring() -> None:
    init_db()
    agent = MonitoringAgent()
    yf_client = YahooFinanceIntegration()

    with SessionLocal() as session:
        companies = [
            c for c in list_companies(session, limit=200)
            if c.score is not None and c.score >= 60
        ]
        logger.info("Monitoring %d portfolio/pipeline companies", len(companies))

        for company in companies:
            try:
                signals = [
                    {"signal_type": s.signal_type, "summary": s.summary}
                    for s in list_signals(session, company_id=company.id, limit=20)
                ]
                kpis = [
                    {"metric_name": k.metric_name, "value": k.value, "period": k.period}
                    for k in list_portfolio_kpis(session, company_id=company.id, limit=10)
                ]

                # Try to get a price quote if it looks like a ticker
                price_data: dict = {}

                result = agent.run({
                    "company": {
                        "id": company.id,
                        "name": company.name,
                        "sector": company.sector,
                        "score": company.score,
                    },
                    "signals": signals,
                    "kpis": kpis,
                    "price_data": price_data,
                })

                # Persist high-severity alerts as signals
                for alert in result.get("alerts", []):
                    if alert.get("severity") == "high":
                        create_signal(
                            session,
                            company_id=company.id,
                            signal_type="monitoring_alert",
                            summary=f"[{alert.get('type', 'alert').upper()}] {alert.get('title', '')}",
                            raw_data=alert,
                            confidence=0.85,
                        )
            except Exception as exc:
                logger.warning("Monitoring failed for %s: %s", company.name, exc)


def monitor_portfolio() -> None:
    """Entry point called by APScheduler."""
    asyncio.run(_run_monitoring())
