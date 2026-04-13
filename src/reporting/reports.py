from __future__ import annotations

import logging

from src.db.schema import SessionLocal, init_db
from src.db.queries import list_companies, list_pipeline_stages, list_portfolio_kpis, list_signals
from src.agents.reporting_agent import ReportingAgent

logger = logging.getLogger(__name__)


def generate_weekly_report() -> str:
    """Generate the weekly PDF report. Entry point called by APScheduler."""
    init_db()

    with SessionLocal() as session:
        companies = [
            {"id": c.id, "name": c.name, "sector": c.sector, "score": c.score, "source": c.source}
            for c in list_companies(session, limit=500)
            if c.name != "_MACRO_DATA_"
        ]
        pipeline_stages = [
            {"stage": ps.stage, "company_id": ps.company_id}
            for ps in list_pipeline_stages(session, limit=1000)
        ]
        portfolio_kpis = [
            {"metric_name": k.metric_name, "value": k.value, "period": k.period}
            for k in list_portfolio_kpis(session, limit=1000)
        ]
        signals = [
            {"signal_type": s.signal_type, "summary": s.summary}
            for s in list_signals(session, limit=100)
        ]

    agent = ReportingAgent()
    result = agent.run({
        "companies": companies,
        "pipeline_stages": pipeline_stages,
        "portfolio_kpis": portfolio_kpis,
        "signals": signals,
    })

    pdf_path = result.get("pdf_path", "")
    logger.info("Weekly report generated: %s", pdf_path)
    return pdf_path
