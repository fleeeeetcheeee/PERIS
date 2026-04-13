from __future__ import annotations

import logging

from src.db.schema import SessionLocal, init_db
from src.db.queries import list_companies, update_company
from src.agents.scoring_agent import ScoringAgent

logger = logging.getLogger(__name__)


def score_new_companies() -> None:
    """Score companies that have no score yet. Called by APScheduler."""
    init_db()
    agent = ScoringAgent()

    with SessionLocal() as session:
        companies = list_companies(session, limit=500)
        unscored = [c for c in companies if c.score is None and c.name != "_MACRO_DATA_"]
        logger.info("Scoring %d unscored companies", len(unscored))

        for company in unscored:
            try:
                profile = {
                    "name": company.name,
                    "sector": company.sector,
                    "country": company.country,
                    "employee_count": company.employee_count,
                    "revenue_estimate": company.revenue_estimate,
                    "source": company.source,
                }
                result = agent.score_company(profile)
                update_company(session, company.id, score=float(result.get("score", 50)))
                logger.debug("Scored %s: %s", company.name, result.get("score"))
            except Exception as exc:
                logger.warning("Scoring failed for company %d (%s): %s", company.id, company.name, exc)
