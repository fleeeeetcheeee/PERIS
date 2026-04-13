from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from apscheduler.schedulers.blocking import BlockingScheduler

from src.db.schema import init_db
from src.ingestion.fred import ingest_fred_macro
from src.ingestion.sec_edgar import ingest_sec_edgar
from src.monitoring.portfolio import monitor_portfolio
from src.reporting.reports import generate_weekly_report
from src.scoring.pipeline import score_new_companies


scheduler = BlockingScheduler(timezone="America/New_York")


def register_jobs() -> None:
    scheduler.add_job(
        ingest_sec_edgar,
        trigger="interval",
        hours=6,
        id="ingest_sec_edgar",
        replace_existing=True,
    )
    scheduler.add_job(
        ingest_fred_macro,
        trigger="cron",
        hour=7,
        minute=0,
        id="ingest_fred_macro",
        replace_existing=True,
    )
    scheduler.add_job(
        score_new_companies,
        trigger="cron",
        hour="0,6,12,18",
        minute=30,
        id="score_new_companies",
        replace_existing=True,
    )
    scheduler.add_job(
        monitor_portfolio,
        trigger="cron",
        hour=8,
        minute=0,
        id="monitor_portfolio",
        replace_existing=True,
    )
    scheduler.add_job(
        generate_weekly_report,
        trigger="cron",
        day_of_week="mon",
        hour=9,
        minute=0,
        id="generate_weekly_report",
        replace_existing=True,
    )


def main() -> None:
    init_db()
    logger.info("Initialised PERIS database")
    register_jobs()
    logger.info(
        "Scheduler starting with %d jobs registered", len(scheduler.get_jobs())
    )
    scheduler.start()


if __name__ == "__main__":
    main()
