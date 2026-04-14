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
from src.reporting.weekly_report import generate_weekly_report
from src.scoring.pipeline import score_new_companies


scheduler = BlockingScheduler(timezone="America/New_York")


def register_jobs() -> None:
    # Ingest SEC EDGAR + FRED daily at 7:00 AM
    scheduler.add_job(
        ingest_sec_edgar,
        trigger="cron",
        hour=7,
        minute=0,
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
    # Score new companies at 7:30 AM daily (after ingest)
    scheduler.add_job(
        score_new_companies,
        trigger="cron",
        hour=7,
        minute=30,
        id="score_new_companies",
        replace_existing=True,
    )
    # Portfolio monitoring at 9:00 AM daily
    scheduler.add_job(
        monitor_portfolio,
        trigger="cron",
        hour=9,
        minute=0,
        id="monitor_portfolio",
        replace_existing=True,
    )
    # Weekly PDF report every Monday at 8:00 AM
    scheduler.add_job(
        generate_weekly_report,
        trigger="cron",
        day_of_week="mon",
        hour=8,
        minute=0,
        id="generate_weekly_report",
        replace_existing=True,
    )


def _print_next_runs() -> None:
    """Print a startup summary showing the next scheduled run for each job.

    Uses trigger.get_next_fire_time() so it works before the scheduler starts.
    """
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc)

    print("\n" + "=" * 58)
    print("  PERIS Scheduler  —  Next Run Times (America/New_York)")
    print("=" * 58)
    for job in scheduler.get_jobs():
        next_fire = job.trigger.get_next_fire_time(None, now)
        next_str = next_fire.strftime("%a %Y-%m-%d %H:%M %Z") if next_fire else "—"
        print(f"  {job.id:<30}  {next_str}")
    print("=" * 58 + "\n")


def main() -> None:
    init_db()
    logger.info("Initialised PERIS database")
    register_jobs()
    _print_next_runs()
    logger.info(
        "Scheduler starting with %d jobs registered", len(scheduler.get_jobs())
    )
    scheduler.start()


if __name__ == "__main__":
    main()
