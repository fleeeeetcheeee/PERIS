from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


@click.group()
def cli() -> None:
    """PERIS — Private Equity Research Intelligence System CLI."""


# ---------------------------------------------------------------------------
# peris ingest
# ---------------------------------------------------------------------------

@cli.group()
def ingest() -> None:
    """Run data ingestion jobs."""


@ingest.command("sec")
@click.option("--query", default="acquisition", show_default=True, help="Search term")
@click.option("--limit", default=20, show_default=True, help="Max filings to fetch")
def ingest_sec(query: str, limit: int) -> None:
    """Ingest SEC EDGAR filings into the companies table."""
    from src.db.schema import init_db
    from src.ingestion.sec_edgar import ingest_sec_edgar

    init_db()
    click.echo(f"Running SEC EDGAR ingestion (query='{query}', limit={limit})…")
    ingest_sec_edgar()
    click.echo("Done.")


@ingest.command("fred")
def ingest_fred() -> None:
    """Ingest FRED macro data snapshot."""
    from src.db.schema import init_db
    from src.ingestion.fred import ingest_fred_macro

    init_db()
    click.echo("Running FRED macro ingestion…")
    ingest_fred_macro()
    click.echo("Done.")


@ingest.command("rss")
def ingest_rss() -> None:
    """Ingest Reuters M&A and SEC 8-K RSS feeds."""
    from src.db.schema import SessionLocal, init_db
    from src.db.queries import list_companies, create_signal
    from src.integrations.rss_feeds import RSSFeedsIntegration

    init_db()
    click.echo("Fetching RSS feeds…")

    async def _run() -> None:
        client = RSSFeedsIntegration()
        items = await client.fetch_default_feeds()
        click.echo(f"Fetched {len(items)} feed items")

        with SessionLocal() as session:
            companies = list_companies(session, limit=10000)
            if not companies:
                click.echo("No companies in DB — run `peris ingest sec` first")
                return

            default_company_id = companies[0].id
            for item in items[:50]:
                create_signal(
                    session,
                    company_id=default_company_id,
                    signal_type=item["signal_type"],
                    summary=f"{item['title']} — {item['summary'][:200]}",
                    raw_data=item,
                    confidence=0.6,
                )
        click.echo("Stored RSS signals.")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# peris score
# ---------------------------------------------------------------------------

@cli.command("score")
@click.option("--company-id", type=int, default=None, help="Score a single company by ID")
@click.option("--all", "score_all", is_flag=True, default=False, help="Score all unscored companies")
def score(company_id: int | None, score_all: bool) -> None:
    """Score companies against the investment thesis."""
    from src.db.schema import SessionLocal, init_db
    from src.db.queries import get_company, list_companies, update_company
    from src.agents.scoring_agent import ScoringAgent

    init_db()
    agent = ScoringAgent()

    with SessionLocal() as session:
        if company_id is not None:
            company = get_company(session, company_id)
            if company is None:
                click.echo(f"Company {company_id} not found.", err=True)
                sys.exit(1)
            targets = [company]
        elif score_all:
            targets = [
                c for c in list_companies(session, limit=500)
                if c.score is None and c.name != "_MACRO_DATA_"
            ]
            click.echo(f"Scoring {len(targets)} unscored companies…")
        else:
            click.echo("Specify --company-id <id> or --all", err=True)
            sys.exit(1)

        for company in targets:
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
            click.echo(
                f"  [{company.id}] {company.name}: score={result.get('score')} "
                f"action={result.get('recommended_action')}"
            )


# ---------------------------------------------------------------------------
# peris report
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option("--output", default=None, help="Override output directory (default: ./reports)")
def report(output: str | None) -> None:
    """Generate the weekly PDF intelligence report."""
    if output:
        os.environ["REPORTS_DIR"] = output

    from src.reporting.weekly_report import generate_weekly_report

    click.echo("Generating weekly PDF report…")
    pdf_path = generate_weekly_report()
    click.echo(f"Report saved to: {pdf_path}")


# ---------------------------------------------------------------------------
# peris monitor
# ---------------------------------------------------------------------------

@cli.command("monitor")
def monitor() -> None:
    """Run the portfolio monitoring agent against all tracked companies."""
    from src.db.schema import init_db
    from src.monitoring.portfolio import monitor_portfolio

    init_db()
    click.echo("Running portfolio monitoring…")
    monitor_portfolio()
    click.echo("Done.")


# ---------------------------------------------------------------------------
# peris companies  (bonus: quick list)
# ---------------------------------------------------------------------------

@cli.command("companies")
@click.option("--limit", default=20, show_default=True)
def companies(limit: int) -> None:
    """List tracked companies."""
    from src.db.schema import SessionLocal, init_db
    from src.db.queries import list_companies

    init_db()
    with SessionLocal() as session:
        cos = list_companies(session, limit=limit)
    if not cos:
        click.echo("No companies yet — run `peris ingest sec`")
        return
    click.echo(f"{'ID':>4}  {'Score':>6}  {'Sector':<20}  Name")
    click.echo("-" * 60)
    for c in cos:
        score_str = f"{c.score:.1f}" if c.score is not None else "    —"
        click.echo(f"{c.id:>4}  {score_str:>6}  {(c.sector or ''):<20}  {c.name}")


if __name__ == "__main__":
    cli()
