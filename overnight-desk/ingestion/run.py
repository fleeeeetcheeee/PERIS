"""Ingest stage: pull EOD prices + macro, validate, promote raw → curated.

Incremental and cached — never re-fetches dates already in data/raw/.

Source strategy: sources are tried in priority order (Tiingo → Stooq → yfinance),
and each run promotes the whole universe from ONE source. Mixing vendors within a
ticker's history would create adjustment-basis seams at every dividend, so a source
that can't cover the run (rate caps, outages) is abandoned for that run and the next
source takes over — loudly. Tiingo's free tier is hourly-capped below our universe
size, so in practice yfinance remains the nightly workhorse; Tiingo raw data still
accumulates for whichever tickers succeed (raw is immutable and per-source).

Also usable directly: python -m ingestion.run --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import logging

import httpx
import pandas as pd

from core import lake, paths
from core.calendar import last_completed_session
from core.config import Config, load_config
from core.universe import load_universe
from ingestion import curate
from ingestion.fred_client import FredClient
from ingestion.stooq_client import StooqClient
from ingestion.tiingo_client import TiingoClient
from ingestion.yahoo_client import YahooClient

logger = logging.getLogger(__name__)

# Abandon a source for the run after this many consecutive failures — a 429 storm
# burns quota (and time) with every further attempt.
MAX_CONSECUTIVE_FAILURES = 4


def _source_candidates() -> list[tuple[str, TiingoClient | StooqClient | YahooClient]]:
    candidates: list[tuple[str, TiingoClient | StooqClient | YahooClient]] = []
    tiingo = TiingoClient()
    if tiingo.available:
        candidates.append(("tiingo", tiingo))
    candidates.append(("stooq", StooqClient()))
    candidates.append(("yahoo", YahooClient()))
    return candidates


def _fetch_universe(
    source: str,
    client: TiingoClient | StooqClient | YahooClient,
    members: list,
    cfg: Config,
    target_end,
) -> tuple[list[str], bool]:
    """Fetch missing history for every ticker from one source.

    Returns (failed_tickers, aborted). aborted=True means the source broke down
    mid-run (rate-limit storm) and remaining tickers were not attempted.
    """
    failures: list[str] = []
    consecutive = 0
    for m in members:
        cached = lake.read_raw_prices(source, m.ticker)
        if cached is not None and not cached.empty:
            last_cached = pd.Timestamp(cached["date"].max()).date()
            if last_cached >= target_end:
                consecutive = 0
                continue
            start = (pd.Timestamp(last_cached) + pd.Timedelta(days=1)).date().isoformat()
        else:
            start = cfg.start.isoformat()
        try:
            df = client.daily_bars(m.ticker, start=start, end=target_end.isoformat())
            if not df.empty:
                lake.write_raw_prices(source, m.ticker, df)
                logger.info("%s: +%d rows from %s", m.ticker, len(df), source)
            consecutive = 0
        except httpx.HTTPStatusError as exc:
            failures.append(m.ticker)
            consecutive += 1
            if exc.response.status_code == 429 and consecutive >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "%s: rate-limit storm (%d consecutive 429s) — abandoning for this run",
                    source,
                    consecutive,
                )
                return failures, True
            logger.warning("%s: fetch failed: %s", m.ticker, exc)
        except Exception as exc:
            failures.append(m.ticker)
            consecutive += 1
            logger.warning("%s: fetch failed: %s", m.ticker, exc)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "%s: %d consecutive failures — abandoning for this run", source, consecutive
                )
                return failures, True
    return failures, False


def ingest_prices(cfg: Config) -> pd.DataFrame:
    paths.ensure_dirs()
    members = load_universe(cfg.universe_file)
    target_end = last_completed_session().date()

    errors: list[str] = []
    for source, client in _source_candidates():
        failures, aborted = _fetch_universe(source, client, members, cfg, target_end)
        if aborted or len(failures) > len(members) / 4:
            msg = f"{source}: {len(failures)} failures{' (aborted)' if aborted else ''}"
            errors.append(msg)
            logger.warning("%s — trying next source", msg)
            continue
        if failures:
            logger.warning("%s: proceeding without %d tickers: %s", source, len(failures), failures)
        logger.info("ingest source for this run: %s", source)
        return curate.promote(source, [m.ticker for m in members])

    raise RuntimeError(f"all price sources failed: {errors}")


def ingest_macro(cfg: Config) -> pd.DataFrame:
    fred = FredClient()
    panel = fred.macro_panel(start=cfg.start.isoformat())
    # FRED revises history in place and the curated table is overwritten on every
    # ingest, so keep one dated raw snapshot per day — a 2023 HY_OAS revision moved
    # the headline Sharpe 0.79 -> 0.68 and was only provable because the prediction
    # divergence sat exactly on a fold boundary. Never lose a macro vintage again.
    snap_dir = paths.RAW / "fred"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / f"macro_{pd.Timestamp.today().date().isoformat()}.parquet"
    if not snap.exists():
        panel.to_parquet(snap, index=False)
    lake.write_curated_macro(panel)
    logger.info("macro: %d rows across %d series", len(panel), panel["series"].nunique())
    return panel


def run(cfg: Config) -> None:
    prices = ingest_prices(cfg)
    ingest_macro(cfg)
    last = prices["date"].max().date()
    logger.info(
        "ingest complete: %d tickers, curated through %s",
        prices["ticker"].nunique(),
        last,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(paths.CONFIGS / "baseline.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))
