"""EDGAR fundamentals + filing-event ingestion (wave-6 data, keyless).

Usage: uv run python -m ingestion.fundamentals [--refresh]

Two curated outputs:
- data/curated/fundamentals.parquet: (ticker, tag, unit, start, end, filed, form, fp, val)
  one row per XBRL fact VINTAGE — a restated value is a new row with its own `filed`
  date, so downstream as-of joins on `filed` are point-in-time correct by
  construction (never join on the fiscal period end).
- data/curated/filing_events.parquet: (ticker, form, filed, accepted, items)
  from the submissions API, `accepted` converted to naive US/Eastern. 8-K rows with
  item 2.02 are earnings announcements — the PEAD event source.

Raw JSON snapshots live in data/raw/edgar/ and are reused unless --refresh: facts
are append-mostly and every row carries its own filed date, so a stale snapshot is
PIT-safe, just shorter.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from core import paths
from core.universe import load_universe
from ingestion.edgar_client import EdgarClient

logger = logging.getLogger(__name__)

RAW_DIR = paths.RAW / "edgar"
FUNDAMENTALS_CURATED = paths.CURATED / "fundamentals.parquet"
FILING_EVENTS_CURATED = paths.CURATED / "filing_events.parquet"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

GAAP_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Assets",
    "StockholdersEquity",
    "NetCashProvidedByUsedInOperatingActivities",
    "EarningsPerShareDiluted",
]
DEI_TAGS = ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"]

EVENT_FORMS = {"10-K", "10-Q", "8-K"}
EARLIEST = "2016-01-01"  # facts/filings before this predate any backtest window


def cik_map(client: EdgarClient, refresh: bool = False) -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "company_tickers.json"
    if path.exists() and not refresh:
        data = json.loads(path.read_text())
    else:
        data = client.fetch(TICKER_MAP_URL)
        path.write_text(json.dumps(data))
    return {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}


def _cached_json(client: EdgarClient, path, endpoint: str, refresh: bool) -> dict:
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = client.fetch(endpoint)
    path.write_text(json.dumps(data))
    return data


def extract_facts(facts_json: dict, ticker: str) -> pd.DataFrame:
    """Selected us-gaap + dei concepts as long vintage rows."""
    rows: list[dict] = []
    for taxonomy, tags in (("us-gaap", GAAP_TAGS), ("dei", DEI_TAGS)):
        concepts = facts_json.get("facts", {}).get(taxonomy, {})
        for tag in tags:
            for unit, unit_rows in concepts.get(tag, {}).get("units", {}).items():
                for r in unit_rows:
                    if "end" not in r or "filed" not in r or "val" not in r:
                        continue
                    if r["end"] < EARLIEST:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "tag": tag,
                            "unit": unit,
                            "start": r.get("start"),
                            "end": r["end"],
                            "filed": r["filed"],
                            "form": r.get("form", ""),
                            "fp": r.get("fp", ""),
                            "val": float(r["val"]),
                        }
                    )
    df = pd.DataFrame(
        rows, columns=["ticker", "tag", "unit", "start", "end", "filed", "form", "fp", "val"]
    )
    if df.empty:
        return df
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    return df.drop_duplicates(subset=["tag", "unit", "start", "end", "filed", "val"]).reset_index(
        drop=True
    )


def _submission_events(data: dict, ticker: str) -> pd.DataFrame:
    rec = data.get("filings", {}).get("recent", {})
    frame = pd.DataFrame(
        {
            "form": rec.get("form", []),
            "filed": rec.get("filingDate", []),
            "accepted": rec.get("acceptanceDateTime", []),
            "items": rec.get("items", []),
        }
    )
    frame = frame[frame["form"].isin(EVENT_FORMS)]
    frame.insert(0, "ticker", ticker)
    return frame


def fetch_events(client: EdgarClient, ticker: str, cik: str, refresh: bool) -> pd.DataFrame:
    """10-K/10-Q/8-K filing events since EARLIEST, following continuation files
    when the recent window doesn't reach back far enough (heavy filers)."""
    cik_padded = str(cik).zfill(10)
    data = _cached_json(
        client,
        RAW_DIR / f"submissions_{ticker}.json",
        f"/submissions/CIK{cik_padded}.json",
        refresh,
    )
    frames = [_submission_events(data, ticker)]
    recent_dates = data.get("filings", {}).get("recent", {}).get("filingDate", [])
    if recent_dates and min(recent_dates) > EARLIEST:
        for extra in data.get("filings", {}).get("files", []):
            if extra.get("filingTo", "") < EARLIEST:
                continue
            more = _cached_json(
                client,
                RAW_DIR / f"submissions_{ticker}_{extra['name']}",
                f"/submissions/{extra['name']}",
                refresh,
            )
            # continuation files hold the columns at top level, not under "recent"
            frames.append(_submission_events({"filings": {"recent": more}}, ticker))
    events = pd.concat(frames, ignore_index=True)
    events = events[events["filed"] >= EARLIEST]
    events["filed"] = pd.to_datetime(events["filed"])
    accepted = pd.to_datetime(events["accepted"], utc=True, errors="coerce")
    events["accepted"] = accepted.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return events.drop_duplicates().sort_values("filed").reset_index(drop=True)


def run(refresh: bool = False, universe_file: str = "data/reference/universe.csv") -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = EdgarClient()
    ciks = cik_map(client, refresh=refresh)
    stocks = [m.ticker for m in load_universe(universe_file) if m.asset_type == "stock"]

    fact_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for i, ticker in enumerate(stocks, 1):
        cik = ciks.get(ticker.upper())
        if cik is None:
            missing.append(ticker)
            continue
        facts_json = _cached_json(
            client,
            RAW_DIR / f"companyfacts_{ticker}.json",
            f"/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json",
            refresh,
        )
        facts = extract_facts(facts_json, ticker)
        events = fetch_events(client, ticker, cik, refresh)
        fact_frames.append(facts)
        event_frames.append(events)
        logger.info(
            "[%d/%d] %s: %d fact rows, %d filing events",
            i,
            len(stocks),
            ticker,
            len(facts),
            len(events),
        )

    if missing:
        logger.warning("no CIK found for: %s", ", ".join(missing))
    fundamentals = pd.concat(fact_frames, ignore_index=True)
    filing_events = pd.concat(event_frames, ignore_index=True)
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    fundamentals.to_parquet(FUNDAMENTALS_CURATED, index=False)
    filing_events.to_parquet(FILING_EVENTS_CURATED, index=False)
    logger.info(
        "wrote %s (%d rows) and %s (%d rows)",
        FUNDAMENTALS_CURATED,
        len(fundamentals),
        FILING_EVENTS_CURATED,
        len(filing_events),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refetch cached raw JSON")
    args = parser.parse_args()
    run(refresh=args.refresh)
