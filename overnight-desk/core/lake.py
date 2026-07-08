"""Parquet lake access. Raw is immutable as-pulled vendor data; curated is validated.

Market data lives here (Parquet + DuckDB), never in SQLite.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from core import paths

PRICES_CURATED = paths.CURATED / "prices.parquet"
MACRO_CURATED = paths.CURATED / "macro.parquet"

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def raw_price_path(source: str, ticker: str) -> Path:
    d = paths.RAW / source
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker.upper()}.parquet"


def write_raw_prices(source: str, ticker: str, df: pd.DataFrame) -> Path:
    """Write as-pulled vendor rows. Never overwrites existing history — raw is immutable;
    new pulls are merged on date with existing rows kept as-is."""
    path = raw_price_path(source, ticker)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df[~df["date"].isin(old["date"])]], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return path


def read_raw_prices(source: str, ticker: str) -> pd.DataFrame | None:
    path = raw_price_path(source, ticker)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_curated_prices(df: pd.DataFrame) -> Path:
    """Replace the curated price table (already validated upstream)."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"curated prices missing columns: {missing}")
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    df = df[PRICE_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(PRICES_CURATED, index=False)
    return PRICES_CURATED


def read_curated_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if not PRICES_CURATED.exists():
        raise FileNotFoundError(
            f"{PRICES_CURATED} not found — run `python -m jobs.nightly --stage ingest` first"
        )
    con = duckdb.connect()
    query = f"SELECT * FROM read_parquet('{PRICES_CURATED}') WHERE 1=1"
    if tickers:
        tickers_sql = ",".join(f"'{t.upper()}'" for t in tickers)
        query += f" AND ticker IN ({tickers_sql})"
    if start:
        query += f" AND date >= DATE '{start}'"
    if end:
        query += f" AND date <= DATE '{end}'"
    query += " ORDER BY ticker, date"
    df = con.execute(query).fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def write_curated_macro(df: pd.DataFrame) -> Path:
    paths.CURATED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MACRO_CURATED, index=False)
    return MACRO_CURATED


def read_curated_macro() -> pd.DataFrame | None:
    if not MACRO_CURATED.exists():
        return None
    df = pd.read_parquet(MACRO_CURATED)
    df["date"] = pd.to_datetime(df["date"])
    return df
