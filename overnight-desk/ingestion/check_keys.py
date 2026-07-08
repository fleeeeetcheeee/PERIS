"""Verify API keys and report which data sources / gates are active.

    uv run python -m ingestion.check_keys

Makes one cheap call per configured key. Exit code 0 always — this is a status
report, not a gate (the pipeline itself degrades loudly where it matters).
"""

from __future__ import annotations

import datetime as dt
import os

import core  # noqa: F401  — importing core loads .env into the environment
from ingestion.finnhub_client import FinnhubClient
from ingestion.fred_client import FredClient
from ingestion.tiingo_client import TiingoClient

OK = "\033[32mACTIVE\033[0m"
OFF = "\033[33minactive\033[0m"
BAD = "\033[31mFAILED\033[0m"


def check_tiingo() -> tuple[str, str]:
    client = TiingoClient()
    if not client.available:
        return OFF, "no TIINGO_API_KEY — prices come from yfinance fallback"
    try:
        df = client.daily_bars("SPY", start="2026-06-25")
        return OK, f"primary EOD source (SPY test: {len(df)} rows)"
    except Exception as exc:
        return BAD, f"key set but request failed: {exc}"


def check_finnhub() -> tuple[str, str]:
    client = FinnhubClient()
    if not client.available:
        return OFF, "no FINNHUB_API_KEY — earnings-exclusion gate is INACTIVE"
    try:
        today = dt.date.today()
        df = client.earnings_calendar(str(today), str(today + dt.timedelta(days=7)))
        return OK, f"earnings gate live ({len(df)} events next 7 days)"
    except Exception as exc:
        return BAD, f"key set but request failed: {exc}"


def check_fred() -> tuple[str, str]:
    client = FredClient()
    keyed = bool(client.api_key)
    try:
        df = client.series_observations("VIXCLS", start="2026-06-25")
        mode = "keyed API" if keyed else "keyless CSV fallback"
        return OK, f"{mode} ({len(df)} VIX rows)"
    except Exception as exc:
        return BAD, f"request failed: {exc}"


def check_ollama() -> tuple[str, str]:
    from llm.ollama_client import OllamaClient

    client = OllamaClient()
    if client.available():
        return OK, f"commentary model: {client.model}"
    return OFF, "server not reachable — tickets render without commentary"


def main() -> None:
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    print(
        f".env file: {'found' if os.path.exists(env_file) else 'NOT FOUND (copy .env.example)'}\n"
    )
    for name, fn in [
        ("Tiingo ", check_tiingo),
        ("Finnhub", check_finnhub),
        ("FRED   ", check_fred),
        ("Ollama ", check_ollama),
    ]:
        status, detail = fn()
        print(f"  {name}  {status:<20} {detail}")


if __name__ == "__main__":
    main()
