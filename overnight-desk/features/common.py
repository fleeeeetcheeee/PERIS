"""Shared helpers for feature modules.

Point-in-time contract for every feature family:
- Input is the long panel (date, ticker, open, high, low, close, volume), sorted.
- A feature value at date t may use only rows with date <= t (trailing windows only).
- Output is (date, ticker) + feature columns, one row per input row.
"""

from __future__ import annotations

import pandas as pd


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def by_ticker(panel: pd.DataFrame):
    return panel.groupby("ticker", group_keys=False, sort=False)


def daily_returns(panel: pd.DataFrame) -> pd.Series:
    return by_ticker(panel)["close"].pct_change()
