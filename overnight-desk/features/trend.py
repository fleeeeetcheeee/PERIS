"""Vol-scaled trend family.

Lookback: 200 sessions. Point-in-time: trailing SMAs and vols only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.common import by_ticker, daily_returns, validate_panel

LOOKBACK = 200
ANN = np.sqrt(252)


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    ret = daily_returns(panel)
    grp_ret = ret.groupby(panel["ticker"], sort=False)

    vol21 = grp_ret.transform(lambda s: s.rolling(21, min_periods=15).std())
    vol63 = grp_ret.transform(lambda s: s.rolling(63, min_periods=40).std())

    close = panel["close"]
    grp_close = by_ticker(panel)["close"]
    r63 = grp_close.pct_change(63)
    r126 = grp_close.pct_change(126)
    sma50 = grp_close.transform(lambda s: s.rolling(50, min_periods=35).mean())
    sma200 = grp_close.transform(lambda s: s.rolling(200, min_periods=150).mean())

    out["trend_r63_vol"] = r63 / (vol63 * np.sqrt(63))
    out["trend_r126_vol"] = r126 / (vol63 * np.sqrt(126))
    out["trend_sma50_gap"] = (close / sma50 - 1) / (vol21 * np.sqrt(21))
    out["trend_sma200_gap"] = (close / sma200 - 1) / (vol63 * np.sqrt(63))
    out["trend_vol21_ann"] = vol21 * ANN
    return out
