"""Short-term reversal family.

Lookback: 21 sessions. Point-in-time: trailing windows on close only.
"""

from __future__ import annotations

import pandas as pd

from features.common import by_ticker, daily_returns, validate_panel

LOOKBACK = 21


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()

    ret = daily_returns(panel)
    out["rev_ret_1d"] = ret
    out["rev_ret_5d"] = by_ticker(panel)["close"].pct_change(5)

    # 5d return z-scored against its own trailing 21d distribution of 5d returns
    r5 = out.groupby(panel["ticker"], group_keys=False, sort=False)["rev_ret_5d"]
    mean21 = r5.transform(lambda s: s.rolling(21, min_periods=15).mean())
    std21 = r5.transform(lambda s: s.rolling(21, min_periods=15).std())
    out["rev_z_5d"] = (out["rev_ret_5d"] - mean21) / std21

    return out
