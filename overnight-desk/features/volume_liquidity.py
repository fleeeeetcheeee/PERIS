"""Volume / liquidity family.

Lookback: 63 sessions. Point-in-time: trailing windows on volume and close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.common import daily_returns, validate_panel

LOOKBACK = 63


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()

    dollar_vol = panel["close"] * panel["volume"]
    grp_dv = dollar_vol.groupby(panel["ticker"], sort=False)
    adv21 = grp_dv.transform(lambda s: s.rolling(21, min_periods=15).mean())
    out["liq_log_adv21"] = np.log1p(adv21)

    grp_vol = panel["volume"].groupby(panel["ticker"], sort=False)
    vmean = grp_vol.transform(lambda s: s.rolling(21, min_periods=15).mean())
    vstd = grp_vol.transform(lambda s: s.rolling(21, min_periods=15).std())
    out["liq_volume_z"] = (panel["volume"] - vmean) / vstd

    ret = daily_returns(panel)
    illiq = (ret.abs() / dollar_vol.replace(0, np.nan)).groupby(panel["ticker"], sort=False)
    out["liq_amihud21"] = illiq.transform(lambda s: s.rolling(21, min_periods=15).mean()) * 1e9
    return out
