"""Calendar family. Lookback: 0 sessions — pure functions of the date itself."""

from __future__ import annotations

import pandas as pd

from features.common import validate_panel

LOOKBACK = 0


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    dates = pd.to_datetime(out["date"])
    out["cal_dow"] = dates.dt.dayofweek
    out["cal_month"] = dates.dt.month
    days_in_month = dates.dt.days_in_month
    dom = dates.dt.day
    out["cal_turn_of_month"] = ((dom <= 3) | (dom >= days_in_month - 2)).astype(int)
    return out
