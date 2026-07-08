"""Macro regime family (FRED series). Same value across all tickers on a date.

Point-in-time: every macro series is LAGGED ONE BUSINESS DAY before joining —
FRED can post same-day values after the US close, so the value used at t is the
one published for t-1. Lookback: 21 sessions + 1 lag day.
"""

from __future__ import annotations

import pandas as pd

from features.common import validate_panel

LOOKBACK = 22


def _pivot_lagged(macro: pd.DataFrame) -> pd.DataFrame:
    wide = macro.pivot_table(index="date", columns="series", values="value", aggfunc="last")
    wide = wide.sort_index().ffill()
    return wide.shift(1)  # one-business-day publication lag


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    if macro is None or macro.empty:
        # Fail soft with NaNs — the model treats missing macro as neutral.
        for col in ("mac_vix", "mac_vix_ts", "mac_t10y2y", "mac_hy_oas_chg21", "mac_dgs10_chg21"):
            out[col] = float("nan")
        return out

    wide = _pivot_lagged(macro)
    feats = pd.DataFrame(index=wide.index)
    feats["mac_vix"] = wide.get("VIX")
    if "VIX" in wide and "VIX3M" in wide:
        feats["mac_vix_ts"] = wide["VIX3M"] / wide["VIX"]  # >1 = contango (calm)
    else:
        feats["mac_vix_ts"] = float("nan")
    feats["mac_t10y2y"] = wide.get("T10Y2Y")
    if "HY_OAS" in wide:
        feats["mac_hy_oas_chg21"] = wide["HY_OAS"].diff(21)
    if "DGS10" in wide:
        feats["mac_dgs10_chg21"] = wide["DGS10"].diff(21)

    merged = out.merge(feats.reset_index().rename(columns={"index": "date"}), on="date", how="left")
    # Trading days with no macro row yet (e.g. holidays in FRED): forward-fill per date order
    macro_cols = [c for c in merged.columns if c.startswith("mac_")]
    date_feats = merged[["date"] + macro_cols].drop_duplicates("date").sort_values("date")
    date_feats[macro_cols] = date_feats[macro_cols].ffill()
    merged = out.merge(date_feats, on="date", how="left")
    return merged
