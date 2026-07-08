"""Label: next-5-day cross-sectional relative return, rank-normalized per date.

The label at date t uses closes at t and t+5 — it exists ONLY for training/backtest
rows and is the one deliberately forward-looking column in the dataset. The
walk-forward harness purges 5 sessions + 2-day embargo so labels never leak into
training folds.
"""

from __future__ import annotations

import pandas as pd

from features.common import by_ticker, validate_panel

HORIZON = 5


def compute_labels(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    fwd = by_ticker(panel)["close"].pct_change(horizon).shift(-horizon)
    out["fwd_ret"] = fwd
    # Cross-sectional rank per date, centered so each date's labels average zero
    pct = out.groupby("date")["fwd_ret"].rank(pct=True)
    out["label"] = pct - pct.groupby(out["date"]).transform("mean")
    return out
