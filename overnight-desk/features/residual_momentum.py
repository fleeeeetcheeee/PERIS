"""Residual momentum family: momentum net of market exposure.

Rolling 63d beta to the benchmark (SPY), residual daily returns, then residual
momentum summed over 126d excluding the most recent 5d (avoids reversal bleed).

Lookback: 189 sessions (63 beta + 126 momentum). Point-in-time: trailing only;
beta at t uses returns through t.
"""

from __future__ import annotations

import pandas as pd

from features.common import daily_returns, validate_panel

LOOKBACK = 189
BENCHMARK = "SPY"


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    out["ret"] = daily_returns(panel)

    mkt = (
        out.loc[out["ticker"] == BENCHMARK, ["date", "ret"]]
        .rename(columns={"ret": "mkt_ret"})
        .set_index("date")["mkt_ret"]
    )
    out["mkt_ret"] = out["date"].map(mkt)

    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        cov = g["ret"].rolling(63, min_periods=40).cov(g["mkt_ret"])
        var = g["mkt_ret"].rolling(63, min_periods=40).var()
        beta = cov / var
        resid = g["ret"] - beta * g["mkt_ret"]
        resid_mom = resid.rolling(126, min_periods=90).sum().shift(5)
        plain_mom = g["ret"].rolling(126, min_periods=90).sum().shift(5)
        return pd.DataFrame(
            {"resmom_beta": beta, "resmom_126x5": resid_mom, "mom_126x5": plain_mom},
            index=g.index,
        )

    feats = out.groupby("ticker", group_keys=False, sort=False).apply(
        per_ticker, include_groups=False
    )
    return pd.concat([out[["date", "ticker"]], feats], axis=1)
