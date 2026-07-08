"""Market-turbulence features: Mahalanobis surprise and topological fragmentation.

Two market-LEVEL series, broadcast to every ticker on the date (the macro_regime
precedent): constant within a date, they can't rank names directly, but they let
the tree condition its cross-sectional splits on the state of the market.

- turb_mahal: Kritzman & Li (2010) turbulence — the Mahalanobis distance of today's
  universe return vector from its trailing-year distribution (mean/covariance
  estimated on the window ENDING YESTERDAY, so today's surprise never explains
  itself). Covariance is shrunk halfway to its diagonal (fixed lambda — estimation-
  free, so lookahead-safe) before inversion.
- turb_h0: topological turbulence a la Gidea & Katz (2018), dependency-free H0
  version — treat the last 21 daily cross-sectional return vectors as a point
  cloud, build its minimum spanning tree, and take the L2 norm of the edge
  lengths (the H0 persistence diagram of the Rips filtration). Calm markets keep
  recent days in one tight cluster (short tree); regime transitions stretch it.

Both series are z-scored against their own trailing year.

Lookback: 252 (estimation) + 21 (cloud) + 252 (z-score window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from features.common import validate_panel

LOOKBACK = 252 + 21 + 252

EST_WINDOW = 252
EST_MIN = 126
CLOUD_WINDOW = 21
SHRINK = 0.5  # fixed halfway-to-diagonal covariance shrinkage
Z_WINDOW = 252
Z_MIN_PERIODS = 126


def mahalanobis_series(rets: pd.DataFrame) -> pd.Series:
    """Daily Kritzman-Li turbulence for a wide (date x ticker) return panel."""
    out = pd.Series(np.nan, index=rets.index)
    values = rets.to_numpy()
    for i in range(EST_MIN, len(rets)):
        window = values[max(0, i - EST_WINDOW) : i]  # ends yesterday
        today = values[i]
        cols = ~np.isnan(today) & ~np.isnan(window).any(axis=0)
        if cols.sum() < 5:
            continue
        w, r = window[:, cols], today[cols]
        mu = w.mean(axis=0)
        cov = np.cov(w, rowvar=False)
        cov = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov))
        d = r - mu
        out.iloc[i] = float(d @ np.linalg.pinv(cov) @ d) / cols.sum()  # per-asset scale
    return out


def h0_persistence_series(rets: pd.DataFrame) -> pd.Series:
    """L2 norm of MST edge lengths over the trailing CLOUD_WINDOW daily return vectors."""
    out = pd.Series(np.nan, index=rets.index)
    values = rets.to_numpy()
    for i in range(CLOUD_WINDOW - 1, len(rets)):
        cloud = values[i - CLOUD_WINDOW + 1 : i + 1]
        cols = ~np.isnan(cloud).any(axis=0)
        if cols.sum() < 5:
            continue
        dist = squareform(pdist(cloud[:, cols] / np.sqrt(cols.sum())))
        mst = minimum_spanning_tree(dist)
        out.iloc[i] = float(np.sqrt((mst.data**2).sum()))
    return out


def _zscore(s: pd.Series) -> pd.Series:
    mean = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    std = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    return (s - mean) / std


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = close.pct_change()

    market = pd.DataFrame(
        {
            "turb_mahal": _zscore(mahalanobis_series(rets)),
            "turb_h0": _zscore(h0_persistence_series(rets)),
        }
    )
    out = panel[["date", "ticker"]].copy()
    return out.merge(market, left_on="date", right_index=True, how="left")
