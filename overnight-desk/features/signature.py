"""Path-signature features: second-level (Lévy area) terms of trailing price paths.

Signatures (Chen iterated integrals) summarize a path's GEOMETRY, not just its
endpoints — information that returns, vol, and momentum all erase. We keep two
depth-2 antisymmetric terms over a trailing 63-session window:

- sig_tp_area: Lévy area of the (time, log price) path — the signed area between
  the path and its chord. Positive when gains arrive late (accelerating), negative
  when early gains stall (fading). Pure momentum TIMING, orthogonal to momentum size.
- sig_pv_levy: Lévy area of the (log price, log volume) path — price/volume
  lead-lag. Positive when price moves precede volume moves within the window.

Depth-2 diagonal terms are quadratic variation (already covered by vol features)
and higher depths explode combinatorially, so we stop at the two areas.

Discrete Chen integrals over every trailing window are computed exactly with a
cumulative-sum identity (no per-window loops). Each raw area is z-scored against
its own trailing year, matching the fracdiff convention.

Lookback: 63 (path window) + 252 (z-score window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.common import validate_panel

LOOKBACK = 63 + 252

SIG_WINDOW = 63
Z_WINDOW = 252
Z_MIN_PERIODS = 126
DLOGV_CLIP = 2.0  # tame split/halt volume spikes; pointwise, so PIT-safe


def levy_area(dx: np.ndarray, dy: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window Lévy area 0.5*(S12 - S21) of the path with increments (dx, dy).

    For the window of increments i in [s, t] with path points X_i = cumsum(dx):
        S12 = sum_i (X_{i-1} - X_{s-1}) dy_i + 0.5 sum_i dx_i dy_i   (and S21 symm.)
    Expanding the inner terms into global cumulative sums makes every window value
    a difference of two cumsum entries — exact, loop-free, and trailing-only.
    NaN increments poison any window containing them (emitted as NaN).
    """
    n = len(dx)
    if n < window:
        return np.full(n, np.nan)
    x = np.nancumsum(np.where(np.isnan(dx), 0.0, dx))
    y = np.nancumsum(np.where(np.isnan(dy), 0.0, dy))
    x_prev = np.concatenate([[0.0], x[:-1]])
    y_prev = np.concatenate([[0.0], y[:-1]])
    p = np.cumsum(x_prev * np.nan_to_num(dy))  # sum x_{i-1} dy_i
    q = np.cumsum(y_prev * np.nan_to_num(dx))  # sum y_{i-1} dx_i

    def wdiff(c: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        out[window - 1 :] = c[window - 1 :] - np.concatenate([[0.0], c[: n - window]])
        return out

    dy_w, dx_w = wdiff(y), wdiff(x)  # window endpoint deltas (cross terms cancel in S12-S21)
    x_start = np.full(n, np.nan)
    y_start = np.full(n, np.nan)
    x_start[window - 1 :] = np.concatenate([[0.0], x[: n - window]])
    y_start[window - 1 :] = np.concatenate([[0.0], y[: n - window]])
    s12 = wdiff(p) - x_start * dy_w
    s21 = wdiff(q) - y_start * dx_w
    area = 0.5 * (s12 - s21)

    bad = np.isnan(dx) | np.isnan(dy)
    poisoned = pd.Series(bad).rolling(window).max().to_numpy()
    area[poisoned != 0] = np.nan
    return area


def _zscore(raw: pd.Series, tickers: pd.Series) -> pd.Series:
    grp = raw.groupby(tickers, sort=False)
    mean = grp.transform(lambda s: s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean())
    std = grp.transform(lambda s: s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std())
    return (raw - mean) / std


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    dlogp = np.log(panel["close"]).groupby(panel["ticker"], sort=False).diff()
    dlogv = (
        np.log(panel["volume"].clip(lower=1.0))
        .groupby(panel["ticker"], sort=False)
        .diff()
        .clip(-DLOGV_CLIP, DLOGV_CLIP)
    )
    dtime = dlogp.notna().astype(float) / SIG_WINDOW  # uniform time steps, NaN-aligned
    dtime[dlogp.isna()] = np.nan

    def per_ticker(dx: pd.Series, dy: pd.Series) -> pd.Series:
        parts = []
        for _, idx in panel.groupby("ticker", sort=False).indices.items():
            parts.append(
                pd.Series(
                    levy_area(dx.iloc[idx].to_numpy(), dy.iloc[idx].to_numpy(), SIG_WINDOW),
                    index=idx,
                )
            )
        return pd.concat(parts).sort_index()

    out["sig_tp_area"] = _zscore(per_ticker(dtime, dlogp), panel["ticker"])
    out["sig_pv_levy"] = _zscore(per_ticker(dlogp, dlogv), panel["ticker"])
    return out
