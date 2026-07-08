"""Fractionally differentiated price features (fixed-width-window FFD).

López de Prado, Advances in Financial ML ch.5 / Hosking (1981): difference log prices
by a real order d in (0,1) so the series is ~stationary but keeps long memory that
integer returns erase. We use FIXED d values (0.4 and 0.7) rather than estimating d
per ticker — estimation on history would need its own point-in-time protocol, while
fixed d is estimation-free and therefore lookahead-safe by construction.

Each FFD value is a trailing convolution of past log closes; features are z-scored
against their own trailing year so scales are cross-sectionally comparable.

Lookback: 63 (FFD window) + 252 (z-score window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.common import validate_panel

LOOKBACK = 63 + 252

FFD_WINDOW = 63
D_VALUES = (0.4, 0.7)
Z_WINDOW = 252
Z_MIN_PERIODS = 126


def ffd_weights(d: float, window: int) -> np.ndarray:
    """Fixed-width FFD weights w_0=1, w_k = -w_{k-1} * (d - k + 1) / k."""
    w = [1.0]
    for k in range(1, window):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def _ffd_series(log_close: pd.Series, weights: np.ndarray) -> pd.Series:
    """Trailing dot-product of the last `len(weights)` log closes (newest first)."""
    values = log_close.to_numpy()
    window = len(weights)
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = float(np.dot(weights, values[i - window + 1 : i + 1][::-1]))
    return pd.Series(out, index=log_close.index)


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()
    log_close = np.log(panel["close"])

    for d in D_VALUES:
        weights = ffd_weights(d, FFD_WINDOW)
        raw = log_close.groupby(panel["ticker"], sort=False).transform(
            lambda s, w=weights: _ffd_series(s, w)
        )
        grp = raw.groupby(panel["ticker"], sort=False)
        mean = grp.transform(lambda s: s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean())
        std = grp.transform(lambda s: s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std())
        out[f"ffd_z_d{int(d * 10):02d}"] = (raw - mean) / std

    return out
