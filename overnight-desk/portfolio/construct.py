"""Portfolio construction: top-K selection, inverse-vol weights, position cap,
vol targeting, and a minimum-trade band as the turnover penalty.

Long-only, cash account: weights are >= 0 and sum to <= 1 (remainder is cash).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import PortfolioConfig

TRADING_DAYS = 252


def select_top_k(scores: pd.Series, eligible: pd.Series, k: int) -> list[str]:
    """scores/eligible indexed by ticker. Returns the k highest-scored eligible tickers."""
    s = scores[eligible.reindex(scores.index, fill_value=False)]
    return list(s.nlargest(k).index)


def inverse_vol_weights(vol: pd.Series, cap: float) -> pd.Series:
    """Inverse-vol weights, normalized, iteratively capped at `cap` per name.

    Once a name hits the cap it stays there; excess redistributes among uncapped
    names only. If everything caps out, the remainder stays in cash (sum < 1).
    """
    return _apply_cap(1.0 / vol.clip(lower=1e-6), cap)


def vol_target_scale(
    weights: pd.Series, trailing_returns: pd.DataFrame, cfg: PortfolioConfig
) -> float:
    """Scale factor <= 1 so realized trailing vol of the target book hits the target.

    trailing_returns: wide (date x ticker) daily returns covering the vol lookback.
    Never levers up (long-only cash account) — scale is capped at 1.
    """
    cols = [t for t in weights.index if t in trailing_returns.columns]
    if not cols:
        return 1.0
    port_ret = trailing_returns[cols].mul(weights[cols], axis=1).sum(axis=1)
    realized = port_ret.tail(cfg.vol_lookback_days).std() * np.sqrt(TRADING_DAYS)
    if realized <= 0 or np.isnan(realized):
        return 1.0
    return float(min(1.0, cfg.vol_target_annual / realized))


def apply_trade_band(target: pd.Series, previous: pd.Series, min_trade_weight: float) -> pd.Series:
    """Skip dust trades: keep the previous weight when |Δw| < band."""
    prev = previous.reindex(target.index.union(previous.index), fill_value=0.0)
    tgt = target.reindex(prev.index, fill_value=0.0)
    small = (tgt - prev).abs() < min_trade_weight
    out = tgt.copy()
    out[small] = prev[small]
    return out[out > 0]


def build_targets(
    scores: pd.Series,
    vol: pd.Series,
    eligible: pd.Series,
    trailing_returns: pd.DataFrame,
    previous: pd.Series,
    cfg: PortfolioConfig,
    exposure_scale: float = 1.0,
    allow_new_entries: bool = True,
    size_mult: pd.Series | None = None,
    size_renormalize: bool = False,
) -> pd.Series:
    """Full construction pipeline for one rebalance. Returns target weights by ticker.

    size_mult: optional per-ticker conviction multiplier in [0, 1] (meta-labeling).
    Missing tickers are neutral (1.0). With size_renormalize the multiplied weights
    are renormalized back to full exposure (a relative tilt); without it the freed
    weight stays in cash (an absolute bet-size / veto).
    """
    picks = select_top_k(scores, eligible, cfg.top_k)
    if not allow_new_entries:
        picks = [t for t in picks if t in previous.index and previous.get(t, 0) > 0]
    if not picks:
        return pd.Series(dtype=float)
    w = _base_weights(picks, vol, trailing_returns, cfg)
    if size_mult is not None:
        mult = size_mult.reindex(w.index).fillna(1.0).clip(lower=0.0, upper=1.0)
        w = w * mult
        if size_renormalize:
            w = _apply_cap(w, cfg.max_weight)  # back to full exposure, cap re-enforced
        else:
            w = w[w > 0]  # vetoed names drop out; their weight stays in cash
        if w.empty:
            return pd.Series(dtype=float)
    scale = vol_target_scale(w, trailing_returns, cfg) * exposure_scale
    w = w * scale
    return apply_trade_band(w, previous, cfg.min_trade_weight)


def _base_weights(
    picks: list[str],
    vol: pd.Series,
    trailing_returns: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.Series:
    """Uncapped-scheme weights for the picks, then the position cap applied uniformly."""
    if cfg.weighting in ("hrp", "rmt_minvar"):
        from portfolio.weighting import hrp_weights, rmt_minvar_weights

        cols = [t for t in picks if t in trailing_returns.columns]
        window = trailing_returns[cols].dropna(how="all")
        # A covariance scheme needs enough observations to say anything; below ~40
        # sessions fall back to inverse-vol rather than trust a junk matrix.
        if len(cols) >= 2 and len(window) >= 40:
            raw = hrp_weights(window) if cfg.weighting == "hrp" else rmt_minvar_weights(window)
            return _apply_cap(raw.reindex(cols).fillna(0.0), cfg.max_weight)
    return inverse_vol_weights(vol.reindex(picks).dropna(), cfg.max_weight)


def _apply_cap(w: pd.Series, cap: float) -> pd.Series:
    """Iterative cap-and-redistribute, same policy as inverse_vol_weights."""
    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    capped = pd.Series(False, index=w.index)
    for _ in range(len(w)):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        capped |= over
        free = ~capped
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w.clip(upper=cap)
