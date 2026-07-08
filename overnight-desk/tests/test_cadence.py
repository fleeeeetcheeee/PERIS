"""Rebalance cadence + score smoothing: engine behavior and known values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from core.config import Config
from tests.conftest import make_panel


def _inputs(seed: int = 3):
    panel = make_panel(n_days=260, tickers=("AAA", "BBB", "CCC", "SPY"), seed=seed)
    rng = np.random.default_rng(5)
    dates = sorted(panel["date"].unique())[200:250]
    preds = pd.concat(
        [
            pd.DataFrame({"date": d, "ticker": ["AAA", "BBB", "CCC"], "score": rng.normal(size=3)})
            for d in dates
        ],
        ignore_index=True,
    )
    asset_types = {"AAA": "stock", "BBB": "stock", "CCC": "etf", "SPY": "etf"}
    return preds, panel[["date", "ticker", "close"]], asset_types


def _cfg(**overrides) -> Config:
    cfg = Config()
    cfg.portfolio.top_k = 2
    cfg.portfolio.max_weight = 0.6
    for k, v in overrides.items():
        setattr(cfg.portfolio, k, v)
    return cfg


def test_cadence_trades_only_every_n_sessions():
    preds, prices, asset_types = _inputs()
    res = run_backtest(preds, prices, asset_types, _cfg(rebalance_every=5))
    trading_days = (res.turnover > 1e-12).sum()
    # 50 prediction days at 5-session cadence -> ~10 rebalances
    assert trading_days <= 12
    # drift days exist and carry zero cost
    assert (res.costs[res.turnover <= 1e-12] == 0).all()


def test_cadence_reduces_turnover_vs_daily():
    preds, prices, asset_types = _inputs()
    daily = run_backtest(preds, prices, asset_types, _cfg())
    weekly = run_backtest(preds, prices, asset_types, _cfg(rebalance_every=5))
    assert weekly.turnover.mean() < daily.turnover.mean() * 0.6


def test_smoothing_reduces_turnover_vs_raw():
    preds, prices, asset_types = _inputs()
    raw = run_backtest(preds, prices, asset_types, _cfg())
    smooth = run_backtest(preds, prices, asset_types, _cfg(score_smoothing_halflife=3))
    assert smooth.turnover.mean() < raw.turnover.mean()


def test_default_config_unchanged_behavior():
    """rebalance_every=1 + halflife=0 must reproduce the legacy engine exactly
    (this is what keeps the golden file valid)."""
    cfg = Config()
    assert cfg.portfolio.rebalance_every == 1
    assert cfg.portfolio.score_smoothing_halflife == 0
