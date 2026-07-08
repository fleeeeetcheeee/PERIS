"""Backtester tests: cost accounting, metrics known values, and the golden file.

The golden file (tests/fixtures/backtest_golden.json) pins the full summary of a
deterministic synthetic backtest. Any change to engine/cost/metric math shows up
as a diff here — update the golden ONLY with walk-forward evidence per CLAUDE2.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.metrics import deflated_sharpe, hit_rate_ci, max_drawdown, sharpe, summarize
from core.config import Config
from tests.conftest import make_panel

GOLDEN = Path(__file__).parent / "fixtures" / "backtest_golden.json"


def _bt_inputs():
    panel = make_panel(n_days=260, tickers=("AAA", "BBB", "CCC", "SPY"), seed=3)
    rng = np.random.default_rng(5)
    dates = sorted(panel["date"].unique())[200:250]  # leave vol warmup
    preds = pd.concat(
        [
            pd.DataFrame({"date": d, "ticker": ["AAA", "BBB", "CCC"], "score": rng.normal(size=3)})
            for d in dates
        ],
        ignore_index=True,
    )
    cfg = Config()
    cfg.portfolio.top_k = 2
    cfg.portfolio.max_weight = 0.6
    asset_types = {"AAA": "stock", "BBB": "stock", "CCC": "etf", "SPY": "etf"}
    return preds, panel, asset_types, cfg


def test_costs_always_on_and_net_below_gross():
    preds, panel, asset_types, cfg = _bt_inputs()
    res = run_backtest(preds, panel[["date", "ticker", "close"]], asset_types, cfg)
    assert (res.costs >= 0).all()
    assert res.costs.sum() > 0
    assert res.net_returns.sum() < res.gross_returns.sum()
    # cost never exceeds turnover * max bps
    max_bps = cfg.costs.stock_bps_per_side / 1e4
    assert (res.costs <= res.turnover * max_bps + 1e-12).all()


def test_backtest_golden_file():
    preds, panel, asset_types, cfg = _bt_inputs()
    res = run_backtest(preds, panel[["date", "ticker", "close"]], asset_types, cfg)
    bench = panel[panel["ticker"] == "SPY"].set_index("date")["close"].pct_change()
    summary = summarize(res.net_returns, bench, res.turnover)
    rounded = {
        k: (round(v, 10) if isinstance(v, float) else v)
        for k, v in summary.items()
        if k != "hit_rate_ci95"
    }
    rounded["hit_rate_ci95"] = [round(x, 10) for x in summary["hit_rate_ci95"]]

    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(rounded, indent=2))
        pytest.skip("golden file created — rerun to compare")

    golden = json.loads(GOLDEN.read_text())
    assert rounded == golden, "backtest output drifted from golden file"


def test_sharpe_and_drawdown_known_values():
    r = pd.Series([0.01, -0.01] * 50)
    assert abs(sharpe(r)) < 0.2
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert np.isclose(max_drawdown(eq), 0.9 / 1.2 - 1)


def test_hit_rate_ci_known_values():
    r = pd.Series([0.01] * 60 + [-0.01] * 40)
    out = hit_rate_ci(r)
    assert np.isclose(out["hit_rate"], 0.6)
    assert out["ci_low"] < 0.6 < out["ci_high"]
    assert out["n"] == 100


def test_deflated_sharpe_deflates_with_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    single = deflated_sharpe(r, n_trials=1)
    many = deflated_sharpe(r, n_trials=100)
    assert many < single  # more trials -> more deflation


def test_summary_reports_spy_underperformance_honestly():
    preds, panel, asset_types, cfg = _bt_inputs()
    res = run_backtest(preds, panel[["date", "ticker", "close"]], asset_types, cfg)
    strong_bench = pd.Series(0.002, index=res.net_returns.index)
    summary = summarize(res.net_returns, strong_bench, res.turnover)
    assert summary["underperforms_spy"] == (
        summary["ann_return_net"] < summary["benchmark_ann_return"]
    )
