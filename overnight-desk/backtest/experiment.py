"""Turnover experiment: portfolio-construction variants on FIXED walk-forward OOS
predictions. Model and features never change — only cadence/band/smoothing — so
cached predictions are reused and every variant sees identical signals.

    uv run python -m backtest.experiment

Honest-reporting: deflated Sharpe is computed with n_trials = number of variants
tried (multiple-testing penalty). Results go to artifacts/experiment_turnover.json.
"""

from __future__ import annotations

import copy
import json
import logging

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import Config, load_config
from core.universe import asset_type_map, load_universe
from features.macro_regime import _pivot_lagged

logger = logging.getLogger(__name__)

PREDS_CACHE = paths.ARTIFACTS / "oos_preds.parquet"

# name -> portfolio-config overrides
VARIANTS: dict[str, dict] = {
    "incumbent_daily": {},
    "cadence3": {"rebalance_every": 3},
    "cadence5": {"rebalance_every": 5},
    "cadence5_band2": {"rebalance_every": 5, "min_trade_weight": 0.02},
    "band3_daily": {"min_trade_weight": 0.03},
    "smooth3_daily": {"score_smoothing_halflife": 3},
    "smooth3_cadence5": {"score_smoothing_halflife": 3, "rebalance_every": 5},
}


def run_experiment(base_cfg: Config) -> dict:
    preds = pd.read_parquet(PREDS_CACHE)
    preds["date"] = pd.to_datetime(preds["date"])
    prices = lake.read_curated_prices()
    members = load_universe(base_cfg.universe_file)
    asset_types = asset_type_map(members)

    macro = lake.read_curated_macro()
    macro_vix = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        macro_vix = wide[["VIX", "VIX3M"]].reset_index()

    bench = (
        prices[prices["ticker"] == base_cfg.benchmark]
        .set_index("date")["close"]
        .pct_change()
        .rename("bench")
    )

    n_trials = len(VARIANTS)
    results: dict[str, dict] = {}
    for name, overrides in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            setattr(cfg.portfolio, k, v)
        res = run_backtest(preds, prices, asset_types, cfg, macro_vix=macro_vix)
        summary = summarize(res.net_returns, bench, res.turnover, n_trials=n_trials)
        summary["overrides"] = overrides
        summary["total_costs_paid"] = float(res.costs.sum())
        results[name] = summary
        logger.info(
            "%-18s ann_net %+7.2f%%  sharpe %+5.2f  dSR %.3f  turnover %5.1f%%/d  maxDD %6.1f%%",
            name,
            summary["ann_return_net"] * 100,
            summary["sharpe_net"],
            summary["deflated_sharpe"],
            summary["avg_daily_turnover"] * 100,
            summary["max_drawdown"] * 100,
        )

    out = paths.ARTIFACTS / "experiment_turnover.json"
    out.write_text(json.dumps({"n_trials": n_trials, "results": results}, indent=2))
    logger.info("wrote %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_experiment(load_config(paths.CONFIGS / "baseline.yaml"))
