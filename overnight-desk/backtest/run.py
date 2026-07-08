"""Backtest CLI: uv run python -m backtest.run configs/baseline.yaml

Backtests the strategy on walk-forward OUT-OF-SAMPLE predictions only, with the cost
model on. Prints the honest summary (deflated Sharpe, hit-rate CI, vs. SPY net) and
writes artifacts/backtest_<config>.json.
"""

from __future__ import annotations

import argparse
import json
import logging

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import load_config
from core.universe import asset_type_map, has_pit_constituents, load_universe
from features.macro_regime import _pivot_lagged
from models.train import load_dataset, oos_predictions

logger = logging.getLogger(__name__)


def main(config_path: str) -> dict:
    cfg = load_config(config_path)
    if not has_pit_constituents():
        logger.warning(
            "SURVIVORSHIP BIAS: no data/reference/constituents_pit.csv — backtest uses "
            "today's universe snapshot into the past. Treat results as optimistic."
        )

    data, cols = load_dataset(cfg)
    logger.info("collecting walk-forward OOS predictions (%d rows)...", len(data))
    preds = oos_predictions(data, cols, cfg)
    logger.info("OOS predictions: %d rows over %d days", len(preds), preds["date"].nunique())

    prices = lake.read_curated_prices()
    members = load_universe(cfg.universe_file)
    macro = lake.read_curated_macro()
    macro_vix = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        macro_vix = wide[["VIX", "VIX3M"]].reset_index()

    meta = None
    if cfg.portfolio.meta_mode != "off":
        from models.meta import meta_probabilities

        meta = meta_probabilities(preds, data, cols, cfg)

    result = run_backtest(
        preds, prices, asset_type_map(members), cfg, macro_vix=macro_vix, meta=meta
    )

    bench = (
        prices[prices["ticker"] == cfg.benchmark]
        .set_index("date")["close"]
        .pct_change()
        .rename("bench")
    )
    summary = summarize(result.net_returns, bench, result.turnover, n_trials=cfg.selection_trials)
    summary["selection_trials"] = cfg.selection_trials
    summary["config"] = str(config_path)
    summary["oos_start"] = str(result.net_returns.index.min().date())
    summary["oos_end"] = str(result.net_returns.index.max().date())
    summary["survivorship_bias_warning"] = not has_pit_constituents()

    out = paths.ARTIFACTS / "backtest_baseline.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out)

    print(json.dumps(summary, indent=2))
    if summary["underperforms_spy"]:
        print("\nNOTE: strategy UNDERPERFORMS SPY buy-and-hold after costs on this window.")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(paths.CONFIGS / "baseline.yaml"))
    args = parser.parse_args()
    main(args.config)
