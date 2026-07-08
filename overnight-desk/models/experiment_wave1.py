"""Wave-1 model experiment: feature families x objective, on the fixed walk-forward.

    uv run python -m models.experiment_wave1

Variants share ONE feature matrix (built with all families); each selects a column
subset and an objective, so data plumbing is identical across variants. Every variant
runs the full purged walk-forward to produce OOS predictions, which then go through
the PROMOTED portfolio config (smooth3 + cadence5) and the cost model.

Honest reporting: deflated Sharpe uses n_trials = 7 (turnover wave) + 6 (this wave).
Results -> artifacts/experiment_wave1.json.
"""

from __future__ import annotations

import copy
import json
import logging

import numpy as np

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import Config, load_config
from core.universe import asset_type_map, load_universe
from features.macro_regime import _pivot_lagged
from models.train import load_dataset, oos_predictions
from models.walkforward import rank_ic

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 13  # 7 (turnover wave) + 6 (this wave)

BASE_PREFIXES = ("rev_", "resmom_", "mom_", "trend_", "liq_", "mac_", "cal_")
FFD_PREFIX = ("ffd_",)
SPILL_PREFIX = ("spill_",)

# name -> (feature prefixes, objective)
VARIANTS: dict[str, tuple[tuple[str, ...], str]] = {
    "incumbent_reg_base": (BASE_PREFIXES, "regression"),
    "reg_ffd": (BASE_PREFIXES + FFD_PREFIX, "regression"),
    "reg_spill": (BASE_PREFIXES + SPILL_PREFIX, "regression"),
    "reg_all": (BASE_PREFIXES + FFD_PREFIX + SPILL_PREFIX, "regression"),
    "rank_base": (BASE_PREFIXES, "lambdarank"),
    "rank_all": (BASE_PREFIXES + FFD_PREFIX + SPILL_PREFIX, "lambdarank"),
}


def _select_cols(all_cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in all_cols if c.startswith(prefixes)]


def run_wave1(base_cfg: Config) -> dict:
    data, all_cols = load_dataset(base_cfg)
    logger.info("dataset: %d rows, %d total feature cols", len(data), len(all_cols))

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

    results: dict[str, dict] = {}
    for name, (prefixes, objective) in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.model.objective = objective
        cols = _select_cols(all_cols, prefixes)
        logger.info("=== %s: %d features, objective=%s ===", name, len(cols), objective)

        preds = oos_predictions(data, cols, cfg)
        # Daily OOS rank IC straight from the predictions (one training pass per fold)
        joined = preds.merge(data[["date", "ticker", "label"]], on=["date", "ticker"])
        daily_ic = rank_ic(joined["score"], joined["label"], joined["date"])
        ic_ir = (
            float(daily_ic.mean() / daily_ic.std() * np.sqrt(252)) if daily_ic.std() > 0 else 0.0
        )

        bt = run_backtest(preds, prices, asset_types, cfg, macro_vix=macro_vix)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)

        results[name] = {
            "objective": objective,
            "n_features": len(cols),
            "mean_ic": float(daily_ic.mean()),
            "ic_ir_annualized": ic_ir,
            "n_oos_days": int(len(daily_ic)),
            "daily_ic": {str(k.date()): round(float(v), 6) for k, v in daily_ic.items()},
            **{
                k: summary[k]
                for k in (
                    "ann_return_net",
                    "ann_vol",
                    "sharpe_net",
                    "deflated_sharpe",
                    "hit_rate",
                    "max_drawdown",
                    "avg_daily_turnover",
                    "excess_ann_return_vs_spy",
                    "underperforms_spy",
                )
            },
        }
        r = results[name]
        logger.info(
            "%-18s IC %+0.4f (IR %4.2f)  ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  TO %4.1f%%/d",
            name,
            r["mean_ic"],
            r["ic_ir_annualized"],
            r["ann_return_net"] * 100,
            r["sharpe_net"],
            r["deflated_sharpe"],
            r["avg_daily_turnover"] * 100,
        )

    # Paired daily-IC comparison vs incumbent for the model-metric readout
    out = paths.ARTIFACTS / "experiment_wave1.json"
    out.write_text(json.dumps({"n_trials": N_TRIALS_TOTAL, "results": results}, indent=2))
    logger.info("wrote %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(42)
    run_wave1(load_config(paths.CONFIGS / "baseline.yaml"))
