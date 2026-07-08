"""Wave-2 experiment: weighting schemes x risk gates on FIXED wave-1 predictions.

    uv run python -m backtest.experiment_wave2

The model (lambdarank + all features) is frozen; only portfolio construction and
gating vary, so the cached OOS predictions are reused across all variants.

Honest reporting: n_trials = 13 (waves 0-1) + 4 (new variants here) = 17.
Results -> artifacts/experiment_wave2.json.
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
from models.regime import regime_series

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 17
PREDS_CACHE = paths.ARTIFACTS / "oos_preds_rank_all.parquet"

# name -> (portfolio overrides, gates overrides)
VARIANTS: dict[str, tuple[dict, dict]] = {
    "incumbent_ivol_vix": ({}, {}),
    "hrp_vix": ({"weighting": "hrp"}, {}),
    "rmt_vix": ({"weighting": "rmt_minvar"}, {}),
    "ivol_jump": ({}, {"vix_term_structure": False, "jump_model": True}),
    "hrp_jump": ({"weighting": "hrp"}, {"vix_term_structure": False, "jump_model": True}),
}


def _cached_predictions(cfg: Config) -> pd.DataFrame:
    if PREDS_CACHE.exists():
        preds = pd.read_parquet(PREDS_CACHE)
        preds["date"] = pd.to_datetime(preds["date"])
        return preds
    from models.train import load_dataset, oos_predictions

    data, cols = load_dataset(cfg)
    preds = oos_predictions(data, cols, cfg)
    preds.to_parquet(PREDS_CACHE, index=False)
    return preds


def run_wave2(base_cfg: Config) -> dict:
    preds = _cached_predictions(base_cfg)
    prices = lake.read_curated_prices()
    members = load_universe(base_cfg.universe_file)
    asset_types = asset_type_map(members)
    macro = lake.read_curated_macro()
    macro_vix = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        macro_vix = wide[["VIX", "VIX3M"]].reset_index()
    bench_close = prices[prices["ticker"] == base_cfg.benchmark].set_index("date")["close"]
    bench = bench_close.pct_change().rename("bench")

    # Regime series computed once (PIT by construction), shared by jump variants.
    regime = regime_series(bench.dropna(), penalty=base_cfg.gates.jump_penalty)
    stress_share = float((regime == 1).mean())
    logger.info(
        "jump model: %d dated states, %.1f%% stress days", len(regime), stress_share * 100
    )

    results: dict[str, dict] = {}
    for name, (port_over, gate_over) in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        for k, v in port_over.items():
            setattr(cfg.portfolio, k, v)
        for k, v in gate_over.items():
            setattr(cfg.gates, k, v)
        bt = run_backtest(preds, prices, asset_types, cfg, macro_vix=macro_vix, regime=regime)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)
        summary["overrides"] = {"portfolio": port_over, "gates": gate_over}
        results[name] = summary
        logger.info(
            "%-20s ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  maxDD %6.1f%%  TO %4.1f%%/d",
            name,
            summary["ann_return_net"] * 100,
            summary["sharpe_net"],
            summary["deflated_sharpe"],
            summary["max_drawdown"] * 100,
            summary["avg_daily_turnover"] * 100,
        )

    out = paths.ARTIFACTS / "experiment_wave2.json"
    out.write_text(
        json.dumps(
            {"n_trials": N_TRIALS_TOTAL, "stress_share": stress_share, "results": results},
            indent=2,
        )
    )
    logger.info("wrote %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_wave2(load_config(paths.CONFIGS / "baseline.yaml"))
