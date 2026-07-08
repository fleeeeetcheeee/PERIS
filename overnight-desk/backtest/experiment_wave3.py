"""Wave-3 experiment: meta-labeling conviction sizing on FIXED wave-1 predictions.

    uv run python -m backtest.experiment_wave3

The primary model (lambdarank + all features) is frozen; a secondary walk-forward
classifier (models/meta.py) estimates P(pick pays off) and the variants differ only
in how that probability touches the book: relative tilt, veto gate, or absolute
calibrated bet sizing. Incumbent construction/gates everywhere else.

Honest reporting: n_trials = 17 (waves 0-2) + 4 (new variants here) = 21.
Results -> artifacts/experiment_wave3.json. Meta probs cached for reuse.
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
from models.meta import meta_probabilities

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 21
PREDS_CACHE = paths.ARTIFACTS / "oos_preds_rank_all.parquet"
META_CACHE = paths.ARTIFACTS / "meta_probs.parquet"

# name -> portfolio overrides
VARIANTS: dict[str, dict] = {
    "incumbent": {},
    "meta_tilt": {"meta_mode": "tilt"},
    "meta_gate": {"meta_mode": "gate"},
    "meta_gate60": {"meta_mode": "gate", "meta_gate_threshold": 0.60},
    "meta_sized": {"meta_mode": "sized"},
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


def _cached_meta(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if META_CACHE.exists():
        meta = pd.read_parquet(META_CACHE)
        meta["date"] = pd.to_datetime(meta["date"])
        return meta
    from models.train import load_dataset

    data, cols = load_dataset(cfg)
    meta = meta_probabilities(preds, data, cols, cfg)
    meta.to_parquet(META_CACHE, index=False)
    return meta


def run_wave3(base_cfg: Config) -> dict:
    preds = _cached_predictions(base_cfg)
    meta = _cached_meta(preds, base_cfg)

    # Meta-model skill diagnostics (OOS, before any portfolio effect).
    base_rate = float(meta["meta_label"].mean())
    by_bucket = meta.groupby(pd.qcut(meta["meta_prob"], 5, labels=False, duplicates="drop"))[
        "meta_label"
    ].mean()
    logger.info(
        "meta skill: base rate %.3f; payoff rate by prob quintile %s",
        base_rate,
        [round(v, 3) for v in by_bucket],
    )

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
    daily_net: dict[str, pd.Series] = {}
    for name, over in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        for k, v in over.items():
            setattr(cfg.portfolio, k, v)
        bt = run_backtest(preds, prices, asset_types, cfg, macro_vix=macro_vix, meta=meta)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)
        summary["overrides"] = over
        results[name] = summary
        daily_net[name] = bt.net_returns
        logger.info(
            "%-14s ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  maxDD %6.1f%%  TO %4.1f%%/d",
            name,
            summary["ann_return_net"] * 100,
            summary["sharpe_net"],
            summary["deflated_sharpe"],
            summary["max_drawdown"] * 100,
            summary["avg_daily_turnover"] * 100,
        )

    out = paths.ARTIFACTS / "experiment_wave3.json"
    out.write_text(
        json.dumps(
            {
                "n_trials": N_TRIALS_TOTAL,
                "meta_base_rate": base_rate,
                "meta_payoff_by_prob_quintile": [float(v) for v in by_bucket],
                "meta_dates_covered": int(meta["date"].nunique()),
                "results": results,
            },
            indent=2,
        )
    )
    pd.DataFrame(daily_net).to_parquet(paths.ARTIFACTS / "experiment_wave3_daily.parquet")
    logger.info("wrote %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_wave3(load_config(paths.CONFIGS / "baseline.yaml"))
