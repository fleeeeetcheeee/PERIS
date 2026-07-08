"""Wave-4 model experiment: path-signature and turbulence features, fixed objective.

    uv run python -m models.experiment_wave4

Waves 2-3 showed construction and sizing layers add nothing on top of the current
signal, so this wave attacks the signal: two NEW information channels on top of the
incumbent (lambdarank, wave-1 feature set) —

- sig_*  path-signature Lévy areas (momentum timing, price/volume lead-lag)
- turb_* market turbulence (Kritzman-Li Mahalanobis, MST H0 persistence)

Variants share ONE feature matrix; each retrains the full purged walk-forward, and
OOS predictions go through the promoted portfolio config and the cost model.

Honest reporting: n_trials = 21 (waves 0-3) + 3 (new variants here) = 24.
Results -> artifacts/experiment_wave4.json (+ per-variant daily nets parquet).
"""

from __future__ import annotations

import copy
import json
import logging

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import Config, load_config
from core.universe import asset_type_map, load_universe
from features.macro_regime import _pivot_lagged
from models.train import load_dataset, oos_predictions
from models.walkforward import rank_ic

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 24  # 21 (waves 0-3) + 3 (this wave)

INCUMBENT_PREFIXES = ("rev_", "resmom_", "mom_", "trend_", "liq_", "mac_", "cal_", "ffd_", "spill_")
SIG_PREFIX = ("sig_",)
TURB_PREFIX = ("turb_",)

# name -> feature prefixes (objective fixed at the promoted lambdarank)
VARIANTS: dict[str, tuple[str, ...]] = {
    "incumbent_rank_all": INCUMBENT_PREFIXES,
    "rank_sig": INCUMBENT_PREFIXES + SIG_PREFIX,
    "rank_turb": INCUMBENT_PREFIXES + TURB_PREFIX,
    "rank_sig_turb": INCUMBENT_PREFIXES + SIG_PREFIX + TURB_PREFIX,
}


def _select_cols(all_cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in all_cols if c.startswith(prefixes)]


def run_wave4(base_cfg: Config) -> dict:
    # sig_/turb_ are NOT in the promoted FAMILIES (this wave rejected them), so the
    # experiment builds its own extended matrix rather than reading the curated one.
    from features.build import EXPERIMENTAL_FAMILIES, FAMILIES, build_matrix

    matrix = build_matrix(
        lake.read_curated_prices(),
        lake.read_curated_macro(),
        with_labels=True,
        families=[*FAMILIES, *EXPERIMENTAL_FAMILIES],
    )
    data, all_cols = load_dataset(base_cfg, matrix=matrix)
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
    daily_net: dict[str, pd.Series] = {}
    for name, prefixes in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        cols = _select_cols(all_cols, prefixes)
        logger.info("=== %s: %d features ===", name, len(cols))

        preds = oos_predictions(data, cols, cfg)
        joined = preds.merge(data[["date", "ticker", "label"]], on=["date", "ticker"])
        daily_ic = rank_ic(joined["score"], joined["label"], joined["date"])

        bt = run_backtest(preds, prices, asset_types, cfg, macro_vix=macro_vix)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)
        daily_net[name] = bt.net_returns
        preds.to_parquet(paths.ARTIFACTS / f"oos_preds_wave4_{name}.parquet", index=False)

        results[name] = {
            "n_features": len(cols),
            "mean_ic": float(daily_ic.mean()),
            "n_oos_days": int(len(daily_ic)),
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
            "%-20s IC %+0.4f  ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  maxDD %6.1f%%  TO %4.1f%%",
            name,
            r["mean_ic"],
            r["ann_return_net"] * 100,
            r["sharpe_net"],
            r["deflated_sharpe"],
            r["max_drawdown"] * 100,
            r["avg_daily_turnover"] * 100,
        )

    out = paths.ARTIFACTS / "experiment_wave4.json"
    out.write_text(json.dumps({"n_trials": N_TRIALS_TOTAL, "results": results}, indent=2))
    pd.DataFrame(daily_net).to_parquet(paths.ARTIFACTS / "experiment_wave4_daily.parquet")
    logger.info("wrote %s", out)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(42)
    run_wave4(load_config(paths.CONFIGS / "baseline.yaml"))
