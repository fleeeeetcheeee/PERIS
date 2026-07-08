"""Train the LightGBM ranker on the purged walk-forward harness.

Usage: uv run python -m models.train configs/model.yaml [--force-promote]

A new model is promoted only if its walk-forward OOS net Sharpe — through the
config's portfolio construction and cost model — beats the incumbent's (mean rank IC
is only a fallback for artifacts predating the portfolio metric; see the Research
entry on why IC alone is the wrong gate). Deterministic: seeds fixed from config.
"""

from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import paths
from core.config import Config, load_config
from features.build import FEATURES_CURATED, feature_columns
from features.build import run as build_features
from models import registry
from models.walkforward import rank_ic, walk_forward_folds

logger = logging.getLogger(__name__)


def load_dataset(
    cfg: Config, matrix: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """matrix: optional prebuilt feature matrix (experiments with non-promoted
    families); default is the curated promoted-family matrix."""
    if matrix is None:
        if FEATURES_CURATED.exists():
            matrix = pd.read_parquet(FEATURES_CURATED)
        else:
            from core.universe import load_universe

            matrix = build_features(
                write=True, tickers=[m.ticker for m in load_universe(cfg.universe_file)]
            )
    matrix["date"] = pd.to_datetime(matrix["date"])
    matrix = matrix[matrix["date"] >= pd.Timestamp(cfg.start)]
    if cfg.end:
        matrix = matrix[matrix["date"] <= pd.Timestamp(cfg.end)]

    # Point-in-time membership: drop (date, ticker) rows where the stock wasn't an
    # index member — no backtesting today's list into the past. Labels are still
    # ranked over the full curated panel; with the current near-gapless demo universe
    # the difference is nil, but flag: a real historical universe should re-rank
    # labels within members only.
    from core.universe import has_pit_constituents, pit_membership_mask

    if has_pit_constituents():
        before = len(matrix)
        matrix = matrix[pit_membership_mask(matrix)].reset_index(drop=True)
        if before - len(matrix):
            logger.info("PIT membership filter dropped %d rows", before - len(matrix))

    cols = feature_columns(matrix)
    labeled = matrix.dropna(subset=["label"]).reset_index(drop=True)
    return labeled, cols


def _lgb_params(cfg: Config) -> dict:
    params = {
        "objective": cfg.model.objective,
        "seed": cfg.model.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    params.update(cfg.model.params)
    return params


N_RANK_GRADES = 5  # lambdarank relevance grades per date (gains 0,1,3,7,15)


def _fit(train: pd.DataFrame, cols: list[str], cfg: Config) -> lgb.Booster:
    if cfg.model.objective == "lambdarank":
        # Listwise ranking: each session is one query group; labels become integer
        # relevance grades from the within-date ordering (our stored label is a
        # centered pct-rank, so its within-date order equals the fwd-return order).
        train = train.sort_values("date", kind="stable")
        grades = (
            train.groupby("date")["label"]
            .rank(pct=True)
            .mul(N_RANK_GRADES)
            .clip(upper=N_RANK_GRADES - 1e-9)
            .astype(int)
        )
        groups = train.groupby("date", sort=True).size().to_numpy()
        params = _lgb_params(cfg)
        params.setdefault("ndcg_eval_at", [12])  # matches portfolio top_k
        dtrain = lgb.Dataset(train[cols], label=grades, group=groups, free_raw_data=True)
        return lgb.train(params, dtrain, num_boost_round=cfg.model.num_boost_round)

    dtrain = lgb.Dataset(train[cols], label=train["label"], free_raw_data=True)
    return lgb.train(_lgb_params(cfg), dtrain, num_boost_round=cfg.model.num_boost_round)


def run_harness(data: pd.DataFrame, cols: list[str], cfg: Config) -> tuple[dict, pd.DataFrame]:
    """Walk-forward evaluation in a single training pass.

    Returns (metrics, oos_predictions). Metrics carry both the model-level rank ICs
    and — because rank IC is provably blind to the channels that move portfolio P&L
    (see Research/2026-07-06) — the portfolio-level walk-forward net Sharpe, which is
    what promotion decisions use.
    """
    fold_ics: list[float] = []
    all_daily_ics: list[pd.Series] = []
    pred_frames: list[pd.DataFrame] = []
    folds = list(walk_forward_folds(pd.DatetimeIndex(data["date"].unique()), cfg.cv))
    for i, fold in enumerate(folds):
        train = data[data["date"].isin(fold.train_dates)]
        test = data[data["date"].isin(fold.test_dates)]
        if train.empty or test.empty:
            continue
        booster = _fit(train, cols, cfg)
        pred = pd.Series(booster.predict(test[cols]), index=test.index)
        daily = rank_ic(pred, test["label"], test["date"])
        fold_ics.append(float(daily.mean()))
        all_daily_ics.append(daily)
        pred_frames.append(
            pd.DataFrame(
                {
                    "date": test["date"].values,
                    "ticker": test["ticker"].values,
                    "score": pred.values,
                }
            )
        )
        logger.info(
            "fold %d/%d: train %d rows to %s, test %s..%s, mean IC %.4f",
            i + 1,
            len(folds),
            len(train),
            fold.train_dates[-1].date(),
            fold.test_dates[0].date(),
            fold.test_dates[-1].date(),
            fold_ics[-1],
        )
    daily_all = pd.concat(all_daily_ics)
    mean_ic = float(daily_all.mean())
    ic_ir = float(mean_ic / daily_all.std() * np.sqrt(252)) if daily_all.std() > 0 else 0.0
    preds = pd.concat(pred_frames, ignore_index=True)
    metrics = {
        "n_folds": len(fold_ics),
        "fold_ics": fold_ics,
        "mean_ic": mean_ic,
        "ic_std": float(daily_all.std()),
        "ic_ir_annualized": ic_ir,
        "n_oos_days": int(len(daily_all)),
    }
    metrics.update(_portfolio_metrics(preds, cfg))
    return metrics, preds


def _portfolio_metrics(preds: pd.DataFrame, cfg: Config) -> dict:
    """Net Sharpe of the OOS predictions run through the config's portfolio + costs."""
    from backtest.engine import run_backtest
    from backtest.metrics import sharpe
    from core import lake
    from core.universe import asset_type_map, load_universe
    from features.macro_regime import _pivot_lagged

    try:
        prices = lake.read_curated_prices()
        members = load_universe(cfg.universe_file)
        macro = lake.read_curated_macro()
        macro_vix = None
        if macro is not None:
            wide = _pivot_lagged(macro)
            macro_vix = wide[["VIX", "VIX3M"]].reset_index()
        bt = run_backtest(preds, prices, asset_type_map(members), cfg, macro_vix=macro_vix)
        return {
            "portfolio_sharpe_net": sharpe(bt.net_returns),
            "portfolio_ann_return_net": float(bt.net_returns.mean() * 252),
            "portfolio_avg_turnover": float(bt.turnover.mean()),
        }
    except Exception as exc:  # metrics are advisory for training; fail soft but loud
        logger.warning("portfolio metrics unavailable (%s)", exc)
        return {}


def oos_predictions(data: pd.DataFrame, cols: list[str], cfg: Config) -> pd.DataFrame:
    """Out-of-sample scores from the walk-forward harness — the only predictions the
    backtester is allowed to consume (no in-sample equity curves)."""
    frames = []
    for fold in walk_forward_folds(pd.DatetimeIndex(data["date"].unique()), cfg.cv):
        train_df = data[data["date"].isin(fold.train_dates)]
        test = data[data["date"].isin(fold.test_dates)]
        if train_df.empty or test.empty:
            continue
        booster = _fit(train_df, cols, cfg)
        frames.append(
            pd.DataFrame(
                {
                    "date": test["date"].values,
                    "ticker": test["ticker"].values,
                    "score": booster.predict(test[cols]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def train(config_path: str, force_promote: bool = False) -> None:
    cfg = load_config(config_path)
    data, cols = load_dataset(cfg)
    logger.info(
        "dataset: %d rows, %d features, %s..%s",
        len(data),
        len(cols),
        data["date"].min().date(),
        data["date"].max().date(),
    )

    harness, _preds = run_harness(data, cols, cfg)
    logger.info(
        "harness: mean IC %.4f over %d OOS days (IR %.2f), portfolio net Sharpe %s",
        harness["mean_ic"],
        harness["n_oos_days"],
        harness["ic_ir_annualized"],
        f"{harness['portfolio_sharpe_net']:.3f}" if "portfolio_sharpe_net" in harness else "n/a",
    )

    final = _fit(data, cols, cfg)
    meta = {
        "data_hash": registry.data_hash(data, cols),
        "feature_cols": cols,
        "seed": cfg.model.seed,
        "cv": cfg.cv.model_dump(),
        "model": cfg.model.model_dump(),
        "train_start": str(data["date"].min().date()),
        "train_end": str(data["date"].max().date()),
        "n_rows": len(data),
        "harness": harness,
    }
    adir = registry.save_artifact(final, meta)
    logger.info("saved artifact %s", adir.name)

    # Promotion gate: walk-forward net Sharpe through the portfolio + costs.
    # Rank IC is NOT the gate — it is blind to magnitude and persistence channels
    # that move net P&L (Research/2026-07-06). IC is kept as a fallback only for
    # comparing against artifacts that predate the portfolio metric.
    challenger = harness.get("portfolio_sharpe_net")
    incumbent = registry.incumbent_metric("portfolio_sharpe_net")
    metric_name = "portfolio net Sharpe"
    if challenger is None or incumbent is None:
        challenger = harness["mean_ic"]
        incumbent = registry.incumbent_metric("mean_ic")
        metric_name = "mean IC (fallback)"

    if force_promote or incumbent is None or challenger > incumbent:
        registry.promote(adir)
        logger.info("PROMOTED on %s: %s -> %.4f", metric_name, incumbent, challenger)
    else:
        logger.info("NOT promoted: %s %.4f <= incumbent %.4f", metric_name, challenger, incumbent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(paths.CONFIGS / "model.yaml"))
    parser.add_argument("--force-promote", action="store_true")
    args = parser.parse_args()
    train(args.config, force_promote=args.force_promote)
