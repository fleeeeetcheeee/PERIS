"""Meta-labeling: a secondary walk-forward classifier over the primary ranker's picks.

Lopez de Prado's meta-labeling separates two concerns the primary model conflates:
the ranker decides WHAT to hold (the top-K names); the meta model decides HOW MUCH
conviction each pick deserves, by predicting P(pick pays off over the label horizon)
from the same point-in-time features plus the pick's cross-sectional score position.

Point-in-time discipline:
- The meta model trains ONLY on the primary model's out-of-sample predictions
  (never in-sample scores), on its own purged expanding walk-forward.
- Raw booster scores are NOT comparable across primary folds (lambdarank scale is
  arbitrary per fold), so the only score-derived feature is the within-date pct rank.
- Dates before the meta walk-forward has enough history get no probability; the
  portfolio layer treats a missing probability as neutral (multiplier 1, no veto).
- Calibration maps are refit every CAL_REFRESH sessions on a trailing window that
  ends GAP sessions in the past, so every outcome used is fully realized.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import Config, CVConfig
from models.walkforward import walk_forward_folds

logger = logging.getLogger(__name__)

# Candidate pool per date: the names the book could plausibly hold. Score smoothing
# pulls entrants from just below the top-K, so the pool is 2x the portfolio size.
POOL_MULTIPLE = 2

META_MIN_TRAIN_DAYS = 252  # binary head on ~24 rows/day needs far less than the ranker
CAL_WINDOW = 252  # trailing sessions feeding a calibration map
CAL_GAP = 7  # label horizon (5) + embargo (2): outcomes inside the gap are unrealized
CAL_REFRESH = 21  # refit the calibration map monthly, frozen in between
CAL_BINS = 5
CAL_MIN_ROWS = 500  # below this the map is refused (probabilities stay uncalibrated NaN)

_META_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
_META_ROUNDS = 200


def build_meta_dataset(
    preds: pd.DataFrame, data: pd.DataFrame, cols: list[str], top_k: int
) -> tuple[pd.DataFrame, list[str]]:
    """Join primary OOS scores onto the feature matrix and keep the candidate pool.

    Returns (meta_df, meta_cols). meta_df carries date, ticker, meta_label
    (1 if fwd_ret > 0), the primary score pct rank, and the feature columns.
    """
    scored = data.merge(preds, on=["date", "ticker"], how="inner", validate="1:1")
    scored["score_pct"] = scored.groupby("date")["score"].rank(pct=True)
    pool = (
        scored.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date")
        .head(POOL_MULTIPLE * top_k)
        .reset_index(drop=True)
    )
    pool["meta_label"] = (pool["fwd_ret"] > 0).astype(int)
    meta_cols = [*cols, "score_pct"]
    return pool[["date", "ticker", "meta_label", *meta_cols]], meta_cols


def meta_probabilities(
    preds: pd.DataFrame,
    data: pd.DataFrame,
    cols: list[str],
    cfg: Config,
    min_train_days: int = META_MIN_TRAIN_DAYS,
) -> pd.DataFrame:
    """Walk-forward OOS meta probabilities for the candidate pool.

    Returns date, ticker, meta_label, meta_prob, meta_prob_cal, meta_base_rate.
    Only dates covered by a meta test fold appear — earlier dates have no row.
    """
    meta_df, meta_cols = build_meta_dataset(preds, data, cols, cfg.portfolio.top_k)
    meta_cv = CVConfig(
        label_horizon_days=cfg.cv.label_horizon_days,
        purge_days=cfg.cv.purge_days,
        embargo_days=cfg.cv.embargo_days,
        min_train_days=min_train_days,
        test_window_days=cfg.cv.test_window_days,
    )
    params = dict(_META_PARAMS, seed=cfg.model.seed)

    frames = []
    dates = pd.DatetimeIndex(meta_df["date"].unique())
    for fold in walk_forward_folds(dates, meta_cv):
        train = meta_df[meta_df["date"].isin(fold.train_dates)]
        test = meta_df[meta_df["date"].isin(fold.test_dates)]
        if train.empty or test.empty:
            continue
        dtrain = lgb.Dataset(train[meta_cols], label=train["meta_label"], free_raw_data=True)
        booster = lgb.train(params, dtrain, num_boost_round=_META_ROUNDS)
        out = test[["date", "ticker", "meta_label"]].copy()
        out["meta_prob"] = booster.predict(test[meta_cols])
        frames.append(out)
    if not frames:
        raise RuntimeError("meta walk-forward produced no folds — not enough history")
    probs = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    probs = calibrate_trailing(probs)
    logger.info(
        "meta probabilities: %d rows over %d dates, base rate %.3f, prob IQR [%.3f, %.3f]",
        len(probs),
        probs["date"].nunique(),
        probs["meta_label"].mean(),
        probs["meta_prob"].quantile(0.25),
        probs["meta_prob"].quantile(0.75),
    )
    return probs.reset_index(drop=True)


def calibrate_trailing(
    probs: pd.DataFrame,
    window: int = CAL_WINDOW,
    gap: int = CAL_GAP,
    bins: int = CAL_BINS,
    refresh: int = CAL_REFRESH,
    min_rows: int = CAL_MIN_ROWS,
) -> pd.DataFrame:
    """Attach meta_prob_cal and meta_base_rate via trailing binned calibration.

    Every `refresh` sessions, quantile-bin the trailing `window` sessions of
    (prob, realized label) pairs ending `gap` sessions in the past and record each
    bin's empirical payoff rate; between refits the map is frozen. Dates with no
    admissible map (warm-up) get NaN — downstream sizing treats NaN as neutral.
    """
    probs = probs.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(probs["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    pos = probs["date"].map(date_pos).to_numpy()
    p = probs["meta_prob"].to_numpy()
    y = probs["meta_label"].to_numpy(dtype=float)

    p_cal = np.full(len(probs), np.nan)
    base = np.full(len(probs), np.nan)
    edges: np.ndarray | None = None
    bin_rate: np.ndarray | None = None
    win_rate = np.nan
    for i in range(len(dates)):
        if i % refresh == 0:
            hi = i - gap  # exclusive: sessions [hi-window, hi) have realized outcomes
            lo = max(0, hi - window)
            mask = (pos >= lo) & (pos < hi)
            if mask.sum() >= min_rows:
                cp, cy = p[mask], y[mask]
                qs = np.quantile(cp, np.linspace(0, 1, bins + 1))
                edges = np.unique(qs[1:-1])  # interior edges; unique guards degenerate probs
                bin_idx = np.searchsorted(edges, cp)
                bin_rate = np.array(
                    [cy[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                     for b in range(len(edges) + 1)]
                )
                win_rate = float(cy.mean())
            else:
                edges, bin_rate, win_rate = None, None, np.nan
        if edges is not None and bin_rate is not None:
            day = pos == i
            p_cal[day] = bin_rate[np.searchsorted(edges, p[day])]
            base[day] = win_rate

    out = probs.copy()
    out["meta_prob_cal"] = p_cal
    out["meta_base_rate"] = base
    return out


def size_multiplier(row_probs: pd.Series, base_rate: float) -> pd.Series:
    """Calibrated-probability bet size, normalized to the unconditional edge.

    A pick at the trailing base payoff rate gets full size (1.0); conviction decays
    linearly to zero at the coin-flip point p = 0.5. NaN (uncalibrated) stays NaN —
    the caller substitutes neutral. Never sizes above 1 (long-only cash account).
    """
    denom = max(base_rate - 0.5, 1e-3) if np.isfinite(base_rate) else np.nan
    return ((row_probs - 0.5) / denom).clip(lower=0.0, upper=1.0)
