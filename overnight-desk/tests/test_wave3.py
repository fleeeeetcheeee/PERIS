"""Wave-3 tests: meta-labeling dataset construction, trailing calibration known
values, mechanical point-in-time safety of the meta walk-forward, and the
size-multiplier semantics in portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import _meta_sizing
from core.config import Config, PortfolioConfig
from models.meta import (
    build_meta_dataset,
    calibrate_trailing,
    meta_probabilities,
    size_multiplier,
)
from portfolio import construct

# ------------------------------------------------------------- synthetic panel


def _synthetic(n_dates: int = 130, n_tickers: int = 10, seed: int = 11):
    """(preds, data, cols) shaped like the real pipeline's outputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = idx.to_frame(index=False)
    data["f1"] = rng.normal(size=len(data))
    data["f2"] = rng.normal(size=len(data))
    # payoff mildly predictable from f1 so the meta model has something to learn
    data["fwd_ret"] = 0.01 * data["f1"] + rng.normal(0, 0.02, len(data))
    data["label"] = data.groupby("date")["fwd_ret"].rank(pct=True) - 0.5
    preds = data[["date", "ticker"]].copy()
    preds["score"] = data["f1"] + rng.normal(0, 0.5, len(data))
    return preds, data, ["f1", "f2"]


def _meta_cfg() -> Config:
    cfg = Config()
    cfg.portfolio.top_k = 3  # pool = 6 of 10 tickers
    return cfg


# --------------------------------------------------------------- meta dataset


def test_meta_dataset_pool_and_labels():
    preds, data, cols = _synthetic()
    meta_df, meta_cols = build_meta_dataset(preds, data, cols, top_k=3)
    per_date = meta_df.groupby("date").size()
    assert (per_date == 6).all()  # POOL_MULTIPLE * top_k
    assert meta_cols == ["f1", "f2", "score_pct"]
    # pool really is the top of the score distribution
    assert (meta_df["score_pct"] > 0.35).all()
    merged = meta_df.merge(data, on=["date", "ticker"])
    assert (merged["meta_label"] == (merged["fwd_ret"] > 0).astype(int)).all()


# ----------------------------------------------------------------- calibration


def _prob_frame(n_dates: int, rows_per_day: int = 10, seed: int = 5) -> pd.DataFrame:
    """Probs uniform in [0,1]; outcome payoff rate = the prob itself (perfectly
    calibrated ground truth), so trailing bin rates must recover ~the bin mean."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    rows = []
    for d in dates:
        p = rng.uniform(0, 1, rows_per_day)
        y = (rng.uniform(0, 1, rows_per_day) < p).astype(int)
        for ticker, (pi, yi) in enumerate(zip(p, y, strict=True)):
            rows.append({"date": d, "ticker": f"T{ticker}", "meta_prob": pi, "meta_label": yi})
    return pd.DataFrame(rows)


def test_calibration_recovers_true_rates_and_warms_up_nan():
    probs = _prob_frame(n_dates=200)
    out = calibrate_trailing(probs, window=120, gap=7, bins=5, refresh=21, min_rows=300)
    first = out[out["date"] == out["date"].min()]
    assert first["meta_prob_cal"].isna().all()  # warm-up: no admissible history
    late = out[out["date"] >= out["date"].unique()[150]].dropna(subset=["meta_prob_cal"])
    assert len(late) > 100
    # ground truth is perfectly calibrated: bin rate should track the prob
    err = (late["meta_prob_cal"] - late["meta_prob"]).abs()
    assert err.median() < 0.15
    assert late["meta_base_rate"].between(0.4, 0.6).all()  # uniform probs -> ~0.5


def test_calibration_map_ignores_unrealized_outcomes():
    """Outcomes inside the gap window must not leak into the calibration map."""
    probs = _prob_frame(n_dates=100)
    dates = pd.DatetimeIndex(probs["date"].unique())
    poisoned = probs.copy()
    # flip every outcome in the last `gap` sessions before a refresh date
    poisoned.loc[poisoned["date"].isin(dates[-7:]), "meta_label"] ^= 1
    a = calibrate_trailing(probs, window=60, gap=7, refresh=21, min_rows=300)
    b = calibrate_trailing(poisoned, window=60, gap=7, refresh=21, min_rows=300)
    # calibrated values on all dates are computed from maps that end >= gap back,
    # so flipping the final gap's outcomes changes nothing anywhere
    pd.testing.assert_series_equal(a["meta_prob_cal"], b["meta_prob_cal"])


def test_meta_probabilities_point_in_time():
    """Mechanical lookahead test: corrupt the future, past probabilities identical."""
    preds, data, cols = _synthetic(n_dates=130)
    cfg = _meta_cfg()
    a = meta_probabilities(preds, data, cols, cfg, min_train_days=60)

    cutoff = pd.DatetimeIndex(data["date"].unique())[-40]
    corrupted = data.copy()
    future = corrupted["date"] >= cutoff
    corrupted.loc[future, "fwd_ret"] = 9.9  # every future pick "pays off" absurdly
    corrupted.loc[future, "f1"] = -5.0
    b = meta_probabilities(preds, corrupted, cols, cfg, min_train_days=60)

    past_a = a[a["date"] < cutoff].reset_index(drop=True)
    past_b = b[b["date"] < cutoff].reset_index(drop=True)
    # training folds are purged+embargoed and calibration maps end gap sessions
    # back, so nothing before the cutoff may move — bit-identical, not "close"
    pd.testing.assert_frame_equal(
        past_a[["date", "ticker", "meta_prob", "meta_prob_cal"]],
        past_b[["date", "ticker", "meta_prob", "meta_prob_cal"]],
    )
    assert len(past_a) > 0


# ------------------------------------------------------------------ bet sizing


def test_size_multiplier_known_values():
    p = pd.Series([0.60, 0.55, 0.50, 0.45, np.nan])
    m = size_multiplier(p, base_rate=0.60)
    assert np.allclose(m[:4], [1.0, 0.5, 0.0, 0.0])
    assert np.isnan(m.iloc[4])  # uncalibrated stays NaN -> neutral downstream
    assert np.isnan(size_multiplier(pd.Series([0.7]), base_rate=np.nan).iloc[0])


def _construct_inputs():
    tickers = ["A", "B", "C", "D"]
    scores = pd.Series([4.0, 3.0, 2.0, 1.0], index=tickers)
    vol = pd.Series(0.2, index=tickers)
    eligible = pd.Series(True, index=tickers)
    # near-zero trailing vol so the vol-target scale stays pinned at 1
    trailing = pd.DataFrame(
        1e-6 * np.ones((70, 4)), columns=tickers, index=pd.bdate_range("2024-01-01", periods=70)
    )
    cfg = PortfolioConfig(top_k=3, max_weight=0.5, min_trade_weight=0.0)
    return scores, vol, eligible, trailing, cfg


def test_build_targets_tilt_renormalizes():
    scores, vol, eligible, trailing, cfg = _construct_inputs()
    mult = pd.Series({"A": 0.9, "B": 0.3, "C": 0.6})
    w = construct.build_targets(
        scores, vol, eligible, trailing, pd.Series(dtype=float), cfg,
        size_mult=mult, size_renormalize=True,
    )
    assert np.isclose(w.sum(), 1.0)  # tilt is exposure-neutral
    assert w["A"] > w["C"] > w["B"]  # ordering follows the tilt (equal base weights)


def test_build_targets_gate_leaves_freed_weight_in_cash():
    scores, vol, eligible, trailing, cfg = _construct_inputs()
    base = construct.build_targets(
        scores, vol, eligible, trailing, pd.Series(dtype=float), cfg
    )
    gated = construct.build_targets(
        scores, vol, eligible, trailing, pd.Series(dtype=float), cfg,
        size_mult=pd.Series({"A": 1.0, "B": 0.0, "C": 1.0}), size_renormalize=False,
    )
    assert "B" not in gated.index  # vetoed, and no replacement pick
    assert np.isclose(gated.sum(), base.sum() - base["B"])  # freed weight -> cash
    assert np.isclose(gated["A"], base["A"])  # survivors untouched


def test_build_targets_missing_prob_is_neutral():
    scores, vol, eligible, trailing, cfg = _construct_inputs()
    base = construct.build_targets(scores, vol, eligible, trailing, pd.Series(dtype=float), cfg)
    partial = construct.build_targets(
        scores, vol, eligible, trailing, pd.Series(dtype=float), cfg,
        size_mult=pd.Series({"A": 1.0}), size_renormalize=False,  # B, C missing
    )
    pd.testing.assert_series_equal(base.sort_index(), partial.sort_index())


# ------------------------------------------------------------- engine plumbing


def test_meta_sizing_modes_and_defaults():
    port = PortfolioConfig(meta_mode="off")
    assert _meta_sizing(pd.DataFrame({"meta_prob": [0.7]}), port) == (None, False)
    assert _meta_sizing(None, PortfolioConfig(meta_mode="gate")) == (None, False)  # warm-up

    day = pd.DataFrame(
        {"meta_prob": [0.7, 0.4], "meta_prob_cal": [0.62, 0.51], "meta_base_rate": [0.58, 0.58]},
        index=["A", "B"],
    )
    mult, renorm = _meta_sizing(day, PortfolioConfig(meta_mode="tilt"))
    assert renorm and np.allclose(mult, [0.7, 0.4])
    mult, renorm = _meta_sizing(day, PortfolioConfig(meta_mode="gate", meta_gate_threshold=0.5))
    assert not renorm and np.allclose(mult, [1.0, 0.0])
    mult, renorm = _meta_sizing(day, PortfolioConfig(meta_mode="sized"))
    assert not renorm and mult["A"] == 1.0 and 0 < mult["B"] < 0.2

    with pytest.raises(ValueError):
        _meta_sizing(day, PortfolioConfig(meta_mode="bogus"))


def test_meta_config_default_is_off():
    assert Config().portfolio.meta_mode == "off"  # goldens depend on this default
