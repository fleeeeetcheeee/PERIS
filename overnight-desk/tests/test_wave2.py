"""Wave-2 tests: HRP/RMT weighting known values, jump-model regime detection and
point-in-time safety, promotion-gate metric selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import Config, GatesConfig
from models.regime import fit_jump_model, market_features, regime_series
from portfolio.gates import GateDecision, jump_regime_gate
from portfolio.weighting import hrp_weights, mp_clip_corr, rmt_minvar_weights

# ------------------------------------------------------------------ weighting


def _two_block_returns(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Two correlated pairs (A,B) and (C,D); D is much higher vol."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "A": f1 + rng.normal(0, 0.003, n),
            "B": f1 + rng.normal(0, 0.003, n),
            "C": f2 + rng.normal(0, 0.003, n),
            "D": 3 * f2 + rng.normal(0, 0.009, n),
        }
    )


def test_hrp_weights_sum_to_one_and_favor_low_vol():
    rets = _two_block_returns()
    w = hrp_weights(rets)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()
    assert w["A"] > w["D"]  # low-vol names get more weight


def test_hrp_splits_across_clusters():
    rets = _two_block_returns()
    w = hrp_weights(rets)
    # neither correlated block should take (nearly) all the weight
    assert 0.2 < w[["A", "B"]].sum() < 0.8


def test_rmt_minvar_long_only_and_normalized():
    rets = _two_block_returns()
    w = rmt_minvar_weights(rets)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    assert w["A"] > w["D"]


def test_mp_clip_preserves_unit_diagonal_and_signal():
    rets = _two_block_returns(n=80)
    corr = rets.corr()
    cleaned = mp_clip_corr(corr, n_obs=80)
    assert np.allclose(np.diag(cleaned), 1.0)
    top_raw = np.linalg.eigvalsh(corr)[-1]
    top_clean = np.linalg.eigvalsh(cleaned)[-1]
    assert abs(top_raw - top_clean) / top_raw < 0.35  # signal eigenvalue roughly kept


def test_weighting_config_default_is_legacy():
    assert Config().portfolio.weighting == "inverse_vol"
    assert Config().gates.jump_model is False  # goldens depend on these defaults


# ------------------------------------------------------------------ jump model


def _regime_switch_returns(n_calm: int = 1200, n_stress: int = 150, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0005, 0.006, n_calm)
    stress = rng.normal(-0.002, 0.025, n_stress)
    dates = pd.bdate_range("2020-01-01", periods=n_calm + n_stress)
    return pd.Series(np.concatenate([calm, stress]), index=dates)


def test_jump_model_detects_stress_block():
    rets = _regime_switch_returns()
    feats = market_features(rets)
    _, states, _, _ = fit_jump_model(feats, penalty=50.0)
    # last 100 obs are deep in the synthetic stress period
    assert states[-100:].mean() > 0.8
    assert states[:800].mean() < 0.2


def test_jump_penalty_reduces_switching():
    rets = _regime_switch_returns()
    feats = market_features(rets)
    _, low, _, _ = fit_jump_model(feats, penalty=1.0)
    _, high, _, _ = fit_jump_model(feats, penalty=100.0)
    assert (np.diff(high) != 0).sum() <= (np.diff(low) != 0).sum()


def test_regime_series_is_point_in_time():
    """Corrupting the future must not change past states."""
    rets = _regime_switch_returns()
    base = regime_series(rets, penalty=50.0)
    cutoff = rets.index[1250]
    corrupted = rets.copy()
    corrupted.loc[corrupted.index > cutoff] *= 10
    after = regime_series(corrupted, penalty=50.0)
    common = base.index[base.index <= cutoff].intersection(after.index)
    assert (base.loc[common] == after.loc[common]).all()


def test_jump_gate_semantics():
    d = jump_regime_gate(GateDecision(), state=1, cfg=GatesConfig(jump_model=True))
    assert d.exposure_scale == 0.5 and not d.allow_new_entries
    d2 = jump_regime_gate(GateDecision(), state=0, cfg=GatesConfig(jump_model=True))
    assert d2.exposure_scale == 1.0 and d2.allow_new_entries
    d3 = jump_regime_gate(GateDecision(), state=1, cfg=GatesConfig(jump_model=False))
    assert d3.exposure_scale == 1.0  # flag off -> no-op


# ------------------------------------------------------------------ promotion gate


def test_promotion_prefers_portfolio_metric(monkeypatch):
    from models import registry

    meta = {"harness": {"mean_ic": 0.02, "portfolio_sharpe_net": 0.9}}
    monkeypatch.setattr(registry, "load_current", lambda: (object(), meta))
    assert registry.incumbent_metric("portfolio_sharpe_net") == 0.9
    assert registry.incumbent_metric("mean_ic") == 0.02
    # artifacts predating the portfolio metric return None for it
    old_meta = {"harness": {"mean_ic": 0.013}}
    monkeypatch.setattr(registry, "load_current", lambda: (object(), old_meta))
    assert registry.incumbent_metric("portfolio_sharpe_net") is None
