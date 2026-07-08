"""Portfolio construction and gate known-value tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import GatesConfig, PortfolioConfig
from portfolio import construct
from portfolio.gates import GateDecision, drawdown_gate, vix_term_structure_gate


def test_inverse_vol_weights_proportions():
    vol = pd.Series({"A": 0.10, "B": 0.20, "C": 0.40})
    w = construct.inverse_vol_weights(vol, cap=1.0)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["A"] / w["B"], 2.0)
    assert np.isclose(w["B"] / w["C"], 2.0)


def test_position_cap_redistributes():
    vol = pd.Series({"A": 0.01, "B": 0.30, "C": 0.30})
    w = construct.inverse_vol_weights(vol, cap=0.10)
    # all names cap out at 10%; the remainder stays in cash, never re-levered
    assert (w <= 0.10 + 1e-9).all()
    assert np.isclose(w.sum(), 0.30)

    vol2 = pd.Series({"A": 0.05, "B": 0.20, "C": 0.20, "D": 0.20})
    w2 = construct.inverse_vol_weights(vol2, cap=0.40)
    assert (w2 <= 0.40 + 1e-9).all()
    assert np.isclose(w2.sum(), 1.0)  # excess redistributed to uncapped names


def test_trade_band_skips_dust():
    prev = pd.Series({"A": 0.100, "B": 0.05})
    target = pd.Series({"A": 0.105, "B": 0.20})
    out = construct.apply_trade_band(target, prev, min_trade_weight=0.01)
    assert np.isclose(out["A"], 0.100)  # 0.5% move skipped
    assert np.isclose(out["B"], 0.20)


def test_top_k_selection_respects_eligibility():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    eligible = pd.Series({"A": True, "B": False, "C": True})
    assert construct.select_top_k(scores, eligible, 2) == ["A", "C"]


def test_vix_gate_backwardation_halves_exposure():
    d = vix_term_structure_gate(GateDecision(), vix=30.0, vix3m=24.0, cfg=GatesConfig())
    assert d.exposure_scale == 0.5
    assert not d.allow_new_entries

    d2 = vix_term_structure_gate(GateDecision(), vix=15.0, vix3m=18.0, cfg=GatesConfig())
    assert d2.exposure_scale == 1.0 and d2.allow_new_entries


def test_drawdown_gate_halts_at_10pct():
    d = drawdown_gate(GateDecision(), equity=89.9, high_water_mark=100.0, cfg=GatesConfig())
    assert not d.allow_new_entries
    d2 = drawdown_gate(GateDecision(), equity=95.0, high_water_mark=100.0, cfg=GatesConfig())
    assert d2.allow_new_entries


def test_vol_target_never_levers_up():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    quiet = pd.DataFrame({"A": [0.0001] * 63, "B": [-0.0001] * 63})
    scale = construct.vol_target_scale(weights, quiet, PortfolioConfig())
    assert scale == 1.0  # would need leverage; capped at 1 (long-only cash account)
