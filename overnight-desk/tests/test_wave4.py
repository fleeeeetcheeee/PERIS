"""Wave-4 tests: path-signature Lévy areas (known geometric values) and turbulence
indices (Mahalanobis calibration, MST persistence separation), plus dedicated
lookahead checks on a panel wide enough to actually exercise the turbulence math
(the shared 3-ticker fixture is below its 5-name floor)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import signature, turbulence
from features.signature import levy_area
from features.turbulence import h0_persistence_series, mahalanobis_series
from tests.conftest import make_panel

# ------------------------------------------------------------------ Lévy areas


def test_levy_area_l_shaped_path():
    # right-then-up unit L: signed area vs the chord is exactly 1/2
    dx = np.array([1.0, 0.0])
    dy = np.array([0.0, 1.0])
    assert np.isclose(levy_area(dx, dy, window=2)[-1], 0.5)
    # up-then-right traverses the other side of the chord: -1/2
    assert np.isclose(levy_area(dy, dx, window=2)[-1], -0.5)


def test_levy_area_straight_line_is_zero():
    rng = np.random.default_rng(1)
    dx = rng.uniform(0.1, 1.0, 50)
    assert np.allclose(levy_area(dx, 3.0 * dx, window=20)[19:], 0.0, atol=1e-12)


def test_levy_area_nan_poisons_only_touching_windows():
    dx = np.ones(30)
    dy = np.ones(30)
    dy[10] = np.nan
    out = levy_area(dx, dy, window=5)
    assert np.isnan(out[10:15]).all()  # windows containing increment 10
    assert np.isfinite(out[15:]).all()
    assert np.isfinite(out[4:10]).all()


def test_sig_tp_area_sign_is_momentum_timing():
    # gains arriving late (accelerating) => positive area; early gains => negative
    w = signature.SIG_WINDOW
    late = np.concatenate([np.zeros(w - 10), np.full(10, 0.01)])
    early = late[::-1].copy()
    dt = np.full(w, 1.0 / w)
    assert levy_area(dt, late, window=w)[-1] > 0
    assert levy_area(dt, early, window=w)[-1] < 0


def test_signature_compute_shapes_and_columns():
    panel = make_panel(n_days=400)
    out = signature.compute(panel)
    assert list(out.columns) == ["date", "ticker", "sig_tp_area", "sig_pv_levy"]
    assert len(out) == len(panel)
    assert out["sig_tp_area"].notna().sum() > 0  # z-warm-up passed within 400 days


# ------------------------------------------------------------------ turbulence


def _wide_rets(n_days: int = 500, n_assets: int = 10, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_assets)),
        index=dates,
        columns=[f"T{i}" for i in range(n_assets)],
    )


def test_mahalanobis_calibration_and_outlier():
    rets = _wide_rets()
    rets.iloc[400] = 0.08  # one day where everything gaps 8%
    turb = mahalanobis_series(rets)
    typical = turb.iloc[150:400].median()
    # normalized d^2/n has expectation ~1 for iid normal data
    assert 0.5 < typical < 2.0
    assert turb.iloc[400] > 10 * typical  # the gap day screams


def test_h0_persistence_separates_dispersion():
    calm = _wide_rets(seed=5)
    wild = calm * 5.0  # same geometry, stretched cloud -> longer spanning tree
    assert h0_persistence_series(wild).iloc[-1] > 3 * h0_persistence_series(calm).iloc[-1]


def test_turbulence_broadcasts_per_date():
    panel = make_panel(n_days=560, tickers=tuple(f"T{i}" for i in range(8)))
    out = turbulence.compute(panel)
    assert list(out.columns) == ["date", "ticker", "turb_mahal", "turb_h0"]
    per_date = out.dropna(subset=["turb_mahal"]).groupby("date")["turb_mahal"].nunique()
    assert (per_date == 1).all()  # market-level: identical across tickers
    assert out["turb_mahal"].notna().sum() > 0


# ------------------------------------------------- lookahead on a wide panel


def _corrupt_future(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = panel.copy()
    future = out["date"] > cutoff
    rng = np.random.default_rng(99)
    for col in ("close", "volume"):
        out.loc[future, col] = out.loc[future, col].values * rng.uniform(0.5, 2.0, future.sum())
    return out


def test_wave4_families_no_lookahead_wide_panel():
    panel = make_panel(n_days=600, tickers=tuple(f"T{i}" for i in range(8)))
    cutoff = pd.DatetimeIndex(panel["date"].unique())[520]
    for family in (signature, turbulence):
        base = family.compute(panel)
        corrupted = family.compute(_corrupt_future(panel, cutoff))
        past = base["date"] <= cutoff
        cols = [c for c in base.columns if c not in ("date", "ticker")]
        assert base.loc[past, cols].notna().any().any()  # the test actually bites
        pd.testing.assert_frame_equal(
            base.loc[past, cols].reset_index(drop=True),
            corrupted.loc[past, cols].reset_index(drop=True),
            check_exact=True,
            obj=f"{family.__name__} lookahead",
        )
