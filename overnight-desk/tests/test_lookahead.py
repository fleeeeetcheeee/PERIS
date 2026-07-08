"""Lookahead tests — the repo's hard constraint #2, enforced mechanically.

For every feature family: corrupt all rows AFTER a cutoff date, recompute, and
require feature values at dates <= cutoff to be bit-identical. If a feature ever
peeks forward, this fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    calendar_features,
    fracdiff,
    macro_regime,
    residual_momentum,
    reversal,
    signature,
    spillover,
    trend,
    turbulence,
    volume_liquidity,
)

FAMILIES = [
    reversal,
    residual_momentum,
    trend,
    volume_liquidity,
    macro_regime,
    calendar_features,
    fracdiff,
    spillover,
    signature,
    turbulence,
]
CUTOFF = pd.Timestamp("2022-12-15")


def _corrupt_future(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = panel.copy()
    future = out["date"] > cutoff
    rng = np.random.default_rng(99)
    for col in ("open", "high", "low", "close", "volume"):
        out.loc[future, col] = out.loc[future, col].values * rng.uniform(0.5, 2.0, future.sum())
    return out


@pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.__name__)
def test_no_lookahead(family, panel, macro):
    base = family.compute(panel, macro)
    corrupted = family.compute(_corrupt_future(panel, CUTOFF), macro)

    past = base["date"] <= CUTOFF
    cols = [c for c in base.columns if c not in ("date", "ticker")]
    pd.testing.assert_frame_equal(
        base.loc[past, cols].reset_index(drop=True),
        corrupted.loc[past, cols].reset_index(drop=True),
        check_exact=True,
        obj=f"{family.__name__} lookahead",
    )


def test_macro_regime_lagged_one_day(panel, macro):
    """Corrupting macro values ON date t must not change macro features at t
    (they must come from t-1 or earlier)."""
    base = macro_regime.compute(panel, macro)
    t = pd.Timestamp("2022-12-15")
    corrupted_macro = macro.copy()
    corrupted_macro.loc[corrupted_macro["date"] >= t, "value"] *= 5.0
    after = macro_regime.compute(panel, corrupted_macro)
    at_t = base["date"] == t
    cols = [c for c in base.columns if c.startswith("mac_")]
    pd.testing.assert_frame_equal(
        base.loc[at_t, cols].reset_index(drop=True),
        after.loc[at_t, cols].reset_index(drop=True),
        check_exact=True,
    )


def test_label_is_forward_looking_only_by_design(panel):
    """Sanity check the inverse: labels MUST change when the future changes."""
    from features import labels

    base = labels.compute_labels(panel)
    corrupted = labels.compute_labels(_corrupt_future(panel, CUTOFF))
    near_cutoff = (base["date"] <= CUTOFF) & (base["date"] > CUTOFF - pd.Timedelta(days=7))
    assert not base.loc[near_cutoff, "fwd_ret"].equals(corrupted.loc[near_cutoff, "fwd_ret"])
