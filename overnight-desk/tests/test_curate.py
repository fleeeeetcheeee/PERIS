"""Curation validation: split-artifact detection vs real market moves.

The >40% jump check exists to catch adjustment artifacts. It must NOT drop real
crashes/earnings gaps — dropping the blowups injects survivorship bias into the
panel (found in wave 5: the old rule excluded AAL, OXY, PCG, SBNY, SMCI...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingestion.curate import ValidationError, _looks_like_split, validate_ticker_frame


def _frame(closes: np.ndarray, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    n = len(closes)
    if dates is None:
        dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
            "ticker": "TST",
        }
    )


def test_split_ratio_shapes():
    assert _looks_like_split(-0.5)  # 2:1 unadjusted split
    assert _looks_like_split(1.0)  # 1:2 reverse split
    assert _looks_like_split(-2 / 3)  # 3:1
    assert not _looks_like_split(-0.47)  # a real crash
    assert not _looks_like_split(0.45)  # a real earnings gap


def test_unadjusted_split_is_rejected():
    closes = np.full(300, 100.0)
    closes[150:] = 50.0  # exact 2:1 split artifact
    with pytest.raises(ValidationError, match="split-ratio-shaped"):
        validate_ticker_frame("TST", _frame(closes))


def test_real_crash_is_kept():
    closes = np.full(300, 100.0)
    closes[150:] = 53.0  # -47% single-day crash (COVID-style), not a ratio
    out = validate_ticker_frame("TST", _frame(closes))
    assert len(out) == 300


def test_garbage_series_is_rejected():
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.choice([-0.45, 0.85], size=300))
    with pytest.raises(ValidationError):
        validate_ticker_frame("TST", _frame(closes))


def test_listing_gap_return_is_exempt():
    # 300 sessions, with a 2-year hole: the cross-gap "return" of -80% is not a
    # daily move and must not fail validation
    d1 = pd.bdate_range("2018-01-02", periods=150)
    d2 = pd.bdate_range("2021-01-04", periods=150)
    closes = np.concatenate([np.full(150, 100.0), np.full(150, 20.0)])
    out = validate_ticker_frame("TST", _frame(closes, d1.append(d2)))
    assert len(out) == 300
