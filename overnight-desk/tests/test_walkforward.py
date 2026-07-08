"""Purged walk-forward splits: expanding, gapped, never overlapping."""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import CVConfig
from models.walkforward import walk_forward_folds


def test_folds_are_purged_and_expanding():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=800))
    cv = CVConfig(
        label_horizon_days=5, purge_days=5, embargo_days=2, min_train_days=504, test_window_days=63
    )
    folds = list(walk_forward_folds(dates, cv))
    assert len(folds) >= 2

    prev_train_len = 0
    for fold in folds:
        assert len(fold.train_dates) > prev_train_len  # expanding window
        prev_train_len = len(fold.train_dates)
        # gap: at least purge+embargo sessions between last train and first test date
        gap = dates.get_loc(fold.test_dates[0]) - dates.get_loc(fold.train_dates[-1])
        assert gap >= cv.purge_days + cv.embargo_days
        assert len(set(fold.train_dates) & set(fold.test_dates)) == 0

    # test windows tile forward without overlap
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_dates[-1] < b.test_dates[0]


def test_purge_must_cover_horizon():
    with pytest.raises(ValueError):
        CVConfig(label_horizon_days=5, purge_days=3)


def test_insufficient_history_raises():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=100))
    with pytest.raises(ValueError, match="not enough history"):
        list(walk_forward_folds(dates, CVConfig()))
