"""Purged expanding-window walk-forward splits. The ONLY CV allowed in this repo.

For a test block starting at session T, training uses sessions up to
T - (purge_days + embargo_days). purge >= label horizon (5) guarantees no training
label window overlaps the test block; the embargo adds 2 extra sessions of slack.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import CVConfig


@dataclass(frozen=True)
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walk_forward_folds(dates: pd.DatetimeIndex, cv: CVConfig) -> Iterator[Fold]:
    """dates: sorted unique trading sessions present in the dataset."""
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    gap = cv.purge_days + cv.embargo_days
    start = cv.min_train_days + gap
    n = len(dates)
    if start >= n:
        raise ValueError(
            f"not enough history: {n} sessions, need > {start} "
            f"(min_train_days={cv.min_train_days} + gap={gap})"
        )
    for test_start in range(start, n, cv.test_window_days):
        test_end = min(test_start + cv.test_window_days, n)
        train_end = test_start - gap
        yield Fold(
            train_dates=dates[:train_end],
            test_dates=dates[test_start:test_end],
        )


def rank_ic(pred: pd.Series, label: pd.Series, dates: pd.Series) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation between predictions and labels."""
    df = pd.DataFrame({"pred": pred, "label": label, "date": dates}).dropna()

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return np.nan
        return g["pred"].rank().corr(g["label"].rank())

    return df.groupby("date").apply(_ic, include_groups=False).dropna()
