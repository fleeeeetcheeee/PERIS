from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(
    n_days: int = 320, tickers: tuple[str, ...] = ("AAA", "BBB", "SPY"), seed: int = 7
) -> pd.DataFrame:
    """Deterministic synthetic long panel on real NYSE-ish weekdays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0004, 0.015, n_days)
        close = 100 * (1 + i) * np.cumprod(1 + rets)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "open": close * (1 - 0.002),
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1e6, 5e6, n_days).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_macro(n_days: int = 340, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-12-01", periods=n_days)
    frames = []
    for series, base in [
        ("VIX", 18.0),
        ("VIX3M", 20.0),
        ("T10Y2Y", 0.5),
        ("DGS10", 3.0),
        ("FEDFUNDS", 4.0),
        ("HY_OAS", 3.5),
    ]:
        vals = base + np.cumsum(rng.normal(0, 0.1, n_days))
        frames.append(pd.DataFrame({"date": dates, "series": series, "value": vals}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture
def macro() -> pd.DataFrame:
    return make_macro()
