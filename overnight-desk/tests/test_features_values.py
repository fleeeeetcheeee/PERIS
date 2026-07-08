"""Known-value fixture tests for feature families and labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import calendar_features, labels, reversal


def tiny_panel(closes: list[float], ticker: str = "AAA", start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        }
    )


def test_reversal_known_values():
    panel = tiny_panel([100, 110, 99, 99, 99, 99])
    out = reversal.compute(panel)
    assert np.isclose(out["rev_ret_1d"].iloc[1], 0.10)
    assert np.isclose(out["rev_ret_1d"].iloc[2], 99 / 110 - 1)
    assert np.isclose(out["rev_ret_5d"].iloc[5], 99 / 100 - 1)
    assert pd.isna(out["rev_ret_1d"].iloc[0])


def test_reversal_independent_across_tickers():
    a = tiny_panel([100, 110, 121], "AAA")
    b = tiny_panel([50, 55, 66], "BBB")
    out = reversal.compute(pd.concat([a, b], ignore_index=True))
    bbb = out[out["ticker"] == "BBB"]
    assert pd.isna(bbb["rev_ret_1d"].iloc[0])  # no bleed from AAA's last close
    assert np.isclose(bbb["rev_ret_1d"].iloc[1], 0.10)


def test_calendar_known_values():
    panel = tiny_panel([100, 100, 100], start="2024-01-30")  # Tue Jan 30 .. Thu Feb 1
    out = calendar_features.compute(panel)
    assert list(out["cal_dow"]) == [1, 2, 3]
    assert list(out["cal_month"]) == [1, 1, 2]
    assert list(out["cal_turn_of_month"]) == [1, 1, 1]


def test_labels_rank_normalized_per_date():
    frames = []
    for ticker, growth in [("AAA", 1.02), ("BBB", 1.00), ("CCC", 0.98)]:
        closes = [100 * growth**i for i in range(8)]
        frames.append(tiny_panel(closes, ticker))
    out = labels.compute_labels(pd.concat(frames, ignore_index=True))
    first = out[out["date"] == out["date"].min()].set_index("ticker")
    # AAA has the best forward 5d return -> highest rank
    assert first.loc["AAA", "label"] > first.loc["BBB", "label"] > first.loc["CCC", "label"]
    assert np.isclose(first["label"].mean(), 0.0, atol=1e-9)
    # last `horizon` rows per ticker have no label
    last_date = out["date"].max()
    assert out.loc[out["date"] == last_date, "fwd_ret"].isna().all()
