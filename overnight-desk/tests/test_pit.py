"""Point-in-time membership mask: window logic and rename handling."""

from __future__ import annotations

import pandas as pd

from core import universe


def test_pit_mask_respects_windows(tmp_path, monkeypatch):
    pit = tmp_path / "constituents_pit.csv"
    pit.write_text("ticker,start,end\nAAA,2020-01-01,2021-12-31\nBBB,2019-01-01,\n")
    monkeypatch.setattr(universe, "PIT_FILE", pit)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-06-01", "2020-06-01", "2022-06-01"] * 2 + ["2020-06-01"]),
            "ticker": ["AAA"] * 3 + ["BBB"] * 3 + ["SPY"],
        }
    )
    mask = universe.pit_membership_mask(panel)
    aaa = mask[panel["ticker"] == "AAA"].tolist()
    assert aaa == [False, True, False]  # before window, inside, after
    assert mask[panel["ticker"] == "BBB"].all()  # open-ended membership
    assert mask[panel["ticker"] == "SPY"].all()  # not in file -> always member (ETF)


def test_real_pit_file_has_no_gaps_for_universe():
    """After the rename patches (FB->META, UTX->RTX), every universe stock must be a
    member for the whole backtest window — else the demo universe silently shrinks."""
    if not universe.has_pit_constituents():
        import pytest

        pytest.skip("no constituents_pit.csv")
    pit = pd.read_csv(universe.PIT_FILE, parse_dates=["start", "end"])
    members = universe.load_universe("data/reference/universe.csv")
    bt_start, bt_end = pd.Timestamp("2018-01-02"), pd.Timestamp("2026-07-02")
    for m in members:
        if m.asset_type != "stock":
            continue
        w = pit[pit["ticker"] == m.ticker].copy()
        assert not w.empty, f"{m.ticker} missing from PIT file"
        w["end"] = w["end"].fillna(pd.Timestamp("2099-01-01"))
        covered = any((r.start <= bt_start) and (r.end >= bt_end) for r in w.itertuples())
        assert covered, f"{m.ticker} has a membership gap inside the backtest window"
