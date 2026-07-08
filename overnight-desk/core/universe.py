"""Universe loading with point-in-time membership support.

The demo universe file is a *current* snapshot. For backtests, if
data/reference/constituents_pit.csv exists (ticker,start,end), membership is applied
point-in-time. Without it, backtests over the snapshot carry survivorship bias — the
backtester logs a warning so results are never mistaken for clean ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core import paths

PIT_FILE = paths.REFERENCE / "constituents_pit.csv"


@dataclass(frozen=True)
class UniverseMember:
    ticker: str
    name: str
    asset_type: str  # "stock" | "etf"
    sector: str


def load_universe(universe_file: str | Path) -> list[UniverseMember]:
    path = Path(universe_file)
    if not path.is_absolute():
        path = paths.ROOT / path
    df = pd.read_csv(path)
    required = {"ticker", "name", "asset_type", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe file missing columns: {missing}")
    return [
        UniverseMember(
            ticker=row.ticker.upper(),
            name=row.name_,
            asset_type=row.asset_type,
            sector=row.sector,
        )
        for row in df.rename(columns={"name": "name_"}).itertuples(index=False)
    ]


def asset_type_map(members: list[UniverseMember]) -> dict[str, str]:
    return {m.ticker: m.asset_type for m in members}


def has_pit_constituents() -> bool:
    return PIT_FILE.exists()


def pit_membership_mask(panel: pd.DataFrame) -> pd.Series:
    """Boolean mask over a (date, ticker) panel: was the ticker a member on that date?

    ETFs and any ticker absent from the PIT file are treated as always-members.
    """
    if not PIT_FILE.exists():
        return pd.Series(True, index=panel.index)
    pit = pd.read_csv(PIT_FILE, parse_dates=["start", "end"])
    mask = pd.Series(True, index=panel.index)
    by_ticker = {t: idx for t, idx in panel.groupby("ticker").indices.items()}
    for ticker, grp in pit.groupby("ticker"):
        idx = by_ticker.get(ticker)
        if idx is None:
            continue
        rows = panel.index[idx]
        dates = panel.loc[rows, "date"]
        member = pd.Series(False, index=rows)
        for _, r in grp.iterrows():
            end = r["end"] if pd.notna(r["end"]) else pd.Timestamp.max
            member |= (dates >= r["start"]) & (dates <= end)
        mask.loc[rows] = member
    return mask
