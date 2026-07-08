"""Build the expanded point-in-time S&P 500 universe from the fja05680/sp500 dataset.

    uv run python -m ingestion.build_sp500_universe

Inputs:  data/reference/sp500_ticker_start_end.csv  (ticker, start_date, end_date)
         data/reference/universe.csv                (kept for its ETF rows)
Outputs: data/reference/universe_sp500.csv          (every ticker with S&P membership
                                                     overlapping the backtest window,
                                                     plus the ETFs)
         data/reference/constituents_pit.csv        (FULL membership windows for all
                                                     tickers — REPLACES the 55-name
                                                     subset originally derived from
                                                     the same dataset)

Symbols are normalized to Yahoo form ('.' -> '-', e.g. BRK.B -> BRK-B) so the
ingestion clients, the price lake, and the PIT file all share one spelling.

RENAMES maps old symbols to their post-rename ticker for pure corporate renames
where the vendor serves the full history under the new symbol (FB -> META). The
old and new membership windows then union naturally in pit_membership_mask.
Mergers/acquisitions are NOT mapped — a departed target's data simply ends, and
whatever the vendor can't serve is measured by the experiment's coverage report.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from core import paths

logger = logging.getLogger(__name__)

WINDOW_START = pd.Timestamp("2018-01-02")  # matches configs' start

# Pure renames within/near the window (old -> new); vendor history lives under new.
# Mergers into genuinely new entities are NOT mapped (their data is honestly gone
# and counted by the coverage report). Verified empirically: the old symbols below
# return nothing from the vendor while the new symbol serves the joined history.
RENAMES = {
    "FB": "META",
    "UTX": "RTX",
    "ANTM": "ELV",
    "WLTW": "WTW",
    "NLOK": "GEN",
    "SYMC": "GEN",  # Symantec -> NortonLifeLock -> Gen Digital
    "FISV": "FI",
    "PKI": "RVTY",
    "BLL": "BALL",
    "ABC": "COR",  # AmerisourceBergen -> Cencora
    "ADS": "BFH",  # Alliance Data -> Bread Financial
    "CDAY": "DAY",  # Ceridian -> Dayforce
    "TMK": "GL",  # Torchmark -> Globe Life
    "HRS": "LHX",  # Harris -> L3Harris
    "KORS": "CPRI",  # Michael Kors -> Capri
    "CTL": "LUMN",  # CenturyLink -> Lumen
    "VIAB": "PARA",  # Viacom -> ViacomCBS -> Paramount
    "VIAC": "PARA",
    "WYND": "TNL",  # Wyndham Destinations -> Travel + Leisure
    "COG": "CTRA",  # Cabot -> Coterra
    "FBHS": "FBIN",  # Fortune Brands
    "FLT": "CPAY",  # Fleetcor -> Corpay
    "RE": "EG",  # Everest Re -> Everest Group
    "PEAK": "DOC",  # Healthpeak
    "Q": "IQV",  # Quintiles -> IQVIA
}

SRC = paths.REFERENCE / "sp500_ticker_start_end.csv"
UNIVERSE_OUT = paths.REFERENCE / "universe_sp500.csv"
PIT_OUT = paths.REFERENCE / "constituents_pit.csv"


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper().replace(".", "-")
    return RENAMES.get(t, t)


COALESCE_GAP = pd.Timedelta(days=7)
OPEN_END = pd.Timestamp("2200-01-01")  # finite sentinel: Timestamp.max overflows on + gap


def _coalesce(pit: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/adjacent membership windows per ticker (renames leave
    back-to-back windows: FB ends 2022-06-09, META starts 2022-06-09)."""
    rows = []
    for ticker, grp in pit.groupby("ticker", sort=True):
        cur_start, cur_end = None, None
        for r in grp.sort_values("start").itertuples():
            end = r.end if pd.notna(r.end) else OPEN_END
            if cur_start is None:
                cur_start, cur_end = r.start, end
            elif r.start <= cur_end + COALESCE_GAP:
                cur_end = max(cur_end, end)
            else:
                rows.append({"ticker": ticker, "start": cur_start, "end": cur_end})
                cur_start, cur_end = r.start, end
        rows.append({"ticker": ticker, "start": cur_start, "end": cur_end})
    out = pd.DataFrame(rows)
    out.loc[out["end"] >= OPEN_END, "end"] = pd.NaT
    return out


def build(window_start: pd.Timestamp = WINDOW_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(SRC, parse_dates=["start_date", "end_date"])
    src["ticker"] = src["ticker"].map(_normalize)
    today = pd.Timestamp(date.today())

    pit = src.rename(columns={"start_date": "start", "end_date": "end"})
    # Windows that closed before the backtest window carry no information here and
    # create symbol collisions with the modern universe (a 1996-97 member used
    # "GLD"; keeping its window would mask out the ETF for the whole backtest).
    pit = pit[pit["end"].isna() | (pit["end"] >= window_start)]
    pit = pit.sort_values(["ticker", "start"]).reset_index(drop=True)
    pit = _coalesce(pit)

    overlap = pit[(pit["start"] <= today) & (pit["end"].isna() | (pit["end"] >= window_start))]
    tickers = sorted(overlap["ticker"].unique())

    etfs = pd.read_csv(paths.REFERENCE / "universe.csv")
    etfs = etfs[etfs["asset_type"] == "etf"]

    universe = pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": tickers,
                    "name": tickers,  # dataset carries no company names; ticker is enough
                    "asset_type": "stock",
                    "sector": "Unknown",
                }
            ),
            etfs[["ticker", "name", "asset_type", "sector"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["ticker"], keep="first")
    return universe, pit


def main() -> None:
    universe, pit = build()
    universe.to_csv(UNIVERSE_OUT, index=False)
    pit.to_csv(PIT_OUT, index=False)
    n_active = universe[universe["asset_type"] == "stock"]["ticker"].nunique()
    logger.info(
        "wrote %s (%d stocks + %d ETFs) and %s (%d membership windows, %d tickers)",
        UNIVERSE_OUT.name,
        n_active,
        (universe["asset_type"] == "etf").sum(),
        PIT_OUT.name,
        len(pit),
        pit["ticker"].nunique(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
