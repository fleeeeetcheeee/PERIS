"""Raw → curated promotion with validation.

Curated data must pass: positive prices, sane OHLC ordering, no duplicate dates,
adjusted-price continuity (flag |daily return| > 40%), and minimum history length.
Failures are per-ticker: a bad ticker is dropped loudly, never silently patched.
"""

from __future__ import annotations

import logging

import pandas as pd

from core import lake

logger = logging.getLogger(__name__)

MIN_ROWS = 260  # ~1 trading year minimum to be usable for features
MAX_ABS_RETURN = 0.40
MAX_JUMPS = 8  # more large moves than any real stock produced here = garbage series
GAP_DAYS = 7  # returns spanning a listing gap are not daily moves; exempt from checks

# An unadjusted split shows as a one-day move of exactly a share ratio. Real crashes
# and earnings gaps land anywhere; ratios land on these points (within tolerance).
SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 1.5, 4 / 3, 5 / 4)
SPLIT_TOL = 0.005


def _looks_like_split(ret: float) -> bool:
    gross = 1.0 + ret
    if gross <= 0:
        return False
    for k in SPLIT_RATIOS:
        if abs(gross - 1 / k) * k < SPLIT_TOL or abs(gross - k) / k < SPLIT_TOL:
            return True
    return False


class ValidationError(Exception):
    pass


def validate_ticker_frame(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if len(df) < MIN_ROWS:
        raise ValidationError(f"{ticker}: only {len(df)} rows (< {MIN_ROWS})")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValidationError(f"{ticker}: non-positive prices")
    bad_ohlc = (df["high"] < df["low"]).sum()
    if bad_ohlc:
        raise ValidationError(f"{ticker}: {bad_ohlc} rows with high < low")

    rets = df["close"].pct_change()
    # Returns spanning a listing gap (delisting, symbol reuse) are not daily moves.
    contiguous = pd.to_datetime(df["date"]).diff().dt.days <= GAP_DAYS
    jumps = rets[(rets.abs() > MAX_ABS_RETURN) & contiguous]
    if len(jumps) > 0:
        # Only fail for the artifacts this check exists to catch: split-ratio-shaped
        # moves (unadjusted split) or a garbage series. Isolated real moves — COVID
        # crashes, earnings gaps, bankruptcies — must stay in the panel: dropping the
        # blowups is survivorship bias injected by our own pipeline.
        split_like = [r for r in jumps if _looks_like_split(float(r))]
        if split_like or len(jumps) >= MAX_JUMPS:
            raise ValidationError(
                f"{ticker}: {len(jumps)} daily moves > {MAX_ABS_RETURN:.0%}, "
                f"{len(split_like)} split-ratio-shaped "
                f"(first at {df.loc[jumps.index[0], 'date'].date()}) — adjustment suspect"
            )
        logger.warning(
            "%s: %d large daily moves look like real events (max %.0f%%) — kept",
            ticker,
            len(jumps),
            jumps.abs().max() * 100,
        )
    return df


def promote(source: str, tickers: list[str]) -> pd.DataFrame:
    """Validate every raw ticker frame and write the curated price table.

    Returns the curated panel. Raises if fewer than half the universe survives —
    a briefing built on a gutted universe is worse than no briefing.
    """
    frames: list[pd.DataFrame] = []
    dropped: list[str] = []
    for ticker in tickers:
        raw = lake.read_raw_prices(source, ticker)
        if raw is None or raw.empty:
            dropped.append(ticker)
            logger.warning("%s: no raw data", ticker)
            continue
        try:
            frames.append(validate_ticker_frame(ticker, raw))
        except ValidationError as exc:
            dropped.append(ticker)
            logger.warning("validation failed: %s", exc)

    if len(frames) < len(tickers) / 2:
        raise RuntimeError(
            f"curation failed: only {len(frames)}/{len(tickers)} tickers valid (dropped: {dropped})"
        )
    if dropped:
        logger.warning("curated without %d tickers: %s", len(dropped), dropped)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    lake.write_curated_prices(panel)
    return panel
