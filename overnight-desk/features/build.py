"""Feature matrix builder: runs every family, merges on (date, ticker), attaches labels.

Usable directly: python -m features.build --config configs/baseline.yaml
Output: data/curated/features.parquet
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from core import lake, paths
from features import (
    calendar_features,
    fracdiff,
    fundamentals,
    labels,
    macro_regime,
    pead,
    residual_momentum,
    reversal,
    signature,
    spillover,
    trend,
    turbulence,
    volume_liquidity,
)

logger = logging.getLogger(__name__)

# The PROMOTED feature set — this is what nightly builds and what any retrain sees.
# Families that exist in the codebase but failed their promotion experiment are NOT
# listed here (wave 4: signature, turbulence — see README experiment history);
# putting them here would let a routine retrain promote on a point estimate that
# the wave-4 significance tests already rejected. Experiments that want them build
# an extended matrix via build_matrix(..., families=[*FAMILIES, signature, ...]).
FAMILIES = [
    reversal,
    residual_momentum,
    trend,
    volume_liquidity,
    macro_regime,
    calendar_features,
    fracdiff,
    spillover,
]
EXPERIMENTAL_FAMILIES = [signature, turbulence, fundamentals, pead]  # wave 4 rejected the
# first two; fundamentals/pead are wave-6 candidates — promotion moves a family to FAMILIES

FEATURES_CURATED = paths.CURATED / "features.parquet"


def feature_columns(df: pd.DataFrame) -> list[str]:
    reserved = {"date", "ticker", "fwd_ret", "label", "close", "volume"}
    return [c for c in df.columns if c not in reserved]


def build_matrix(
    panel: pd.DataFrame,
    macro: pd.DataFrame | None,
    with_labels: bool = True,
    families: list | None = None,
) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    matrix = panel[["date", "ticker", "close"]].copy()
    for family in families if families is not None else FAMILIES:
        feats = family.compute(panel, macro)
        matrix = matrix.merge(feats, on=["date", "ticker"], how="left", validate="1:1")
        logger.info("family %s: %d cols", family.__name__, feats.shape[1] - 2)
    if with_labels:
        lab = labels.compute_labels(panel)
        matrix = matrix.merge(lab, on=["date", "ticker"], how="left", validate="1:1")
    return matrix


def run(write: bool = True, tickers: list[str] | None = None) -> pd.DataFrame:
    """tickers: pin the panel to a universe (None = whole lake). The CLI always
    pins to a config's universe so experiment backfills in the lake can't leak
    into the curated matrix."""
    panel = lake.read_curated_prices(tickers=tickers)
    macro = lake.read_curated_macro()
    matrix = build_matrix(panel, macro, with_labels=True)
    if write:
        matrix.to_parquet(FEATURES_CURATED, index=False)
        logger.info("wrote %s: %s", FEATURES_CURATED, matrix.shape)
    return matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(paths.CONFIGS / "baseline.yaml"))
    args = parser.parse_args()
    from core.config import load_config
    from core.universe import load_universe

    cfg = load_config(args.config)
    run(tickers=[m.ticker for m in load_universe(cfg.universe_file)])
