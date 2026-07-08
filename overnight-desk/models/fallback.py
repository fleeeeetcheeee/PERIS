"""Rule-based fallback: reversal + residual momentum composite (the PERIS pattern).

Used when no model artifact exists or the artifact fails validation checks.
Briefings built on this scorer are marked "fallback mode".
"""

from __future__ import annotations

import pandas as pd


def _xs_z(s: pd.Series, dates: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date."""
    g = s.groupby(dates)
    return (s - g.transform("mean")) / g.transform("std")


def score(features: pd.DataFrame) -> pd.Series:
    """Composite score: buy recent losers with strong residual momentum.

    Higher = better expected next-5-day relative return.
    """
    rev = -_xs_z(features["rev_z_5d"], features["date"])
    mom = _xs_z(features["resmom_126x5"], features["date"])
    return (rev.fillna(0) + mom.fillna(0)) / 2
