"""Scoring with validation checks and rule-based fallback (the PERIS pattern).

If the model artifact is missing, its feature schema doesn't match, or its output is
degenerate, scoring falls back to the reversal+momentum composite and reports
fallback_mode=True so briefings are labeled honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from models import fallback, registry

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    scores: pd.Series  # aligned to the input frame's index
    fallback_mode: bool
    reason: str


def score(features: pd.DataFrame, feature_cols: list[str]) -> ScoreResult:
    loaded = registry.load_current()
    if loaded is None:
        return _fallback(features, "no model artifact promoted")

    booster, meta = loaded
    expected = meta.get("feature_cols", [])
    missing = set(expected) - set(feature_cols)
    if missing:
        # A superset is fine (new feature families added since training); only
        # genuinely absent columns break the schema.
        return _fallback(features, f"feature schema mismatch (missing: {missing})")

    pred = pd.Series(booster.predict(features[expected]), index=features.index)
    if pred.isna().mean() > 0.5:
        return _fallback(features, "model produced >50% NaN scores")
    if pred.nunique() <= 1:
        return _fallback(features, "model produced constant scores")
    return ScoreResult(scores=pred, fallback_mode=False, reason="model")


def _fallback(features: pd.DataFrame, reason: str) -> ScoreResult:
    logger.warning("falling back to rule-based composite: %s", reason)
    return ScoreResult(scores=fallback.score(features), fallback_mode=True, reason=reason)
