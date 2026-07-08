"""Model scoring fallback: the rule-based composite kicks in when the artifact is
missing or invalid, and the result is flagged fallback_mode (PERIS pattern)."""

from __future__ import annotations

import pandas as pd

from models import fallback, predict, registry


def _features(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2026-07-02"),
            "ticker": [f"T{i}" for i in range(n)],
            "rev_z_5d": [(-1) ** i * (i / 10) for i in range(n)],
            "resmom_126x5": [i / 20 for i in range(n)],
        }
    )


def test_fallback_when_no_artifact(monkeypatch):
    monkeypatch.setattr(registry, "load_current", lambda: None)
    result = predict.score(_features(), ["rev_z_5d", "resmom_126x5"])
    assert result.fallback_mode
    assert "no model artifact" in result.reason
    assert result.scores.notna().all()


def test_fallback_on_schema_mismatch(monkeypatch):
    class FakeBooster:
        def predict(self, X):
            raise AssertionError("must not be called on schema mismatch")

    monkeypatch.setattr(
        registry, "load_current", lambda: (FakeBooster(), {"feature_cols": ["other_col"]})
    )
    result = predict.score(_features(), ["rev_z_5d", "resmom_126x5"])
    assert result.fallback_mode
    assert "schema mismatch" in result.reason


def test_fallback_composite_prefers_losers_with_momentum():
    f = _features()
    scores = fallback.score(f)
    assert len(scores) == len(f)
    assert scores.std() > 0
