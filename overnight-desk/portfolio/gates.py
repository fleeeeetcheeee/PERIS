"""Risk gates applied before every rebalance.

1. Earnings exclusion — skip names reporting within N sessions (needs Finnhub key;
   degrades to a loud no-op without one).
2. VIX term structure — VIX3M/VIX < 1 (backwardation) halves exposure and blocks
   new entries; the briefing states which gate fired.
3. Max-drawdown circuit breaker — halt new entries when equity is >= 10% below the
   ledger high-water mark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from core.config import GatesConfig

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    excluded: set[str] = field(default_factory=set)
    exposure_scale: float = 1.0
    allow_new_entries: bool = True
    notes: list[str] = field(default_factory=list)


def earnings_gate(
    decision: GateDecision,
    earnings: pd.DataFrame | None,
    asof: pd.Timestamp,
    next_sessions: list[pd.Timestamp],
    cfg: GatesConfig,
) -> GateDecision:
    if earnings is None or earnings.empty:
        decision.notes.append("earnings gate INACTIVE (no earnings calendar — set FINNHUB_API_KEY)")
        return decision
    window = set(
        pd.Timestamp(s).normalize() for s in next_sessions[: cfg.earnings_exclusion_sessions]
    )
    window.add(pd.Timestamp(asof).normalize())
    hits = earnings[earnings["earnings_date"].dt.normalize().isin(window)]
    if not hits.empty:
        excluded = set(hits["ticker"])
        decision.excluded |= excluded
        decision.notes.append(f"earnings exclusion: {sorted(excluded)}")
    return decision


def vix_term_structure_gate(
    decision: GateDecision, vix: float | None, vix3m: float | None, cfg: GatesConfig
) -> GateDecision:
    if not cfg.vix_term_structure:
        return decision
    if vix is None or vix3m is None or vix <= 0:
        decision.notes.append("VIX gate INACTIVE (missing VIX/VIX3M data)")
        return decision
    ratio = vix3m / vix
    if ratio < 1.0:
        decision.exposure_scale *= 0.5
        decision.allow_new_entries = False
        decision.notes.append(
            f"VIX term structure inverted (VIX3M/VIX={ratio:.2f}) — exposure halved, no new entries"
        )
    return decision


def jump_regime_gate(decision: GateDecision, state: int | None, cfg: GatesConfig) -> GateDecision:
    """Stress state (1) from the jump model halves exposure and blocks new entries."""
    if not cfg.jump_model:
        return decision
    if state is None:
        decision.notes.append("jump-model gate INACTIVE (no regime state available)")
        return decision
    if state == 1:
        decision.exposure_scale *= 0.5
        decision.allow_new_entries = False
        decision.notes.append("jump model: STRESS regime — exposure halved, no new entries")
    return decision


def drawdown_gate(
    decision: GateDecision, equity: float | None, high_water_mark: float | None, cfg: GatesConfig
) -> GateDecision:
    if equity is None or high_water_mark is None or high_water_mark <= 0:
        return decision
    dd = equity / high_water_mark - 1.0
    if dd <= -cfg.drawdown_halt_pct:
        decision.allow_new_entries = False
        decision.notes.append(f"drawdown circuit breaker: {dd:.1%} from HWM — new entries halted")
    return decision
