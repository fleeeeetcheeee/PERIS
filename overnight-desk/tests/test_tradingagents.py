"""TradingAgents tests: graph structure, parsing fail-safes, snapshot PIT, memory.

All LLM calls go through ScriptedLLM — deterministic, offline. The PIT tests
mirror the repo's mechanical lookahead style: corrupt data the decision must
not see and require bit-identical snapshot blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_wave6_features import make_events, make_facts
from tradingagents import roles
from tradingagents.config import TradingAgentsConfig
from tradingagents.graph import Decision, TradingAgentsGraph, target_weight
from tradingagents.llm import ScriptedLLM
from tradingagents.memory import DecisionMemory
from tradingagents.snapshot import build_snapshot

MANAGER = "STANCE: BULLISH\nPLAN: Momentum and fundamentals align; buy dips.\nWEAKNESS: valuation."
TRADER_BUY = "DECISION: BUY\nCONVICTION: HIGH\nRATIONALE: Trend intact, earnings reaction positive."
PM_APPROVE = (
    "VERDICT: APPROVE\nFINAL DECISION: BUY\nFINAL CONVICTION: HIGH\nREASON: Risk acceptable."
)
PM_REJECT = "VERDICT: REJECT\nFINAL DECISION: BUY\nFINAL CONVICTION: HIGH\nREASON: Too crowded."


def _cfg(**overrides) -> TradingAgentsConfig:
    base = dict(debate_rounds=1, risk_rounds=1, tickers=["AAA"], benchmark="SPY")
    base.update(overrides)
    return TradingAgentsConfig(**base)


def _quick_responses(cfg: TradingAgentsConfig) -> list[str]:
    analysts = [
        f"SIGNAL: BULLISH\nKEY POINTS: momentum\nRISKS: reversal (analyst {i})" for i in range(4)
    ]
    debate = [f"argument {i}" for i in range(2 * cfg.debate_rounds)]
    risk = [f"risk view {i}" for i in range(3 * cfg.risk_rounds)]
    return analysts + debate + risk


def test_graph_call_structure_and_decision(panel, macro):
    cfg = _cfg()
    quick = ScriptedLLM(_quick_responses(cfg))
    deep = ScriptedLLM([MANAGER, TRADER_BUY, PM_APPROVE])
    graph = TradingAgentsGraph(cfg, quick, deep)
    asof = panel["date"].max()
    d = graph.propagate(
        "AAA", asof, panel, macro=macro, fundamentals=make_facts("AAA"), events=make_events("AAA")
    )

    assert d.action == "BUY" and d.conviction == "HIGH" and d.verdict == "APPROVE"
    assert d.target_weight == 1.0
    assert d.trader_action == "BUY" and d.trader_conviction == "HIGH"
    assert len(quick.calls) == 4 + 2 * cfg.debate_rounds + 3 * cfg.risk_rounds
    assert len(deep.calls) == 3
    for key in (
        "analyst_technical",
        "analyst_fundamental",
        "analyst_news",
        "analyst_sentiment",
        "bull_round1",
        "bear_round1",
        "research_manager",
        "trader",
        "risk_aggressive_round1",
        "risk_conservative_round1",
        "risk_neutral_round1",
        "portfolio_manager",
    ):
        assert key in d.transcript, key
    # numbers reach the analysts, stances reach the decision-makers
    assert "RSI(14)" in quick.calls[0][1]
    assert "STANCE: BULLISH" in deep.calls[1][1]  # manager plan flows into trader prompt


def test_pm_reject_forces_hold(panel, macro):
    cfg = _cfg()
    quick = ScriptedLLM(_quick_responses(cfg))
    deep = ScriptedLLM([MANAGER, TRADER_BUY, PM_REJECT])
    graph = TradingAgentsGraph(cfg, quick, deep)
    d = graph.propagate("AAA", panel["date"].max(), panel, macro=macro)
    assert d.verdict == "REJECT"
    assert d.action == "HOLD"
    assert d.target_weight == 0.0  # prev_weight was 0


def test_parsers_fail_safe():
    assert roles.parse_decision("DECISION: **BUY** because...") == "BUY"
    assert roles.parse_decision("decision - sell everything") == "SELL"
    assert roles.parse_decision("I feel great about this stock!") == "HOLD"
    assert roles.parse_decision("FINAL DECISION: Hold\n", tag="FINAL DECISION") == "HOLD"
    assert roles.parse_conviction("CONVICTION: medium-ish") == "MEDIUM"
    assert roles.parse_conviction("no tag at all") == "LOW"
    assert roles.parse_verdict("VERDICT: APPROVE") == "APPROVE"
    assert roles.parse_verdict("hmm") == "ADJUST"
    assert roles.parse_signal("SIGNAL: BEARISH") == "BEARISH"


def test_target_weight_hold_carries_position():
    cfg = _cfg()
    assert target_weight("HOLD", "LOW", cfg, prev=0.66) == 0.66
    assert target_weight("SELL", "HIGH", cfg, prev=0.66) == 0.0
    assert target_weight("BUY", "MEDIUM", cfg, prev=0.0) == 0.66


def test_snapshot_is_point_in_time(panel, macro):
    asof = pd.Timestamp("2022-12-15")
    facts, events = make_facts("AAA"), make_events("AAA")

    base = build_snapshot("AAA", asof, panel, macro=macro, fundamentals=facts, events=events)

    corrupted = panel.copy()
    future = corrupted["date"] > asof
    rng = np.random.default_rng(1)
    for col in ("close", "volume"):
        corrupted.loc[future, col] = corrupted.loc[future, col].values * rng.uniform(
            0.5, 2.0, future.sum()
        )
    cmacro = macro.copy()
    cmacro.loc[cmacro["date"] >= asof, "value"] *= 9.0  # ON asof must be invisible too
    cfacts = facts.copy()
    cfacts.loc[cfacts["filed"] >= asof, "val"] *= 7.0
    cevents = events.copy()
    cevents.loc[cevents["filed"] > asof, "filed"] += pd.Timedelta(days=30)

    after = build_snapshot(
        "AAA", asof, corrupted, macro=cmacro, fundamentals=cfacts, events=cevents
    )
    assert base.technical == after.technical
    assert base.fundamental == after.fundamental
    assert base.macro == after.macro
    assert base.mood == after.mood


def test_snapshot_rejects_nonsession(panel):
    with pytest.raises(ValueError):
        build_snapshot("AAA", "2022-12-25", panel)  # Christmas: not a session


def test_memory_roundtrip(panel, tmp_path):
    mem = DecisionMemory(path=tmp_path / "decisions.jsonl", horizon_sessions=5)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    asof = sessions[-10]
    d = Decision(
        ticker="AAA",
        asof=str(asof.date()),
        action="BUY",
        conviction="HIGH",
        verdict="APPROVE",
        target_weight=1.0,
        trader_action="BUY",
        trader_conviction="HIGH",
        transcript={"trader": "x"},
    )
    mem.record(d)
    mem.record(d)  # idempotent
    assert len(mem._load()) == 1

    filled = mem.update_outcomes(panel, benchmark="SPY")
    assert filled == 1
    row = mem._load()[0]
    close = panel.pivot(index="date", columns="ticker", values="close")
    expected = close["AAA"].loc[sessions[-5]] / close["AAA"].loc[asof] - 1
    assert row["fwd_return"] == pytest.approx(expected)
    assert "alpha_vs_spy" in row

    text = mem.lessons("AAA")
    assert "BUY/HIGH" in text and str(asof.date()) in text
