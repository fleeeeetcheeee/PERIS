"""Desk server tests: event bus, graph instrumentation, worker cycle, endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from tests.test_tradingagents import MANAGER, PM_APPROVE, TRADER_BUY, _cfg, _quick_responses
from tradingagents.app import create_app
from tradingagents.events import EventBus
from tradingagents.graph import Decision, TradingAgentsGraph
from tradingagents.llm import ScriptedLLM
from tradingagents.worker import DeskWorker


def test_event_bus_pubsub_and_history():
    bus = EventBus(history=3)
    q = bus.subscribe()
    bus.emit("a", n=1)
    bus.emit_dict({"type": "b", "n": 2})
    assert q.get_nowait()["type"] == "a"
    assert q.get_nowait()["n"] == 2
    for i in range(5):
        bus.emit("spam", i=i)
    assert len(bus.recent()) == 3  # ring buffer bounded
    assert q.qsize() == 5  # the spam events were delivered before unsubscribing
    bus.unsubscribe(q)
    bus.emit("after")
    assert q.qsize() == 5  # nothing delivered after unsubscribe


def test_graph_emits_stage_and_decision_events(panel, macro):
    cfg = _cfg()
    events: list[dict] = []
    graph = TradingAgentsGraph(
        cfg,
        ScriptedLLM(_quick_responses(cfg)),
        ScriptedLLM([MANAGER, TRADER_BUY, PM_APPROVE]),
        on_event=events.append,
    )
    graph.propagate("AAA", panel["date"].max(), panel, macro=macro)

    stages = [e for e in events if e["type"] == "stage"]
    agents = {e["agent"] for e in stages}
    assert agents == {
        "technical",
        "fundamental",
        "news",
        "sentiment",
        "bull",
        "bear",
        "manager",
        "trader",
        "risk_aggressive",
        "risk_neutral",
        "risk_conservative",
        "pm",
    }
    # every agent emits a start and a done, done carries a detail snippet
    for agent in agents:
        statuses = [e["status"] for e in stages if e["agent"] == agent]
        assert statuses == ["start", "done"]
    assert all(e["detail"] for e in stages if e["status"] == "done")
    final = [e for e in events if e["type"] == "decision"]
    assert len(final) == 1 and final[0]["action"] == "BUY"


def _stub_graph(cfg, bus):
    class StubGraph:
        memory = None

        def propagate(self, ticker, asof, panel, **kw):
            d = Decision(
                ticker=ticker,
                asof=str(pd.Timestamp(asof).date()),
                action="BUY",
                conviction="HIGH",
                verdict="APPROVE",
                target_weight=1.0,
                trader_action="BUY",
                trader_conviction="HIGH",
                transcript={
                    "analyst_technical": "RSI is neutral.",
                    "bull_round1": "Momentum is strong.",
                    "risk_conservative_round1": "Trim the size.",
                    "portfolio_manager": "APPROVE. FINAL DECISION: BUY",
                },
            )
            bus.emit_dict(
                {"type": "decision", **{k: v for k, v in d.to_dict().items() if k != "transcript"}}
            )
            return d

    return StubGraph()


def test_worker_cycle_caches_and_replays(panel, macro, tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.evaluate.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("tradingagents.worker.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("tradingagents.worker.lake.read_curated_prices", lambda: panel)
    monkeypatch.setattr("tradingagents.worker.lake.read_curated_macro", lambda: macro)
    monkeypatch.setattr("tradingagents.worker.paths.CURATED", tmp_path)  # no fundamentals/events

    cfg = _cfg(tickers=["AAA", "BBB"])
    bus = EventBus()
    worker = DeskWorker(cfg, _stub_graph(cfg, bus), bus)

    worker.cycle()
    types = [e["type"] for e in bus.recent()]
    assert types.count("decision") == 2
    assert (tmp_path / "cache" / f"AAA_{panel['date'].max().date()}.json").exists()

    worker.cycle()  # second cycle: everything cached, replayed with cached=True
    replays = [e for e in bus.recent() if e["type"] == "decision" and e.get("cached")]
    assert len(replays) == 2
    # cached transcripts replay as stage events so the floor shows what was said
    stages = [e for e in bus.recent() if e["type"] == "stage" and e.get("cached")]
    assert {e["agent"] for e in stages} == {"technical", "bull", "risk_conservative", "pm"}
    assert all(e["status"] == "done" and e["detail"] for e in stages)


def test_agent_for_transcript_key():
    from tradingagents.worker import _agent_for_transcript_key

    assert _agent_for_transcript_key("analyst_technical") == "technical"
    assert _agent_for_transcript_key("analyst_sentiment") == "sentiment"
    assert _agent_for_transcript_key("bull_round2") == "bull"
    assert _agent_for_transcript_key("bear_round1") == "bear"
    assert _agent_for_transcript_key("risk_aggressive_round1") == "risk_aggressive"
    assert _agent_for_transcript_key("research_manager") == "manager"
    assert _agent_for_transcript_key("trader") == "trader"
    assert _agent_for_transcript_key("portfolio_manager") == "pm"
    assert _agent_for_transcript_key("unknown_key") is None


def test_app_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr("tradingagents.app.CACHE_DIR", tmp_path)  # empty board
    app = create_app(_cfg(), worker=False)
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200 and "TRADINGAGENTS DESK" in page.text

    state = client.get("/state").json()
    assert state["tickers"] == ["AAA"]
    assert state["board"] == [] and "memory" in state and "events" in state


def test_intraday_block_reaches_snapshot(panel):
    from tradingagents.snapshot import build_snapshot

    q = {
        "last": 101.0,
        "prev_close": 97.0,
        "change_pct": 4.1,
        "day_high": 102.0,
        "day_low": 96.5,
        "from_high_pct": -1.0,
        "volume": 2e6,
        "prev_volume": 1e6,
    }
    snap = build_snapshot(
        "AAA", panel["date"].max(), panel, intraday=q, intraday_reason="unit test trigger"
    )
    assert "INTRADAY UPDATE" in snap.technical
    assert "+4.1%" in snap.technical and "unit test trigger" in snap.technical
    assert "Intraday flash" in snap.mood
    assert snap.numbers["intraday_change_pct"] == 4.1
    base = build_snapshot("AAA", panel["date"].max(), panel)
    assert "INTRADAY" not in base.technical  # absent unless provided


def _watch_worker(panel, macro, tmp_path, monkeypatch, quotes, **cfg_overrides):
    monkeypatch.setattr("tradingagents.evaluate.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("tradingagents.worker.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("tradingagents.worker.lake.read_curated_prices", lambda **kw: panel)
    monkeypatch.setattr("tradingagents.worker.lake.read_curated_macro", lambda: macro)
    monkeypatch.setattr("tradingagents.worker.paths.CURATED", tmp_path)
    monkeypatch.setattr("tradingagents.worker.fetch_quotes", lambda tickers: quotes)
    notifications: list[tuple] = []
    monkeypatch.setattr(
        "tradingagents.worker.mac_notify", lambda *a: notifications.append(a) or True
    )
    cfg = _cfg(tickers=list(quotes), watch_move_pct=2.5, **cfg_overrides)
    bus = EventBus()
    calls: list[str] = []

    class WatchStubGraph:
        memory = None

        def propagate(self, ticker, asof, panel, intraday=None, **kw):
            assert intraday is not None  # revisions must carry the live quote
            calls.append(ticker)
            return Decision(
                ticker=ticker,
                asof=str(pd.Timestamp(asof).date()),
                action="SELL",
                conviction="HIGH",
                verdict="APPROVE",
                target_weight=0.0,
                trader_action="SELL",
                trader_conviction="HIGH",
                transcript={},
            )

    return DeskWorker(cfg, WatchStubGraph(), bus), bus, calls, notifications


def test_watch_triggers_review_cooldown_and_alert(panel, macro, tmp_path, monkeypatch):
    quotes = {
        "AAA": {
            "last": 1,
            "prev_close": 1,
            "change_pct": -3.2,
            "day_high": 1,
            "day_low": 1,
            "from_high_pct": -3.2,
            "volume": 1,
            "prev_volume": 1,
        },
        "BBB": {
            "last": 1,
            "prev_close": 1,
            "change_pct": 0.4,
            "day_high": 1,
            "day_low": 1,
            "from_high_pct": 0.0,
            "volume": 1,
            "prev_volume": 1,
        },
    }
    worker, bus, calls, notifications = _watch_worker(panel, macro, tmp_path, monkeypatch, quotes)

    t0 = pd.Timestamp("2023-03-01 10:00")
    worker.watch_once(now=t0)
    assert calls == ["AAA"]  # only the big mover triggers, BBB (+0.4%) does not
    session = panel["date"].max().date()
    assert (tmp_path / "cache" / f"AAA_{session}_r1000.json").exists()
    alerts = [e for e in bus.recent() if e["type"] == "alert"]
    assert len(alerts) == 1 and alerts[0]["action"] == "SELL"
    assert len(notifications) == 1 and "SELL AAA" in notifications[0][1]

    worker.watch_once(now=t0 + pd.Timedelta(minutes=30))
    assert calls == ["AAA"]  # inside the 90-min cooldown: no second review

    worker.watch_once(now=t0 + pd.Timedelta(minutes=120))
    assert calls == ["AAA", "AAA"]  # cooldown passed
    # same SELL stance again -> alert only fired once (notify_all off)
    assert len([e for e in bus.recent() if e["type"] == "alert"]) == 1


def test_notify_only_on_buy_sell_changes(panel, macro, tmp_path, monkeypatch):
    quotes = {
        "AAA": {
            "last": 1,
            "prev_close": 1,
            "change_pct": 5.0,
            "day_high": 1,
            "day_low": 1,
            "from_high_pct": 0.0,
            "volume": 1,
            "prev_volume": 1,
        },
    }
    worker, bus, calls, notifications = _watch_worker(panel, macro, tmp_path, monkeypatch, quotes)
    hold = Decision(
        ticker="AAA",
        asof="2023-03-01",
        action="HOLD",
        conviction="LOW",
        verdict="ADJUST",
        target_weight=0.0,
        trader_action="HOLD",
        trader_conviction="LOW",
        transcript={},
    )
    worker._maybe_notify(hold, trigger="t")
    assert notifications == []  # HOLD never notifies
    buy = Decision(
        ticker="AAA",
        asof="2023-03-01",
        action="BUY",
        conviction="HIGH",
        verdict="APPROVE",
        target_weight=1.0,
        trader_action="BUY",
        trader_conviction="HIGH",
        transcript={},
    )
    worker._maybe_notify(buy, trigger="t")
    worker._maybe_notify(buy, trigger="t")
    assert len(notifications) == 1  # change to BUY notifies once, repeat is silent


def test_decision_board_prefers_intraday_revision(monkeypatch, tmp_path):
    import json as _json

    from tradingagents.app import decision_board

    monkeypatch.setattr("tradingagents.app.CACHE_DIR", tmp_path)
    daily = {"ticker": "AAA", "asof": "2023-03-01", "action": "BUY", "transcript": {}}
    revision = {"ticker": "AAA", "asof": "2023-03-01", "action": "SELL", "transcript": {}}
    (tmp_path / "AAA_2023-03-01.json").write_text(_json.dumps(daily))
    (tmp_path / "AAA_2023-03-01_r1435.json").write_text(_json.dumps(revision))
    board = decision_board()
    assert len(board) == 1 and board[0]["action"] == "SELL"
