"""Ledger tests: idempotent signal recording, fills, paper-trading gate."""

from __future__ import annotations

from ledger import db


def _signal_row(date: str = "2026-07-02", ticker: str = "AAPL") -> dict:
    return {
        "ticket_date": date,
        "ticker": ticker,
        "score": 0.05,
        "rank": 1,
        "weight": 0.075,
        "shares": 3,
        "limit_price": "212.34",
        "side": "buy",
        "rationale": "test",
        "fallback_mode": 0,
    }


def test_record_signals_idempotent(tmp_path):
    p = tmp_path / "ledger.db"
    assert db.record_signals([_signal_row()], db_path=p) == 1
    assert db.record_signals([_signal_row()], db_path=p) == 0  # duplicate ignored
    rows = db.signals_for_date("2026-07-02", db_path=p)
    assert len(rows) == 1
    assert rows[0]["limit_price"] == "212.34"  # decimal string, not float
    assert isinstance(rows[0]["shares"], int)


def test_fill_marks_signal_filled(tmp_path):
    p = tmp_path / "ledger.db"
    db.record_signals([_signal_row()], db_path=p)
    sid = db.signals_for_date("2026-07-02", db_path=p)[0]["id"]
    db.record_fill(sid, "2026-07-02", 3, "212.50", db_path=p)
    assert db.signals_for_date("2026-07-02", db_path=p)[0]["status"] == "filled"


def test_paper_gate_blocks_live_until_three_months(tmp_path):
    p = tmp_path / "ledger.db"
    status = db.paper_gate_status(db_path=p)
    assert not status["live_allowed"]

    db.record_signals([_signal_row(date="2026-07-01")], db_path=p)
    status = db.paper_gate_status(db_path=p)
    assert not status["live_allowed"]  # one recent ticket is nowhere near 3 months


def test_equity_high_water_mark(tmp_path):
    p = tmp_path / "ledger.db"
    db.mark_equity("2026-07-01", 100.0, db_path=p)
    db.mark_equity("2026-07-02", 120.0, db_path=p)
    db.mark_equity("2026-07-03", 90.0, db_path=p)
    eq, hwm = db.latest_equity(db_path=p)
    assert eq == 90.0 and hwm == 120.0


def test_high_water_mark_is_trailing_not_all_time(tmp_path):
    """The drawdown breaker must release after a recovery window: HWM looks back
    252 marks, so an ancient peak cannot freeze the strategy forever."""
    p = tmp_path / "ledger.db"
    db.mark_equity("2020-01-01", 200.0, db_path=p)  # ancient peak
    for i in range(252):
        db.mark_equity(f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", 90.0, db_path=p)
    eq, hwm = db.latest_equity(db_path=p)
    assert eq == 90.0
    assert hwm == 90.0  # the 200.0 peak has aged out of the window
