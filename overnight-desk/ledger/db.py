"""SQLite ledger: signals, paper/real fills, equity marks, slippage reconciliation.

Market data never lands here — Parquet/DuckDB only. Per repo conventions the ledger
stores share counts as INTEGER and prices as decimal TEXT.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from core import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL,
    weight REAL NOT NULL,
    shares INTEGER NOT NULL,
    limit_price TEXT NOT NULL,
    side TEXT NOT NULL DEFAULT 'buy',
    rationale TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    fallback_mode INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (ticket_date, ticker, side)
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    fill_date TEXT NOT NULL,
    shares INTEGER NOT NULL,
    price TEXT NOT NULL,
    paper INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS equity_marks (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    high_water_mark REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS slippage (
    month TEXT PRIMARY KEY,
    modeled_bps REAL NOT NULL,
    realized_bps REAL,
    n_fills INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def connect(db_path: Path | None = None):
    path = db_path or paths.LEDGER_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def record_signals(rows: list[dict], db_path: Path | None = None) -> int:
    """Insert a ticket's signals as `pending`. Idempotent per (date, ticker, side)."""
    with connect(db_path) as con:
        n = 0
        for r in rows:
            cur = con.execute(
                """INSERT OR IGNORE INTO signals
                   (ticket_date, ticker, score, rank, weight, shares, limit_price,
                    side, rationale, fallback_mode)
                   VALUES (:ticket_date, :ticker, :score, :rank, :weight, :shares,
                           :limit_price, :side, :rationale, :fallback_mode)""",
                r,
            )
            n += cur.rowcount
        return n


def signals_for_date(ticket_date: str, db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as con:
        cur = con.execute(
            "SELECT * FROM signals WHERE ticket_date = ? ORDER BY rank", (ticket_date,)
        )
        return [dict(r) for r in cur.fetchall()]


def record_fill(
    signal_id: int,
    fill_date: str,
    shares: int,
    price: str,
    paper: bool = True,
    db_path: Path | None = None,
) -> None:
    with connect(db_path) as con:
        con.execute(
            "INSERT INTO fills (signal_id, fill_date, shares, price, paper) VALUES (?,?,?,?,?)",
            (signal_id, fill_date, shares, price, int(paper)),
        )
        con.execute("UPDATE signals SET status = 'filled' WHERE id = ?", (signal_id,))


def mark_equity(d: str, equity: float, db_path: Path | None = None) -> None:
    with connect(db_path) as con:
        row = con.execute("SELECT MAX(high_water_mark) hwm FROM equity_marks").fetchone()
        hwm = max(row["hwm"] or equity, equity)
        con.execute(
            "INSERT OR REPLACE INTO equity_marks (date, equity, high_water_mark) VALUES (?,?,?)",
            (d, equity, hwm),
        )


def latest_equity(db_path: Path | None = None) -> tuple[float, float] | None:
    """Returns (equity, trailing HWM). The drawdown breaker compares against the max
    equity of the last 252 marks, not all-time — an all-time HWM would make the halt
    permanent if the book ever went flat below the threshold."""
    with connect(db_path) as con:
        row = con.execute("SELECT equity FROM equity_marks ORDER BY date DESC LIMIT 1").fetchone()
        if row is None:
            return None
        hwm = con.execute(
            "SELECT MAX(equity) m FROM ("
            " SELECT equity FROM equity_marks ORDER BY date DESC LIMIT 252)"
        ).fetchone()
        return (row["equity"], hwm["m"])


def paper_gate_status(db_path: Path | None = None) -> dict:
    """Live briefing mode stays disabled until >= 3 months of tracked signals.

    Signals land only on rebalance days, so the ticket count is calibrated to the
    5-session cadence: >= 90 calendar days AND >= 12 rebalance tickets (~one quarter
    of weekly tickets). Do not remove or shortcut this gate.
    """
    with connect(db_path) as con:
        row = con.execute(
            "SELECT MIN(ticket_date) first, COUNT(DISTINCT ticket_date) days FROM signals"
        ).fetchone()
    first, days = row["first"], row["days"]
    if not first:
        return {"live_allowed": False, "reason": "no tracked signals yet", "tracked_days": 0}
    elapsed = date.today() - datetime.strptime(first, "%Y-%m-%d").date()
    ok = elapsed >= timedelta(days=90) and days >= 12
    return {
        "live_allowed": ok,
        "reason": "gate satisfied"
        if ok
        else f"paper ledger has {elapsed.days} days / {days} rebalance tickets "
        "(need >= 90 days and 12 tickets)",
        "tracked_days": days,
        "first_signal": first,
    }
