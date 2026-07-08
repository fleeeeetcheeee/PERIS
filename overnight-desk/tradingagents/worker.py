"""Background desk worker: keeps the agent floor running continuously.

Two duties:
- Daily cycle: when the lake gains a new session (nightly ingest), refresh
  realized outcomes and decide every configured ticker for that session.
  Decisions are cached per (ticker, session), so repeat cycles are free.
- Live watch (market hours): poll delayed quotes every watch_poll_minutes;
  a move >= watch_move_pct vs yesterday's close triggers a REVISION — a full
  agent pipeline run with the intraday update injected — at most one per
  ticker per revision_cooldown_minutes. Revisions are cached alongside the
  daily decision as {ticker}_{session}_r{HHMM}.json.

Whenever a fresh decision's action is BUY or SELL and differs from the desk's
previous stance on that ticker (or notify_all is set), a macOS notification
fires and an `alert` event reaches the UI.
"""

from __future__ import annotations

import json
import logging
import threading

import pandas as pd

from core import calendar, lake, paths
from tradingagents.config import TradingAgentsConfig
from tradingagents.evaluate import CACHE_DIR, _cache_decision, _cached_decision
from tradingagents.events import EventBus
from tradingagents.graph import Decision, TradingAgentsGraph
from tradingagents.intraday import fetch_quotes
from tradingagents.notify import mac_notify

logger = logging.getLogger(__name__)


def latest_cached_weight(ticker: str, before: pd.Timestamp) -> float:
    """Most recent cached target weight strictly before `before` (0.0 if none)."""
    if not CACHE_DIR.exists():
        return 0.0
    best: tuple[str, float] | None = None
    for p in CACHE_DIR.glob(f"{ticker}_*.json"):
        day = p.stem.removeprefix(f"{ticker}_")
        if day < str(before.date()):
            if best is None or day > best[0]:
                best = (day, json.loads(p.read_text()).get("target_weight", 0.0))
    return best[1] if best else 0.0


def _agent_for_transcript_key(key: str) -> str | None:
    """Map a Decision.transcript key to the UI agent id it belongs to."""
    if key.startswith("analyst_"):
        return key.removeprefix("analyst_")
    if key.startswith(("bull_", "bear_")):
        return key.split("_", 1)[0]
    if key.startswith("risk_"):
        return key.rsplit("_round", 1)[0]
    return {"research_manager": "manager", "trader": "trader", "portfolio_manager": "pm"}.get(key)


def _cache_revision(decision: Decision, tag: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{decision.ticker}_{decision.asof}_r{tag}.json"
    p.write_text(json.dumps(decision.to_dict(), indent=2))


class DeskWorker(threading.Thread):
    def __init__(self, cfg: TradingAgentsConfig, graph: TradingAgentsGraph, bus: EventBus) -> None:
        super().__init__(daemon=True, name="tradingagents-desk-worker")
        self.cfg = cfg
        self.graph = graph
        self.bus = bus
        self._stop = threading.Event()
        self._last_session: str | None = None
        self._last_action: dict[str, str] = {}
        self._last_revision: dict[str, pd.Timestamp] = {}

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        while not self._stop.is_set():
            market_open = False
            try:
                self._cycle_if_new_session()
                market_open = calendar.is_open_now()
                if market_open and self.cfg.watch_enabled:
                    self.watch_once()
            except Exception as exc:  # keep the floor alive; surface to the UI
                logger.exception("worker loop failed")
                self.bus.emit("error", detail=str(exc)[:300])
            minutes = self.cfg.watch_poll_minutes if market_open else self.cfg.worker_poll_minutes
            self.bus.emit(
                "worker",
                status="watching" if market_open else "sleeping",
                minutes=minutes,
            )
            self._stop.wait(minutes * 60)

    def _load_lake(self):
        panel = lake.read_curated_prices()
        macro = lake.read_curated_macro()
        fundamentals = events = None
        if (paths.CURATED / "fundamentals.parquet").exists():
            fundamentals = pd.read_parquet(paths.CURATED / "fundamentals.parquet")
        if (paths.CURATED / "filing_events.parquet").exists():
            events = pd.read_parquet(paths.CURATED / "filing_events.parquet")
        return panel, macro, fundamentals, events

    def _cycle_if_new_session(self) -> None:
        panel = lake.read_curated_prices(tickers=[self.cfg.benchmark])
        asof = str(panel["date"].max().date())
        if asof != self._last_session:
            self.cycle()
            self._last_session = asof

    # ----------------------------------------------------------- daily cycle

    def cycle(self) -> None:
        panel, macro, fundamentals, events = self._load_lake()
        asof = panel["date"].max()
        self.bus.emit("worker", status="cycle_start", asof=str(asof.date()))

        if self.graph.memory is not None:
            filled = self.graph.memory.update_outcomes(panel, benchmark=self.cfg.benchmark)
            if filled:
                self.bus.emit("worker", status="outcomes_filled", count=filled)

        for ticker in self.cfg.tickers:
            if self._stop.is_set():
                return
            cached = _cached_decision(ticker, asof)
            if cached is not None:
                self._last_action.setdefault(ticker, cached.action)
                # replay the transcript so the floor shows what each agent said
                # even when the decision came from cache (e.g. after a restart)
                for key, text in cached.transcript.items():
                    agent = _agent_for_transcript_key(key)
                    if agent is not None and text:
                        self.bus.emit(
                            "stage",
                            agent=agent,
                            status="done",
                            ticker=ticker,
                            asof=cached.asof,
                            detail=text[:400],
                            cached=True,
                        )
                # replay so late-joining UIs still see today's board
                self.bus.emit(
                    "decision",
                    ticker=ticker,
                    asof=cached.asof,
                    action=cached.action,
                    conviction=cached.conviction,
                    verdict=cached.verdict,
                    target_weight=cached.target_weight,
                    trader_action=cached.trader_action,
                    cached=True,
                )
                continue
            decision = self.graph.propagate(
                ticker,
                asof,
                panel,
                macro=macro,
                fundamentals=fundamentals,
                events=events,
                prev_weight=latest_cached_weight(ticker, asof),
            )
            _cache_decision(decision)
            self._maybe_notify(decision, trigger=f"new session {decision.asof}")
        self.bus.emit("worker", status="cycle_done", asof=str(asof.date()))

    # ------------------------------------------------------------ live watch

    def watch_once(self, now: pd.Timestamp | None = None) -> None:
        quotes = fetch_quotes(self.cfg.tickers)
        if not quotes:
            return
        now = now if now is not None else pd.Timestamp.now(tz="America/New_York").tz_localize(None)
        self.bus.emit("quotes", changes={t: round(q["change_pct"], 2) for t, q in quotes.items()})
        lake_data = None
        for ticker, q in quotes.items():
            if self._stop.is_set():
                return
            if abs(q["change_pct"]) < self.cfg.watch_move_pct:
                continue
            last = self._last_revision.get(ticker)
            if last is not None and (now - last) < pd.Timedelta(
                minutes=self.cfg.revision_cooldown_minutes
            ):
                continue
            if lake_data is None:
                lake_data = self._load_lake()
            panel, macro, fundamentals, events = lake_data
            asof = panel["date"].max()
            reason = f"intraday move {q['change_pct']:+.1f}% vs yesterday's close"
            self.bus.emit("worker", status="reviewing", ticker=ticker, detail=reason)
            decision = self.graph.propagate(
                ticker,
                asof,
                panel,
                macro=macro,
                fundamentals=fundamentals,
                events=events,
                prev_weight=latest_cached_weight(ticker, asof + pd.Timedelta(days=1)),
                intraday=q,
                intraday_reason=reason,
            )
            _cache_revision(decision, tag=now.strftime("%H%M"))
            self._last_revision[ticker] = now
            self._maybe_notify(decision, trigger=reason)

    # ---------------------------------------------------------- notifications

    def _maybe_notify(self, decision: Decision, trigger: str) -> None:
        prev = self._last_action.get(decision.ticker)
        self._last_action[decision.ticker] = decision.action
        if decision.action not in ("BUY", "SELL"):
            return
        if prev == decision.action and not self.cfg.notify_all:
            return
        subtitle = f"{decision.action} {decision.ticker} ({decision.conviction})"
        message = (
            f"PM {decision.verdict.lower()}s | target weight {decision.target_weight:.0%} | "
            f"{trigger}"
        )
        self.bus.emit(
            "alert",
            ticker=decision.ticker,
            action=decision.action,
            conviction=decision.conviction,
            trigger=trigger,
        )
        if self.cfg.notify:
            mac_notify("TradingAgents Desk", subtitle, message)
