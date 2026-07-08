"""Desk server: pixel-floor UI + live event stream + always-on worker.

    uv run python -m tradingagents.app          # http://localhost:8102

Endpoints:
- GET /        the pixel trading floor (self-contained HTML, no CDN)
- GET /events  Server-Sent Events: stage/decision/worker events as they happen
- GET /state   recent event history + today's decision board + memory stats

The DeskWorker starts with the app and keeps deciding in the background
(set TRADINGAGENTS_NO_WORKER=1 to serve the UI without it — used by tests).
"""

from __future__ import annotations

import json
import logging
import os
import queue
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from tradingagents.config import TradingAgentsConfig, load_ta_config
from tradingagents.evaluate import CACHE_DIR
from tradingagents.events import EventBus
from tradingagents.memory import DecisionMemory

logger = logging.getLogger(__name__)

UI_INDEX = Path(__file__).parent / "ui" / "index.html"

SSE_HEARTBEAT_SECONDS = 15.0


def decision_board() -> list[dict]:
    """Most recent cached decision per ticker. Filenames sort chronologically and
    intraday revisions ({t}_{date}_rHHMM.json) sort after the daily file for the
    same date, so with >= the newest stance — revision included — wins."""
    latest: dict[str, dict] = {}
    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.glob("*_*.json")):
            d = json.loads(p.read_text())
            t = d.get("ticker")
            if t and (t not in latest or d["asof"] >= latest[t]["asof"]):
                latest[t] = {k: v for k, v in d.items() if k != "transcript"}
    return sorted(latest.values(), key=lambda d: d["ticker"])


def memory_stats(memory: DecisionMemory) -> dict:
    rows = [r for r in memory._load() if "fwd_return" in r]
    if not rows:
        return {"decided": 0}
    alphas = [r["alpha_vs_spy"] for r in rows if "alpha_vs_spy" in r]
    return {
        "decided": len(rows),
        "positive_alpha_share": round(sum(a > 0 for a in alphas) / len(alphas), 3)
        if alphas
        else None,
        "mean_alpha_bps": round(sum(alphas) / len(alphas) * 1e4, 1) if alphas else None,
    }


def create_app(cfg: TradingAgentsConfig | None = None, worker: bool | None = None) -> FastAPI:
    cfg = cfg or load_ta_config()
    if worker is None:
        worker = os.getenv("TRADINGAGENTS_NO_WORKER", "") != "1"
    bus = EventBus()
    memory = DecisionMemory(horizon_sessions=cfg.horizon_sessions)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        desk = None
        if worker:
            from tradingagents.graph import TradingAgentsGraph
            from tradingagents.llm import OllamaLLM
            from tradingagents.worker import DeskWorker

            quick = OllamaLLM(cfg.quick_model, cfg.temperature, cfg.llm_timeout)
            deep = OllamaLLM(cfg.deep_model, cfg.temperature, cfg.llm_timeout)
            graph = TradingAgentsGraph(cfg, quick, deep, memory=memory, on_event=bus.emit_dict)
            desk = DeskWorker(cfg, graph, bus)
            desk.start()
            logger.info("desk worker started (poll every %d min)", cfg.worker_poll_minutes)
        yield
        if desk is not None:
            desk.stop()

    app = FastAPI(title="TradingAgents Desk", lifespan=lifespan)
    app.state.bus = bus

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI_INDEX.read_text()

    @app.get("/state")
    def state() -> dict:
        return {
            "tickers": cfg.tickers,
            "events": bus.recent(),
            "board": decision_board(),
            "memory": memory_stats(memory),
        }

    @app.get("/events")
    async def events() -> StreamingResponse:
        q = bus.subscribe()

        async def stream():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        event = await anyio.to_thread.run_sync(
                            lambda: q.get(timeout=SSE_HEARTBEAT_SECONDS)
                        )
                        yield f"data: {json.dumps(event)}\n\n"
                    except queue.Empty:
                        yield ": heartbeat\n\n"
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_ta_config()
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.server_port, log_level="info")


if __name__ == "__main__":
    main()
