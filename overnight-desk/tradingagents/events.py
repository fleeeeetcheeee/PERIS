"""Thread-safe event bus: the worker publishes pipeline events, SSE clients and
the UI's initial /state call consume them.

Events are plain dicts with at least {type, ts}; stage events add
{stage, status, ticker, asof, detail}. A bounded ring buffer replays recent
history to newly connected clients so the UI can reconstruct the floor state.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque


class EventBus:
    def __init__(self, history: int = 400) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: deque[dict] = deque(maxlen=history)

    def emit(self, type: str, **fields) -> dict:
        event = {"type": type, "ts": time.time(), **fields}
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:  # slow client: drop for them, never block the worker
                pass
        return event

    def emit_dict(self, event: dict) -> dict:
        """Adapter for callers that already build the full event dict
        (TradingAgentsGraph.on_event passes {"type": ..., ...})."""
        e = dict(event)
        return self.emit(e.pop("type", "event"), **e)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._history)
