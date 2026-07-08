"""Decision memory: the paper's reflection component, kept numeric-honest.

Every final decision is appended to artifacts/tradingagents/decisions.jsonl.
Once the outcome horizon has passed, `update_outcomes` fills in the realized
forward return and the alpha vs SPY — computed from the lake, never by the LLM.
`lessons` renders the most recent same-ticker outcomes (plus one cross-ticker
line) as the block injected into the portfolio manager prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from core import paths

logger = logging.getLogger(__name__)

MEMORY_DIR = paths.ARTIFACTS / "tradingagents"


class DecisionMemory:
    def __init__(self, path: Path | None = None, horizon_sessions: int = 5) -> None:
        self.path = path or MEMORY_DIR / "decisions.jsonl"
        self.horizon = horizon_sessions
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ io

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _save(self, rows: list[dict]) -> None:
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def record(self, decision) -> None:
        rows = self._load()
        entry = decision.to_dict()
        entry.pop("transcript", None)  # transcripts live in the eval cache, not memory
        # rerunning a cached day must not duplicate the memory row
        rows = [
            r for r in rows if not (r["ticker"] == entry["ticker"] and r["asof"] == entry["asof"])
        ]
        rows.append(entry)
        rows.sort(key=lambda r: (r["asof"], r["ticker"]))
        self._save(rows)

    # ------------------------------------------------------------ outcomes

    def update_outcomes(self, panel: pd.DataFrame, benchmark: str = "SPY") -> int:
        """Fill realized fwd return + alpha for decisions whose horizon has passed."""
        rows = self._load()
        pending = [r for r in rows if "fwd_return" not in r]
        if not pending:
            return 0
        close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
        filled = 0
        for r in pending:
            t, d = r["ticker"], pd.Timestamp(r["asof"])
            if t not in close.columns or d not in close.index:
                continue
            pos = close.index.get_loc(d)
            if pos + self.horizon >= len(close.index):
                continue  # horizon not yet elapsed
            fwd = float(close[t].iloc[pos + self.horizon] / close[t].iloc[pos] - 1)
            r["fwd_return"] = fwd
            if benchmark in close.columns:
                bench = float(
                    close[benchmark].iloc[pos + self.horizon] / close[benchmark].iloc[pos] - 1
                )
                r["alpha_vs_spy"] = fwd - bench
            filled += 1
        if filled:
            self._save(rows)
            logger.info("memory: filled %d realized outcomes", filled)
        return filled

    # ------------------------------------------------------------- lessons

    def lessons(self, ticker: str, k: int = 3) -> str:
        rows = [r for r in self._load() if "fwd_return" in r]
        if not rows:
            return ""
        lines: list[str] = []
        same = [r for r in rows if r["ticker"] == ticker][-k:]
        for r in same:
            alpha = f", alpha vs SPY {r['alpha_vs_spy'] * 100:+.1f}%" if "alpha_vs_spy" in r else ""
            lines.append(
                f"- {r['asof']} {r['ticker']}: decided {r['action']}/{r['conviction']}; "
                f"{self.horizon}-session return {r['fwd_return'] * 100:+.1f}%{alpha}"
            )
        other = [r for r in rows if r["ticker"] != ticker]
        if other:
            r = other[-1]
            alpha = f", alpha vs SPY {r['alpha_vs_spy'] * 100:+.1f}%" if "alpha_vs_spy" in r else ""
            lines.append(
                f"- (cross-ticker) {r['asof']} {r['ticker']}: {r['action']}/{r['conviction']}; "
                f"return {r['fwd_return'] * 100:+.1f}%{alpha}"
            )
        return "\n".join(lines)
