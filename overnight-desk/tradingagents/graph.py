"""Orchestration of one trading decision.

Plain sequential Python instead of LangGraph (not a repo dependency; the paper's
graph is a fixed pipeline, so a framework buys nothing here):

    4 analysts (quick LLM, parallel-safe but run serially against local Ollama)
    -> bull/bear debate, debate_rounds each (quick)
    -> research manager verdict (deep)
    -> trader proposal (deep)
    -> risk debate: aggressive/neutral/conservative x risk_rounds (quick)
    -> portfolio manager final decision with memory lessons (deep)

Every intermediate output lands in Decision.transcript for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tradingagents import roles
from tradingagents.config import TradingAgentsConfig
from tradingagents.snapshot import Snapshot, build_snapshot

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    ticker: str
    asof: str
    action: str  # BUY / HOLD / SELL (portfolio manager's final word)
    conviction: str  # LOW / MEDIUM / HIGH
    verdict: str  # APPROVE / ADJUST / REJECT
    target_weight: float  # long-only fraction of the per-ticker book
    trader_action: str
    trader_conviction: str
    transcript: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "asof": self.asof,
            "action": self.action,
            "conviction": self.conviction,
            "verdict": self.verdict,
            "target_weight": self.target_weight,
            "trader_action": self.trader_action,
            "trader_conviction": self.trader_conviction,
            "transcript": self.transcript,
        }


def target_weight(action: str, conviction: str, cfg: TradingAgentsConfig, prev: float) -> float:
    if action == "BUY":
        return cfg.conviction_weights.get(conviction, 0.33)
    if action == "SELL":
        return 0.0
    return prev  # HOLD carries the existing position


class TradingAgentsGraph:
    def __init__(
        self, cfg: TradingAgentsConfig, quick_llm, deep_llm, memory=None, on_event=None
    ) -> None:
        self.cfg = cfg
        self.quick = quick_llm
        self.deep = deep_llm
        self.memory = memory
        self.on_event = on_event  # callable(dict) -> None; UI/telemetry hook

    def _emit(self, agent: str, status: str, snap: Snapshot, detail: str = "") -> None:
        if self.on_event is None:
            return
        self.on_event(
            {
                "type": "stage",
                "agent": agent,
                "status": status,
                "ticker": snap.ticker,
                "asof": str(snap.asof.date()),
                "detail": detail[:400],
            }
        )

    def _run(self, llm, agent: str, snap: Snapshot, system: str, prompt: str) -> str:
        self._emit(agent, "start", snap)
        out = llm.generate(system, prompt)
        self._emit(agent, "done", snap, detail=out)
        return out

    # ---------------------------------------------------------------- stages

    def _analysts(self, snap: Snapshot, transcript: dict[str, str]) -> str:
        reports = []
        for role, (system, block) in roles.ANALYST_ROLES.items():
            prompt = roles.ANALYST_TEMPLATE.format(
                ticker=snap.ticker, asof=snap.asof.date(), block=getattr(snap, block)
            )
            out = self._run(self.quick, role, snap, system + " " + roles.GROUNDING, prompt)
            transcript[f"analyst_{role}"] = out
            reports.append(f"[{role.upper()} ANALYST — signal {roles.parse_signal(out)}]\n{out}")
        return "\n\n".join(reports)

    def _research_debate(self, snap: Snapshot, reports: str, transcript: dict[str, str]) -> str:
        debate: list[str] = []
        for rnd in range(1, self.cfg.debate_rounds + 1):
            for side in ("bull", "bear"):
                prompt = roles.RESEARCHER_TEMPLATE.format(
                    ticker=snap.ticker,
                    asof=snap.asof.date(),
                    reports=reports,
                    debate="\n\n".join(debate) or "(debate opens)",
                    round=rnd,
                    rounds=self.cfg.debate_rounds,
                )
                out = self._run(self.quick, side, snap, roles.RESEARCHER_SYSTEM[side], prompt)
                transcript[f"{side}_round{rnd}"] = out
                debate.append(f"[{side.upper()} r{rnd}] {out}")
        return "\n\n".join(debate)

    def _manager(self, snap: Snapshot, reports: str, debate: str, transcript: dict) -> str:
        out = self._run(
            self.deep,
            "manager",
            snap,
            roles.MANAGER_SYSTEM,
            roles.MANAGER_TEMPLATE.format(
                ticker=snap.ticker, asof=snap.asof.date(), reports=reports, debate=debate
            ),
        )
        transcript["research_manager"] = out
        return out

    def _trader(self, snap: Snapshot, plan: str, position: str, transcript: dict) -> str:
        out = self._run(
            self.deep,
            "trader",
            snap,
            roles.TRADER_SYSTEM,
            roles.TRADER_TEMPLATE.format(
                ticker=snap.ticker, asof=snap.asof.date(), plan=plan, position=position
            ),
        )
        transcript["trader"] = out
        return out

    def _risk_debate(self, snap: Snapshot, proposal: str, transcript: dict) -> str:
        debate: list[str] = []
        for rnd in range(1, self.cfg.risk_rounds + 1):
            for stance in ("aggressive", "conservative", "neutral"):
                prompt = roles.RISK_TEMPLATE.format(
                    ticker=snap.ticker,
                    asof=snap.asof.date(),
                    proposal=proposal,
                    macro=snap.macro,
                    technical=snap.technical,
                    debate="\n\n".join(debate) or "(discussion opens)",
                    round=rnd,
                    rounds=self.cfg.risk_rounds,
                )
                out = self._run(
                    self.quick, f"risk_{stance}", snap, roles.RISK_SYSTEM[stance], prompt
                )
                transcript[f"risk_{stance}_round{rnd}"] = out
                debate.append(f"[{stance.upper()} r{rnd}] {out}")
        return "\n\n".join(debate)

    def _portfolio_manager(
        self, snap: Snapshot, proposal: str, risk_debate: str, transcript: dict
    ) -> str:
        lessons = "(no recorded past decisions)"
        if self.memory is not None:
            lessons = self.memory.lessons(snap.ticker) or lessons
        out = self._run(
            self.deep,
            "pm",
            snap,
            roles.PM_SYSTEM,
            roles.PM_TEMPLATE.format(
                ticker=snap.ticker,
                asof=snap.asof.date(),
                proposal=proposal,
                debate=risk_debate,
                lessons=lessons,
            ),
        )
        transcript["portfolio_manager"] = out
        return out

    # ------------------------------------------------------------- pipeline

    def propagate(
        self,
        ticker: str,
        asof: str | pd.Timestamp,
        panel: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
        prev_weight: float = 0.0,
        intraday: dict | None = None,
        intraday_reason: str = "",
    ) -> Decision:
        snap = build_snapshot(
            ticker,
            asof,
            panel,
            macro=macro,
            fundamentals=fundamentals,
            events=events,
            benchmark=self.cfg.benchmark,
            intraday=intraday,
            intraday_reason=intraday_reason,
        )
        transcript: dict[str, str] = {}
        reports = self._analysts(snap, transcript)
        debate = self._research_debate(snap, reports, transcript)
        plan = self._manager(snap, reports, debate, transcript)
        position = f"{prev_weight:.0%} of the per-ticker book" if prev_weight else "flat"
        proposal = self._trader(snap, plan, position, transcript)
        trader_action = roles.parse_decision(proposal)
        trader_conviction = roles.parse_conviction(proposal)
        risk = self._risk_debate(snap, proposal, transcript)
        final = self._portfolio_manager(snap, proposal, risk, transcript)

        verdict = roles.parse_verdict(final)
        if verdict == "REJECT":
            action, conviction = "HOLD", "LOW"
        else:
            action = roles.parse_decision(final, tag="FINAL DECISION")
            conviction = roles.parse_conviction(final, tag="FINAL CONVICTION")
        decision = Decision(
            ticker=ticker,
            asof=str(pd.Timestamp(asof).date()),
            action=action,
            conviction=conviction,
            verdict=verdict,
            target_weight=target_weight(action, conviction, self.cfg, prev_weight),
            trader_action=trader_action,
            trader_conviction=trader_conviction,
            transcript=transcript,
        )
        logger.info(
            "%s %s: %s/%s (trader said %s/%s, verdict %s) -> weight %.2f",
            ticker,
            decision.asof,
            action,
            conviction,
            trader_action,
            trader_conviction,
            verdict,
            decision.target_weight,
        )
        if self.on_event is not None:
            self.on_event(
                {
                    "type": "decision",
                    "ticker": ticker,
                    "asof": decision.asof,
                    "action": action,
                    "conviction": conviction,
                    "verdict": verdict,
                    "target_weight": decision.target_weight,
                    "trader_action": trader_action,
                }
            )
        if self.memory is not None:
            self.memory.record(decision)
        return decision
