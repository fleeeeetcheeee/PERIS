from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .base import BaseAgent

SIGNAL_SYSTEM = """You are a portfolio monitoring analyst for a PE fund.
Given company data, KPIs, and recent signals, identify monitoring alerts.
Return ONLY valid JSON:
{
  "alerts": [
    {
      "type": "price|news|filing|kpi",
      "severity": "high|medium|low",
      "title": "...",
      "description": "...",
      "action_required": true|false
    }
  ]
}"""

ALERT_SYSTEM = """You are a PE portfolio manager reviewing monitoring alerts.
Given a list of alerts, write a concise daily briefing paragraph summarizing
what requires immediate attention and what can wait. Be direct and actionable."""

PRICE_CHANGE_THRESHOLD = 0.05  # 5% move triggers an alert


class MonitoringAgent(BaseAgent):
    """Daily portfolio monitoring: price moves, news signals, 8-K filings."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm=llm)
        self._chain = self.build_chain()

    def build_chain(self) -> Any:
        return self.llm

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        Expected input: {company: dict, kpis: list, signals: list, price_data: dict}
        Returns: {alerts: list, briefing: str, generated_at: str}
        """
        company = input.get("company", {})
        signals = input.get("signals", [])
        kpis = input.get("kpis", [])
        price_data = input.get("price_data", {})

        # Rule-based checks first (fast, deterministic)
        rule_alerts = self._rule_based_checks(company, kpis, signals, price_data)

        # LLM-based signal detection for nuanced issues
        llm_result = self.detect_signals({
            "company": company,
            "kpis": kpis,
            "signals": signals[:15],
            "price_data": price_data,
        })
        llm_alerts = llm_result.get("alerts", [])

        all_alerts = rule_alerts + llm_alerts
        briefing = self.generate_alerts(all_alerts)

        return {
            "company_name": company.get("name", "Unknown"),
            "alerts": all_alerts,
            "alert_count": len(all_alerts),
            "high_severity_count": sum(1 for a in all_alerts if a.get("severity") == "high"),
            "briefing": briefing.get("briefing", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _rule_based_checks(
        self,
        company: dict[str, Any],
        kpis: list[dict[str, Any]],
        signals: list[dict[str, Any]],
        price_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []

        # Price move check
        price = price_data.get("price")
        prev_price = price_data.get("prev_close")
        if price and prev_price and prev_price > 0:
            change_pct = abs(price - prev_price) / prev_price
            if change_pct >= PRICE_CHANGE_THRESHOLD:
                direction = "up" if price > prev_price else "down"
                alerts.append({
                    "type": "price",
                    "severity": "high" if change_pct >= 0.10 else "medium",
                    "title": f"Price moved {direction} {change_pct:.1%}",
                    "description": f"{company.get('name')} price {direction} {change_pct:.1%} to {price}",
                    "action_required": change_pct >= 0.10,
                })

        # 8-K filing check
        for signal in signals:
            if signal.get("signal_type") == "sec_8k":
                alerts.append({
                    "type": "filing",
                    "severity": "medium",
                    "title": "New 8-K filing detected",
                    "description": signal.get("summary", "")[:300],
                    "action_required": False,
                })

        # KPI deviation check
        if kpis:
            latest = kpis[-1] if kpis else {}
            for kpi in kpis[-3:]:
                if kpi.get("metric_name") == "revenue_growth" and kpi.get("value", 0) < -0.1:
                    alerts.append({
                        "type": "kpi",
                        "severity": "high",
                        "title": "Revenue growth turned negative",
                        "description": f"Revenue growth: {kpi['value']:.1%}",
                        "action_required": True,
                    })

        return alerts

    def detect_signals(self, company_data: dict[str, Any]) -> dict[str, Any]:
        """LLM-based detection of monitoring signals."""
        user_prompt = (
            f"Portfolio company data:\n{json.dumps(company_data, indent=2, default=str)}\n\n"
            "Identify monitoring alerts. Return JSON only."
        )
        raw = self._invoke(SIGNAL_SYSTEM, user_prompt)

        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

        return {"alerts": []}

    def generate_alerts(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a human-readable briefing from a list of alerts."""
        if not signals:
            return {"briefing": "No monitoring alerts detected."}

        user_prompt = (
            f"Today's alerts:\n{json.dumps(signals, indent=2, default=str)}\n\n"
            "Write the daily briefing paragraph now."
        )
        briefing = self._invoke(ALERT_SYSTEM, user_prompt)
        return {"briefing": briefing.strip()}
