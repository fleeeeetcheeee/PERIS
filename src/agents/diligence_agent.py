from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .base import BaseAgent

DILIGENCE_SYSTEM = """You are a senior private equity due diligence analyst.
Given company data and signals, produce a structured diligence memo in Markdown.
Structure it as:
# Diligence Memo: {company_name}

## Executive Summary
## Business Overview
## Key Investment Highlights
## Risk Factors
## Key Diligence Questions
## Recommended Next Steps

Be concise, data-driven, and focus on material risks and opportunities."""

QUESTION_SYSTEM = """You are a PE analyst generating due diligence questions.
Return ONLY a JSON array of strings, each a specific diligence question.
Example: ["What is the revenue breakdown by customer?", "..."]"""

RISKS_SYSTEM = """You are a PE risk analyst.
Return ONLY valid JSON: {"risks": [{"title": "...", "severity": "high|medium|low", "description": "..."}]}"""


class DiligenceAgent(BaseAgent):
    """Pulls all signals for a company and produces a structured diligence memo."""

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__(llm=llm)
        self._chain = self.build_chain()

    def build_chain(self) -> Any:
        return self.llm

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """
        Expected input keys: company (dict), signals (list[dict])
        Returns: {memo: str, questions: list[str], risks: list[dict]}
        """
        company = input.get("company", {})
        signals = input.get("signals", [])

        memo = self._generate_memo(company, signals)
        questions = self.generate_diligence_questions(company)
        risks = self.summarize_risks(signals)

        return {
            "company_name": company.get("name", "Unknown"),
            "memo": memo,
            "questions": questions.get("questions", []),
            "risks": risks.get("risks", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_memo(
        self, company: dict[str, Any], signals: list[dict[str, Any]]
    ) -> str:
        context = json.dumps(
            {"company": company, "signals": signals[:20]}, indent=2, default=str
        )
        user_prompt = (
            f"Company name: {company.get('name', 'Unknown')}\n\n"
            f"Context data:\n{context}\n\n"
            "Generate the diligence memo now."
        )
        return self._invoke(DILIGENCE_SYSTEM, user_prompt).strip()

    def generate_diligence_questions(
        self, company_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate diligence questions for a company."""
        user_prompt = (
            f"Company: {json.dumps(company_data, indent=2, default=str)}\n\n"
            "Generate 10-15 specific due diligence questions."
        )
        raw = self._invoke(QUESTION_SYSTEM, user_prompt)

        try:
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                questions = json.loads(match.group())
                return {"questions": questions}
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split by newlines/numbers
        lines = [
            re.sub(r"^\s*[\d\.\-\*]+\s*", "", line).strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return {"questions": lines[:15]}

    def summarize_risks(
        self, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Extract material risks from a list of signals/findings."""
        if not findings:
            return {"risks": []}

        user_prompt = (
            f"Signals and findings:\n{json.dumps(findings[:20], indent=2, default=str)}\n\n"
            "Identify the top material risks. Return JSON only."
        )
        raw = self._invoke(RISKS_SYSTEM, user_prompt)

        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

        return {"risks": [{"title": "Review required", "severity": "medium", "description": raw[:200]}]}
