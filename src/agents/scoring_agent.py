from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import BaseAgent

DEFAULT_THESIS = """
Investment Thesis:
- Target sectors: B2B SaaS, healthcare tech, industrial automation, fintech
- Revenue range: $5M - $100M ARR
- Growth: >20% YoY preferred
- Geography: US and Western Europe
- Avoid: retail, restaurants, oil & gas, crypto
- Positive signals: recurring revenue, high gross margins (>60%), strong NRR
- Red flags: customer concentration >30%, declining revenue, regulatory risk
"""

SCORING_SYSTEM = """You are a private equity analyst scoring companies against an investment thesis.
Given a company profile and the investment thesis, return ONLY valid JSON with these fields:
{
  "score": <integer 0-100>,
  "rationale": "<2-3 sentence explanation>",
  "strengths": ["<strength1>", "<strength2>"],
  "risks": ["<risk1>", "<risk2>"],
  "recommended_action": "<one of: pass, watch, pursue>"
}
Do not include any text outside the JSON object."""


class ScoringAgent(BaseAgent):
    """Scores companies 0-100 against a configurable investment thesis."""

    def __init__(self, llm: Any | None = None, thesis: str | None = None) -> None:
        super().__init__(llm=llm)
        self.thesis = thesis or os.getenv("INVESTMENT_THESIS", DEFAULT_THESIS)
        self._chain = self.build_chain()

    def build_chain(self) -> Any:
        # Chain is just the LLM; prompting handled in run()
        return self.llm

    def run(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run scoring on input["company"]. Returns score dict."""
        company = input.get("company", input)
        return self.score_company(company)

    def score_company(self, company_profile: dict[str, Any]) -> dict[str, Any]:
        """Score a company dict against the investment thesis."""
        profile_text = json.dumps(company_profile, indent=2, default=str)
        user_prompt = (
            f"Investment Thesis:\n{self.thesis}\n\n"
            f"Company Profile:\n{profile_text}\n\n"
            "Score this company and return JSON only."
        )

        raw = self._invoke(SCORING_SYSTEM, user_prompt)
        return self._parse_score_response(raw, company_profile)

    def explain_score(self, scoring_result: dict[str, Any]) -> dict[str, Any]:
        """Generate a brief plain-English explanation of the score."""
        user_prompt = (
            f"Scoring result:\n{json.dumps(scoring_result, indent=2)}\n\n"
            "Write a 3-5 sentence plain-English investment summary for a GP memo."
        )
        explanation = self._invoke(
            "You are a private equity analyst writing concise deal memos.", user_prompt
        )
        return {**scoring_result, "explanation": explanation.strip()}

    def _parse_score_response(
        self, raw: str, company: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract JSON from LLM response, falling back to a default."""
        try:
            # Try to find a JSON block in the response
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                data = json.loads(match.group())
                score = int(data.get("score", 50))
                data["score"] = max(0, min(100, score))
                data["company_name"] = company.get("name", "Unknown")
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract a numeric score if JSON parsing fails
        nums = re.findall(r"\b([0-9]{1,3})\b", raw)
        score = int(nums[0]) if nums else 50
        return {
            "score": max(0, min(100, score)),
            "rationale": raw[:300],
            "strengths": [],
            "risks": [],
            "recommended_action": "watch",
            "company_name": company.get("name", "Unknown"),
        }
