"""Tests for the ScoringAgent and DiligenceAgent."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ScoringAgent
# ---------------------------------------------------------------------------

class TestScoringAgent:
    def _make_agent(self, llm_response: str):
        """Build a ScoringAgent with a mocked LLM."""
        from src.agents.scoring_agent import ScoringAgent

        mock_llm = MagicMock()
        agent = ScoringAgent.__new__(ScoringAgent)
        agent.llm = mock_llm
        agent.thesis = "Prefer B2B SaaS companies with >20% growth"
        agent._chain = mock_llm

        # Patch _invoke to return controlled text
        agent._invoke = MagicMock(return_value=llm_response)
        return agent

    def test_score_company_valid_json(self):
        from src.agents.scoring_agent import ScoringAgent

        response_json = json.dumps({
            "score": 78,
            "rationale": "Strong SaaS profile with recurring revenue.",
            "strengths": ["Recurring revenue", "High margins"],
            "risks": ["Customer concentration"],
            "recommended_action": "pursue",
        })
        agent = self._make_agent(response_json)
        result = agent.score_company({
            "name": "SaaSCo",
            "sector": "SaaS",
            "revenue_estimate": 25_000_000,
        })
        assert result["score"] == 78
        assert result["recommended_action"] == "pursue"
        assert result["company_name"] == "SaaSCo"

    def test_score_company_json_embedded_in_prose(self):
        from src.agents.scoring_agent import ScoringAgent

        prose = 'Based on the thesis, I give it: {"score": 65, "rationale": "OK fit", "strengths": [], "risks": [], "recommended_action": "watch"}'
        agent = self._make_agent(prose)
        result = agent.score_company({"name": "AverageCo"})
        assert result["score"] == 65

    def test_score_company_fallback_on_bad_json(self):
        from src.agents.scoring_agent import ScoringAgent

        agent = self._make_agent("The score is 42 out of 100.")
        result = agent.score_company({"name": "BrokenCo"})
        assert result["score"] == 42
        assert "recommended_action" in result

    def test_score_clamped_to_0_100(self):
        from src.agents.scoring_agent import ScoringAgent

        response_json = json.dumps({
            "score": 150,
            "rationale": "Off-the-charts",
            "strengths": [],
            "risks": [],
            "recommended_action": "pursue",
        })
        agent = self._make_agent(response_json)
        result = agent.score_company({"name": "OverflowCo"})
        assert result["score"] == 100

    def test_explain_score(self):
        from src.agents.scoring_agent import ScoringAgent

        agent = self._make_agent("This company is well-positioned for acquisition.")
        scoring = {
            "score": 80,
            "rationale": "Strong fit",
            "recommended_action": "pursue",
            "company_name": "FitCo",
        }
        result = agent.explain_score(scoring)
        assert "explanation" in result
        assert result["score"] == 80

    def test_run_delegates_to_score_company(self):
        from src.agents.scoring_agent import ScoringAgent

        response_json = json.dumps({
            "score": 55,
            "rationale": "Mid-tier",
            "strengths": [],
            "risks": [],
            "recommended_action": "watch",
        })
        agent = self._make_agent(response_json)
        result = agent.run({"company": {"name": "MidCo", "sector": "Manufacturing"}})
        assert result["score"] == 55


# ---------------------------------------------------------------------------
# DiligenceAgent
# ---------------------------------------------------------------------------

class TestDiligenceAgent:
    def _make_agent(self, llm_response: str):
        from src.agents.diligence_agent import DiligenceAgent

        agent = DiligenceAgent.__new__(DiligenceAgent)
        agent.llm = MagicMock()
        agent._invoke = MagicMock(return_value=llm_response)
        agent._chain = agent.llm
        return agent

    def test_generate_diligence_questions_json(self):
        from src.agents.diligence_agent import DiligenceAgent

        questions_json = json.dumps([
            "What is the customer concentration?",
            "What is the ARR growth rate?",
            "Who are the key competitors?",
        ])
        agent = self._make_agent(questions_json)
        result = agent.generate_diligence_questions({"name": "TestCo"})
        assert "questions" in result
        assert len(result["questions"]) == 3

    def test_generate_diligence_questions_fallback(self):
        from src.agents.diligence_agent import DiligenceAgent

        prose = "1. What is the churn rate?\n2. Who owns the IP?\n3. What is the sales cycle?"
        agent = self._make_agent(prose)
        result = agent.generate_diligence_questions({"name": "TestCo"})
        assert "questions" in result
        assert len(result["questions"]) >= 2

    def test_summarize_risks_json(self):
        from src.agents.diligence_agent import DiligenceAgent

        risks_json = json.dumps({
            "risks": [
                {"title": "Customer concentration", "severity": "high", "description": ">50% from one customer"},
                {"title": "Management depth", "severity": "medium", "description": "Thin team"},
            ]
        })
        agent = self._make_agent(risks_json)
        result = agent.summarize_risks([{"summary": "High customer concentration detected"}])
        assert len(result["risks"]) == 2
        assert result["risks"][0]["severity"] == "high"

    def test_summarize_risks_empty_input(self):
        from src.agents.diligence_agent import DiligenceAgent

        agent = self._make_agent("")
        result = agent.summarize_risks([])
        assert result == {"risks": []}

    def test_run_structure(self):
        from src.agents.diligence_agent import DiligenceAgent

        # First call: memo, second: questions JSON, third: risks JSON
        memo_text = "# Diligence Memo\n## Executive Summary\nStrong company."
        questions_json = json.dumps(["What is ARR?", "Who are the customers?"])
        risks_json = json.dumps({"risks": [{"title": "Key man risk", "severity": "high", "description": "CEO dependency"}]})

        agent = DiligenceAgent.__new__(DiligenceAgent)
        agent.llm = MagicMock()
        agent._chain = agent.llm
        call_count = [0]

        def side_effect(system, user):
            call_count[0] += 1
            if call_count[0] == 1:
                return memo_text
            elif call_count[0] == 2:
                return questions_json
            else:
                return risks_json

        agent._invoke = side_effect

        result = agent.run({
            "company": {"name": "DiligenceCo", "sector": "SaaS"},
            "signals": [{"signal_type": "ma_news", "summary": "Acquisition interest noted"}],
        })
        assert "memo" in result
        assert "questions" in result
        assert "risks" in result
        assert result["company_name"] == "DiligenceCo"
