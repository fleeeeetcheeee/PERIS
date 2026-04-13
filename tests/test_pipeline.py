"""Tests for the CRM pipeline: DB CRUD, FastAPI routes, and MonitoringAgent."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# DB / Queries
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session(tmp_path):
    """Provide a fresh in-memory SQLite session per test."""
    import os
    os.environ["DATABASE_URL"] = "sqlite://"  # in-memory

    # Re-import to pick up env var
    import importlib
    import src.db.schema as schema_mod
    importlib.reload(schema_mod)

    schema_mod.init_db()
    session = schema_mod.SessionLocal()
    yield session
    session.close()


class TestCompanyCRUD:
    def test_create_and_get(self, db_session):
        from src.db.queries import create_company, get_company

        company = create_company(
            db_session,
            name="Pipeline Corp",
            sector="SaaS",
            country="US",
            score=72.0,
        )
        assert company.id is not None
        assert company.name == "Pipeline Corp"

        fetched = get_company(db_session, company.id)
        assert fetched is not None
        assert fetched.score == 72.0

    def test_list_companies(self, db_session):
        from src.db.queries import create_company, list_companies

        for i in range(5):
            create_company(db_session, name=f"Company {i}", source="test")

        companies = list_companies(db_session, limit=3)
        assert len(companies) == 3

    def test_update_company(self, db_session):
        from src.db.queries import create_company, update_company

        company = create_company(db_session, name="OldName")
        updated = update_company(db_session, company.id, name="NewName", score=85.0)
        assert updated.name == "NewName"
        assert updated.score == 85.0

    def test_delete_company(self, db_session):
        from src.db.queries import create_company, delete_company, get_company

        company = create_company(db_session, name="ToDelete")
        assert delete_company(db_session, company.id) is True
        assert get_company(db_session, company.id) is None

    def test_delete_nonexistent(self, db_session):
        from src.db.queries import delete_company

        assert delete_company(db_session, 99999) is False


class TestPipelineStageCRUD:
    def test_create_and_list(self, db_session):
        from src.db.queries import create_company, create_pipeline_stage, list_pipeline_stages

        company = create_company(db_session, name="DealCo")
        stage = create_pipeline_stage(
            db_session,
            company_id=company.id,
            stage="initial_screen",
            owner="alice",
        )
        assert stage.stage == "initial_screen"

        stages = list_pipeline_stages(db_session, company_id=company.id)
        assert len(stages) == 1
        assert stages[0].owner == "alice"

    def test_cascade_delete(self, db_session):
        from src.db.queries import create_company, create_pipeline_stage, delete_company, list_pipeline_stages

        company = create_company(db_session, name="CascadeCo")
        create_pipeline_stage(db_session, company_id=company.id, stage="diligence")
        create_pipeline_stage(db_session, company_id=company.id, stage="loi")

        delete_company(db_session, company.id)
        stages = list_pipeline_stages(db_session, company_id=company.id)
        assert len(stages) == 0


class TestSignalCRUD:
    def test_create_and_retrieve(self, db_session):
        from src.db.queries import create_company, create_signal, get_signal

        company = create_company(db_session, name="SignalCo")
        signal = create_signal(
            db_session,
            company_id=company.id,
            signal_type="ma_news",
            summary="Acquisition announced",
            raw_data={"source": "reuters"},
            confidence=0.85,
        )
        assert signal.id is not None

        fetched = get_signal(db_session, signal.id)
        assert fetched.signal_type == "ma_news"
        assert fetched.confidence == 0.85
        assert fetched.raw_data == {"source": "reuters"}

    def test_list_signals_by_company(self, db_session):
        from src.db.queries import create_company, create_signal, list_signals

        co1 = create_company(db_session, name="Co1")
        co2 = create_company(db_session, name="Co2")

        create_signal(db_session, company_id=co1.id, signal_type="news", summary="s1")
        create_signal(db_session, company_id=co1.id, signal_type="news", summary="s2")
        create_signal(db_session, company_id=co2.id, signal_type="news", summary="s3")

        co1_signals = list_signals(db_session, company_id=co1.id)
        assert len(co1_signals) == 2

        all_signals = list_signals(db_session)
        assert len(all_signals) == 3


# ---------------------------------------------------------------------------
# FastAPI routes (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """TestClient with an isolated in-memory DB engine injected via dependency override."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from src.db.schema import Base
    from src.api.main import app
    from src.api.routes import companies, pipeline, portfolio, signals

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def override_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[companies.get_db] = override_db
    app.dependency_overrides[pipeline.get_db] = override_db
    app.dependency_overrides[portfolio.get_db] = override_db
    app.dependency_overrides[signals.get_db] = override_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


class TestCompaniesAPI:
    def test_list_empty(self, api_client):
        resp = api_client.get("/companies/")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_create_and_get(self, api_client):
        payload = {"name": "API Corp", "sector": "SaaS", "country": "US"}
        create_resp = api_client.post("/companies/", json=payload)
        assert create_resp.status_code == 201
        data = create_resp.json()
        assert data["name"] == "API Corp"
        company_id = data["id"]

        get_resp = api_client.get(f"/companies/{company_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["sector"] == "SaaS"

    def test_get_not_found(self, api_client):
        resp = api_client.get("/companies/99999")
        assert resp.status_code == 404

    def test_update_company(self, api_client):
        create_resp = api_client.post("/companies/", json={"name": "BeforeUpdate"})
        cid = create_resp.json()["id"]

        update_resp = api_client.patch(f"/companies/{cid}", json={"score": 88.0})
        assert update_resp.status_code == 200
        assert update_resp.json()["score"] == 88.0

    def test_delete_company(self, api_client):
        create_resp = api_client.post("/companies/", json={"name": "ToDelete"})
        cid = create_resp.json()["id"]

        del_resp = api_client.delete(f"/companies/{cid}")
        assert del_resp.status_code == 204

        get_resp = api_client.get(f"/companies/{cid}")
        assert get_resp.status_code == 404


class TestPipelineAPI:
    def test_create_pipeline_stage(self, api_client):
        # Create a company first
        co = api_client.post("/companies/", json={"name": "DealFlow"}).json()
        payload = {"company_id": co["id"], "stage": "initial_screen", "owner": "bob"}
        resp = api_client.post("/pipeline/", json=payload)
        assert resp.status_code == 201
        assert resp.json()["stage"] == "initial_screen"

    def test_list_pipeline_stages(self, api_client):
        co = api_client.post("/companies/", json={"name": "FunnelCo"}).json()
        for stage in ("sourced", "screen", "loi"):
            api_client.post("/pipeline/", json={"company_id": co["id"], "stage": stage})

        resp = api_client.get(f"/pipeline/?company_id={co['id']}")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3


class TestSignalsAPI:
    def test_create_and_list_signals(self, api_client):
        co = api_client.post("/companies/", json={"name": "SigCo"}).json()
        payload = {
            "company_id": co["id"],
            "signal_type": "ma_news",
            "summary": "M&A activity detected",
            "confidence": 0.75,
        }
        resp = api_client.post("/signals/", json=payload)
        assert resp.status_code == 201
        assert resp.json()["signal_type"] == "ma_news"

        list_resp = api_client.get(f"/signals/?company_id={co['id']}")
        assert list_resp.json()["count"] == 1


# ---------------------------------------------------------------------------
# MonitoringAgent (rule-based checks only — no LLM calls)
# ---------------------------------------------------------------------------

class TestMonitoringAgent:
    def _make_agent(self):
        from src.agents.monitoring_agent import MonitoringAgent

        agent = MonitoringAgent.__new__(MonitoringAgent)
        agent.llm = MagicMock()
        agent._chain = agent.llm
        agent._invoke = MagicMock(return_value='{"alerts": []}')
        return agent

    def test_price_alert_triggered(self):
        from src.agents.monitoring_agent import MonitoringAgent

        agent = self._make_agent()
        alerts = agent._rule_based_checks(
            company={"name": "PriceCo"},
            kpis=[],
            signals=[],
            price_data={"price": 110.0, "prev_close": 100.0},
        )
        assert any(a["type"] == "price" for a in alerts)
        price_alert = next(a for a in alerts if a["type"] == "price")
        assert "up" in price_alert["title"]

    def test_no_alert_for_small_move(self):
        from src.agents.monitoring_agent import MonitoringAgent

        agent = self._make_agent()
        alerts = agent._rule_based_checks(
            company={"name": "StableCo"},
            kpis=[],
            signals=[],
            price_data={"price": 102.0, "prev_close": 100.0},
        )
        assert not any(a["type"] == "price" for a in alerts)

    def test_8k_signal_triggers_alert(self):
        from src.agents.monitoring_agent import MonitoringAgent

        agent = self._make_agent()
        alerts = agent._rule_based_checks(
            company={"name": "FilingCo"},
            kpis=[],
            signals=[{"signal_type": "sec_8k", "summary": "8-K filed: material event"}],
            price_data={},
        )
        assert any(a["type"] == "filing" for a in alerts)

    def test_generate_alerts_empty(self):
        from src.agents.monitoring_agent import MonitoringAgent

        agent = self._make_agent()
        agent._invoke = MagicMock(return_value="No monitoring alerts detected.")
        result = agent.generate_alerts([])
        assert "No monitoring" in result["briefing"]
