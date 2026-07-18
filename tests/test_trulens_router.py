"""
Tests for src/api/routers/trulens.py — the manually-triggered TruLens
capture tab. Mocks run_with_trulens/build_demo_payload entirely; no live
OpenAI/TruLens session work happens here (that's covered by
tests/test_trulens_wrapper.py and tests/test_trulens_cli.py).
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agents.state import GlobalState, RiskClassificationResult
from src.api.main import app
from src.api.routers import trulens as trulens_router

client = TestClient(app)


def _fake_final_state(risk_label: str = "HIGH") -> GlobalState:
    return GlobalState(
        risk_classification=RiskClassificationResult(
            mode="live", composite_score=0.5,
            geo_component=0.4, supply_component=0.5, freight_component=0.4, defect_component=0.3,
            duration_days=None, base_label=risk_label, final_label=risk_label,
            escalated=False, rationale="test", critical_flag=risk_label == "CRITICAL",
        ),
    )


def setup_function():
    trulens_router._RUNS.clear()


def _fake_run_with_trulens(payload, capture=None):
    if capture is not None:
        capture["node_latencies_ms"] = {"L1": 100.0, "L4": 250.0, "total": 350.0}
        capture["cost_summary"] = {
            "prompt_tokens": 120, "completion_tokens": 40, "cost_usd": 0.0021, "models": ["gpt-4o"],
        }
    return _fake_final_state()


def test_run_then_status_reports_complete_with_captured_metrics():
    with patch("src.api.routers.trulens.build_demo_payload", return_value={"run_id": "x"}):
        with patch("src.api.routers.trulens.run_with_trulens", side_effect=_fake_run_with_trulens):
            resp = client.post("/api/trulens/run", json={"demo_scenario_id": "taiwan_earthquake"})
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]

            status_resp = client.get(f"/api/trulens/status/{run_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "complete"
    assert body["risk_label"] == "HIGH"
    assert body["cost_summary"]["cost_usd"] == 0.0021
    assert body["node_latencies_ms"]["total"] == 350.0
    assert 0.0 <= body["node_latency_check"] <= 1.0


def test_status_unknown_run_id_404s():
    resp = client.get("/api/trulens/status/does-not-exist")
    assert resp.status_code == 404


def test_run_reports_failed_status_when_capture_raises():
    with patch("src.api.routers.trulens.build_demo_payload", return_value={"run_id": "x"}):
        with patch("src.api.routers.trulens.run_with_trulens", side_effect=RuntimeError("boom")):
            resp = client.post("/api/trulens/run", json={"demo_scenario_id": "red_sea_crisis"})
            run_id = resp.json()["run_id"]
            status_resp = client.get(f"/api/trulens/status/{run_id}")
    assert status_resp.json()["status"] == "failed"
    assert "boom" in status_resp.json()["error"]


def test_run_rejects_unknown_demo_scenario_id():
    resp = client.post("/api/trulens/run", json={"demo_scenario_id": "not_a_real_scenario"})
    assert resp.status_code == 422


def test_metrics_reads_recent_composite_scores(monkeypatch):
    monkeypatch.setattr(
        "src.api.routers.trulens.fetch_recent_composite_scores",
        lambda days: [0.5, 0.55, 0.52],
    )
    resp = client.get("/api/trulens/metrics?days=7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["days"] == 7
    assert body["n_runs"] == 3
    assert 0.0 <= body["risk_score_stability"] <= 1.0
