"""
Day 9 — POST /api/pipeline/run + GET /api/pipeline/status, and the
snapshot_run_outputs() bridge that writes L4/L6/L7 into run_id-keyed
dashboard tables. No live OpenAI/LangGraph calls: run_pipeline() is
mocked everywhere except test_run_calls_run_pipeline_not_run_agent_graph,
which exists specifically to assert it (not run_agent_graph) is the
BackgroundTask target.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agents.pipeline_bridge import snapshot_run_outputs
from src.agents.state import (
    GlobalState,
    MitigationAction,
    RiskClassificationResult,
    SimulationResult,
)
from src.api.main import app

client = TestClient(app)


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)
    from src.utils.db_utils import ensure_schema

    ensure_schema()
    yield db_path


def _fake_final_state(run_id: str) -> GlobalState:
    return GlobalState(
        run_id=run_id,
        active_record={"port": "Eastern Asia", "demand": 500},
        risk_classification=RiskClassificationResult(
            mode="live",
            composite_score=0.82,
            geo_component=0.5,
            supply_component=0.5,
            freight_component=0.5,
            defect_component=0.1,
            duration_days=14,
            base_label="HIGH",
            final_label="CRITICAL",
            escalated=True,
            rag_citations=[],
            rationale="test",
            critical_flag=True,
        ),
        simulation_result=SimulationResult(
            stockout_probability_pct=42.0,
            expected_inventory_gap_pct=10.0,
            alternate_route="Cape of Good Hope",
        ),
        mitigation_action=MitigationAction(
            summary="test",
            recommendations=["Reroute via alternate port"],
            cost_delta="180000",
            urgency="CRITICAL",
        ),
        forecast_result=None,  # L5 skipped — expected, not an error
    )


def test_run_requires_scenario_id_in_demo_mode():
    resp = client.post("/api/pipeline/run", json={"mode": "demo"})
    assert resp.status_code == 422


def test_run_requires_replay_run_id_in_replay_mode():
    resp = client.post("/api/pipeline/run", json={"mode": "replay"})
    assert resp.status_code == 422


def test_replay_mode_short_circuits_no_background_task(seeded_db):
    from src.utils.db_utils import execute_non_query

    execute_non_query(
        "INSERT INTO agent_execution_log (run_id, agent_name, status, started_at, completed_at) "
        "VALUES ('replay-run-1', 'L7_mitigation', 'Complete', 't0', 't1')"
    )
    with patch("src.api.routers.pipeline.run_pipeline") as mock_run:
        resp = client.post(
            "/api/pipeline/run", json={"mode": "replay", "replay_run_id": "replay-run-1"}
        )
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "replay-run-1"
    mock_run.assert_not_called()


def test_run_calls_run_pipeline_not_run_agent_graph(seeded_db):
    with patch("src.api.routers.pipeline.run_pipeline") as mock_seq, patch(
        "src.agents.langgraph_engine.run_agent_graph"
    ) as mock_graph, patch(
        "src.api.routers.pipeline.build_demo_payload", return_value={"run_id": "x", "mode": "demo"}
    ):
        mock_seq.return_value = _fake_final_state("x")
        resp = client.post(
            "/api/pipeline/run", json={"mode": "demo", "demo_scenario_id": "clean_baseline"}
        )
    assert resp.status_code == 200
    mock_seq.assert_called_once()
    mock_graph.assert_not_called()


def test_live_mode_refreshes_live_data_before_pipeline(seeded_db):
    """mode='live' should fetch fresh news/weather (DataIngestionAgent.run_batch())
    before running L1-L7, so the Live Feed tab has something new to show."""
    with patch("src.api.routers.pipeline.run_pipeline") as mock_seq, patch(
        "src.api.routers.pipeline.DataIngestionAgent"
    ) as mock_agent_cls, patch(
        "src.api.routers.pipeline.fetch_scenario_options",
        return_value=[{"port": "Eastern Asia", "sku": "sku1", "event_date": "2025-01-01", "history_points": 20}],
    ):
        mock_agent_cls.return_value.run_batch.return_value.status = "ok"
        mock_seq.return_value = _fake_final_state("y")
        resp = client.post("/api/pipeline/run", json={"mode": "live"})
    assert resp.status_code == 200
    mock_agent_cls.return_value.run_batch.assert_called_once()
    mock_seq.assert_called_once()


def test_demo_mode_does_not_refresh_live_data(seeded_db):
    """Demo scenarios use fixed historical baselines — no reason to hit
    live connectors, so run_batch() must not be called for mode='demo'."""
    with patch("src.api.routers.pipeline.run_pipeline") as mock_seq, patch(
        "src.api.routers.pipeline.DataIngestionAgent"
    ) as mock_agent_cls, patch(
        "src.api.routers.pipeline.build_demo_payload", return_value={"run_id": "z", "mode": "demo"}
    ):
        mock_seq.return_value = _fake_final_state("z")
        resp = client.post(
            "/api/pipeline/run", json={"mode": "demo", "demo_scenario_id": "clean_baseline"}
        )
    assert resp.status_code == 200
    mock_agent_cls.return_value.run_batch.assert_not_called()


def test_live_mode_status_shows_fetching_phase_before_l1_starts(seeded_db):
    """Before the connector sweep finishes there are zero agent_execution_log
    rows for the run_id yet. GET /status must not 404 in that window — it
    should report all-Idle agents plus a human-readable current_phase, so
    the status bar has something to show instead of going blank."""
    import src.api.routers.pipeline as pipeline_module

    captured: Dict[str, Any] = {}

    def _capture_phase_mid_sweep(*args, **kwargs):
        # Runs synchronously inside POST /run's BackgroundTask (TestClient
        # executes it inline) — this is exactly the window a real client's
        # GET /status poll would land in during a live sweep.
        captured["run_id"] = pipeline_module._RUN_PHASE and next(iter(pipeline_module._RUN_PHASE))
        captured["status_response"] = client.get(
            "/api/pipeline/status", params={"run_id": captured["run_id"]}
        )
        return MagicMock(status="ok")

    with patch("src.api.routers.pipeline.run_pipeline") as mock_seq, patch(
        "src.api.routers.pipeline.DataIngestionAgent"
    ) as mock_agent_cls, patch(
        "src.api.routers.pipeline.fetch_scenario_options",
        return_value=[{"port": "Eastern Asia", "sku": "sku1", "event_date": "2025-01-01", "history_points": 20}],
    ):
        mock_agent_cls.return_value.run_batch.side_effect = _capture_phase_mid_sweep
        mock_seq.return_value = _fake_final_state("live-phase-run")
        resp = client.post("/api/pipeline/run", json={"mode": "live"})

    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert captured["run_id"] == run_id

    mid_sweep_status = captured["status_response"]
    assert mid_sweep_status.status_code == 200
    body = mid_sweep_status.json()
    assert body["current_phase"] == "Fetching live news & weather data…"
    assert body["is_complete"] is False
    assert all(a["status"] == "Idle" for a in body["agents"])

    # Phase is cleared once the sweep finishes (before run_pipeline runs).
    assert run_id not in pipeline_module._RUN_PHASE


def test_status_unknown_run_id_returns_404(seeded_db):
    resp = client.get("/api/pipeline/status", params={"run_id": "does-not-exist"})
    assert resp.status_code == 404


def test_status_reflects_real_agent_execution_log_rows(seeded_db):
    from src.utils.db_utils import execute_non_query

    rows = [
        ("run-mix", "L1_ingestion", "Complete", "t0", "t1", None),
        ("run-mix", "L2_news", "Complete", "t1", "t2", None),
        ("run-mix", "L3_weather", "Running", "t2", None, None),
        ("run-mix", "L5_forecast", "Failed-Fallback", "t2", "t3", "prophet not installed"),
    ]
    for run_id, agent, status, started, completed, err in rows:
        execute_non_query(
            "INSERT INTO agent_execution_log (run_id, agent_name, status, started_at, completed_at, error_message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, agent, status, started, completed, err),
        )

    resp = client.get("/api/pipeline/status", params={"run_id": "run-mix"})
    assert resp.status_code == 200
    body = resp.json()
    by_id = {a["id"]: a["status"] for a in body["agents"]}
    assert by_id["L1"] == "Complete"
    assert by_id["L3"] == "Running"
    # L5 is optional: agent_span() wrote Failed-Fallback, translated to
    # Skipped-Optional for the frontend contract.
    assert by_id["L5"] == "Skipped-Optional"
    assert by_id["L4"] == "Idle"  # never ran for this run_id
    assert body["is_complete"] is False  # L7 never ran


def test_snapshot_writes_l4_l5_l6_l7(seeded_db):
    from src.utils.db_utils import fetch_forecast, fetch_mitigation, fetch_risk_classification_output, fetch_simulation

    state = _fake_final_state("bridge-run-1")
    with patch("src.agents.pipeline_bridge.insert_simulation_output") as mock_sim, patch(
        "src.agents.pipeline_bridge.insert_mitigation_output"
    ) as mock_mit, patch(
        "src.agents.pipeline_bridge.insert_risk_classification_output"
    ) as mock_risk, patch(
        "src.agents.pipeline_bridge.insert_forecast_output"
    ) as mock_forecast:
        snapshot_run_outputs("bridge-run-1", state)
        mock_sim.assert_called_once()
        mock_mit.assert_called_once()
        mock_risk.assert_called_once()
        mock_forecast.assert_called_once()

    # Real (non-mocked) run: rows actually land in the L4/L5/L6/L7 tables.
    snapshot_run_outputs("bridge-run-2", state)
    assert fetch_risk_classification_output("bridge-run-2") is not None
    assert fetch_simulation("bridge-run-2") is not None
    assert fetch_mitigation("bridge-run-2") is not None
    # forecast_result was None (L5 skipped) on _fake_final_state, so this is
    # the fixture-shaped fallback row, not a fabricated real forecast.
    forecast = fetch_forecast("bridge-run-2")
    assert forecast is not None
    assert forecast["category"] == "Skipped-Optional"


def test_forecast_endpoint_handles_l5_skip_gracefully(seeded_db):
    """A run_id that was never snapshotted at all (pipeline never ran for
    it) still 404s, not a 500. A run_id where L5 itself was Skipped-Optional
    now has a fallback row (see test_snapshot_writes_l4_l5_l6_l7) and does
    not hit this path."""
    resp = client.get("/api/forecast/run-with-no-forecast")
    assert resp.status_code in (200, 404)
    assert resp.status_code != 500


def test_simulation_snapshot_carries_revenue_and_days_to_stockout_spread(seeded_db):
    """revenue_at_risk_p10/p90_usd and days_to_stockout_p10/50/90 are real
    SimulationResult fields the Monte Carlo engine always computes; they
    were previously dropped at the pipeline_bridge boundary. Assert they
    survive the persist_simulation_output -> fetch_simulation round trip."""
    from src.utils.db_utils import fetch_simulation

    state = _fake_final_state("bridge-run-3")
    state.simulation_result = SimulationResult(
        stockout_probability_pct=42.0,
        expected_inventory_gap_pct=10.0,
        alternate_route="Cape of Good Hope",
        revenue_impact_usd_p10=50_000.0,
        revenue_impact_usd_p50=150_000.0,
        revenue_impact_usd_p90=400_000.0,
        days_to_stockout_p10=3.0,
        days_to_stockout_p50=7.0,
        days_to_stockout_p90=14.0,
    )
    snapshot_run_outputs("bridge-run-3", state)

    sim = fetch_simulation("bridge-run-3")
    assert sim["revenue_at_risk_p10_usd"] == 50_000.0
    assert sim["revenue_at_risk_p90_usd"] == 400_000.0
    assert sim["days_to_stockout_p10"] == 3.0
    assert sim["days_to_stockout_p50"] == 7.0
    assert sim["days_to_stockout_p90"] == 14.0


def test_simulation_endpoint_reads_real_data_not_fixture(seeded_db):
    state = _fake_final_state("sim-run-1")
    snapshot_run_outputs("sim-run-1", state)
    resp = client.get("/api/simulation/sim-run-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "sim-run-1"
    assert body["p50"] == 42.0
    assert body["alternate_route"] == "Cape of Good Hope"
