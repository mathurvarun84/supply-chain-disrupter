"""
Tests for the mitigation fix pass: server-computed slack_alert_fired, the
structured RAG trace (mitigation_rag_trace table), and mitigation_window /
cost_delta_usd derivation in pipeline_bridge.persist_mitigation_output().

Uses a temp SQLite file via monkeypatch on db_utils.DB_PATH — no live OpenAI
or network calls.
"""

from __future__ import annotations

import pytest

from src.agents.pipeline_bridge import persist_mitigation_output
from src.agents.state import (
    EventMetadata,
    GlobalState,
    MitigationAction,
    RiskClassificationResult,
    SimulationResult,
)
from src.utils import db_utils


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_utils, "DB_PATH", tmp_path / "test.db")
    db_utils.ensure_schema()
    return tmp_path / "test.db"


def _state(critical_flag: bool, risk_label: str = "HIGH", with_trace: bool = True) -> GlobalState:
    risk = RiskClassificationResult(
        mode="live", composite_score=0.6,
        geo_component=0.4, supply_component=0.5, freight_component=0.6, defect_component=0.4,
        duration_days=None, base_label=risk_label, final_label=risk_label,
        escalated=False, rationale="test", critical_flag=critical_flag,
    )
    action = MitigationAction(
        summary="Test mitigation summary.",
        recommendations=["Reroute via backup lane.", "Raise safety stock."],
        cost_delta="HIGH: expedite required.",
        urgency="IMMEDIATE" if critical_flag else "HIGH",
        rag_citations=["historical_precedents: red_sea_disruption_2023_2024.txt"],
        india_sourcing_recommendations=["CG Power-Kaynes OSAT (India)."],
    )
    trace = [
        {
            "query_name": "historical_disruption_lookup",
            "query_text": "q1",
            "fired": True,
            "fire_condition": "Always queried",
            "chunks": [{
                "collection": "historical_precedents",
                "source_file": "red_sea_disruption_2023_2024.txt",
                "similarity_score": 0.31,
                "snippet": "Houthi attacks forced rerouting.",
            }],
        },
        {
            "query_name": "export_control_check",
            "query_text": "",
            "fired": False,
            "fire_condition": "Fires only when export_control_level is in the top quartile",
            "chunks": [],
        },
        {
            "query_name": "india_sourcing_query",
            "query_text": "q3",
            "fired": True,
            "fire_condition": "Always queried",
            "chunks": [],
        },
    ] if with_trace else []

    return GlobalState(
        event_metadata=EventMetadata(
            disruption_type="port closure", affected_port="Rotterdam", affected_route="test",
            severity=0.6, shock_duration_days=6, recovery_window_days=45, synthetic_ratio=0.0,
        ),
        active_record={"event_date": "2024-01-01", "port": "Rotterdam", "sku": "CHIP_AP"},
        risk_classification=risk,
        simulation_result=SimulationResult(
            stockout_probability_pct=40.0,
            expected_inventory_gap_pct=20.0,
            alternate_route="Cape of Good Hope",
            revenue_impact_usd_p50=250_000.0,
        ),
        mitigation_action=action,
        mitigation_rag_trace=trace,
    )


def test_slack_alert_fired_true_only_for_critical_flag(temp_db):
    """slack_alert_fired must come from risk.critical_flag, never the LLM output,
    and slack_preview_text must only be populated when it's true."""
    persist_mitigation_output("run-critical", _state(critical_flag=True))
    persist_mitigation_output("run-low", _state(critical_flag=False, risk_label="LOW"))

    critical_row = db_utils.fetch_mitigation("run-critical")
    low_row = db_utils.fetch_mitigation("run-low")

    assert critical_row["slack_alert_fired"] is True
    assert critical_row["slack_preview"] is not None
    assert "IMMEDIATE" in critical_row["slack_preview"]

    assert low_row["slack_alert_fired"] is False
    assert low_row["slack_preview"] is None


def test_rag_query_trace_always_has_exactly_three_entries_in_fixed_order(temp_db):
    persist_mitigation_output("run-trace", _state(critical_flag=False))
    result = db_utils.fetch_mitigation("run-trace")
    trace = result["rag_query_trace"]
    assert len(trace) == 3
    assert [q["query_name"] for q in trace] == [
        "historical_disruption_lookup",
        "export_control_check",
        "india_sourcing_query",
    ]
    export_control = trace[1]
    assert export_control["fired"] is False
    assert export_control["retrieved_chunks"] == []
    historical = trace[0]
    assert historical["fired"] is True
    assert len(historical["retrieved_chunks"]) == 1
    assert historical["retrieved_chunks"][0]["source_file"] == "red_sea_disruption_2023_2024.txt"


def test_rag_query_trace_falls_back_to_three_unfired_rows_when_nothing_persisted(temp_db):
    """A run predating trace capture (or where the LLM+RAG path never ran) still
    gets exactly 3 rows back, not an empty list."""
    persist_mitigation_output("run-no-trace", _state(critical_flag=False, with_trace=False))
    result = db_utils.fetch_mitigation("run-no-trace")
    trace = result["rag_query_trace"]
    assert len(trace) == 3
    assert all(q["fired"] is False for q in trace)


def test_mitigation_window_and_cost_delta_usd_are_derived_not_hardcoded(temp_db):
    persist_mitigation_output("run-window", _state(critical_flag=True))
    result = db_utils.fetch_mitigation("run-window")
    assert result["mitigation_window"] == "6-day disruption window, 45-day recovery"
    assert result["cost_delta_usd"] == 250_000.0


def test_ranked_action_citations_are_structured(temp_db):
    state = _state(critical_flag=True)
    action = state.mitigation_action
    # mitigation_agent.py writes mitigation_actions itself (native persistence);
    # pipeline_bridge only writes the run_id-keyed snapshot — seed both, as a
    # real pipeline run would, so fetch_mitigation's richer action_row path runs.
    db_utils.insert_mitigation_action(
        run_id="run-citations",
        event_date="2024-01-01", port="Rotterdam", sku="CHIP_AP",
        risk_label="HIGH", summary=action.summary, recommendations=action.recommendations,
        urgency=action.urgency, cost_delta=action.cost_delta,
        rag_citations=action.rag_citations,
        india_sourcing_recommendations=action.india_sourcing_recommendations,
    )
    persist_mitigation_output("run-citations", state)
    result = db_utils.fetch_mitigation("run-citations")
    first_action = result["ranked_actions"][0]
    assert first_action["citations"] == [
        {"collection": "historical_precedents", "source_file": "red_sea_disruption_2023_2024.txt"}
    ]
    assert first_action["action_type"] in (
        "INVENTORY", "ROUTING", "SOURCING", "INDIA-SOURCING", "MONITOR", "FINANCIAL",
    )


def test_mitigation_endpoint_response_matches_contract(temp_db):
    from fastapi.testclient import TestClient

    from src.api.main import app

    persist_mitigation_output("run-endpoint", _state(critical_flag=True))
    client = TestClient(app)
    resp = client.get("/api/mitigation/run-endpoint")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slack_alert_fired"] is True
    assert len(body["rag_query_trace"]) == 3
    assert body["ranked_actions"][0]["action_type"]
    assert body["mitigation_window"] == "6-day disruption window, 45-day recovery"


def test_mitigation_endpoint_unknown_run_id_404s(temp_db):
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    resp = client.get("/api/mitigation/does-not-exist")
    assert resp.status_code == 404
