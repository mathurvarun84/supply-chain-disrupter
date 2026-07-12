"""
Day 8 — verify all 17 endpoints read real SQLite/ChromaDB data, not fixtures.

Seeds a temp SQLite file via monkeypatch on db_utils.DB_PATH. No live OpenAI
or network calls.
"""

from __future__ import annotations

import ast
import inspect
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.fixtures import COST_DATA, FORECAST_SERIES, NEWS_GROUPS
from src.api.main import app

client = TestClient(app)
ROUTER_DIR = Path(__file__).resolve().parents[1] / "src" / "api" / "routers"


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """Temp SQLite with rows for all Day-8 read paths."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    conn.execute(
        """
        CREATE TABLE live_news_ingest (
            hub_city TEXT, hub_country TEXT, supplier_country TEXT,
            headline TEXT, published_at TEXT, relevance_score REAL,
            query_term TEXT, source_feed TEXT, fetched_at_utc TEXT, run_id TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO live_news_ingest VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("Hsinchu", "Taiwan", "Taiwan", "Alpha headline", "2026-07-01T00:00:00Z",
             0.91, "chip", "rss_a", "2026-07-01T00:05:00Z", "news-run-1"),
            ("Osaka", "Japan", "Japan", "Beta headline", "2026-07-01T00:01:00Z",
             0.55, "fab", "rss_b", "2026-07-01T00:05:00Z", "news-run-1"),
            ("Austin", "USA", "USA", "Gamma headline", "2026-07-01T00:02:00Z",
             0.22, "supply", "rss_c", "2026-07-01T00:05:00Z", "news-run-1"),
        ],
    )

    conn.execute(
        """
        CREATE TABLE live_weather_ingest (
            hub_city TEXT, wind_speed_kmh REAL, precipitation_mm REAL,
            weather_code INTEGER, temperature_c REAL, raw_severity_score REAL,
            is_trigger_hub INTEGER, fetched_at_utc TEXT, run_id TEXT
        )
        """
    )
    hubs = [
        ("Hsinchu", 62.0, 18.4, 95, 24.0, 9.2, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Osaka", 14.0, 2.1, 3, 18.0, 2.1, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Austin", 22.0, 0.3, 2, 31.0, 1.4, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Shanghai", 19.0, 4.7, 61, 27.0, 3.8, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Singapore", 11.0, 6.2, 63, 29.0, 2.3, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Rotterdam", 28.0, 1.1, 3, 16.0, 1.8, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
        ("Mumbai", 30.0, 5.0, 61, 30.0, 4.0, 0, "2026-07-01T00:05:00Z", "wx-run-1"),
    ]
    conn.executemany("INSERT INTO live_weather_ingest VALUES (?,?,?,?,?,?,?,?,?)", hubs)

    conn.execute(
        """
        CREATE TABLE agent_execution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, agent_name TEXT, status TEXT,
            started_at TEXT, completed_at TEXT, duration_ms REAL,
            error_message TEXT, langfuse_trace_id TEXT, langfuse_span_id TEXT,
            updated_at TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO agent_execution_log
        (run_id, agent_name, status, started_at, completed_at, duration_ms)
        VALUES (?,?,?,?,?,?)
        """,
        [
            ("pipe-run-1", "L1_ingestion", "Complete", "2026-07-01T10:00:00+00:00",
             "2026-07-01T10:00:04+00:00", 4200),
            ("pipe-run-1", "L4_risk_classifier", "Complete", "2026-07-01T10:00:07+00:00",
             "2026-07-01T10:00:14+00:00", 7200),
        ],
    )

    conn.execute(
        """
        CREATE TABLE risk_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER, mode TEXT, composite_score REAL,
            geo_component REAL, supply_component REAL, freight_component REAL,
            defect_component REAL, duration_days REAL, base_label TEXT,
            final_label TEXT, escalated INTEGER, rag_citations TEXT,
            rationale TEXT, full_result_json TEXT, run_ts TEXT
        )
        """
    )
    full_json = json.dumps(
        {
            "judge_verdict": {
                "verdict_type": "majority_rule",
                "final_label": "CRITICAL",
            }
        }
    )
    conn.execute(
        """
        INSERT INTO risk_classifications
        (order_id, mode, composite_score, geo_component, supply_component,
         freight_component, defect_component, duration_days, base_label,
         final_label, escalated, rag_citations, rationale, full_result_json)
        VALUES (9001,'replay',0.84,0.7,0.8,0.5,0.2,14,'HIGH','CRITICAL',1,'[]','test',?)
        """,
        (full_json,),
    )

    conn.execute(
        """
        CREATE TABLE forecast_output (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, category TEXT, categories_json TEXT, series_json TEXT, created_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO forecast_output (run_id, category, categories_json, series_json)
        VALUES (?,?,?,?)
        """,
        (
            "pipe-run-1",
            "Laptops",
            json.dumps(["Laptops", "Phones"]),
            json.dumps([{"day": "D+1", "baseline": 1000, "adjusted": 900}]),
        ),
    )

    conn.execute(
        """
        CREATE TABLE simulation_output (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE, p10 REAL, p50 REAL, p90 REAL,
            revenue_at_risk_usd REAL, alternate_route TEXT, histogram_json TEXT, created_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO simulation_output
        (run_id, p10, p50, p90, revenue_at_risk_usd, alternate_route, histogram_json)
        VALUES (?,?,?,?,?,?,?)
        """,
        ("pipe-run-1", 12.0, 33.0, 55.0, 1_500_000.0, "Suez Canal",
         json.dumps([{"range": "30-40%", "count": 42}])),
    )

    conn.execute(
        """
        CREATE TABLE mitigation_output (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE, urgency TEXT, ranked_actions_json TEXT,
            rag_query_trace_json TEXT, india_sourcing_json TEXT,
            slack_preview TEXT, cost_delta_usd REAL, created_at TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO mitigation_output
        (run_id, urgency, ranked_actions_json, rag_query_trace_json,
         india_sourcing_json, slack_preview, cost_delta_usd)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            "pipe-run-1",
            "IMMEDIATE",
            json.dumps([{"rank": 1, "text": "Reroute via Singapore", "citations": ["RouteMap"]}]),
            json.dumps(["historical_disruption_lookup → historical_precedents"]),
            json.dumps(["Kaynes Technology — Mysuru"]),
            "CRITICAL disruption detected",
            180000.0,
        ),
    )

    conn.execute(
        """
        CREATE TABLE llm_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, agent_name TEXT, model TEXT, prompt_preview TEXT,
            full_prompt TEXT, full_response TEXT, input_tokens INTEGER,
            output_tokens INTEGER, total_tokens INTEGER, cost_usd REAL,
            latency_ms REAL, status TEXT, retry_count INTEGER,
            error_message TEXT, langfuse_trace_id TEXT,
            langfuse_generation_id TEXT, ts TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO llm_call_log
        (run_id, agent_name, model, prompt_preview, full_prompt, full_response,
         input_tokens, output_tokens, total_tokens, cost_usd, latency_ms, status, ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            ("pipe-run-1", "L2_news", "gpt-4o-mini", "news prompt", "full", "resp",
             100, 50, 150, 0.003, 3100.0, "success", "10:00:01"),
            ("pipe-run-1", "L4_risk_classifier", "gpt-4o", "risk prompt", "full", "resp",
             200, 80, 280, 0.009, 7200.0, "success", "10:00:08"),
        ],
    )

    conn.execute(
        """
        CREATE TABLE guardrail_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, dir TEXT, agent TEXT,
            pass_count INTEGER, fail_count INTEGER, reason TEXT, updated_at TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO guardrail_events (name, dir, agent, pass_count, fail_count, reason)
        VALUES (?,?,?,?,?,?)
        """,
        [
            ("prompt-injection-screen", "input", "L2", 10, 1, "seed reason"),
            ("faithfulness-gate", "output", "L7", 8, 2, "seed fail"),
        ],
    )

    conn.execute(
        """
        CREATE TABLE ingestion_run_log (
            run_id TEXT, run_ts_utc TEXT, source TEXT, connector_class TEXT,
            rows_fetched INTEGER, rows_inserted INTEGER, rows_skipped INTEGER,
            duration_ms INTEGER, status TEXT, error_detail TEXT, last_fetched_key TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO ingestion_run_log VALUES
        ('ingest-1','2026-07-01T00:05:00Z','google_news_rss','GoogleNewsRSSConnector',
         12,12,0,1450,'success',NULL,NULL)
        """
    )

    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)
    return db_path


def test_live_news_reads_from_table(seeded_db):
    resp = client.get("/api/live-feed/news")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    headlines = [item["headline"] for item in body["items"]]
    assert headlines == ["Alpha headline", "Beta headline", "Gamma headline"]
    fixture_headline = NEWS_GROUPS[0]["items"][0]["headline"]
    assert fixture_headline not in headlines


def test_live_weather_returns_six_hubs_only(seeded_db):
    resp = client.get("/api/live-feed/weather")
    assert resp.status_code == 200
    hubs = resp.json()["hubs"]
    assert len(hubs) == 6
    hub_names = {h["hub_city"] for h in hubs}
    assert "Mumbai" not in hub_names
    hsinchu = next(h for h in hubs if h["hub_city"] == "Hsinchu")
    assert hsinchu["is_trigger_hub"] is True


def test_run_logs_from_agent_execution_log(seeded_db):
    resp = client.get("/api/live-feed/logs?run_id=pipe-run-1")
    assert resp.status_code == 200
    lines = resp.json()["lines"]
    assert len(lines) == 2
    assert all(line["source"] == "real" for line in lines)
    assert lines[0]["level"] == "L1"


def test_run_gantt_from_agent_execution_log(seeded_db):
    resp = client.get("/api/live-feed/gantt?run_id=pipe-run-1")
    bars = resp.json()["bars"]
    assert len(bars) == 2
    assert bars[0]["id"] == "L1"
    assert bars[0]["source"] == "real"


def test_risk_classification_slack_should_fire_is_server_computed(seeded_db):
    """CRITICAL final_label must force slack_should_fire=True regardless of storage."""
    resp = client.get("/api/risk-classification/9001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_label"] == "CRITICAL"
    assert body["slack_should_fire"] is True


def test_risk_classification_unknown_run_id_404s(seeded_db):
    resp = client.get("/api/risk-classification/doesnotexist")
    assert resp.status_code == 404


def test_forecast_filters_by_category(seeded_db):
    resp = client.get("/api/forecast/pipe-run-1?category=Phones")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "Phones"
    assert body["series"][0]["baseline"] == 1000
    assert body["series"] != FORECAST_SERIES


def test_simulation_reads_seeded_row(seeded_db):
    resp = client.get("/api/simulation/pipe-run-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["p50"] == 33.0
    assert body["alternate_route"] == "Suez Canal"


def test_mitigation_reads_seeded_row(seeded_db):
    resp = client.get("/api/mitigation/pipe-run-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["urgency"] == "IMMEDIATE"
    assert body["ranked_actions"][0]["text"] == "Reroute via Singapore"


def test_observability_cost_sums_correctly(seeded_db):
    resp = client.get("/api/observability/cost")
    assert resp.status_code == 200
    body = resp.json()
    assert body != COST_DATA
    total = round(sum(row["cost"] for row in body), 6)
    assert total == 0.012


def test_observability_verdicts_from_db(seeded_db):
    resp = client.get("/api/observability/verdicts")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_guardrail_events_returns_seeded_rows(seeded_db):
    resp = client.get("/api/guardrails/events")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    dirs = {row["dir"] for row in body}
    assert dirs == {"input", "output"}


def test_rag_corpus_health_reflects_live_collection_count(monkeypatch):
    mock_col = MagicMock()
    mock_col.count.return_value = 42
    mock_col.get.return_value = {"documents": ["real doc"] * 42}
    mock_col.metadata = {}

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_col

    monkeypatch.setattr("src.rag.utils.get_chroma_client", lambda: mock_client)
    monkeypatch.setattr("src.rag.utils.get_embedding_model", lambda: MagicMock())

    resp = client.get("/api/rag/corpus")
    assert resp.status_code == 200
    historical = next(c for c in resp.json() if c["name"] == "historical_precedents")
    assert historical["docs"] == 42


def test_rag_scorecard_reads_persisted_json(monkeypatch, tmp_path):
    scores_path = tmp_path / "ragas_scores_full.json"
    scores_path.write_text(
        json.dumps(
            {
                "overall": {
                    "faithfulness": 0.88,
                    "answer_relevancy": 0.81,
                    "context_precision": 0.79,
                    "context_recall": 0.84,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.rag.utils.RAGAS_SCORES_PATH", scores_path)
    resp = client.get("/api/rag/scorecard")
    assert resp.status_code == 200
    faithfulness = next(t for t in resp.json() if t["metric"] == "Faithfulness")
    assert faithfulness["score"] == 0.88


def test_no_fixture_constants_referenced_in_any_handler():
    """Static check: router handlers must not return FIXTURE_/MOCK_ constants."""
    offenders = []
    for path in ROUTER_DIR.glob("*.py"):
        if path.name == "pipeline.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value is not None:
                        ret_src = ast.get_source_segment(source, child.value) or ""
                        if "FIXTURE_" in ret_src or "MOCK_" in ret_src:
                            offenders.append(f"{path.name}:{node.name}")
    assert offenders == []
