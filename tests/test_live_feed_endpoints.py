"""
Tests for Screen 1 (Live Feed) endpoints — src/api/routers/live_feed.py.

Mocks DataIngestionAgent and seeds a temp SQLite DB via monkeypatching
src.utils.db_utils.DB_PATH — no real network calls to Open-Meteo, GDELT,
FRED, RSS feeds, or yfinance happen in this test module.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE live_news_ingest (hub_city TEXT, hub_country TEXT, "
        "supplier_country TEXT, headline TEXT, published_at TEXT, "
        "relevance_score REAL, query_term TEXT, source_feed TEXT, "
        "fetched_at_utc TEXT, run_id TEXT)"
    )
    conn.execute(
        "INSERT INTO live_news_ingest VALUES "
        "('Hsinchu','Taiwan','Taiwan','TSMC output disrupted','2026-07-01T00:00:00Z',"
        "0.82,'chip shortage','google_news_rss','2026-07-01T00:05:00Z','run-abc')"
    )
    conn.execute(
        "CREATE TABLE live_weather_ingest (hub_city TEXT, wind_speed_kmh REAL, "
        "precipitation_mm REAL, weather_code INTEGER, temperature_c REAL, "
        "raw_severity_score REAL, is_trigger_hub INTEGER, fetched_at_utc TEXT, run_id TEXT)"
    )
    conn.execute(
        "INSERT INTO live_weather_ingest VALUES "
        "('Hsinchu',65.0,12.0,95,28.0,9.2,1,'2026-07-01T00:05:00Z','run-abc')"
    )
    conn.execute(
        "CREATE TABLE ingestion_run_log (run_id TEXT, run_ts_utc TEXT, source TEXT, "
        "connector_class TEXT, rows_fetched INTEGER, rows_inserted INTEGER, "
        "rows_skipped INTEGER, duration_ms INTEGER, status TEXT, error_detail TEXT, "
        "last_fetched_key TEXT)"
    )
    conn.execute(
        "INSERT INTO ingestion_run_log VALUES "
        "('run-abc','2026-07-01T00:05:00Z','google_news_rss','GoogleNewsRSSConnector',"
        "12,12,0,1450,'success',NULL,NULL)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)
    return db_path


def test_news_endpoint_returns_seeded_rows(seeded_db):
    resp = client.get("/api/live-feed/news")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["run_id"] == "run-abc"
    assert body["items"][0]["headline"] == "TSMC output disrupted"
    assert body["items"][0]["score_tier"] == "high"
    assert body["items"][0]["source_feed"] == "google_news_rss"


def test_news_endpoint_empty_state(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE live_news_ingest (hub_city TEXT, hub_country TEXT, "
        "supplier_country TEXT, headline TEXT, published_at TEXT, "
        "relevance_score REAL, query_term TEXT, source_feed TEXT, "
        "fetched_at_utc TEXT, run_id TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

    resp = client.get("/api/live-feed/news")
    assert resp.status_code == 200
    assert resp.json() == {"run_id": None, "count": 0, "fetched_at": None, "items": []}


def test_weather_endpoint_always_returns_all_six_configured_hubs(seeded_db):
    resp = client.get("/api/live-feed/weather")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hubs"]) == 6
    hub_names = {h["hub_city"] for h in body["hubs"]}
    assert hub_names == {"Hsinchu", "Osaka", "Austin", "Shanghai", "Singapore", "Rotterdam"}

    hsinchu = next(h for h in body["hubs"] if h["hub_city"] == "Hsinchu")
    assert hsinchu["is_trigger_hub"] is True
    assert hsinchu["raw_severity_score"] == 9.2

    osaka = next(h for h in body["hubs"] if h["hub_city"] == "Osaka")
    assert osaka["raw_severity_score"] is None
    assert osaka["is_trigger_hub"] is False


def test_weather_endpoint_empty_db_still_returns_all_six_hubs(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE live_weather_ingest (hub_city TEXT, wind_speed_kmh REAL, "
        "precipitation_mm REAL, weather_code INTEGER, temperature_c REAL, "
        "raw_severity_score REAL, is_trigger_hub INTEGER, fetched_at_utc TEXT, run_id TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

    resp = client.get("/api/live-feed/weather")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hubs"]) == 6
    assert body["run_id"] is None
    assert all(h["raw_severity_score"] is None for h in body["hubs"])


def test_refresh_triggers_background_task_not_full_langgraph():
    with patch("src.api.routers.live_feed.DataIngestionAgent") as mock_agent_cls:
        mock_agent_cls.return_value.run_batch.return_value = MagicMock()
        resp = client.post("/api/live-feed/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        # The whole point of this screen: run_agent_graph/LangGraph is never imported here.
        import src.api.routers.live_feed as live_feed_module

        assert not hasattr(live_feed_module, "run_agent_graph")


def test_refresh_returns_skipped_when_already_running():
    from src.agents.data_ingestion_agent import _INGESTION_LOCK

    _INGESTION_LOCK.acquire()
    try:
        resp = client.post("/api/live-feed/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped_already_running"
    finally:
        _INGESTION_LOCK.release()


def test_ingest_status_reflects_recent_runs(seeded_db):
    resp = client.get("/api/live-feed/ingest-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_running"] is False
    assert body["last_run"]["source"] == "google_news_rss"
    assert body["last_run"]["rows_inserted"] == 12
    assert len(body["recent_runs"]) == 1


def test_ingest_status_reflects_running_lock(seeded_db):
    from src.agents.data_ingestion_agent import _INGESTION_LOCK

    _INGESTION_LOCK.acquire()
    try:
        resp = client.get("/api/live-feed/ingest-status")
        assert resp.json()["is_running"] is True
    finally:
        _INGESTION_LOCK.release()


def test_logs_endpoint_has_real_l1_line_and_unmodified_stub_l2_l7(seeded_db):
    from src.api.fixtures import LOG_LINES

    resp = client.get("/api/live-feed/logs")
    assert resp.status_code == 200
    lines = resp.json()["lines"]

    assert lines[0]["level"] == "L1"
    assert lines[0]["source"] == "real"
    assert "run-abc" in lines[0]["text"]
    assert "12 rows" in lines[0]["text"]

    stub_lines = lines[1:]
    assert len(stub_lines) == len(LOG_LINES) - 1
    for actual, expected in zip(stub_lines, LOG_LINES[1:]):
        assert actual["level"] == expected["level"]
        assert actual["text"] == expected["text"]
        assert actual["tab"] == expected["tab"]
        assert actual["source"] == "stub"


def test_logs_endpoint_empty_run_shows_placeholder(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE ingestion_run_log (run_id TEXT, run_ts_utc TEXT, source TEXT, "
        "connector_class TEXT, rows_fetched INTEGER, rows_inserted INTEGER, "
        "rows_skipped INTEGER, duration_ms INTEGER, status TEXT, error_detail TEXT, "
        "last_fetched_key TEXT)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

    resp = client.get("/api/live-feed/logs")
    assert resp.status_code == 200
    assert "No ingestion run yet" in resp.json()["lines"][0]["text"]


def test_gantt_endpoint_has_real_l1_bar_and_unmodified_stub_l2_l7(seeded_db):
    from src.api.fixtures import GANTT

    resp = client.get("/api/live-feed/gantt")
    assert resp.status_code == 200
    bars = resp.json()["bars"]

    assert bars[0]["id"] == "L1"
    assert bars[0]["source"] == "real"
    assert bars[0]["dur"] == 1.4  # 1450ms / 1000, round(1.45, 1) == 1.4

    stub_bars = bars[1:]
    assert len(stub_bars) == len(GANTT) - 1
    for actual, expected in zip(stub_bars, GANTT[1:]):
        assert actual["id"] == expected["id"]
        assert actual["start"] == expected["start"]
        assert actual["dur"] == expected["dur"]
        assert actual["color"] == expected["color"]
        assert actual["source"] == "stub"
