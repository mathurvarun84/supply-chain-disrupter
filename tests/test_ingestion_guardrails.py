"""
Unit tests for the L1 data ingestion agent's MCP-backed connectors and
data-quality guardrails (outlier quarantine, enrichment conflict hold-back,
per-connector circuit breaker).

All external calls (MCP tool functions, SQLite) are mocked — no live network
or database access.
"""

from unittest.mock import patch

import pytest

from src.utils.ingestion_connectors import HUB_CITIES, NewsMcpConnector, WeatherMcpConnector
from src.utils.ingestion_validator import DataValidator


# ── WeatherMcpConnector ───────────────────────────────────────────────────────

def test_weather_mcp_connector_fetch_and_normalize():
    canned = {
        "city": "Hsinchu", "latitude": 24.80, "longitude": 120.97,
        "wind_speed_kmh": 42.0, "precipitation_mm": 5.0,
        "weather_code": 3, "temperature_c": 22.5, "raw_severity_score": 1.0,
    }
    with patch("src.utils.ingestion_connectors.fetch_hub_weather", return_value=canned) as mock_fetch:
        connector = WeatherMcpConnector()
        raw = connector.fetch()
        assert mock_fetch.call_count == len(HUB_CITIES)
        rows = connector.normalize(raw)

    assert len(rows) == len(HUB_CITIES)
    assert rows[0]["hub_city"] == "Hsinchu"
    assert rows[0]["raw_severity_score"] == 1.0
    assert rows[0]["is_trigger_hub"] == 0  # trigger flag only set in persist()


def test_weather_mcp_connector_fetch_skips_failed_city():
    def flaky_fetch(city, lat, lon):
        if city == "Osaka":
            raise RuntimeError("503 Service Unavailable")
        return {
            "city": city, "latitude": lat, "longitude": lon,
            "wind_speed_kmh": 1.0, "precipitation_mm": 0.0,
            "weather_code": 0, "temperature_c": 20.0, "raw_severity_score": 0.0,
        }

    with patch("src.utils.ingestion_connectors.fetch_hub_weather", side_effect=flaky_fetch):
        connector = WeatherMcpConnector()
        raw = connector.fetch()

    assert len(raw) == len(HUB_CITIES) - 1
    assert all(r["city"] != "Osaka" for r in raw)


def test_weather_mcp_connector_persist_marks_trigger_hub():
    rows = [
        {"fetched_at_utc": "t", "hub_city": "Hsinchu", "latitude": 24.8, "longitude": 120.97,
         "wind_speed_kmh": 70.0, "precipitation_mm": 60.0, "weather_code": 95,
         "temperature_c": 20.0, "raw_severity_score": 8.0, "is_trigger_hub": 0},
        {"fetched_at_utc": "t", "hub_city": "Osaka", "latitude": 34.69, "longitude": 135.5,
         "wind_speed_kmh": 5.0, "precipitation_mm": 0.0, "weather_code": 0,
         "temperature_c": 25.0, "raw_severity_score": 0.0, "is_trigger_hub": 0},
    ]
    connector = WeatherMcpConnector()
    connector._run_id = "test-run"
    with patch("src.utils.ingestion_connectors.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = mock_conn.return_value
        inserted, skipped = connector.persist(rows)

    assert inserted == 2
    assert skipped == 0
    assert rows[0]["is_trigger_hub"] == 1  # severity 8.0 >= 6.0 threshold
    assert rows[1]["is_trigger_hub"] == 0


# ── NewsMcpConnector ──────────────────────────────────────────────────────────

def test_news_mcp_connector_fetch_and_normalize():
    canned_article = {
        "headline": "Taiwan chip export controls tighten",
        "summary": "Some summary text",
        "published_at": "2026-01-01",
        "url": "https://example.com/a",
        "relevance_score": 0.5,
        "hub_city": "Hsinchu",
        "hub_country": None,
        "supplier_country": None,
    }
    with patch("src.utils.ingestion_connectors.fetch_news_headlines", return_value=[canned_article]) as mock_fetch:
        connector = NewsMcpConnector()
        raw = connector.fetch()
        assert mock_fetch.called
        rows = connector.normalize(raw)

    assert len(rows) == len(raw)
    assert rows[0]["headline"] == canned_article["headline"]
    assert rows[0]["source_feed"] == "news_mcp"
    assert rows[0]["relevance_score"] == 0.5


def test_news_mcp_connector_normalize_skips_rows_without_headline():
    connector = NewsMcpConnector()
    rows = connector.normalize([{"headline": "", "url": "https://x"}])
    assert rows == []


# ── Guardrails: outlier detection ────────────────────────────────────────────

def test_check_outlier_zscore_flags_extreme_value():
    # Tight cluster around 0.2 with slight variance (std=0 would skip the check entirely).
    history = [(0.19,), (0.20,), (0.21,)] * 5
    with patch("src.utils.ingestion_validator.execute_query", return_value=history):
        assert DataValidator.check_outlier_zscore(5.0, "derived_weather_severity", "weather_events") is True
        assert DataValidator.check_outlier_zscore(0.2, "derived_weather_severity", "weather_events") is False


def test_check_outlier_zscore_requires_min_history():
    with patch("src.utils.ingestion_validator.execute_query", return_value=[(0.2,)] * 5):
        # Fewer than 10 historical points — never flags, regardless of value.
        assert DataValidator.check_outlier_zscore(999.0, "derived_weather_severity", "weather_events") is False


def test_baseline_does_not_drift_across_a_bulk_batch():
    """
    Regression test: a connector persisting many rows in one run (e.g. FRED's
    full historical CSV backfill) must check every row against a single
    snapshot of the pre-run baseline, not re-query the table per row. Re-
    querying per row would let earlier-in-batch inserts skew the baseline for
    later rows, cascading into mass false-positive quarantine.
    """
    history = [(8.6,), (8.7,), (8.8,), (8.65,), (8.75,)] * 3  # tight legit cluster, mean~8.694

    with patch("src.utils.ingestion_validator.execute_query", return_value=history):
        baseline = DataValidator.get_zscore_baseline("normalized_sdi", "freight_signals")
    assert baseline is not None
    mean, std = baseline
    assert mean == pytest.approx(8.694, abs=0.01)

    # A batch of 50 legitimate values near the baseline mean, checked WITHOUT
    # re-querying the DB per row (baseline is reused as-is).
    batch = [8.69 + (i % 5) * 0.01 for i in range(50)]
    outlier_flags = [DataValidator.is_outlier(v, baseline) for v in batch]
    assert not any(outlier_flags), "Legitimate clustered values should never be flagged when baseline is stable"

    # An actual outlier within the same batch is still caught.
    assert DataValidator.is_outlier(50.0, baseline) is True


# ── Guardrails: enrichment conflict hold-back ────────────────────────────────

def test_check_enrichment_conflict_flags_large_shifts():
    prior_row = [(0.10, 5.0)]  # (weather_severity_live, supply_disruption_index_live)
    new_row = {"weather_severity_live": 0.90, "supply_disruption_index_live": 8.0}
    with patch("src.utils.ingestion_validator.execute_query", return_value=prior_row):
        conflicts = DataValidator.check_enrichment_conflict("JNPT", new_row)
    assert "weather_severity_live" in conflicts
    assert "supply_disruption_index_live" in conflicts


def test_check_enrichment_conflict_no_conflict_for_small_shift():
    prior_row = [(0.40, 5.0)]
    new_row = {"weather_severity_live": 0.42, "supply_disruption_index_live": 5.1}
    with patch("src.utils.ingestion_validator.execute_query", return_value=prior_row):
        conflicts = DataValidator.check_enrichment_conflict("JNPT", new_row)
    assert conflicts == []


def test_check_enrichment_conflict_no_prior_row():
    with patch("src.utils.ingestion_validator.execute_query", return_value=[]):
        conflicts = DataValidator.check_enrichment_conflict("JNPT", {"weather_severity_live": 0.9})
    assert conflicts == []


# ── Guardrails: quarantine persistence ───────────────────────────────────────

def test_quarantine_row_inserts_into_ingestion_quarantine():
    with patch("src.utils.ingestion_validator.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = mock_conn.return_value
        DataValidator.quarantine_row(
            source="fred", target_table="freight_signals", field="normalized_sdi",
            value=999.0, reason="outlier: |z| > 3.0", row={"x": 1}, run_id="run-1",
        )
    execute_call = mock_conn.return_value.execute.call_args
    assert "INSERT INTO ingestion_quarantine" in execute_call.args[0]
    params = execute_call.args[1]
    assert params[2] == "fred"
    assert params[3] == "freight_signals"
    assert params[4] == "normalized_sdi"


# ── Guardrails: circuit breaker ───────────────────────────────────────────────

def test_circuit_breaker_opens_after_consecutive_failures():
    from src.agents.data_ingestion_agent import DataIngestionAgent, _CIRCUIT_BREAKER_THRESHOLD

    with patch("src.agents.data_ingestion_agent.ensure_ingestion_schema"), \
         patch("src.agents.data_ingestion_agent.load_config", return_value={"ports": {}}):
        agent = DataIngestionAgent()

    failed_rows = [("failed",)] * _CIRCUIT_BREAKER_THRESHOLD
    with patch("src.agents.data_ingestion_agent.execute_query", return_value=failed_rows):
        assert agent._is_circuit_open("fred") is True

    mixed_rows = [("failed",), ("success",), ("failed",)]
    with patch("src.agents.data_ingestion_agent.execute_query", return_value=mixed_rows):
        assert agent._is_circuit_open("fred") is False

    with patch("src.agents.data_ingestion_agent.execute_query", return_value=[]):
        assert agent._is_circuit_open("fred") is False
