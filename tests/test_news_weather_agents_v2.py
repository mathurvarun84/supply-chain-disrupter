"""
Tests for SQLite-first L2 News and L3 Weather agents (v2).

Run: python -m pytest tests/test_news_weather_agents_v2.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import (
    EventMetadata,
    GlobalState,
    NewsAnalysisLLMOutput,
    NewsRiskSignal,
    WeatherRiskLLMOutput,
)


def _news_state(**overrides):
    defaults = dict(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="Hsinchu to Singapore",
            severity=0.8,
            shock_duration_days=6,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        active_record={"order_region": "Eastern Asia", "year": 2022},
    )
    defaults.update(overrides)
    return GlobalState(**defaults)


def _weather_state(**overrides):
    defaults = dict(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="test",
            severity=0.9,
            shock_duration_days=0,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        config={"ports": {"Hsinchu": {"latitude": 24.8, "longitude": 120.97}}},
        active_record={"latitude": 24.8, "longitude": 120.97, "order_region": "Eastern Asia"},
    )
    defaults.update(overrides)
    return GlobalState(**defaults)


def _sqlite_weather_row(severity=0.743, max_wind=57.2):
    return {
        "hub": "Hsinchu",
        "severity": severity,
        "wind_score": 0.65,
        "precipitation_score": 0.09,
        "weather_code_score": 0.003,
        "max_wind_speed": max_wind,
        "max_precipitation": 2.1,
        "weather_summary": "Max wind 57.2 km/h; severity 0.743.",
    }


# ── News Agent ──────────────────────────────────────────────────────────────


def test_news_agent_sqlite_primary_path():
    """Step 1: reads news_signals from SQLite; never calls live news APIs."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    state = _news_state()
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=[]) as mock_news:
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
            with patch("src.agents.news_agent.agent.build_news_signals", return_value=[]):
                news_event_analysis_agent(state)
    mock_news.assert_called_once()
    assert mock_news.call_args.kwargs.get("limit") == 20


def test_news_agent_llm_path_produces_structured_output():
    """Step 4–5: LLM success populates news_analysis_llm and uses news_severity_component."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    mock_llm = NewsAnalysisLLMOutput(
        category="weather",
        severity=0.72,
        affected_regions=["Eastern Asia", "Southeast Asia"],
        affected_commodities=["advanced logic chips"],
        news_severity_component=0.55,
        expected_duration_days=45.0,
        summary="TSMC disruption test summary.",
        signal_tags=["earthquake", "tsmc", "taiwan"],
    )
    state = _news_state()
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=[]):
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
                with patch("src.agents.news_agent.agent.execute_query", return_value=[]):
                    with patch(
                        "src.agents.news_agent.agent.call_openai_structured",
                        return_value=mock_llm,
                    ):
                        result = news_event_analysis_agent(state)

    assert result["news_analysis_llm"] is not None
    assert result["news_signals"][0].severity == 0.55
    assert len(result["news_signals"]) == 3  # primary + 2 regional


def test_news_agent_rule_based_fallback_on_llm_error():
    """Step 6: LLM failure uses FALLBACK_PARAMS for earthquake."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    state = _news_state()
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=[]):
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
                with patch("src.agents.news_agent.agent.execute_query", return_value=[]):
                    with patch(
                        "src.agents.news_agent.agent.call_openai_structured",
                        side_effect=RuntimeError("rate limit"),
                    ):
                        with patch("src.agents.news_agent.agent.build_news_signals", return_value=[]):
                            result = news_event_analysis_agent(state)

    assert result["news_analysis_llm"] is None
    assert result["news_signals"][0].severity == 0.55
    assert result["news_signals"][0].source_id == "fallback-primary"


def test_news_agent_rule_based_fallback_no_api_key():
    """Step 4 skipped when no API key; rule-based fallback used."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    state = _news_state()
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=[]):
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
            with patch("src.agents.news_agent.agent.build_news_signals", return_value=[]):
                with patch("src.agents.news_agent.agent.call_openai_structured") as mock_llm:
                    result = news_event_analysis_agent(state)

    mock_llm.assert_not_called()
    assert result["news_analysis_llm"] is None
    assert result["news_signals"][0].source_id == "fallback-primary"


def test_news_agent_news_count_bumps_fallback_severity():
    """>5 live news rows bumps rule-based severity by +0.05."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    live_rows = [{"title": f"article-{i}"} for i in range(6)]
    state = _news_state()
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=live_rows):
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
            with patch("src.agents.news_agent.agent.build_news_signals", return_value=[]):
                result = news_event_analysis_agent(state)

    assert result["news_signals"][0].severity == 0.60  # 0.55 + 0.05 for earthquake


def test_news_agent_regional_signals_derived():
    """Step 5: up to 3 regional signals at 0.75× primary severity."""
    from src.agents.news_agent.agent import news_event_analysis_agent

    mock_llm = NewsAnalysisLLMOutput(
        category="logistics",
        severity=0.65,
        affected_regions=["Western Europe", "Southern Europe", "West Asia", "India"],
        affected_commodities=["PCBs"],
        news_severity_component=0.72,
        expected_duration_days=180.0,
        summary="Red Sea routing disruption.",
        signal_tags=["red-sea", "logistics"],
    )
    state = _news_state(
        event_metadata=EventMetadata(
            disruption_type="port closure",
            affected_port="Rotterdam",
            affected_route="Asia-Europe",
            severity=0.6,
            shock_duration_days=2,
            recovery_window_days=90,
            synthetic_ratio=0.0,
        ),
    )
    with patch("src.agents.news_agent.agent.fetch_recent_news", return_value=[]):
        with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
                with patch("src.agents.news_agent.agent.execute_query", return_value=[]):
                    with patch(
                        "src.agents.news_agent.agent.call_openai_structured",
                        return_value=mock_llm,
                    ):
                        result = news_event_analysis_agent(state)

    assert len(result["news_signals"]) == 4  # primary + 3 regional (max 3)
    assert result["news_signals"][1].severity == round(0.72 * 0.75, 3)


def test_news_agent_requires_metadata():
    from src.agents.news_agent.agent import news_event_analysis_agent

    with pytest.raises(ValueError, match="Event metadata"):
        news_event_analysis_agent(GlobalState(event_metadata=None))


# ── Weather Agent ───────────────────────────────────────────────────────────


def test_weather_agent_sqlite_primary_path():
    """Step 3: SQLite row used; live Open-Meteo never called."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(),
    ):
        with patch("src.agents.weather_agent.agent.fetch_open_meteo") as mock_api:
            with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
                result = weather_risk_monitoring_agent(state)

    mock_api.assert_not_called()
    assert result["live_weather_severity"] == 0.743


def test_weather_agent_llm_overrides_sqlite_severity():
    """Step 7: LLM geo_risk_component overrides SQLite numeric severity."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    mock_output = WeatherRiskLLMOutput(
        event_classification="extreme",
        geo_risk_component=0.91,
        affected_semiconductor_hubs=["Hsinchu"],
        supply_chain_narrative="Test narrative",
        rag_escalation_warranted=True,
    )
    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value="ctx"):
                with patch(
                    "src.agents.weather_agent.agent.call_openai_structured",
                    return_value=mock_output,
                ):
                    result = weather_risk_monitoring_agent(state)

    assert result["live_weather_severity"] == 0.91
    assert result["weather_risk_llm"] is not None


def test_weather_agent_rule_based_fallback_on_llm_error():
    """Step 8: LLM failure returns SQLite severity unchanged."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(severity=0.55),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value=""):
                with patch(
                    "src.agents.weather_agent.agent.call_openai_structured",
                    side_effect=RuntimeError("rate limit"),
                ):
                    result = weather_risk_monitoring_agent(state)

    assert result["live_weather_severity"] == 0.55
    assert result["weather_risk_llm"] is None


def test_weather_agent_rule_based_fallback_no_api_key():
    """No API key: SQLite severity returned without LLM call."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(severity=0.42),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
            with patch("src.agents.weather_agent.agent.call_openai_structured") as mock_llm:
                result = weather_risk_monitoring_agent(state)

    mock_llm.assert_not_called()
    assert result["live_weather_severity"] == 0.42


def test_weather_agent_falls_back_to_live_api_when_no_sqlite_row():
    """Step 4: live Open-Meteo when weather_signals has no matching row."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    state = _weather_state()
    payload = {
        "hourly": {
            "windspeed_10m": [10, 12],
            "precipitation": [1, 2],
            "weathercode": [95, 95],
        }
    }
    with patch("src.agents.weather_agent.agent.fetch_latest_weather_signal", return_value=None):
        with patch("src.agents.weather_agent.agent.fetch_open_meteo", return_value=payload) as mock_api:
            with patch("src.agents.weather_agent.agent.compute_weather_severity", return_value=0.5):
                with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
                    result = weather_risk_monitoring_agent(state)

    mock_api.assert_called_once()
    assert result["live_weather_severity"] == 0.5


def test_weather_agent_rag_fetched_only_above_threshold():
    """Step 6: RAG only when numeric_severity >= 0.40."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(severity=0.55),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value="") as mock_rag:
                with patch(
                    "src.agents.weather_agent.agent.call_openai_structured",
                    return_value=WeatherRiskLLMOutput(
                        event_classification="severe",
                        geo_risk_component=0.7,
                        affected_semiconductor_hubs=["Hsinchu"],
                        supply_chain_narrative="n",
                        rag_escalation_warranted=True,
                    ),
                ):
                    weather_risk_monitoring_agent(state)
            mock_rag.assert_called_once()

    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(severity=0.25),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value="") as mock_rag:
                with patch(
                    "src.agents.weather_agent.agent.call_openai_structured",
                    return_value=WeatherRiskLLMOutput(
                        event_classification="minor",
                        geo_risk_component=0.2,
                        affected_semiconductor_hubs=[],
                        supply_chain_narrative="n",
                        rag_escalation_warranted=False,
                    ),
                ):
                    weather_risk_monitoring_agent(state)
            mock_rag.assert_not_called()


def test_weather_agent_typhoon_flag_present_in_prompt():
    """Step 5: max_wind > 32 adds typhoon warning to LLM user message."""
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    captured = {}

    def _capture_call(**kwargs):
        captured["user_message"] = kwargs.get("user_message", "")
        return WeatherRiskLLMOutput(
            event_classification="extreme",
            geo_risk_component=0.91,
            affected_semiconductor_hubs=["Hsinchu"],
            supply_chain_narrative="n",
            rag_escalation_warranted=True,
        )

    state = _weather_state()
    with patch(
        "src.agents.weather_agent.agent.fetch_latest_weather_signal",
        return_value=_sqlite_weather_row(max_wind=57.2),
    ):
        with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=True):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value=""):
                with patch(
                    "src.agents.weather_agent.agent.call_openai_structured",
                    side_effect=lambda **kw: _capture_call(**kw),
                ):
                    weather_risk_monitoring_agent(state)

    assert "TYPHOON THRESHOLD" in captured["user_message"]


def test_weather_agent_nearest_hub_hsinchu():
    """Step 2: coordinates near Hsinchu resolve to Hsinchu hub."""
    from src.agents.weather_agent.agent import _nearest_hub

    hub, _ = _nearest_hub(24.8, 120.97)
    assert hub == "Hsinchu"


def test_weather_agent_requires_metadata_and_config():
    from src.agents.weather_agent.agent import weather_risk_monitoring_agent

    with pytest.raises(ValueError, match="Event metadata and config"):
        weather_risk_monitoring_agent(
            GlobalState(
                event_metadata=None,
                config={"ports": {}},
            )
        )
    with pytest.raises(ValueError, match="Event metadata and config"):
        weather_risk_monitoring_agent(
            GlobalState(
                event_metadata=EventMetadata(
                    disruption_type="earthquake",
                    affected_port="Hsinchu",
                    affected_route="x",
                    severity=0.5,
                    shock_duration_days=0,
                    recovery_window_days=30,
                    synthetic_ratio=0.0,
                ),
                config=None,
            )
        )
