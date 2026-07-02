"""
QA-10 | Taiwan earthquake — L2 + L3 standalone scenario
=========================================================
Agents tested : L2 News (news_event_analysis_agent)
                L3 Weather (weather_risk_monitoring_agent)
Data source   : Scenario fixture modelled on April 2024 Hualien M7.4 event
                (TSMC Hsinchu fabs; complements QA-05 which tests L4 downstream)

What this file verifies
-----------------------
Documented capstone scenario for the enrichment layer BEFORE risk classification:

  L2 (News):
    - disruption_type=earthquake triggers FALLBACK_PARAMS calibration
      (news_severity_component=0.55, duration=45d, category=weather)
    - Produces at least one NewsRiskSignal suitable for L4 freight component

  L3 (Weather):
    - Resolves nearest hub to Hsinchu from coordinates
    - Uses SQLite weather severity when available; otherwise scenario fixture row
    - Sets live_weather_severity in high-risk range for earthquake context

LLM is disabled for deterministic PASS/FAIL output suitable for evaluation reports.
Open-Meteo is blocked to prove SQLite-first design.

Expected outcome: all 8 assertions PASS.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.agents.state import EventMetadata, GlobalState
from src.utils.db_utils import ensure_ingestion_schema, fetch_latest_weather_signal
from src.utils.yaml_utils import load_config

# ---------------------------------------------------------------------------
# Helper: print PASS/FAIL and track overall result
# ---------------------------------------------------------------------------
all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


# Scenario fixture: Open-Meteo-equivalent severity for Hsinchu earthquake context
SCENARIO_WEATHER_ROW = {
    "hub": "Hsinchu",
    "severity": 0.743,
    "wind_score": 0.65,
    "precipitation_score": 0.09,
    "weather_code_score": 0.003,
    "max_wind_speed": 57.2,
    "max_precipitation": 2.1,
    "weather_summary": "Max wind 57.2 km/h; severity 0.743 (scenario fixture).",
}

print("=== QA-10: Taiwan Earthquake — L2 + L3 Standalone Scenario ===")
print()

config = load_config()
config = {
    **config,
    "ports": {
        **config.get("ports", {}),
        "Hsinchu": {"latitude": 24.80, "longitude": 120.97},
    },
}

state = GlobalState(
    event_metadata=EventMetadata(
        disruption_type="earthquake",
        affected_port="Hsinchu",
        affected_route="Hsinchu to Singapore",
        severity=0.95,
        shock_duration_days=6,
        recovery_window_days=90,
        synthetic_ratio=1.0,
    ),
    config=config,
    active_record={
        "order_id": None,
        "order_date": "2024-04-03",
        "event_date": "2024-04-03",
        "port": "Eastern Asia",
        "order_region": "Eastern Asia",
        "year": 2024,
        "latitude": 24.80,
        "longitude": 120.97,
        "natural_disaster_risk": 9.8,
        "supply_disruption_index": 9.5,
    },
)

# Use real SQLite row when present; otherwise inject scenario fixture for the report.
ensure_ingestion_schema()
try:
    real_weather = fetch_latest_weather_signal("Hsinchu")
except RuntimeError:
    real_weather = None
weather_source = "sqlite" if real_weather else "scenario_fixture"
weather_row = real_weather or SCENARIO_WEATHER_ROW

print(f"Weather data source: {weather_source}")
print(f"  hub=Hsinchu  severity={weather_row.get('severity')}")
print()

# ---------------------------------------------------------------------------
# L2 — News agent (fallback path = deterministic earthquake calibration)
# ---------------------------------------------------------------------------
with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
    with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
        news_delta = news_event_analysis_agent(state)

state = state.model_copy(update=news_delta)
signals = state.news_signals or []
primary = signals[0]
l2_log = next((log for log in state.agent_logs if log.startswith("L2:")), "")

print("--- L2 News output ---")
print(f"  category:   {primary.category}")
print(f"  severity:   {primary.severity:.3f}  (news_severity_component for L4 freight)")
print(f"  duration:   {primary.expected_duration_days} days")
print(f"  summary:    {primary.summary[:90]}...")
print(f"  log:        {l2_log}")
print()

chk(len(signals) >= 1, f"L2 produced news_signals (got {len(signals)})")
chk(primary.category == "weather", f"earthquake maps to category='weather' (got {primary.category!r})")
chk(
    abs(primary.severity - 0.55) < 0.01,
    f"earthquake fallback news_severity_component ~= 0.55 (got {primary.severity})",
)
chk(
    primary.expected_duration_days == 45.0,
    f"earthquake fallback duration=45d (got {primary.expected_duration_days})",
)

# ---------------------------------------------------------------------------
# L3 — Weather agent (SQLite or scenario fixture; block live API)
# ---------------------------------------------------------------------------
open_meteo_called = False


def _block_open_meteo(*_args, **_kwargs):
    global open_meteo_called
    open_meteo_called = True
    raise AssertionError("Open-Meteo must not be called in Taiwan earthquake scenario QA")


with patch("src.agents.weather_agent.agent.fetch_latest_weather_signal", return_value=weather_row):
    with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
        with patch("src.agents.weather_agent.agent.build_rag_context", return_value=""):
            with patch("src.agents.weather_agent.agent.fetch_open_meteo", side_effect=_block_open_meteo):
                weather_delta = weather_risk_monitoring_agent(state)

state = state.model_copy(update=weather_delta)
geo = state.live_weather_severity
l3_log = next((log for log in state.agent_logs if log.startswith("L3:")), "")

print("--- L3 Weather output ---")
print(f"  live_weather_severity: {geo:.3f}  (geo component for L4)")
print(f"  log:                   {l3_log}")
print()

chk(not open_meteo_called, "Open-Meteo was NOT called (SQLite-first path)")
chk("hub=Hsinchu" in l3_log, f"L3 resolved nearest hub to Hsinchu (log={l3_log!r})")
chk(geo is not None and geo >= 0.40, f"earthquake scenario severity >= 0.40 (got {geo})")
chk(
    abs(float(geo) - float(weather_row["severity"])) < 0.001,
    f"fallback path preserves SQLite/fixture numeric severity (got {geo})",
)

print()
print(f"All pass: {all_pass}")
if not all_pass:
    sys.exit(1)
