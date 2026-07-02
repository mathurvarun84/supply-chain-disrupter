"""
QA-11 | LangGraph slice — L1 -> L2 -> L3 integration
===================================================
Agents tested : L1 Data Ingestion (data_ingestion_agent_v2 or legacy shim)
                L2 News (news_event_analysis_agent)
                L3 Weather (weather_risk_monitoring_agent)
Data source   : Real daily_records row from outputs/supply_chain.db

What this file verifies
-----------------------
State propagation across the first three LangGraph layers (critical path prefix
of run_agent_graph() in langgraph_engine.py):

  1. L1 loads active_record + config + event_metadata from payload + DB.
  2. L2 reads that active_record (order_region, year) and writes news_signals.
  3. L3 reads coordinates/config and writes live_weather_severity.
  4. agent_logs accumulate L1, L2, L3 entries in order.

Stops before L4 Risk Classifier. LLM and live Open-Meteo are patched off so
the integration test is deterministic and does not require OPENAI_API_KEY.

Prerequisites: python scripts/build_databases.py

Expected outcome: all 7 assertions PASS.
"""

from __future__ import annotations

import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from src.agents.data_ingestion.agent import data_ingestion_agent
from src.agents.langgraph_engine import _INGESTION_V2_AVAILABLE
from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.agents.state import GlobalState
from src.utils.db_utils import DB_PATH, ensure_risk_classification_table

# ---------------------------------------------------------------------------
# Helper: print PASS/FAIL and track overall result
# ---------------------------------------------------------------------------
all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


def _run_l1(state: GlobalState, payload: dict) -> GlobalState:
    """Mirror langgraph_engine L1 branch without importing side-effect-heavy v2."""
    if _INGESTION_V2_AVAILABLE:
        try:
            from src.agents.data_ingestion_agent import data_ingestion_agent_v2

            delta = data_ingestion_agent_v2(state, payload)
        except Exception:
            delta = data_ingestion_agent(state, payload)
    else:
        delta = data_ingestion_agent(state, payload)
    return state.model_copy(update=delta)


print("=== QA-11: LangGraph L1 -> L2 -> L3 Integration ===")
print()

if not DB_PATH.exists():
    print(f"FAIL | Database not found at {DB_PATH} — run: python scripts/build_databases.py")
    sys.exit(1)

ensure_risk_classification_table()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
row = conn.execute(
    """
    SELECT event_date, port, sku, order_region, year
    FROM daily_records
    WHERE order_region IS NOT NULL
    ORDER BY record_id
    LIMIT 1
    """
).fetchone()
conn.close()

if row is None:
    print("FAIL | No daily_records row found — run ETL first.")
    sys.exit(1)

record_preview = dict(row)
print(
    f"Using daily_records row: event_date={record_preview['event_date']} "
    f"port={record_preview['port']!r} sku={record_preview.get('sku')!r} "
    f"region={record_preview.get('order_region')!r}"
)
print()

payload = {
    "disruption_type": "geopolitical",
    "affected_port": record_preview["port"],
    "affected_route": f"{record_preview['port']} to Singapore",
    "severity": 0.65,
    "shock_duration_days": 14,
    "recovery_window_days": 90,
    "synthetic_ratio": 0.0,
    "event_date": record_preview["event_date"],
    "sku": record_preview.get("sku") or "CHIP_AP",
}

state = GlobalState()

# ---------------------------------------------------------------------------
# L1 → L2 → L3 (LLM off; block Open-Meteo unless QA-09-style skip needed)
# ---------------------------------------------------------------------------
open_meteo_called = False


def _track_open_meteo(*_args, **_kwargs):
    global open_meteo_called
    open_meteo_called = True
    return {
        "hourly": {
            "windspeed_10m": [12.0],
            "precipitation": [0.5],
            "weathercode": [61],
        }
    }


with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
    with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
        with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
            with patch("src.agents.weather_agent.agent.build_rag_context", return_value=""):
                with patch(
                    "src.agents.weather_agent.agent.fetch_open_meteo",
                    side_effect=_track_open_meteo,
                ):
                    state = _run_l1(state, payload)
                    l1_logs = len(state.agent_logs)

                    news_delta = news_event_analysis_agent(state)
                    state = state.model_copy(update=news_delta)
                    l2_logs = len(state.agent_logs)

                    weather_delta = weather_risk_monitoring_agent(state)
                    state = state.model_copy(update=weather_delta)
                    l3_logs = len(state.agent_logs)

# ---------------------------------------------------------------------------
# Assertions — state fields and log chain
# ---------------------------------------------------------------------------
print("--- Pipeline output ---")
print(f"  active_record loaded:  {state.active_record is not None}")
print(f"  news_signals count:    {len(state.news_signals or [])}")
print(f"  live_weather_severity: {state.live_weather_severity}")
print(f"  agent_logs ({l3_logs} entries):")
for entry in state.agent_logs:
    print(f"    - {entry}")
print()

chk(state.event_metadata is not None, "L1 set event_metadata on state")
chk(state.config is not None, "L1 loaded config on state")
chk(state.active_record is not None, "L1 loaded active_record from daily_records")
chk(l1_logs >= 1 and l2_logs > l1_logs and l3_logs > l2_logs, "agent_logs grew after each layer")
chk(any(log.startswith("L1:") for log in state.agent_logs), "agent_logs contains L1 entry")
chk(any(log.startswith("L2:") for log in state.agent_logs), "agent_logs contains L2 entry")
chk(any(log.startswith("L3:") for log in state.agent_logs), "agent_logs contains L3 entry")
chk(len(state.news_signals or []) >= 1, "L2 populated news_signals")
chk(
    state.live_weather_severity is not None
    and 0.0 <= float(state.live_weather_severity) <= 1.0,
    f"L3 populated live_weather_severity in [0,1] (got {state.live_weather_severity})",
)

if open_meteo_called:
    print("WARN | L3 used Open-Meteo fallback (no weather_signals row for nearest hub)")
else:
    chk(True, "L3 used SQLite weather path (Open-Meteo not called)")

print()
print(f"All pass: {all_pass}")
if not all_pass:
    sys.exit(1)
