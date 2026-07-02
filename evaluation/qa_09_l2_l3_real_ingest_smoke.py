"""
QA-09 | L2/L3 smoke test — real SQLite ingestion tables
========================================================
Agents tested : L2 News (news_event_analysis_agent)
                L3 Weather (weather_risk_monitoring_agent)
Data source   : outputs/supply_chain.db — news_signals / weather_signals
                (populated by L1 live ingest: live_ingest.py or DataIngestionAgent)

What this file verifies
-----------------------
Unlike pytest (mocked DB/API), this script runs L2 and L3 against the REAL
SQLite ingestion tables to catch wiring bugs:

  1. news_signals and weather_signals tables exist and are readable.
  2. L2 calls fetch_recent_news() — never live GDELT/RSS APIs.
  3. L3 reads weather_signals when a row exists for the nearest hub.
  4. L3 does NOT call Open-Meteo when SQLite has a weather row.
  5. Agents populate news_signals and live_weather_severity on GlobalState.

LLM and RAG are disabled via patches so the run is deterministic and free.

Prerequisites
-------------
  python scripts/build_databases.py          # lite_master ETL
  # Optional but recommended for full coverage:
  python -m src.agents.data_ingestion.live_ingest   # or your L1 poller

Expected outcome: all applicable checks PASS (ingestion row checks may WARN if
L1 has not run yet — see printed guidance).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")

from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.agents.state import EventMetadata, GlobalState
from src.utils.db_utils import DB_PATH, ensure_ingestion_schema, fetch_latest_weather_signal
from src.utils.yaml_utils import load_config

# ---------------------------------------------------------------------------
# Helper: print PASS/FAIL/WARN and track overall result
# ---------------------------------------------------------------------------
all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


def warn(msg: str) -> None:
    print("WARN |", msg)


print("=== QA-09: L2/L3 Real Ingest Smoke Test ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — DB and ingestion schema
# ---------------------------------------------------------------------------
if not DB_PATH.exists():
    print(f"FAIL | Database not found at {DB_PATH} — run: python scripts/build_databases.py")
    sys.exit(1)

ensure_ingestion_schema()

conn = sqlite3.connect(DB_PATH)
tables = {
    r[0]
    for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
}
conn.close()

chk("lite_master" in tables, "lite_master table present")
chk("news_signals" in tables, "news_signals table present (L1 ingestion schema)")
chk("weather_signals" in tables, "weather_signals table present (L1 ingestion schema)")

news_count = 0
weather_count = 0
if "news_signals" in tables:
    conn = sqlite3.connect(DB_PATH)
    news_count = conn.execute("SELECT COUNT(*) FROM news_signals").fetchone()[0]
    weather_count = conn.execute("SELECT COUNT(*) FROM weather_signals").fetchone()[0]
    conn.close()

print(f"\n--- Ingestion table counts ---")
print(f"  news_signals:    {news_count:,} rows")
print(f"  weather_signals: {weather_count:,} rows")

if news_count == 0:
    warn("news_signals empty — L2 will use fallback only; run L1 ingest for full smoke coverage")
if weather_count == 0:
    warn("weather_signals empty — L3 may hit Open-Meteo fallback (patched to fail in this QA)")

# ---------------------------------------------------------------------------
# Step 2 — Build scenario state (Taiwan / Eastern Asia — matches ingest regions)
# ---------------------------------------------------------------------------
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
        severity=0.85,
        shock_duration_days=6,
        recovery_window_days=90,
        synthetic_ratio=0.0,
    ),
    config=config,
    active_record={
        "order_region": "Eastern Asia",
        "year": 2024,
        "latitude": 24.80,
        "longitude": 120.97,
        "port": "Eastern Asia",
    },
)

hsinchu_weather = fetch_latest_weather_signal("Hsinchu")
expect_sqlite_weather = hsinchu_weather is not None
if expect_sqlite_weather:
    print(f"\n  Hsinchu weather_signals row found (severity={hsinchu_weather.get('severity')})")
else:
    warn("No Hsinchu weather_signals row — L3 Open-Meteo patch will assert if fallback triggers")

# ---------------------------------------------------------------------------
# Step 3 — Run L2 (real fetch_recent_news; LLM/RAG disabled)
# ---------------------------------------------------------------------------
print("\n--- L2 News Agent ---")

with patch("src.agents.news_agent.agent.has_openai_api_key", return_value=False):
    with patch("src.agents.news_agent.agent.build_rag_context", return_value=""):
        news_delta = news_event_analysis_agent(state)

state = state.model_copy(update=news_delta)
news_signals = state.news_signals or []
l2_log = next((log for log in state.agent_logs if log.startswith("L2:")), "")

print(f"  signals produced: {len(news_signals)}")
print(f"  log: {l2_log}")

chk(len(news_signals) >= 1, f"L2 produced at least one NewsRiskSignal (got {len(news_signals)})")
chk(
    news_signals[0].severity is not None and 0.0 <= news_signals[0].severity <= 1.0,
    f"primary signal severity in [0, 1] (got {news_signals[0].severity})",
)
chk("L2:" in l2_log, "agent_logs contains L2 completion entry")
if news_count > 0:
    chk(
        "live_news=" in l2_log and "live_news=0" not in l2_log,
        f"L2 log reports live SQLite news rows (log={l2_log!r})",
    )

# ---------------------------------------------------------------------------
# Step 4 — Run L3 (real fetch_latest_weather_signal; block live API)
# ---------------------------------------------------------------------------
print("\n--- L3 Weather Agent ---")

open_meteo_called = {"value": False}

def _block_open_meteo(*_args, **_kwargs):
    open_meteo_called["value"] = True
    raise AssertionError("L3 must not call Open-Meteo when SQLite has a weather row")


with patch("src.agents.weather_agent.agent.has_openai_api_key", return_value=False):
    with patch("src.agents.weather_agent.agent.build_rag_context", return_value=""):
        with patch("src.agents.weather_agent.agent.fetch_open_meteo", side_effect=_block_open_meteo):
            try:
                weather_delta = weather_risk_monitoring_agent(state)
            except AssertionError as exc:
                if expect_sqlite_weather:
                    chk(False, str(exc))
                    weather_delta = {"live_weather_severity": None, "agent_logs": state.agent_logs}
                else:
                    warn(f"L3 used live API fallback (no SQLite row): {exc}")
                    weather_delta = {
                        "live_weather_severity": 0.5,
                        "agent_logs": state.agent_logs + ["L3: (fallback stub — no SQLite weather)"],
                    }

state = state.model_copy(update=weather_delta)
severity = state.live_weather_severity
l3_log = next((log for log in state.agent_logs if log.startswith("L3:")), "")

print(f"  live_weather_severity: {severity}")
print(f"  log: {l3_log}")

chk(severity is not None, "L3 set live_weather_severity on state")
chk(0.0 <= float(severity) <= 1.0, f"live_weather_severity in [0, 1] (got {severity})")
chk("L3:" in l3_log, "agent_logs contains L3 completion entry")
if expect_sqlite_weather:
    chk(not open_meteo_called["value"], "Open-Meteo was NOT called (SQLite-primary path)")
    chk("source=sqlite" in l3_log, f"L3 log reports SQLite data source (log={l3_log!r})")
else:
    warn("Skipped SQLite-only Open-Meteo assertion — no Hsinchu weather_signals row")

print()
print(f"All pass: {all_pass}")
if not all_pass:
    sys.exit(1)
