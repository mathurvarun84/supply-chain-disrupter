"""
QA-12 | L6 Simulation Agent — Monte Carlo impact ranges
========================================================
Verifies L6 simulation_agent produces percentile bands, persists to
simulation_runs, and propagates through L7 mitigation.

Prerequisites: python -m src.build_databases
"""

from __future__ import annotations

import os
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, ".")

from src.agents.data_ingestion.agent import data_ingestion_agent
from src.agents.langgraph_engine import _INGESTION_V2_AVAILABLE
from src.agents.mitigation_agent import mitigation_recommendation_agent
from src.agents.simulation_agent import simulation_agent
from src.agents.state import GlobalState, RiskClassificationResult
from src.utils.db_utils import DB_PATH, ensure_schema, ensure_simulation_schema

all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


def _run_l1(state: GlobalState, payload: dict) -> GlobalState:
    if _INGESTION_V2_AVAILABLE:
        try:
            from src.agents.data_ingestion_agent import data_ingestion_agent_v2

            delta = data_ingestion_agent_v2(state, payload)
        except Exception:
            delta = data_ingestion_agent(state, payload)
    else:
        delta = data_ingestion_agent(state, payload)
    return state.model_copy(update=delta)


print("=== QA-12: L6 Simulation Agent ===")
print()

if not DB_PATH.exists():
    print(f"FAIL | Database not found at {DB_PATH} — run: python -m src.build_databases")
    sys.exit(1)

ensure_schema()
ensure_simulation_schema()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
row = conn.execute(
    """
    SELECT port, sku, MAX(event_date) AS event_date
    FROM daily_records
    WHERE category_name IN ('Cameras', 'Computers', 'Consumer Electronics', 'Video Games')
    GROUP BY port, sku
    HAVING COUNT(DISTINCT event_date) >= 3
    ORDER BY COUNT(DISTINCT event_date) DESC
    LIMIT 1
    """
).fetchone()
conn.close()

if row is None:
    print("FAIL | No scenario row found in daily_records")
    sys.exit(1)

payload = {
    "disruption_type": "port closure",
    "affected_port": row["port"],
    "affected_route": "Supplier to destination",
    "severity": 0.75,
    "shock_duration_days": 7,
    "recovery_window_days": 45,
    "synthetic_ratio": 0.0,
    "sku": row["sku"],
    "event_date": row["event_date"],
}

os.environ["SIMULATION_TRIALS"] = "300"

before_count = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM simulation_runs").fetchone()[0]

state = GlobalState()
state = _run_l1(state, payload)
chk(state.active_record is not None, "L1 active_record loaded")

risk = RiskClassificationResult(
    mode="live",
    composite_score=0.62,
    geo_component=0.5,
    supply_component=0.6,
    freight_component=0.55,
    defect_component=0.4,
    duration_days=7.0,
    base_label="HIGH",
    final_label="HIGH",
    escalated=False,
    rationale="QA-12 injected risk for L6 path",
    critical_flag=False,
)
state = state.model_copy(update={"risk_classification": risk})
chk(state.risk_classification is not None, "L4 risk_classification populated")

l6_delta = simulation_agent(state)
state = state.model_copy(update=l6_delta)
sim = state.simulation_result

chk(sim is not None, "L6 simulation_result populated")
if sim:
    chk(sim.trials_run >= 100, f"trials_run >= 100 (got {sim.trials_run})")
    chk(sim.stockout_probability_p10 is not None, "stockout P10 present")
    chk(sim.stockout_probability_p90 is not None, "stockout P90 present")
    chk(sim.revenue_impact_usd_p50 is not None, "revenue P50 present")
    chk(
        sim.stockout_probability_p10 <= sim.stockout_probability_p90,
        "severity percentiles ordered P10 <= P90",
    )
    chk(len(sim.revenue_impact_samples) > 0, "revenue samples for histogram")

with patch("src.agents.mitigation_agent.insert_mitigation_action"):
    l7_delta = mitigation_recommendation_agent(state)
state = state.model_copy(update=l7_delta)

chk(state.mitigation_action is not None, "L7 mitigation_action populated")
if state.mitigation_action:
    joined = " ".join(state.mitigation_action.recommendations).lower()
    chk("p50" in joined or "stockout" in joined, "L7 references simulation stockout ranges")

after_count = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM simulation_runs").fetchone()[0]
chk(after_count > before_count, f"simulation_runs row persisted ({before_count} -> {after_count})")

print()
if all_pass:
    print("RESULT: PASS — QA-12 complete")
    sys.exit(0)
print("RESULT: FAIL — see failures above")
sys.exit(1)
