#!/usr/bin/env python3
"""
One-time offline seed: run L1–L7 via run_pipeline() and persist outputs
to forecast_output, simulation_output, mitigation_output, guardrail_events.

Usage:
    python scripts/seed_demo_run.py

Prints the pipeline run_id to use in curl checks and the manual verification
checklist. Does NOT expose an HTTP endpoint — Day 9 wires the live Run button.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.langgraph_engine import run_pipeline
from src.agents.pipeline_bridge import (
    persist_forecast_output,
    persist_mitigation_output,
    persist_risk_classification_output,
    persist_simulation_output,
)
from src.utils.db_utils import (
    DB_PATH,
    ensure_schema,
    execute_query,
    fetch_guardrail_events,
    fetch_scenario_options,
    insert_guardrail_event,
)


def _pick_scenario() -> dict:
    """Choose a daily_records row with enough Prophet history (>= 10 points)."""
    options = fetch_scenario_options()
    if not options:
        raise RuntimeError("No scenario options — run: python scripts/build_databases.py")
    best = max(options, key=lambda r: r.get("history_points") or 0)
    return best


def _seed_guardrails_if_empty() -> None:
    """Insert aggregate guardrail rows when guardrail_events is empty."""
    if fetch_guardrail_events():
        return
    defaults = [
        ("prompt-injection-screen", "input", "L2", 142, 3, "Adversarial suffix detected in headline seed"),
        ("length-cap-4096", "input", "L2/L4", 145, 0, "—"),
        ("structured-output-schema", "output", "L4", 141, 4, "Missing field: verdict_confidence"),
        ("fallback-on-failure", "output", "L4", 145, 0, "—"),
        ("faithfulness-gate", "output", "L7", 138, 7, "faithfulness=0.61 < 0.75 → routed to human review"),
        ("slack-critical-flag-guard", "output", "L7", 145, 0, "—"),
    ]
    for row in defaults:
        insert_guardrail_event(*row)


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run: python scripts/build_databases.py")
        sys.exit(1)

    ensure_schema()
    scenario = _pick_scenario()
    run_id = str(uuid.uuid4())

    payload = {
        "run_id": run_id,
        "mode": "replay",
        "source_type": "REPLAY",
        "disruption_type": "geopolitical",
        "affected_port": scenario["port"],
        "affected_route": f"{scenario['port']} to Singapore",
        "severity": 0.75,
        "shock_duration_days": 14,
        "recovery_window_days": 90,
        "synthetic_ratio": 0.0,
        "event_date": scenario["event_date"],
        "sku": scenario["sku"],
    }

    print(f"Seeding pipeline run_id={run_id}")
    print(
        f"  scenario: port={scenario['port']!r} sku={scenario['sku']!r} "
        f"date={scenario['event_date']} history={scenario.get('history_points')}"
    )

    state = run_pipeline(payload)

    persist_forecast_output(run_id, state)
    persist_risk_classification_output(run_id, state)
    persist_simulation_output(run_id, state)
    persist_mitigation_output(run_id, state)
    _seed_guardrails_if_empty()

    agent_rows = execute_query(
        "SELECT COUNT(*) AS n FROM agent_execution_log WHERE run_id = ?", (run_id,)
    )
    llm_rows = execute_query(
        "SELECT COUNT(*) AS n FROM llm_call_log WHERE run_id = ?", (run_id,)
    )

    print("\nSeed complete.")
    print(f"  run_id:           {run_id}")
    print(f"  agent_exec rows:  {agent_rows[0]['n'] if agent_rows else 0}")
    print(f"  llm_call_log rows:{llm_rows[0]['n'] if llm_rows else 0}")
    print(f"  risk_label:       {state.risk_label}")
    print("\nUse this run_id in Day 8 curl checks, e.g.:")
    print(f'  curl -s "http://localhost:8000/api/forecast/{run_id}?category=Laptops"')


if __name__ == "__main__":
    main()
