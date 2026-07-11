"""
trulens_runner.py — Runs a small set of pipeline scenarios through
run_with_trulens() and writes their risk outcomes to
evaluation/trulens_scores.json.

evaluation/ragas/test_dataset.json holds RAG gold Q&A pairs (question/
ground_truth/source_collection), not port/sku/event_date pipeline
scenarios, so it can't drive run_with_trulens. These two scenarios mirror
the real payload shape used by the Streamlit Scenario Analyzer at
src/dashboard/dashboard.py:216-229.

`affected_port`/`sku`/`event_date` must be an EXACT match against a real
`daily_records` row (verified via `SELECT DISTINCT port, sku, event_date
FROM daily_records`) — L1 finds no active_record on a non-match, and L3's
get_port_coordinates() only resolves the 7 India ports in
config/india_electronics.yaml when active_record has no lat/lon of its own
(a row that matched supplies its own). Discovered during Task 13 manual
verification: the original placeholder values ("Eastern Asia"/"CHIP_AP"/
"2024-04-03", styled after a Taiwan-earthquake scenario) don't exist in the
real workbook and raised KeyError/ValueError instead of running.

Usage: python -m evaluation.trulens_runner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.wrapper import run_with_trulens

OUTPUT_PATH = Path(__file__).parent / "trulens_scores.json"

SCENARIOS: list[dict] = [
    {
        "name": "central_america_earbuds",
        "payload": {
            "disruption_type": "logistics",
            "affected_port": "Central America",
            "affected_route": "Panama Canal to Central America",
            "severity": 0.6,
            "shock_duration_days": 3,
            "recovery_window_days": 30,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "Samsung Galaxy Buds Wireless Earbuds",
            "event_date": "2015-01-01",
        },
    },
    {
        "name": "western_europe_laptop",
        "payload": {
            "disruption_type": "geopolitical",
            "affected_port": "Western Europe",
            "affected_route": "Suez Canal to Rotterdam",
            "severity": 0.85,
            "shock_duration_days": 14,
            "recovery_window_days": 120,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "HP Spectre x360 14",
            "event_date": "2025-12-29",
        },
    },
]


def main() -> int:
    results = []
    for scenario in SCENARIOS:
        state = run_with_trulens(scenario["payload"])
        results.append({
            "scenario": scenario["name"],
            "risk_label": state.risk_label,
            "composite_score": state.risk_score_composite,
        })

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} scenario result(s) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
