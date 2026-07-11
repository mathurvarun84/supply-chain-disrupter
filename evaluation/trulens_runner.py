"""
trulens_runner.py — Runs a small set of pipeline scenarios through
run_with_trulens() and writes their risk outcomes to
evaluation/trulens_scores.json.

evaluation/ragas/test_dataset.json holds RAG gold Q&A pairs (question/
ground_truth/source_collection), not port/sku/event_date pipeline
scenarios, so it can't drive run_with_trulens. These two scenarios mirror
the real payload shape used by the Streamlit Scenario Analyzer at
src/dashboard/dashboard.py:216-229.

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
        "name": "taiwan_earthquake",
        "payload": {
            "disruption_type": "earthquake",
            "affected_port": "Eastern Asia",
            "affected_route": "Hsinchu to Singapore",
            "severity": 0.95,
            "shock_duration_days": 6,
            "recovery_window_days": 90,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "CHIP_AP",
            "event_date": "2024-04-03",
        },
    },
    {
        "name": "red_sea_crisis",
        "payload": {
            "disruption_type": "geopolitical",
            "affected_port": "Western Europe",
            "affected_route": "Suez Canal to Rotterdam",
            "severity": 0.85,
            "shock_duration_days": 14,
            "recovery_window_days": 120,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "ELECTRONICS_EU",
            "event_date": "2024-01-15",
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
