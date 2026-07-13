import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.trulens_runner import SCENARIOS, main


def test_scenarios_have_required_payload_keys():
    required = {"disruption_type", "affected_port", "affected_route", "severity",
                "shock_duration_days", "recovery_window_days", "synthetic_ratio",
                "simulation_trials", "sku", "event_date"}
    assert len(SCENARIOS) >= 2
    for scenario in SCENARIOS:
        assert required.issubset(scenario["payload"].keys())


def test_main_writes_trulens_scores_json(tmp_path):
    output_path = tmp_path / "trulens_scores.json"
    fake_result = MagicMock()
    fake_result.risk_label = "CRITICAL"
    fake_result.risk_score_composite = 0.9

    with patch("evaluation.trulens_runner.run_with_trulens", return_value=fake_result):
        with patch("evaluation.trulens_runner.OUTPUT_PATH", output_path):
            exit_code = main()

    assert exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert len(data) == len(SCENARIOS)
    assert data[0]["risk_label"] == "CRITICAL"
