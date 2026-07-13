import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.cli import main


def test_run_command_calls_run_with_trulens_with_scenario_payload():
    with patch("src.evaluation.trulens_integration.cli.run_with_trulens") as mock_run:
        mock_run.return_value = MagicMock(risk_label="HIGH")
        exit_code = main([
            "run", "--port", "Chennai", "--sku", "CHIP-001", "--event-date", "2024-03-15",
        ])

    assert exit_code == 0
    mock_run.assert_called_once()
    payload = mock_run.call_args[0][0]
    assert payload["affected_port"] == "Chennai"
    assert payload["sku"] == "CHIP-001"
    assert payload["event_date"] == "2024-03-15"
    # EventMetadata (src/agents/state.py) requires these too — discovered as
    # a real gap during Task 13 manual verification (L1 raised a pydantic
    # ValidationError with only port/sku/event_date supplied).
    required_event_metadata_fields = {
        "disruption_type", "affected_route", "severity",
        "shock_duration_days", "recovery_window_days", "synthetic_ratio",
    }
    assert required_event_metadata_fields.issubset(payload.keys())


def test_dashboard_command_calls_launch_dashboard():
    with patch("src.evaluation.trulens_integration.cli.launch_dashboard") as mock_launch:
        exit_code = main(["dashboard"])

    assert exit_code == 0
    mock_launch.assert_called_once_with(port=8502)


def test_no_command_prints_help_and_returns_nonzero():
    exit_code = main([])
    assert exit_code != 0


def test_query_command_prints_risk_drift_score():
    with patch(
        "src.evaluation.trulens_integration.cli.fetch_recent_composite_scores",
        return_value=[0.5, 0.5, 0.5],
    ) as mock_fetch:
        exit_code = main(["query", "--metric", "risk_drift", "--days", "14"])

    assert exit_code == 0
    mock_fetch.assert_called_once_with(14)


def test_query_command_rejects_unknown_metric():
    exit_code = main(["query", "--metric", "not_a_real_metric", "--days", "30"])
    assert exit_code != 0
