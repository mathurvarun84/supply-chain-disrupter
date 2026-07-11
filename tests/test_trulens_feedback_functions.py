import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.feedback_functions import (
    ensemble_agreement, forecast_accuracy, node_latency_check, risk_score_stability,
)


def test_risk_score_stability_is_1_when_scores_identical():
    assert risk_score_stability([0.5, 0.5, 0.5]) == 1.0


def test_risk_score_stability_drops_with_high_variance():
    score = risk_score_stability([0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= score < 0.7


def test_risk_score_stability_handles_fewer_than_two_scores():
    assert risk_score_stability([]) == 1.0
    assert risk_score_stability([0.5]) == 1.0


def test_ensemble_agreement_all_agree():
    triples = [("HIGH", "HIGH", "HIGH"), ("LOW", "LOW", "LOW")]
    assert ensemble_agreement(triples) == 1.0


def test_ensemble_agreement_majority_counts_as_agreement():
    # 2 of 3 match -> counted as agreement
    triples = [("HIGH", "HIGH", "MEDIUM")]
    assert ensemble_agreement(triples) == 1.0


def test_ensemble_agreement_no_majority_counts_as_disagreement():
    triples = [("HIGH", "MEDIUM", "LOW")]
    assert ensemble_agreement(triples) == 0.0


def test_ensemble_agreement_empty_list_is_perfect_by_convention():
    assert ensemble_agreement([]) == 1.0


def test_node_latency_check_all_pass():
    latencies = {"L2": 1000.0, "L3": 1500.0, "L4": 4000.0, "total": 10000.0}
    assert node_latency_check(latencies) == 1.0


def test_node_latency_check_partial_pass():
    latencies = {"L2": 3000.0, "L3": 1500.0, "L4": 4000.0, "total": 10000.0}
    score = node_latency_check(latencies)
    assert 0.0 < score < 1.0


def test_node_latency_check_empty_dict_is_perfect_by_convention():
    assert node_latency_check({}) == 1.0


def test_forecast_accuracy_perfect_prediction():
    assert forecast_accuracy(predicted_drop_pct=10.0, actual_drop_pct=10.0) == 1.0


def test_forecast_accuracy_worst_case_clips_to_zero():
    score = forecast_accuracy(predicted_drop_pct=0.0, actual_drop_pct=100.0)
    assert score == 0.0

def test_forecast_accuracy_both_zero_is_perfect_by_convention():
    assert forecast_accuracy(predicted_drop_pct=0.0, actual_drop_pct=0.0) == 1.0
