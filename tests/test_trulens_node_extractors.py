import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import (
    DistilBERTSignal, ForecastResult, GlobalState, LLMSignal,
    RiskClassificationResult, RuleBasedSignal,
)
from src.evaluation.trulens_integration.node_extractors import (
    NODE_LATENCY_LABELS, extract_l4_signals, extract_l5_forecast,
)


def _rule_signal(label: str) -> RuleBasedSignal:
    return RuleBasedSignal(
        composite_score=0.8, geo_component=0.4, supply_component=0.2,
        freight_component=0.1, defect_component=0.1, base_label=label,
        escalated_label=label, escalated=False, duration_days=None,
    )


def test_extract_l4_signals_returns_none_without_classification():
    state = GlobalState()
    assert extract_l4_signals(state) is None


def test_extract_l4_signals_pulls_all_three_labels():
    state = GlobalState(
        risk_classification=RiskClassificationResult(
            mode="live", composite_score=0.82, geo_component=0.4,
            supply_component=0.2, freight_component=0.1, defect_component=0.1,
            duration_days=None, base_label="CRITICAL", final_label="CRITICAL",
            escalated=False, rationale="r", critical_flag=True,
            rule_signal=_rule_signal("CRITICAL"),
            distilbert_signal=DistilBERTSignal(
                predicted_label="HIGH", confidence=0.7,
                probability_distribution={"HIGH": 0.7}, model_source="ft",
                inference_ms=20.0,
            ),
            llm_signal=LLMSignal(
                predicted_label="CRITICAL", rationale="r", rag_citations=[],
                rag_chunks_used=2, confidence_level="high", primary_driver="geo",
            ),
        )
    )

    result = extract_l4_signals(state)

    assert result == {
        "composite_score": 0.82,
        "rule_label": "CRITICAL",
        "distilbert_label": "HIGH",
        "llm_label": "CRITICAL",
    }


def test_extract_l5_forecast_returns_none_without_result():
    assert extract_l5_forecast(GlobalState()) is None


def test_extract_l5_forecast_pulls_expected_drop():
    state = GlobalState(
        forecast_result=ForecastResult(prophet_forecast=[], expected_drop_pct=12.5)
    )
    assert extract_l5_forecast(state) == {"expected_drop_pct": 12.5}


def test_node_latency_labels_cover_critical_path():
    assert NODE_LATENCY_LABELS["l2_news_analysis"] == "L2"
    assert NODE_LATENCY_LABELS["l3_weather_monitoring"] == "L3"
    assert NODE_LATENCY_LABELS["l4_risk_classifier"] == "L4"
