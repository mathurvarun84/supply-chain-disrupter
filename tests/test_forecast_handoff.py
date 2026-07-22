from types import SimpleNamespace

from src.agents.forecast.agent import demand_forecasting_agent


class DummyState:
    def __init__(self, handoff=None, active_record=None):
        self.forecast_handoff = handoff
        self.active_record = active_record or {}
        self.agent_logs = []
        self.risk_classification = None


def test_demand_forecasting_prefers_forecast_handoff(monkeypatch):
    captured = {}

    class DummyAgent:
        def __init__(self):
            pass

        def run(self, sku_id, disruption_scenario=None):
            captured["sku_id"] = sku_id
            captured["disruption_scenario"] = disruption_scenario
            return SimpleNamespace(
                sku_id=sku_id,
                demand_forecast=[],
                expected_drop_pct=0.0,
                stockout_prob=0.0,
                model_selected="prophet",
                model_comparison_scores={},
                regressors_used=[],
                regressor_selection_method="backtest_ablation",
                disruption_scenario=disruption_scenario,
                agent_logs=[],
                mape_prophet_trend_only=None,
                mape_prophet_selected=None,
                mape_dataset_baseline_avg=None,
                mape_dataset_ai_avg=None,
                mape_improvement_pct_vs_dataset_baseline=None,
            )

    monkeypatch.setattr("src.agents.forecast.agent.DemandForecastingAgent", DummyAgent)
    monkeypatch.setattr("src.agents.forecast.agent._write_forecast_to_db", lambda result: None)

    state = DummyState(
        handoff=SimpleNamespace(
            sku_id="SKU777", risk_score_composite=0.91, risk_label="CRITICAL", duration_days=21.0
        ),
        active_record={"sku_id": "SKU001", "SKU_ID": "SKU001"},
    )

    demand_forecasting_agent(state)

    assert captured["sku_id"] == "SKU777"
    assert captured["disruption_scenario"] == {
        "disruption_flag": 1,
        "risk_score_composite": 0.91,
        "duration_days": 21.0,
    }


def test_demand_forecasting_does_not_force_disruption_flag_for_low_risk(monkeypatch):
    """disruption_flag must reflect L4's real label, not be hardcoded to 1 —
    a LOW-risk handoff should feed the disrupted-scenario regressor/stockout
    classifier disruption_flag=0, or L5 keeps reporting a disruption even
    when L4 found none."""
    captured = {}

    class DummyAgent:
        def __init__(self):
            pass

        def run(self, sku_id, disruption_scenario=None):
            captured["disruption_scenario"] = disruption_scenario
            return SimpleNamespace(
                sku_id=sku_id,
                demand_forecast=[],
                expected_drop_pct=0.0,
                stockout_prob=0.0,
                model_selected="prophet",
                model_comparison_scores={},
                regressors_used=[],
                regressor_selection_method="backtest_ablation",
                disruption_scenario=disruption_scenario,
                agent_logs=[],
                mape_prophet_trend_only=None,
                mape_prophet_selected=None,
                mape_dataset_baseline_avg=None,
                mape_dataset_ai_avg=None,
                mape_improvement_pct_vs_dataset_baseline=None,
            )

    monkeypatch.setattr("src.agents.forecast.agent.DemandForecastingAgent", DummyAgent)
    monkeypatch.setattr("src.agents.forecast.agent._write_forecast_to_db", lambda result: None)

    state = DummyState(
        handoff=SimpleNamespace(
            sku_id="SKU777", risk_score_composite=0.12, risk_label="LOW", duration_days=None
        ),
        active_record={"sku_id": "SKU001", "SKU_ID": "SKU001"},
    )

    demand_forecasting_agent(state)

    assert captured["disruption_scenario"] == {
        "disruption_flag": 0,
        "risk_score_composite": 0.12,
        "duration_days": None,
    }
