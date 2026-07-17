import sys
import types
from types import SimpleNamespace

m = types.ModuleType('statsmodels')
m.tsa = types.ModuleType('statsmodels.tsa')
m.tsa.statespace = types.ModuleType('statsmodels.tsa.statespace')
m.tsa.statespace.sarimax = types.ModuleType('statsmodels.tsa.statespace.sarimax')
m.tsa.statespace.sarimax.SARIMAX = object
sys.modules['statsmodels'] = m
sys.modules['statsmodels.tsa'] = m.tsa
sys.modules['statsmodels.tsa.statespace'] = m.tsa.statespace
sys.modules['statsmodels.tsa.statespace.sarimax'] = m.tsa.statespace.sarimax

import src.agents.forecast.agent as forecast_agent

captured = {}

class DummyAgent:
    def __init__(self):
        pass

    def run(self, sku_id, disruption_scenario=None):
        captured['sku_id'] = sku_id
        captured['disruption_scenario'] = disruption_scenario
        return SimpleNamespace(
            sku_id=sku_id,
            demand_forecast=[],
            expected_drop_pct=0.0,
            stockout_prob=0.0,
            model_selected='prophet',
            model_comparison_scores={},
            regressors_used=[],
            regressor_selection_method='backtest_ablation',
            disruption_scenario=disruption_scenario,
            agent_logs=[],
            mape_prophet_trend_only=None,
            mape_prophet_selected=None,
            mape_dataset_baseline_avg=None,
            mape_dataset_ai_avg=None,
            mape_improvement_pct_vs_dataset_baseline=None,
        )

forecast_agent.DemandForecastingAgent = DummyAgent
forecast_agent._write_forecast_to_db = lambda result: None
state = SimpleNamespace(
    forecast_handoff=SimpleNamespace(sku_id='SKU777', risk_score_composite=0.91),
    active_record={'sku_id': 'SKU001', 'SKU_ID': 'SKU001'},
    agent_logs=[],
    risk_classification=None,
)
forecast_agent.demand_forecasting_agent(state)
print(captured)
