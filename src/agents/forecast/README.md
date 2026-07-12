# L5 Demand Forecasting Agent (v4)

Multi-model weekly demand forecasting agent, integrated as the **L5 node** in
the LangGraph supply-chain disruption pipeline. Architecture reference:
`Agent_Architecture_Spec_P8_v2.docx` → "L5 Demand Forecasting Agent".

Includes a standalone Flask dashboard (`src/forecast_dashboard.py`) for
interactive model comparison and live disruption scenario exploration.

---

## Pipeline position

```
L1 Data Ingestion
      ↓
L2 News Agent  +  L3 Weather Agent
      ↓
L4 Risk Classifier  ──→  composite_score (0–1), final_label
      ↓
L5 Demand Forecasting Agent  ◄── (this agent)
      ↓
L6 Simulation Agent  (Monte Carlo stockout)
      ↓
L7 Mitigation Agent
```

L5 runs immediately after L4. The `composite_score` from L4's
`RiskClassificationResult` is passed in as the disruption scenario's
`risk_score_composite`, so the disrupted forecast reflects the actual
classified risk level rather than a hard-coded assumption.

---

## What it does

For each SKU, backtests **four candidate models** on an 8-week holdout and
picks the production forecaster by lowest SMAPE:

| Model | Exogenous inputs | Role |
|---|---|---|
| Naive seasonal (52-week lag) | none | Floor check only — never selected as winner; just confirms the real models earn their complexity |
| **Prophet** | Per-SKU ablation-selected subset of `{Disruption_Flag, risk_score_composite, chip_price_index, market_growth_rate}` | Multiplicative yearly seasonality, no weekly (already weekly-aggregated input) |
| **SARIMAX** | `Disruption_Flag`, `risk_score_composite` | ARIMA(1,1,1) with exogenous regressors via statsmodels |
| **TimeGPT** (Nixtla) | `Disruption_Flag`, `risk_score_composite` | Optional — gracefully skipped and logged if `NIXTLA_API_KEY` is not set or network is unreachable |

Every candidate is scored on **RMSE, RMSLE, SMAPE, MAPE, and wall-clock
latency** (fit + predict). Latency matters because SARIMAX typically runs
5–10× slower than Prophet per SKU — worth knowing before scaling to hundreds
of SKUs.

The **winning model** is then fit on the **full history** and produces:
- A **5-week forecast**: baseline (business-as-usual) vs. disrupted
  (driven by `Disruption_Flag=1` + the L4 risk score or a custom scenario)
- `expected_drop_pct` — percentage demand drop over the 5-week horizon
- `stockout_prob` — from a lightweight logistic regression
  (`Disruption_Flag + risk_score_composite → Stockout_Flag`)

If TimeGPT wins but its production forecast call fails, the agent falls back
to SARIMAX automatically and logs the reason.

### Pooled LightGBM challenger

`evaluate_lightgbm_challenger()` is available for **backtest-only monitoring**
— it pools all SKUs, trains one LGBMRegressor with lag features, week-of-year,
and all regressors, and returns a per-SKU score DataFrame. It is **not** wired
into `run()` and is not part of the production pipeline. Install with
`pip install lightgbm` to use it.

---

## Data scope

| Sheet | Rows | Key columns |
|---|---|---|
| `Ops KPI (Filled)` | 46 SKUs × ~160 weeks | `SKU_ID`, `Week_Start`, `Demand_Actual`, `Disruption_Flag`, `Stockout_Flag`, `MAPE_Baseline`, `MAPE_AI`, `Is_Synthetic` |
| `Lite Master` | ~7 079 rows | `Order_Date`, `Risk_Score_Composite`, `Chip_Price_Index`, `Market_Growth_Rate` |
| `SKU_Product_Mapping` | 46 rows | `sku_id`, `product_name`, `sku_price`, `category_name` |

**Important join caveat:** There is no verified shared key between `Ops KPI`
(SKU001-style IDs) and `Lite Master` (product names). Lite Master regressors
(`risk_score_composite`, `chip_price_index`, `market_growth_rate`) are joined
to ops_kpi **by date only** via `merge_asof` — they act as market-wide weekly
signals, not per-SKU signals. The `SKU_Product_Mapping` crosswalk was built
via price-proximity optimal matching (Hungarian algorithm) and is a
best-effort synthetic construct.

`Is_Synthetic` flags rows added to extend the series to the present; these
rows are clearly marked and visible in the train/test evaluation output.

---

## Regressor selection (Prophet only)

Six regressor combinations are tested per SKU via nested holdout backtest
ablation. The combination with the lowest holdout MAPE is selected:

| Candidate name | Regressors |
|---|---|
| `trend_only` | *(none — trend + yearly seasonality only)* |
| `disruption_only` | `Disruption_Flag` |
| `risk_only` | `risk_score_composite` |
| `disruption_risk` | `Disruption_Flag`, `risk_score_composite` |
| `chip_growth_only` | `chip_price_index`, `market_growth_rate` |
| `all_four` | `Disruption_Flag`, `risk_score_composite`, `chip_price_index`, `market_growth_rate` |

SARIMAX and TimeGPT always use the fixed exogenous set
`[Disruption_Flag, risk_score_composite]`.

Earlier category-proxy-based regressor gating (assigning chip/growth
regressors only to "chip-relevant" price bands) was tested and did not
reliably reduce MAPE because the category proxy itself is unverified.
Per-SKU data-driven selection is used instead.

---

## Output contract

### `_DFAResult` (internal dataclass, within this module)

```python
@dataclass
class _DFAResult:
    sku_id: str
    model_selected: str               # "prophet" | "sarimax" | "timegpt" | "sarimax (timegpt fallback)"
    model_comparison_scores: dict     # {model: {rmse, rmsle, smape, mape, latency_sec}}
    regressors_used: list             # regressors fed to the winning model
    regressor_selection_method: str   # "backtest_ablation"
    demand_forecast: list             # 5-week [{week_start, demand_baseline, demand_disrupted}]
    expected_drop_pct: float          # % demand drop (disrupted vs baseline)
    stockout_prob: float              # logistic regression P(stockout)
    mape_prophet_trend_only: float    # holdout MAPE, trend-only Prophet (baseline reference)
    mape_prophet_selected: float      # holdout MAPE of the winning model
    mape_dataset_baseline_avg: float  # dataset pre-computed MAPE_Baseline column mean
    mape_dataset_ai_avg: float        # dataset pre-computed MAPE_AI column mean
    mape_improvement_pct_vs_dataset_baseline: float
    disruption_scenario: dict         # {disruption_flag, risk_score_composite, calm_period_risk_score_composite}
    agent_logs: list                  # audit trail strings
```

### `ForecastResult` (Pydantic, written to `GlobalState`)

```python
class ForecastResult(BaseModel):
    demand_forecast: List[Dict]       # primary — 5-week list from winning model
    prophet_forecast: List[Dict]      # legacy alias (backward-compatible)
    expected_drop_pct: float
    model_selected: str
    model_comparison_scores: Dict
    sku_id: Optional[str]
    regressors_used: List[str]
    regressor_selection_method: str
    stockout_prob: Optional[float]
    mape_prophet_trend_only: Optional[float]
    mape_prophet_selected: Optional[float]
    mape_dataset_baseline_avg: Optional[float]
    mape_dataset_ai_avg: Optional[float]
    mape_improvement_pct_vs_dataset_baseline: Optional[float]
    disruption_scenario: Optional[Dict]
    forecast_agent_logs: List[str]
```

### Forecast week record format

Each entry in `demand_forecast` (and the legacy `prophet_forecast`) is:

```json
{
  "week_start": "2025-03-17",
  "demand_baseline": 142.30,
  "demand_disrupted": 118.75
}
```

---

## Setup

Dependencies are in the project `requirements.txt`. Key packages for this
agent:

```
prophet
statsmodels>=0.14
scikit-learn
numpy
pandas
```

Optional:
```bash
pip install nixtla        # TimeGPT
pip install lightgbm      # LightGBM challenger only
pip install flask         # model comparison dashboard
```

Point the agent at a different workbook via environment variable:
```bash
export DFA_XLSX_PATH=/path/to/your_data.xlsx
```

Enable TimeGPT:
```bash
export NIXTLA_API_KEY=your_key_here
```

---

## Usage

### As part of the LangGraph pipeline (normal path)

L5 runs automatically when the pipeline executes. It reads `sku_id` from
`state.active_record` and `composite_score` from `state.risk_classification`:

```python
from src.agents.langgraph_engine import build_graph

graph = build_graph()
final_state = graph.invoke(initial_state)

fc = final_state.forecast_result
print(fc.model_selected)          # e.g. "sarimax"
print(fc.expected_drop_pct)       # e.g. 18.4
print(fc.model_comparison_scores) # {naive: {...}, prophet: {...}, sarimax: {...}}
for week in fc.demand_forecast:
    print(week["week_start"], week["demand_baseline"], week["demand_disrupted"])
```

If `active_record` contains no `sku_id`, or the SKU is a Lite Master product
name rather than an `ops_kpi` SKU001-style ID, L5 logs a skip reason and
returns without populating `forecast_result`.

### Standalone from Python

```python
from src.agents.forecast.agent import DemandForecastingAgent

agent = DemandForecastingAgent()

# Single SKU, default disruption scenario (uses dataset's own disruption levels)
result = agent.run("SKU045")
print(result.model_selected, result.expected_drop_pct, result.stockout_prob)

# Single SKU, custom disruption scenario (e.g. from L4 risk score)
result = agent.run("SKU045", disruption_scenario={
    "disruption_flag": 1,
    "risk_score_composite": 0.85,
})

# Model comparison scores for a SKU
scores = agent.get_model_comparison_chart_data("SKU045")
# Returns chart-ready JSON: labels, actual, prophet, sarimax, timegpt, winner, scores

# All eligible SKUs (>=26 weeks of history)
results = agent.run_all(min_weeks=26)
skipped = results.pop("_skipped")   # [(sku_id, reason), ...]

# Explicit train/test report (12-week holdout)
report = agent.train_test_evaluate("SKU045", test_weeks=12)
print(report["test_mape_baseline"], report["test_mape_disrupted"])
print(report["test_table"])        # DataFrame with actual vs predicted, week by week

# LightGBM pooled challenger (requires: pip install lightgbm)
lgb_scores = agent.evaluate_lightgbm_challenger()
print(lgb_scores)                  # DataFrame: sku_id, rmse, smape, mape
```

### List eligible SKUs

```python
agent = DemandForecastingAgent()
agent._load_source_tables()
print(agent.list_skus(min_weeks=26))   # SKUs with enough history
```

---

## Model comparison dashboard

A standalone Flask app at [src/forecast_dashboard.py](../../../forecast_dashboard.py)
visualises the backtest comparison and lets you run live forecasts interactively.

```bash
# From the project root
py -m src.forecast_dashboard
```

Then open **http://localhost:5000**.

### Dashboard endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Full dashboard page |
| `GET /api/skus` | JSON list of eligible SKU IDs |
| `GET /api/model-comparison/<sku_id>` | Chart-ready backtest data: actual vs Prophet vs SARIMAX vs TimeGPT, plus full score table |
| `GET /api/forecast/<sku_id>?risk_score=0.7&disruption_flag=1` | Live 5-week production forecast from the winning model |

### What you see on the dashboard

**Top section — Backtest: which model wins?**
- Overlay line chart: actual demand (black), Prophet (blue), SARIMAX (orange),
  TimeGPT (green, if configured) over the 8-week holdout window
- Score table: RMSE, RMSLE, SMAPE, MAPE, latency (seconds) for each model
- Winner highlighted
- Note shown if TimeGPT was skipped and why

**Bottom section — 5-week forecast: baseline vs disrupted**
- Bar chart: baseline vs disrupted demand week-by-week
- `Expected demand drop` and `Stockout probability` KPI cards
- **Disruption risk slider** (0–1) — drag to re-query the agent and re-render
  the forecast in real time without reloading the page
- Disruption active/inactive toggle

The dashboard uses a single shared `DemandForecastingAgent` instance with
source tables loaded once at startup. Each SKU dropdown selection re-runs the
full backtest (Prophet ablation + all model candidates), so expect 10–30 s
per request depending on series length.

---

## Streamlit integration

The Streamlit main app (`py -m streamlit run src/main.py`) also shows
pre-computed forecast results on the **Demand Forecasts** page.

- Reads JSON files from `data/forecast_outputs/forecast_result_<SKU_ID>.json`
- Uses `demand_forecast` field with `prophet_forecast` as a backward-compatible
  fallback (v3 JSON files use `prophet_forecast`)
- Shows model name, expected drop %, stockout probability, and week-by-week
  chart via `_render_demand_forecast()`

---

## SQLite persistence

Every successful `run()` call writes forecast rows to the project-central
SQLite database (`outputs/supply_chain.db`, table `demand_forecasts`):

```sql
CREATE TABLE demand_forecasts (
    sku_id TEXT, week_start TEXT, demand_baseline REAL, demand_disrupted REAL,
    expected_drop_pct REAL, stockout_prob REAL, mape_selected REAL,
    created_at TEXT,
    PRIMARY KEY (sku_id, week_start)
);
```

DB writes are non-fatal — if the write fails (e.g. the table doesn't exist
yet), L5 logs a warning and continues.

---

## FastAPI endpoints

The REST API at `src/api/routers/forecast.py` exposes forecast results:

| Endpoint | Description |
|---|---|
| `GET /forecast/{sku_id}` | Returns `SkuForecastResponse` with `demand_forecast`, `model_selected`, `model_comparison_scores`, `expected_drop_pct`, `stockout_prob`, and backward-compat `prophet_forecast` |

---

## Key constants

| Constant | Default | Meaning |
|---|---|---|
| `FORECAST_HORIZON_WEEKS` | 5 | Number of future weeks to forecast |
| `HOLDOUT_WEEKS` | 8 | Holdout window size for all backtests |
| `MIN_HISTORY_WEEKS` | 26 | Minimum ops_kpi rows to attempt forecasting |
| `SARIMAX_EXOG` | `[Disruption_Flag, risk_score_composite]` | Fixed exog for SARIMAX |
| `TIMEGPT_EXOG` | `[Disruption_Flag, risk_score_composite]` | Exog passed to TimeGPT |
| `DFA_XLSX_PATH` env var | `data/raw/supply_chain_lite_master.xlsx` | Override workbook path |

---

## Notes and assumptions

- **SKU scope**: Only SKUs with `SKU001`-style IDs in `ops_kpi` are
  forecastable. Scenario Analyzer inputs that use Lite Master product names
  are silently skipped with a helpful log message; use the Demand Forecasts
  page to browse pre-computed SKU results instead.
- **Date join**: Lite Master regressors are joined to ops_kpi by date only
  (nearest match via `merge_asof`). There is no verified per-SKU or
  per-product join — treat category-derived signals as low-confidence.
- **Naive floor**: If the winner does not beat the naive seasonal baseline in
  SMAPE, L5 logs a warning. The winner is still used — the warning flags that
  this SKU's series may be too flat or noisy for the models to add value.
- **TimeGPT fallback**: If TimeGPT wins the backtest but the production
  forecast call fails, the agent automatically falls back to SARIMAX and
  sets `model_selected = "sarimax (timegpt fallback)"`.
- **Synthetic data**: Rows with `Is_Synthetic=True` in `Ops KPI` are
  date-extended synthetic rows. They are used in training but flagged in
  `train_test_evaluate()` output so you can assess their impact.
- **Dashboard CDN**: Chart.js is loaded from `cdnjs.cloudflare.com` in the
  Flask dashboard. If your network blocks that CDN, vendor the file locally.

---

## File map

```
src/agents/forecast/
├── agent.py               # DemandForecastingAgent, _DFAResult, demand_forecasting_agent (LangGraph node)
├── __init__.py            # re-exports DemandForecastingAgent, InsufficientHistoryError, demand_forecasting_agent
└── README.md              # this file

src/
├── forecast_dashboard.py  # Flask model comparison dashboard (py -m src.forecast_dashboard)

src/agents/
├── state.py               # ForecastResult (Pydantic, GlobalState field)

src/api/
├── schemas.py             # SkuForecastResponse (FastAPI response model)
└── routers/forecast.py    # GET /forecast/{sku_id}

src/dashboard/
└── dashboard.py           # Streamlit: show_demand_forecasts(), _render_demand_forecast()

data/
└── raw/
    └── supply_chain_lite_master.xlsx   # source workbook (3 sheets: Ops KPI, Lite Master, SKU_Product_Mapping)

data/forecast_outputs/
└── forecast_result_<SKU_ID>.json       # pre-computed v3/v4 forecast JSON files (46 SKUs)

outputs/
└── supply_chain.db                     # SQLite, table: demand_forecasts
```
