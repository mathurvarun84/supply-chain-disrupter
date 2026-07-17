"""
L5 Demand Forecasting Agent (v3) — LangGraph-integrated
=========================================================
Architecture reference: Agent_Architecture_Spec_P8_v2.docx -> "L5 Demand Forecasting Agent"

Scope
-----
Weekly-level Prophet model, 5-week forecast horizon, scoped to Electronics
category products.

Data source findings (see docs/Agent_Architecture_Spec_P8_v2.docx for full audit):
  1. ops_kpi (Ops KPI (Filled)) has everything needed for the target series:
     Week_Start, SKU_ID, Demand_Actual, Disruption_Flag, Stockout_Flag.
  2. The model also needs lite_master for risk_score_composite (and
     optionally chip_price_index / market_growth_rate) -- ops_kpi has none
     of these.
  3. There is NO shared join key between ops_kpi (SKU_ID) and lite_master
     (Product_Name / Category_Name). Region fields don't align either; the
     only literally shared field is Disruption_Event_Label (a severity tag,
     not a unique key). Confirmed by full column audit.
  4. Because there's no product-level relation, lite_master regressors are
     joined to ops_kpi by DATE only (market-wide weekly signals), not by
     product/category. Evidence: all 46 ops_kpi SKUs sit in a $48-133
     price band, matching lite_master's broad "Electronics" Category_Name
     bucket (median $75) far better than Cameras/Computers/Consumer
     Electronics/Video Games -- so all ops_kpi SKUs are treated as in the
     Electronics category scope for this agent (price-based inference,
     not a verified join).

Regressor selection strategy
-----------------------------
Earlier category-proxy-based regressor gating (assigning chip/growth
regressors only to "chip-relevant" categories inferred from price) was
tested and did not reliably reduce MAPE, because the category proxy itself
is unverified. What did work empirically: letting each SKU's own holdout
backtest decide which regressor combination minimizes its MAPE, out of
{trend-only, disruption-only, risk-only, disruption+risk, chip+growth,
all four}. This agent uses that data-driven selection instead of
hardcoded category gating.

Output contract (GlobalState.forecast_result):
    prophet_forecast   -> 5-week list of {week_start, demand_baseline, demand_disrupted}
    expected_drop_pct  -> float, % demand drop disrupted vs baseline over the 5-week horizon
    stockout_prob      -> float, probability of stockout over the horizon
    (plus MAPE metrics, regressor selection audit, disruption_scenario)
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _smape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmsle(y_true, y_pred) -> float:
    y_true = np.clip(np.array(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

# ---------------------------------------------------------------------------
# Path resolution: env override → project data/raw/ directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parents[3]
XLSX_PATH = os.environ.get(
    "DFA_XLSX_PATH",
    str(_PROJECT_ROOT / "data" / "raw" / "supply_chain_lite_master.xlsx"),
)

FORECAST_HORIZON_WEEKS = 5
HOLDOUT_WEEKS = 8            # for MAPE backtesting / regressor selection
MIN_HISTORY_WEEKS = 26       # Prophet needs enough history for trend + yearly seasonality

# Candidate regressor sets tested per-SKU via backtest to pick the best one.
REGRESSOR_CANDIDATES = {
    "trend_only": [],
    "disruption_only": ["Disruption_Flag"],
    "risk_only": ["risk_score_composite"],
    "disruption_risk": ["Disruption_Flag", "risk_score_composite"],
    "chip_growth_only": ["chip_price_index", "market_growth_rate"],
    "all_four": ["Disruption_Flag", "risk_score_composite", "chip_price_index", "market_growth_rate"],
}


class InsufficientHistoryError(ValueError):
    """Raised when a SKU has too few weeks of ops_kpi history for Prophet
    to fit a meaningful trend + yearly seasonality."""


@dataclass
class _DFAResult:
    """Internal per-run result produced by DemandForecastingAgent.run().

    This is a private dataclass used only within this module.  The LangGraph
    node converts it to the GlobalState Pydantic ForecastResult before
    returning the state delta.
    """
    sku_id: str
    model_selected: str = "prophet"          # "prophet", "sarimax", or "timegpt"
    model_comparison_scores: dict = field(default_factory=dict)
    regressors_used: list = field(default_factory=list)
    regressor_selection_method: str = "backtest_ablation"
    demand_forecast: list = field(default_factory=list)   # 5-week list from selected model
    expected_drop_pct: Optional[float] = None
    stockout_prob: Optional[float] = None
    mape_prophet_trend_only: Optional[float] = None
    mape_prophet_selected: Optional[float] = None
    mape_dataset_baseline_avg: Optional[float] = None
    mape_dataset_ai_avg: Optional[float] = None
    mape_improvement_pct_vs_dataset_baseline: Optional[float] = None
    disruption_scenario: Optional[dict] = None
    agent_logs: list = field(default_factory=list)


class DemandForecastingAgent:
    """L5 agent. Optional, runs in parallel with L6 after L4."""

    def __init__(self, xlsx_path: str = XLSX_PATH):
        self.xlsx_path = xlsx_path
        self.ops_kpi = None
        self.lite_master = None

    # ---------------------------------------------------------------
    # Data access layer
    # ---------------------------------------------------------------
    def _load_source_tables(self):
        self.ops_kpi = pd.read_excel(self.xlsx_path, sheet_name="Ops KPI (Filled)", header=1)
        self.ops_kpi["Week_Start"] = pd.to_datetime(self.ops_kpi["Week_Start"])

        self.lite_master = pd.read_excel(self.xlsx_path, sheet_name="Lite Master", header=1)
        self.lite_master["Order_Date"] = pd.to_datetime(self.lite_master["Order_Date"])

    def _weekly_market_regressors(self) -> pd.DataFrame:
        """Weekly market-wide regressors from lite_master: risk_score_composite,
        chip_price_index, market_growth_rate. Joined to ops_kpi by DATE only --
        there is no per-SKU/per-product relation available (see module docstring)."""
        lm = self.lite_master[
            ["Order_Date", "Risk_Score_Composite", "Chip_Price_Index", "Market_Growth_Rate"]
        ].dropna()
        lm = lm.set_index("Order_Date").resample("W-MON", label="left", closed="left").mean()
        lm = lm.rename(columns={
            "Risk_Score_Composite": "risk_score_composite",
            "Chip_Price_Index": "chip_price_index",
            "Market_Growth_Rate": "market_growth_rate",
        }).reset_index().rename(columns={"Order_Date": "Week_Start"})
        return lm

    def _build_sku_series(self, sku_id: str) -> pd.DataFrame:
        df = self.ops_kpi[self.ops_kpi["SKU_ID"] == sku_id].copy()
        if df.empty:
            raise ValueError(f"No ops_kpi rows found for sku_id={sku_id}")
        df = df.sort_values("Week_Start")

        market = self._weekly_market_regressors()
        df = pd.merge_asof(
            df.sort_values("Week_Start"), market.sort_values("Week_Start"),
            on="Week_Start", direction="nearest",
        )
        for col in ("risk_score_composite", "chip_price_index", "market_growth_rate"):
            df[col] = df[col].ffill().bfill()
        return df

    def list_skus(self, min_weeks: int = MIN_HISTORY_WEEKS) -> list:
        if self.ops_kpi is None:
            self._load_source_tables()
        counts = self.ops_kpi.groupby("SKU_ID").size()
        eligible = counts[counts >= min_weeks].sort_values(ascending=False)
        return eligible.index.tolist()

    # ---------------------------------------------------------------
    # Modeling
    # ---------------------------------------------------------------
    @staticmethod
    def _fit_prophet(train_df: pd.DataFrame, regressors: list) -> Prophet:
        m = Prophet(
            weekly_seasonality=False,   # input is already weekly-aggregated
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.8,
        )
        for r in regressors:
            m.add_regressor(r)
        m.fit(train_df[["ds", "y"] + regressors])
        return m

    def _backtest_mape(self, weekly: pd.DataFrame, regressors: list, holdout: int = HOLDOUT_WEEKS) -> float:
        if len(weekly) <= holdout + 10:
            holdout = max(4, len(weekly) // 5)
        train, test = weekly.iloc[:-holdout], weekly.iloc[-holdout:]
        model = self._fit_prophet(train, regressors)
        pred = model.predict(test[["ds"] + regressors])
        return mean_absolute_percentage_error(test["y"].values, pred["yhat"].values) * 100

    # -- SARIMAX candidate model -----------------------------------------
    @staticmethod
    def _fit_sarimax(train_df: pd.DataFrame, exog_cols: list):
        exog = train_df[exog_cols] if exog_cols else None
        model = SARIMAX(
            train_df["y"], exog=exog, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        return model.fit(disp=False)

    # -- Naive seasonal floor check (backtest only) -----------------------
    @staticmethod
    def _naive_seasonal_predict(train_df: pd.DataFrame, n_steps: int) -> np.ndarray:
        if len(train_df) > 52:
            pred = train_df["y"].iloc[-52:-52 + n_steps].values
            if len(pred) < n_steps:
                pred = np.pad(pred, (0, n_steps - len(pred)), mode="edge")
        else:
            pred = np.full(n_steps, train_df["y"].iloc[-1])
        return pred

    # -- TimeGPT candidate model (Nixtla) -- gracefully skipped if unavailable
    _timegpt_client = None
    _timegpt_unavailable_reason = None

    def _get_timegpt_client(self):
        if self._timegpt_client is not None:
            return self._timegpt_client
        if self._timegpt_unavailable_reason is not None:
            return None
        api_key = os.environ.get("NIXTLA_API_KEY")
        if not api_key:
            self._timegpt_unavailable_reason = "NIXTLA_API_KEY not set"
            return None
        try:
            from nixtla import NixtlaClient
            client = NixtlaClient(api_key=api_key)
            if not client.validate_api_key():
                self._timegpt_unavailable_reason = "NIXTLA_API_KEY failed validation"
                return None
        except Exception as e:
            self._timegpt_unavailable_reason = f"TimeGPT unreachable: {type(e).__name__}: {e}"
            return None
        self._timegpt_client = client
        return client

    def _fit_predict_timegpt(
        self, train_df: pd.DataFrame, future_df: pd.DataFrame, exog_cols: list
    ) -> Optional[np.ndarray]:
        client = self._get_timegpt_client()
        if client is None:
            return None
        try:
            hist = train_df.rename(columns={"ds": "ds", "y": "y"}).copy()
            hist["unique_id"] = "series"
            cols = ["unique_id", "ds", "y"] + exog_cols
            x_df = None
            if exog_cols:
                x_df = future_df[["ds"] + exog_cols].copy()
                x_df["unique_id"] = "series"
                x_df = x_df[["unique_id", "ds"] + exog_cols]
            fc = client.forecast(
                df=hist[cols], h=len(future_df), freq="W-MON",
                X_df=x_df, time_col="ds", target_col="y", id_col="unique_id",
            )
            return fc["TimeGPT"].clip(lower=0).values
        except Exception as e:
            self._timegpt_unavailable_reason = f"TimeGPT call failed: {type(e).__name__}: {e}"
            return None

    TIMEGPT_EXOG = ["Disruption_Flag", "risk_score_composite"]
    SARIMAX_EXOG = ["Disruption_Flag", "risk_score_composite"]

    def _backtest_candidate(
        self, weekly_p: pd.DataFrame, kind: str, features: list, holdout: int = HOLDOUT_WEEKS
    ) -> Optional[dict]:
        if len(weekly_p) <= holdout + 10:
            holdout = max(4, len(weekly_p) // 5)
        train, test = weekly_p.iloc[:-holdout], weekly_p.iloc[-holdout:]
        y_true = test["y"].values
        t0 = time.perf_counter()
        if kind == "naive":
            y_pred = self._naive_seasonal_predict(train, len(test))
        elif kind == "prophet":
            model = self._fit_prophet(train, features)
            y_pred = model.predict(test[["ds"] + features])["yhat"].clip(lower=0).values
        elif kind == "sarimax":
            try:
                fit = self._fit_sarimax(train, features)
                exog = test[features] if features else None
                y_pred = fit.forecast(steps=len(test), exog=exog).clip(lower=0).values
            except Exception:
                y_pred = np.full(len(test), train["y"].mean())
        elif kind == "timegpt":
            y_pred = self._fit_predict_timegpt(train, test, features)
            if y_pred is None:
                return None
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        latency_sec = time.perf_counter() - t0
        return {
            "rmse": round(_rmse(y_true, y_pred), 2),
            "rmsle": round(_rmsle(y_true, y_pred), 4),
            "smape": round(_smape(y_true, y_pred), 2),
            "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
            "latency_sec": round(latency_sec, 3),
        }

    def select_best_model(self, weekly_p: pd.DataFrame) -> dict:
        """Backtest naive/Prophet/SARIMAX/TimeGPT and pick winner by lowest holdout SMAPE."""
        prophet_regressors, _ = self.select_regressors(weekly_p)
        scores = {
            "naive": self._backtest_candidate(weekly_p, "naive", []),
            "prophet": self._backtest_candidate(weekly_p, "prophet", prophet_regressors),
            "sarimax": self._backtest_candidate(weekly_p, "sarimax", self.SARIMAX_EXOG),
        }
        timegpt_scores = self._backtest_candidate(weekly_p, "timegpt", self.TIMEGPT_EXOG)
        timegpt_available = timegpt_scores is not None
        if timegpt_available:
            scores["timegpt"] = timegpt_scores
        candidates = [c for c in ("prophet", "sarimax", "timegpt") if c in scores]
        winner = min(candidates, key=lambda k: scores[k]["smape"])
        return {
            "winner": winner,
            "prophet_regressors": prophet_regressors,
            "sarimax_exog": self.SARIMAX_EXOG,
            "timegpt_exog": self.TIMEGPT_EXOG,
            "timegpt_available": timegpt_available,
            "timegpt_unavailable_reason": self._timegpt_unavailable_reason,
            "scores": scores,
        }

    def get_model_comparison_chart_data(self, sku_id: str, holdout: int = HOLDOUT_WEEKS) -> dict:
        """Return chart-ready backtest data for the Flask dashboard."""
        if self.ops_kpi is None:
            self._load_source_tables()
        weekly_p = self._build_sku_series(sku_id).rename(
            columns={"Week_Start": "ds", "Demand_Actual": "y"}
        )
        if len(weekly_p) <= holdout + 10:
            holdout = max(4, len(weekly_p) // 5)
        train, test = weekly_p.iloc[:-holdout], weekly_p.iloc[-holdout:]
        selection = self.select_best_model(weekly_p)
        prophet_model = self._fit_prophet(train, selection["prophet_regressors"])
        prophet_pred = prophet_model.predict(
            test[["ds"] + selection["prophet_regressors"]] if selection["prophet_regressors"] else test[["ds"]]
        )["yhat"].clip(lower=0).values
        sarimax_fit = self._fit_sarimax(train, selection["sarimax_exog"])
        sarimax_pred = sarimax_fit.forecast(
            steps=len(test), exog=test[selection["sarimax_exog"]]
        ).clip(lower=0).values
        timegpt_pred = self._fit_predict_timegpt(train, test, selection["timegpt_exog"])
        return {
            "sku_id": sku_id,
            "labels": test["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": test["y"].round(2).tolist(),
            "prophet": [round(float(v), 2) for v in prophet_pred],
            "sarimax": [round(float(v), 2) for v in sarimax_pred],
            "timegpt": [round(float(v), 2) for v in timegpt_pred] if timegpt_pred is not None else None,
            "timegpt_status": "ok" if timegpt_pred is not None else selection["timegpt_unavailable_reason"],
            "winner": selection["winner"],
            "scores": selection["scores"],
        }

    def evaluate_lightgbm_challenger(
        self, sku_ids: Optional[List[str]] = None, test_weeks: int = HOLDOUT_WEEKS
    ) -> "pd.DataFrame":
        """Backtest-only pooled LightGBM challenger. NOT used in run()."""
        import lightgbm as lgb
        if self.ops_kpi is None:
            self._load_source_tables()
        if sku_ids is None:
            sku_ids = self.list_skus()
        frames = []
        for sku in sku_ids:
            df = self._build_sku_series(sku).sort_values("Week_Start")
            df["SKU_ID"] = sku
            df["y_lag1"] = df["Demand_Actual"].shift(1)
            df["y_lag2"] = df["Demand_Actual"].shift(2)
            frames.append(df)
        pooled = pd.concat(frames, ignore_index=True).dropna(subset=["y_lag1", "y_lag2"])
        pooled["SKU_ID"] = pooled["SKU_ID"].astype("category")
        pooled["week_of_year"] = pooled["Week_Start"].dt.isocalendar().week.astype(int)
        features = ["SKU_ID", "Disruption_Flag", "risk_score_composite", "chip_price_index",
                    "market_growth_rate", "Price_USD", "Weather_Index", "Promo_Flag",
                    "y_lag1", "y_lag2", "week_of_year"]
        train_parts, test_parts = [], []
        for sku, g in pooled.groupby("SKU_ID", observed=True):
            g = g.sort_values("Week_Start")
            train_parts.append(g.iloc[:-test_weeks])
            test_parts.append(g.iloc[-test_weeks:])
        train_all, test_all = pd.concat(train_parts), pd.concat(test_parts).copy()
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                   min_child_samples=10, verbosity=-1)
        model.fit(train_all[features], train_all["Demand_Actual"])
        test_all["pred"] = np.clip(model.predict(test_all[features]), 0, None)
        rows = []
        for sku, g in test_all.groupby("SKU_ID", observed=True):
            rows.append({
                "sku_id": sku,
                "rmse": round(_rmse(g["Demand_Actual"], g["pred"]), 2),
                "smape": round(_smape(g["Demand_Actual"], g["pred"]), 2),
                "mape": round(mean_absolute_percentage_error(g["Demand_Actual"], g["pred"]) * 100, 2),
            })
        return pd.DataFrame(rows)

    def train_test_evaluate(self, sku_id: str, test_weeks: int = 12) -> dict:
        """Explicit chronological train/test split for reporting purposes.

        - Test set = the most recent `test_weeks` weeks (held out entirely).
        - Regressor selection is done via backtest ablation INSIDE the
          training set only (nested holdout), so the reported test result
          isn't contaminated by having used the test set to pick regressors.
        - Two models are fit on the training set: trend-only (baseline) and
          the selected-regressor model (disrupted-aware).
        - Both are scored twice: on the training set itself (in-sample fit,
          a diagnostic -- not a generalization measure) and on the held-out
          test set (the actual generalization measure).
        """
        weekly = self._build_sku_series(sku_id)
        if len(weekly) < test_weeks + MIN_HISTORY_WEEKS:
            raise InsufficientHistoryError(
                f"{sku_id} has {len(weekly)} weeks; need at least {test_weeks + MIN_HISTORY_WEEKS} "
                f"for a {test_weeks}-week test split plus minimum training history"
            )
        weekly_p = weekly.rename(columns={"Week_Start": "ds", "Demand_Actual": "y"}).reset_index(drop=True)

        train = weekly_p.iloc[:-test_weeks].reset_index(drop=True)
        test = weekly_p.iloc[-test_weeks:].reset_index(drop=True)

        # Regressor selection nested inside the training set (no test leakage)
        regressors, selection_scores = self.select_regressors(train)

        baseline_model = self._fit_prophet(train, regressors=[])
        disrupted_model = self._fit_prophet(train, regressors=regressors)

        # --- In-sample (training) fit ---
        train_pred_baseline = baseline_model.predict(train[["ds"]])
        train_pred_disrupted = disrupted_model.predict(train[["ds"] + regressors]) if regressors else train_pred_baseline
        train_mape_baseline = mean_absolute_percentage_error(train["y"], train_pred_baseline["yhat"]) * 100
        train_mape_disrupted = mean_absolute_percentage_error(train["y"], train_pred_disrupted["yhat"]) * 100

        # --- Held-out test evaluation ---
        test_pred_baseline = baseline_model.predict(test[["ds"]])
        test_pred_disrupted = disrupted_model.predict(test[["ds"] + regressors]) if regressors else test_pred_baseline
        test_mape_baseline = mean_absolute_percentage_error(test["y"], test_pred_baseline["yhat"]) * 100
        test_mape_disrupted = mean_absolute_percentage_error(test["y"], test_pred_disrupted["yhat"]) * 100

        test_table = pd.DataFrame({
            "week_start": test["ds"].dt.strftime("%Y-%m-%d"),
            "actual": test["y"].round(2),
            "predicted_baseline": test_pred_baseline["yhat"].clip(lower=0).round(2).values,
            "predicted_disrupted": test_pred_disrupted["yhat"].clip(lower=0).round(2).values,
            "is_synthetic": weekly["Is_Synthetic"].iloc[-test_weeks:].values if "Is_Synthetic" in weekly.columns else False,
        })
        test_table["abs_pct_error_disrupted"] = (
            (test_table["actual"] - test_table["predicted_disrupted"]).abs() / test_table["actual"] * 100
        ).round(1)

        return {
            "sku_id": sku_id,
            "train_weeks": len(train),
            "test_weeks": len(test),
            "regressors_selected": regressors,
            "regressor_selection_scores": {k: round(v, 2) for k, v in selection_scores.items()},
            "train_mape_baseline": round(train_mape_baseline, 2),
            "train_mape_disrupted": round(train_mape_disrupted, 2),
            "test_mape_baseline": round(test_mape_baseline, 2),
            "test_mape_disrupted": round(test_mape_disrupted, 2),
            "test_table": test_table,
        }

    def select_regressors(self, weekly_p: pd.DataFrame) -> tuple:
        """Backtest every candidate regressor combination and return the
        one with the lowest holdout MAPE, plus the full score table."""
        scores = {
            name: self._backtest_mape(weekly_p, regs) for name, regs in REGRESSOR_CANDIDATES.items()
        }
        best_name = min(scores, key=scores.get)
        return REGRESSOR_CANDIDATES[best_name], scores

    def _train_stockout_classifier(self, weekly: pd.DataFrame):
        X = weekly[["Disruption_Flag", "risk_score_composite"]].values
        y = weekly["Stockout_Flag"].values
        if len(np.unique(y)) < 2:
            return None
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(X, y)
        return clf

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------
    def run(self, sku_id: str, disruption_scenario: Optional[dict] = None) -> _DFAResult:
        logs = [
            f"L5 DemandForecastingAgent started for {sku_id} "
            f"(Electronics scope, weekly, {FORECAST_HORIZON_WEEKS}-week horizon)"
        ]

        if self.ops_kpi is None:
            self._load_source_tables()
            logs.append("Loaded ops_kpi and lite_master source tables")

        weekly = self._build_sku_series(sku_id)
        if len(weekly) < MIN_HISTORY_WEEKS:
            raise InsufficientHistoryError(
                f"{sku_id} has only {len(weekly)} weeks of history (minimum {MIN_HISTORY_WEEKS} required)"
            )

        weekly_p = weekly.rename(columns={"Week_Start": "ds", "Demand_Actual": "y"})

        # ---- 1. Model selection: naive floor check, Prophet, SARIMAX, TimeGPT ----
        selection = self.select_best_model(weekly_p)
        winner = selection["winner"]
        regressors = selection["prophet_regressors"]
        sarimax_exog = selection["sarimax_exog"]
        timegpt_exog = selection["timegpt_exog"]
        scores = selection["scores"]
        logs.append(f"Model backtest scores (SMAPE/RMSE/MAPE): {scores}")
        logs.append(f"Selected production model: {winner}")
        if not selection["timegpt_available"]:
            logs.append(f"TimeGPT not evaluated: {selection['timegpt_unavailable_reason']}")
        if scores[winner]["smape"] >= scores["naive"]["smape"]:
            logs.append(
                f"WARNING: {winner} did not beat the naive seasonal floor check "
                f"({scores[winner]['smape']}% vs {scores['naive']['smape']}% SMAPE) -- "
                f"this SKU's series may be too flat/noisy."
            )
        mape_trend_only = self._backtest_mape(weekly_p, [])
        mape_selected = scores[winner]["mape"]

        # ---- 2. Build the forward 5-week frame ----
        last_date = weekly_p["ds"].max()
        future_dates = pd.date_range(last_date + timedelta(weeks=1), periods=FORECAST_HORIZON_WEEKS, freq="W-MON")
        future = pd.DataFrame({"ds": future_dates})

        calm_risk = weekly_p.loc[weekly_p["Disruption_Flag"] == 0, "risk_score_composite"].tail(8).mean()
        if np.isnan(calm_risk):
            calm_risk = weekly_p["risk_score_composite"].quantile(0.25)
        future_baseline = future.copy()
        future_baseline["Disruption_Flag"] = 0
        future_baseline["risk_score_composite"] = calm_risk

        scenario = disruption_scenario or {}
        disruption_flag = scenario.get("disruption_flag", 1)
        disrupted_risk = scenario.get(
            "risk_score_composite",
            float(weekly_p.loc[weekly_p["Disruption_Flag"] == 1, "risk_score_composite"].quantile(0.75)),
        )
        future_disrupted = future.copy()
        future_disrupted["Disruption_Flag"] = disruption_flag
        future_disrupted["risk_score_composite"] = disrupted_risk

        if "chip_price_index" in regressors:
            trailing_chip = weekly_p["chip_price_index"].tail(8).mean()
            trailing_growth = weekly_p["market_growth_rate"].tail(8).mean()
            future_baseline["chip_price_index"] = trailing_chip
            future_baseline["market_growth_rate"] = trailing_growth
            future_disrupted["chip_price_index"] = trailing_chip
            future_disrupted["market_growth_rate"] = trailing_growth

        # ---- 3. Fit the SELECTED model on full history and predict ----
        if winner == "prophet":
            baseline_model = self._fit_prophet(weekly_p, regressors=[])
            fc_baseline = baseline_model.predict(future_baseline[["ds"]])["yhat"].clip(lower=0).values
        elif winner == "sarimax":
            baseline_fit = self._fit_sarimax(weekly_p, exog_cols=[])
            fc_baseline = baseline_fit.forecast(steps=FORECAST_HORIZON_WEEKS).clip(lower=0).values
        elif winner == "timegpt":
            fc_baseline = self._fit_predict_timegpt(weekly_p, future_baseline, timegpt_exog)
            if fc_baseline is None:
                logs.append("TimeGPT production call failed -- falling back to SARIMAX")
                winner = "sarimax (timegpt fallback)"
                baseline_fit = self._fit_sarimax(weekly_p, exog_cols=[])
                fc_baseline = baseline_fit.forecast(steps=FORECAST_HORIZON_WEEKS).clip(lower=0).values
        else:
            raise ValueError(f"Unexpected winning model: {winner}")

        # disruption_flag == 0 means L4 did not classify this as a real event
        # (HIGH/CRITICAL) — see demand_forecasting_agent() in this module.
        # Running a separately-fit "disrupted" model here is misleading in
        # that case: baseline/disrupted are two independently fitted models
        # (not the same model evaluated at two input points), so they diverge
        # from model-estimation differences alone, even fed matching inputs —
        # confirmed empirically, feeding the disrupted scenario the exact
        # calm-period composite still produced a large "drop" on this SKU's
        # own history. Reporting the disrupted series as identical to
        # baseline is the only way to guarantee a non-event shows ~0% change.
        if not disruption_flag:
            fc_disrupted = np.array(fc_baseline, dtype=float)
        elif winner == "prophet":
            disrupted_model = self._fit_prophet(weekly_p, regressors=regressors)
            fc_disrupted = disrupted_model.predict(future_disrupted[["ds"] + regressors])["yhat"].clip(lower=0).values
        elif winner in ("sarimax", "sarimax (timegpt fallback)"):
            disrupted_fit = self._fit_sarimax(weekly_p, exog_cols=sarimax_exog)
            fc_disrupted = disrupted_fit.forecast(
                steps=FORECAST_HORIZON_WEEKS, exog=future_disrupted[sarimax_exog]
            ).clip(lower=0).values
        elif winner == "timegpt":
            fc_disrupted = self._fit_predict_timegpt(weekly_p, future_disrupted, timegpt_exog)
            if fc_disrupted is None:
                logs.append("TimeGPT disrupted-scenario call failed -- falling back to SARIMAX")
                winner = "sarimax (timegpt fallback)"
                disrupted_fit = self._fit_sarimax(weekly_p, exog_cols=sarimax_exog)
                fc_disrupted = disrupted_fit.forecast(
                    steps=FORECAST_HORIZON_WEEKS, exog=future_disrupted[sarimax_exog]
                ).clip(lower=0).values
        else:
            raise ValueError(f"Unexpected winning model: {winner}")

        if not disruption_flag:
            logs.append(
                "L5: disruption_flag=0 (L4 did not classify HIGH/CRITICAL) — "
                "reporting baseline-only forecast, no separately-fit disrupted scenario."
            )

        demand_forecast = [
            {
                "week_start": d.strftime("%Y-%m-%d"),
                "demand_baseline": round(float(b), 2),
                "demand_disrupted": round(float(dstd), 2),
            }
            for d, b, dstd in zip(future_dates, fc_baseline, fc_disrupted)
        ]

        total_baseline = float(np.sum(fc_baseline))
        total_disrupted = float(np.sum(fc_disrupted))
        expected_drop_pct = round(float((total_baseline - total_disrupted) / total_baseline * 100), 2)
        logs.append(
            f"Expected {FORECAST_HORIZON_WEEKS}-week demand deviation "
            f"(disrupted vs baseline): {expected_drop_pct}%"
        )

        # ---- 5. Stockout probability ----
        # disruption_flag == 0: same reasoning as the forecast above — go
        # straight to the historical base rate rather than the classifier,
        # which would otherwise be scored against the live composite
        # (disrupted_risk) as if this were a real event.
        if not disruption_flag:
            stockout_prob = float(weekly_p["Stockout_Flag"].mean())
        else:
            clf = self._train_stockout_classifier(weekly_p)
            if clf is not None:
                stockout_prob = float(
                    clf.predict_proba([[future_disrupted["Disruption_Flag"].iloc[0], disrupted_risk]])[0][1]
                )
            else:
                stockout_prob = float(weekly_p["Stockout_Flag"].mean())
        stockout_prob = round(stockout_prob, 3)
        logs.append(f"Stockout probability under disruption scenario: {stockout_prob}")

        # ---- 6. Benchmark vs dataset pre-computed MAPE columns ----
        mape_dataset_baseline_avg = float(weekly["MAPE_Baseline"].mean() * 100)
        mape_dataset_ai_avg = float(weekly["MAPE_AI"].mean() * 100)
        mape_improvement = round(
            (mape_dataset_baseline_avg - mape_dataset_ai_avg) / mape_dataset_baseline_avg * 100, 2
        )

        return _DFAResult(
            sku_id=sku_id,
            model_selected=winner,
            model_comparison_scores=scores,
            regressors_used=(
                regressors if winner == "prophet"
                else timegpt_exog if winner == "timegpt"
                else sarimax_exog
            ),
            demand_forecast=demand_forecast,
            expected_drop_pct=expected_drop_pct,
            stockout_prob=stockout_prob,
            mape_prophet_trend_only=round(mape_trend_only, 2),
            mape_prophet_selected=round(mape_selected, 2),
            mape_dataset_baseline_avg=round(mape_dataset_baseline_avg, 2),
            mape_dataset_ai_avg=round(mape_dataset_ai_avg, 2),
            mape_improvement_pct_vs_dataset_baseline=mape_improvement,
            disruption_scenario={
                "disruption_flag": int(future_disrupted["Disruption_Flag"].iloc[0]),
                "risk_score_composite": round(float(disrupted_risk), 3),
                "calm_period_risk_score_composite": round(float(calm_risk), 3),
            },
            agent_logs=logs,
        )

    def run_all(
        self,
        sku_ids: Optional[List[str]] = None,
        min_weeks: int = MIN_HISTORY_WEEKS,
        disruption_scenario: Optional[dict] = None,
        skip_errors: bool = True,
    ) -> dict:
        if sku_ids is None:
            sku_ids = self.list_skus(min_weeks=min_weeks)
        results, skipped = {}, []
        for sku in sku_ids:
            try:
                results[sku] = self.run(sku, disruption_scenario=disruption_scenario)
            except InsufficientHistoryError as e:
                skipped.append((sku, str(e)))
            except Exception as e:
                if not skip_errors:
                    raise
                skipped.append((sku, f"{type(e).__name__}: {e}"))
        results["_skipped"] = skipped
        return results


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def demand_forecasting_agent(state: Any) -> Dict[str, Any]:
    """L5 — Full DemandForecastingAgent v3 (weekly, 5-week, per-SKU regressor selection).

    Replaces the previous inline Prophet stub in langgraph_engine.py.
    Reads active_record.sku_id, runs DemandForecastingAgent.run(), converts
    the internal _DFAResult to the GlobalState Pydantic ForecastResult, and
    persists forecast rows to the project SQLite DB.
    """
    if state.active_record is None:
        raise ValueError("L5: active_record is required for demand forecasting.")

    handoff = getattr(state, "forecast_handoff", None)
    sku_id: Optional[str] = None
    if handoff is not None and getattr(handoff, "sku_id", None):
        sku_id = str(handoff.sku_id)
    if not sku_id:
        sku_id = (
            state.active_record.get("sku_id")    # daily_records VIEW: SKU001-style crosswalk key
            or state.active_record.get("SKU_ID")
            or state.active_record.get("sku")    # fallback: product_name alias — not an ops_kpi ID
        )
    if not sku_id:
        return {
            "agent_logs": state.agent_logs + [
                "L5: SKIPPED – no sku_id found in active_record or forecast_handoff."
            ],
        }

    # Build disruption scenario from L4 risk-classifier handoff/result if available.
    # Prefer the explicit handoff payload so L5 uses the exact SKU and composite that
    # L4 selected for this event, rather than reusing a different record from the state.
    #
    # disruption_flag was previously hardcoded to 1 here unconditionally — Disruption_Flag
    # is a binary regressor the Prophet/SARIMAX/TimeGPT models learned a separate
    # coefficient for (see future_disrupted["Disruption_Flag"] at line ~567 and the
    # stockout classifier's predict_proba() call at line ~627), independent of
    # risk_score_composite's own coefficient. Forcing it to 1 injected the model's
    # learned "a real disruption happened this week" effect into every run's
    # demand_disrupted/stockout_prob output, even when L4 classified the run LOW —
    # the same "always assume a disruption" bug already fixed in L2/L6. Only treat
    # it as a real disruption for the tiers L4 itself treats as one (matches
    # risk_classifier_agent._gather_rag_citations()'s HIGH/CRITICAL threshold).
    disruption_scenario: Optional[dict] = None
    if handoff is not None:
        disruption_scenario = {
            "disruption_flag": 1 if handoff.risk_label in ("HIGH", "CRITICAL") else 0,
            "risk_score_composite": float(handoff.risk_score_composite),
        }
    elif state.risk_classification is not None:
        disruption_scenario = {
            "disruption_flag": 1 if state.risk_classification.final_label in ("HIGH", "CRITICAL") else 0,
            "risk_score_composite": float(state.risk_classification.composite_score),
        }

    agent = DemandForecastingAgent()
    try:
        result: _DFAResult = agent.run(sku_id, disruption_scenario=disruption_scenario)
    except InsufficientHistoryError as exc:
        return {
            "agent_logs": state.agent_logs + [f"L5: SKIPPED – {exc}"],
        }
    except ValueError as exc:
        # Covers "No ops_kpi rows found for sku_id=<product-name>" — happens when
        # the Scenario Analyzer passes a Lite Master product name instead of an
        # ops_kpi SKU_ID (SKU001-SKU050).  Not an error — just a different dataset.
        return {
            "agent_logs": state.agent_logs + [
                f"L5: SKIPPED – sku '{sku_id}' not found in ops_kpi "
                f"(ops_kpi uses SKU001-style IDs; Lite Master uses product names). "
                f"Use the Demand Forecasts page to browse pre-computed SKU forecasts."
            ],
        }

    # Persist to project-central SQLite (non-fatal on failure)
    _write_forecast_to_db(result)

    # Convert to GlobalState Pydantic ForecastResult
    from src.agents.state import ForecastResult as _PydanticForecastResult
    forecast_result = _PydanticForecastResult(
        demand_forecast=result.demand_forecast,
        # legacy alias — must mirror demand_forecast so downstream readers
        # (L6's _build_forecast_demands(), pipeline_bridge's forecast
        # snapshot) that still key off prophet_forecast see real data
        # instead of silently falling back to a fabricated series.
        prophet_forecast=result.demand_forecast,
        expected_drop_pct=result.expected_drop_pct or 0.0,
        sku_id=result.sku_id,
        model_selected=result.model_selected,
        model_comparison_scores=result.model_comparison_scores,
        regressors_used=result.regressors_used,
        regressor_selection_method=result.regressor_selection_method,
        stockout_prob=result.stockout_prob,
        mape_prophet_trend_only=result.mape_prophet_trend_only,
        mape_prophet_selected=result.mape_prophet_selected,
        mape_dataset_baseline_avg=result.mape_dataset_baseline_avg,
        mape_dataset_ai_avg=result.mape_dataset_ai_avg,
        mape_improvement_pct_vs_dataset_baseline=result.mape_improvement_pct_vs_dataset_baseline,
        disruption_scenario=result.disruption_scenario,
        forecast_agent_logs=result.agent_logs,
    )

    return {
        "forecast_result": forecast_result,
        "agent_logs": state.agent_logs + result.agent_logs + ["L5: Demand forecasting v4 completed."],
    }


def _write_forecast_to_db(result: _DFAResult) -> None:
    """Persist forecast rows to the project-central SQLite DB.  Non-fatal."""
    try:
        from src.utils.db_utils import get_connection, ensure_forecast_schema
        ensure_forecast_schema()
        now = pd.Timestamp.utcnow().isoformat()
        rows = [
            (
                result.sku_id,
                r["week_start"],
                r["demand_baseline"],
                r["demand_disrupted"],
                result.expected_drop_pct,
                result.stockout_prob,
                result.mape_prophet_selected,
                now,
            )
            for r in result.demand_forecast
        ]
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO demand_forecasts VALUES (?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()
    except Exception as exc:
        logger.warning("L5: DB write skipped – %s", exc)
