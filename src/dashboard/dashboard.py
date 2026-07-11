import json
from pathlib import Path

import streamlit as st

from src.agents.langgraph_engine import run_agent_graph
from src.agents.state import RiskClassificationResult
from src.dashboard.data_loader import show_data_loader
from src.dashboard.ingestion_dashboard import show_ingestion_dashboard
from src.utils.db_utils import ensure_schema, fetch_scenario_options
from src.utils.ingestion_schema import ensure_ingestion_schema
from src.utils.openai_utils import has_openai_api_key
from src.rag.utils import query_chroma_rag

_FORECAST_OUTPUTS_DIR = Path(__file__).parents[2] / "data" / "forecast_outputs"


def _render_impact_simulation(result) -> None:
    """Render Monte Carlo impact ranges from L6 simulation_result."""
    sim = result.simulation_result
    if sim is None:
        return

    st.subheader("Impact Simulation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stockout Severity P10", f"{sim.stockout_probability_p10:.1f}%")
    c2.metric("Stockout Severity P50", f"{sim.stockout_probability_pct:.1f}%")
    c3.metric("Stockout Severity P90", f"{sim.stockout_probability_p90:.1f}%")

    r1, r2, r3 = st.columns(3)
    if sim.revenue_impact_usd_p10 is not None:
        r1.metric("Revenue at Risk P10", f"${sim.revenue_impact_usd_p10:,.0f}")
    if sim.revenue_impact_usd_p50 is not None:
        r2.metric("Revenue at Risk P50", f"${sim.revenue_impact_usd_p50:,.0f}")
    if sim.revenue_impact_usd_p90 is not None:
        r3.metric("Revenue at Risk P90", f"${sim.revenue_impact_usd_p90:,.0f}")

    if sim.days_to_stockout_p50 is not None:
        d1, d2, d3 = st.columns(3)
        if sim.days_to_stockout_p10 is not None:
            d1.metric("Days to Stockout P10", f"{sim.days_to_stockout_p10:.0f}")
        d2.metric("Days to Stockout P50", f"{sim.days_to_stockout_p50:.0f}")
        if sim.days_to_stockout_p90 is not None:
            d3.metric("Days to Stockout P90", f"{sim.days_to_stockout_p90:.0f}")

    st.caption(
        f"Alternate route: {sim.alternate_route} · "
        f"Trials: {sim.trials_run:,} · Model: {sim.model_version}"
    )

    if sim.revenue_impact_samples:
        try:
            import matplotlib.pyplot as plt

            with st.expander("Revenue impact distribution"):
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(sim.revenue_impact_samples, bins=min(20, len(sim.revenue_impact_samples)))
                ax.set_xlabel("Revenue impact (USD)")
                ax.set_ylabel("Trial count")
                ax.set_title("Monte Carlo revenue-at-risk samples")
                st.pyplot(fig)
                plt.close(fig)
        except ImportError:
            pass


def _render_demand_forecast(forecast_result) -> None:
    """Render the full L5 DemandForecastingAgent v4 output block."""
    if forecast_result is None:
        return

    fc = forecast_result
    # v4 uses demand_forecast; v3 used prophet_forecast — support both
    weeks = (
        fc.demand_forecast if (hasattr(fc, "demand_forecast") and fc.demand_forecast)
        else getattr(fc, "prophet_forecast", [])
    )
    model_label = getattr(fc, "model_selected", "prophet")

    st.subheader(f"L5 — Demand Forecast (5-Week · model: {model_label})")

    # ── Top metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Demand Drop", f"{fc.expected_drop_pct:.1f}%")
    if fc.stockout_prob is not None:
        m2.metric("Stockout Probability (L5)", f"{fc.stockout_prob:.1%}")
    if fc.mape_prophet_selected is not None:
        m3.metric("MAPE (selected model)", f"{fc.mape_prophet_selected:.1f}%")
    if fc.mape_improvement_pct_vs_dataset_baseline is not None:
        m4.metric(
            "MAPE improvement vs baseline",
            f"{fc.mape_improvement_pct_vs_dataset_baseline:.1f}%",
        )

    # ── Regressor selection ───────────────────────────────────────────────────
    regs = fc.regressors_used or []
    st.caption(
        f"Regressors selected (backtest ablation): "
        + (", ".join(f"`{r}`" for r in regs) if regs else "`trend-only`")
    )

    # ── 5-week forecast chart ─────────────────────────────────────────────────
    if weeks:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            labels = [w["week_start"] for w in weeks]
            baseline = [w["demand_baseline"] for w in weeks]
            disrupted = [w["demand_disrupted"] for w in weeks]

            fig, ax = plt.subplots(figsize=(9, 3))
            x = range(len(labels))
            ax.plot(x, baseline, marker="o", label="Baseline (calm)", color="#1f77b4")
            ax.plot(x, disrupted, marker="s", linestyle="--", label="Disrupted scenario", color="#d62728")
            ax.fill_between(x, disrupted, baseline, alpha=0.12, color="#d62728")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Demand (units)")
            ax.set_title(f"5-Week Demand Forecast — {fc.sku_id or 'SKU'}")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except ImportError:
            # matplotlib not available — fall back to table
            pass

        # ── Week-by-week table ────────────────────────────────────────────────
        with st.expander("Week-by-week forecast table"):
            import pandas as pd
            df = pd.DataFrame(weeks)
            df["demand_drop_%"] = (
                (df["demand_baseline"] - df["demand_disrupted"]) / df["demand_baseline"] * 100
            ).round(1)
            st.dataframe(df, use_container_width=True)

    # ── MAPE comparison ───────────────────────────────────────────────────────
    if fc.mape_prophet_trend_only is not None and fc.mape_prophet_selected is not None:
        with st.expander("MAPE benchmark (Prophet vs dataset baseline)"):
            cols = st.columns(4)
            cols[0].metric("Trend-only MAPE", f"{fc.mape_prophet_trend_only:.1f}%")
            cols[1].metric("Selected model MAPE", f"{fc.mape_prophet_selected:.1f}%")
            if fc.mape_dataset_baseline_avg is not None:
                cols[2].metric("Dataset baseline MAPE", f"{fc.mape_dataset_baseline_avg:.1f}%")
            if fc.mape_dataset_ai_avg is not None:
                cols[3].metric("Dataset AI MAPE", f"{fc.mape_dataset_ai_avg:.1f}%")


def show_demand_forecasts() -> None:
    """Standalone page: browse the 46 pre-computed L5 Prophet forecasts by SKU."""
    st.title("Demand Forecasts (L5)")
    st.caption(
        "Pre-generated 5-week Prophet forecasts for 46 ops_kpi SKUs "
        "(Electronics scope, weekly grain). Produced by DemandForecastingAgent v3 "
        "using backtest-ablation regressor selection."
    )

    json_files = sorted(_FORECAST_OUTPUTS_DIR.glob("forecast_result_SKU*.json"))
    if not json_files:
        st.warning(
            f"No pre-generated forecast files found in `{_FORECAST_OUTPUTS_DIR}`. "
            "Run `python -m src.agents.forecast.agent --all` from the project root to generate them."
        )
        return

    sku_ids = [p.stem.replace("forecast_result_", "") for p in json_files]
    selected_sku = st.selectbox("Select SKU", sku_ids)

    json_path = _FORECAST_OUTPUTS_DIR / f"forecast_result_{selected_sku}.json"
    with json_path.open() as fh:
        data = json.load(fh)

    # ── Top metrics ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected Demand Drop", f"{data.get('expected_drop_pct', 0):.1f}%")
    if data.get("stockout_prob") is not None:
        m2.metric("Stockout Probability", f"{data['stockout_prob']:.1%}")
    if data.get("mape_prophet_selected") is not None:
        m3.metric("MAPE (selected model)", f"{data['mape_prophet_selected']:.1f}%")
    if data.get("mape_improvement_pct_vs_dataset_baseline") is not None:
        m4.metric("MAPE improvement vs baseline", f"{data['mape_improvement_pct_vs_dataset_baseline']:.1f}%")

    regs = data.get("regressors_used") or []
    st.caption(
        "Regressors selected: "
        + (", ".join(f"`{r}`" for r in regs) if regs else "`trend-only`")
        + f"  ·  Method: `{data.get('regressor_selection_method', 'backtest_ablation')}`"
    )

    # ── Forecast chart ────────────────────────────────────────────────────────
    weeks = data.get("demand_forecast") or data.get("prophet_forecast", [])
    if weeks:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            labels = [w["week_start"] for w in weeks]
            baseline = [w["demand_baseline"] for w in weeks]
            disrupted = [w["demand_disrupted"] for w in weeks]

            fig, ax = plt.subplots(figsize=(9, 3))
            x = range(len(labels))
            ax.plot(x, baseline, marker="o", label="Baseline (calm)", color="#1f77b4")
            ax.plot(x, disrupted, marker="s", linestyle="--", label="Disrupted scenario", color="#d62728")
            ax.fill_between(x, disrupted, baseline, alpha=0.12, color="#d62728")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Demand (units)")
            ax.set_title(f"5-Week Demand Forecast — {selected_sku}")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except ImportError:
            st.info("Install matplotlib (`pip install matplotlib`) to see the chart.")

        import pandas as pd
        df = pd.DataFrame(weeks)
        df["demand_drop_%"] = (
            (df["demand_baseline"] - df["demand_disrupted"]) / df["demand_baseline"] * 100
        ).round(1)
        st.dataframe(df, use_container_width=True)

    # ── MAPE detail ───────────────────────────────────────────────────────────
    with st.expander("MAPE benchmark"):
        cols = st.columns(4)
        cols[0].metric("Trend-only MAPE", f"{data.get('mape_prophet_trend_only', 0):.1f}%")
        cols[1].metric("Selected model MAPE", f"{data.get('mape_prophet_selected', 0):.1f}%")
        if data.get("mape_dataset_baseline_avg") is not None:
            cols[2].metric("Dataset baseline MAPE", f"{data['mape_dataset_baseline_avg']:.1f}%")
        if data.get("mape_dataset_ai_avg") is not None:
            cols[3].metric("Dataset AI MAPE", f"{data['mape_dataset_ai_avg']:.1f}%")

    # ── Disruption scenario used ──────────────────────────────────────────────
    if data.get("disruption_scenario"):
        ds = data["disruption_scenario"]
        with st.expander("Disruption scenario parameters"):
            st.json(ds)

    # ── Agent log ─────────────────────────────────────────────────────────────
    if data.get("agent_logs"):
        with st.expander("Agent log"):
            for line in data["agent_logs"]:
                st.text(line)


def _render_ensemble_signals(rc: RiskClassificationResult) -> None:
    """Render three-signal ensemble breakdown and judge verdict panel."""
    st.markdown("---")
    st.markdown("#### 🔬 Ensemble Signal Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Signal 1 — Rule-based**")
        if rc.rule_signal:
            rs = rc.rule_signal
            st.metric("Label", rs.escalated_label)
            st.caption(f"Composite: {rs.composite_score:.4f}")
            st.caption(f"Delivery override: {rs.delivery_status_override or 'none'}")
            st.caption(f"Escalated: {'yes' if rs.escalated else 'no'}")

    with col2:
        st.markdown("**Signal 2 — DistilBERT**")
        if rc.distilbert_signal:
            ds = rc.distilbert_signal
            if ds.model_source == "fine-tuned":
                st.metric("Label", ds.predicted_label, delta=f"{ds.confidence:.0%} conf")
                probs = ds.probability_distribution
                st.caption(
                    f"LOW {probs.get('LOW', 0):.0%} | "
                    f"MED {probs.get('MEDIUM', 0):.0%} | "
                    f"HIGH {probs.get('HIGH', 0):.0%} | "
                    f"CRIT {probs.get('CRITICAL', 0):.0%}"
                )
                st.caption(f"Inference: {ds.inference_ms:.0f}ms (CPU, no API)")
            else:
                st.metric("Label", "N/A")
                st.caption(f"Status: {ds.model_source}")
                st.caption("Run finetune_distilbert.py to enable")

    with col3:
        st.markdown("**Signal 3 — GPT-4o + RAG**")
        if rc.llm_signal:
            ls = rc.llm_signal
            st.metric("Label", ls.predicted_label, delta=ls.confidence_level)
            st.caption(f"Driver: {ls.primary_driver}")
            st.caption(f"RAG chunks used: {ls.rag_chunks_used} (after cross-encoder)")
        else:
            st.metric("Label", "N/A")
            st.caption("LLM signal failed or not run")

    if rc.judge_verdict:
        jv = rc.judge_verdict
        icon = "🟢" if jv.signals_agreed else "🟡"
        st.markdown(f"**{icon} Judge Verdict: `{jv.final_label}` — `{jv.verdict_type}`**")
        with st.expander("📋 Judge reasoning (meta-reasoning about signal disagreements)"):
            st.write(jv.reasoning)
            if jv.disagreement_explanation:
                st.warning(f"⚠️ Disagreement: {jv.disagreement_explanation}")
    else:
        if has_openai_api_key():
            st.info(
                "Judge verdict not available — OpenAI API call failed (often corporate TLS "
                "blocking api.openai.com). You already have `INGEST_INSECURE_SSL=1` in `.env`; "
                "restart Streamlit so Signal 3 and the Judge use the same SSL bypass."
            )
        else:
            st.info(
                "Judge verdict not available — set `OPENAI_API_KEY` in `.env` at the project "
                "root and restart Streamlit. Signal 3 (GPT-4o) uses the same key."
            )


def show_rag_search() -> None:
    st.title("Electronics Knowledge Search")
    query = st.text_input(
        "Search semiconductor events, mitigation guidance, or field definitions",
        "semiconductor factory shutdown risk",
    )
    result_count = st.slider("Results", 1, 10, 5)
    if st.button("Search ChromaDB"):
        hits = query_chroma_rag(query, n_results=result_count)
        if not hits:
            st.warning("No ChromaDB results. Build the databases first.")
            return
        for hit in hits:
            metadata = hit["metadata"]
            with st.expander(
                f"{metadata.get('type', 'document')} · "
                f"distance {hit.get('distance', 0):.3f}"
            ):
                st.write(hit["text"])
                st.json(metadata)


def show_scenario_analyzer() -> None:
    st.title("Scenario Analyzer")
    st.caption(
        "Runs against records from Varun's electronics workbook and live "
        "Open-Meteo weather data."
    )
    ensure_schema()
    ensure_ingestion_schema()
    try:
        options = fetch_scenario_options()
    except Exception:
        options = []

    if not options:
        st.warning("Build the SQLite database before running a scenario.")
        return

    with st.form(key="scenario_form"):
        disruption_type = st.selectbox(
            "Disruption type",
            [
                "earthquake",
                "port closure",
                "chip shortage",
                "geopolitical",
                "extreme weather",
                "supplier lockdown",
            ],
        )
        selected = st.selectbox(
            "Historical scenario baseline",
            options,
            format_func=lambda row: (
                f"{row['port']} · {row['sku']} · {row['event_date']} "
                f"({row['history_points']} history points)"
            ),
        )
        affected_route = st.text_input("Affected route", "Supplier to destination")
        severity = st.slider("Severity", 0.0, 1.0, 0.6)
        shock_duration_days = st.number_input(
            "Shock duration (days)",
            min_value=0,
            max_value=180,
            value=0,
            help="Set only when modeling a confirmed disruption duration; 0 skips duration escalation.",
        )
        recovery_window_days = st.number_input(
            "Recovery window (days)", min_value=1, max_value=180, value=60
        )
        simulation_trials = st.number_input(
            "Simulation trials",
            min_value=100,
            max_value=10000,
            value=2000,
            step=100,
            help="Monte Carlo trials for impact ranges (P10/P50/P90). Higher = smoother bands, slower run.",
        )
        submit = st.form_submit_button("Run scenario")

    if not submit:
        return

    with st.spinner("Running workflow..."):
        try:
            result = run_agent_graph(
                {
                    "disruption_type": disruption_type,
                    "affected_port": selected["port"],
                    "affected_route": affected_route,
                    "severity": severity,
                    "shock_duration_days": shock_duration_days,
                    "recovery_window_days": recovery_window_days,
                    "synthetic_ratio": 0.0,
                    "simulation_trials": int(simulation_trials),
                    "sku": selected["sku"],
                    "event_date": selected["event_date"],
                }
            )
        except Exception as exc:
            st.error(f"Scenario failed: {exc}")
            return

    # ── Risk Classifier ───────────────────────────────────────────────────────
    st.subheader("Risk Classifier")
    if result.risk_classification:
        rc = result.risk_classification

        _LABEL_COLOR = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red",
            "CRITICAL": "darkred",
        }
        color = _LABEL_COLOR.get(rc.final_label, "grey")
        escalation_note = (
            f" *(escalated from **{rc.base_label}** — duration {rc.duration_days:.0f}d)*"
            if rc.escalated
            else ""
        )
        st.markdown(
            f"### :{color}[{rc.final_label}]{escalation_note}"
        )

        # Top-line metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite Score", f"{rc.composite_score:.3f}")
        m2.metric("Mode", rc.mode.upper())
        m3.metric("Base Label", rc.base_label)
        m4.metric("Escalated", "Yes" if rc.escalated else "No")

        # Component breakdown
        st.markdown("**Component Breakdown** *(each normalized 0 → 1)*")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Geo Risk", f"{rc.geo_component:.3f}")
        c1.progress(rc.geo_component)
        c2.metric("Supply Disruption", f"{rc.supply_component:.3f}")
        c2.progress(rc.supply_component)
        c3.metric("Freight / News", f"{rc.freight_component:.3f}")
        c3.progress(rc.freight_component)
        c4.metric("Defect Rate", f"{rc.defect_component:.3f}")
        c4.progress(rc.defect_component)

        # Composite formula callout
        st.caption(
            f"Composite = 0.40 × {rc.geo_component:.3f} (geo)"
            f" + 0.30 × {rc.supply_component:.3f} (supply)"
            f" + 0.15 × {rc.freight_component:.3f} (freight)"
            f" + 0.15 × {rc.defect_component:.3f} (defect)"
            f" = **{rc.composite_score:.3f}**"
        )

        if rc.duration_days is not None:
            st.info(f"Disruption duration signal: **{rc.duration_days:.0f} days** (used for escalation matrix)")

        if rc.rationale:
            with st.expander("Rationale / RAG grounding"):
                st.write(rc.rationale)
                if rc.rag_citations:
                    st.markdown("**Citations:**")
                    for cite in rc.rag_citations:
                        st.markdown(f"- `{cite}`")

        # GPT-4o one-liner and enhanced rationale
        if rc.llm_evaluator_one_liner:
            st.info(f"🤖 GPT-4o: {rc.llm_evaluator_one_liner}")
        if rc.llm_enhanced_rationale:
            with st.expander("GPT-4o Risk Rationale (L4 Signal 3 + Judge)"):
                st.write(rc.llm_enhanced_rationale)
                cols = st.columns(3)
                cols[0].metric("Primary Driver", rc.llm_primary_driver or "N/A")
                cols[1].metric("Confidence", rc.llm_confidence or "N/A")
                cols[2].metric("Label", rc.final_label)

        _render_ensemble_signals(rc)
    else:
        st.warning("Risk classification result not available.")

    st.divider()

    # ── LLM agent outputs ─────────────────────────────────────────────────────
    if result.weather_risk_llm:
        with st.expander("GPT-4.1-mini Weather Interpretation (L3)"):
            st.write(result.weather_risk_llm.supply_chain_narrative)
            st.caption(f"Hubs: {', '.join(result.weather_risk_llm.affected_semiconductor_hubs)}")
            st.caption(f"Event class: {result.weather_risk_llm.event_classification}")

    if result.news_analysis_llm:
        with st.expander("GPT-4.1-mini News Analysis (L2)"):
            st.write(result.news_analysis_llm.summary)
            st.caption(
                f"Category: {result.news_analysis_llm.category} | "
                f"Severity: {result.news_analysis_llm.severity:.3f} | "
                f"Component: {result.news_analysis_llm.news_severity_component:.3f} | "
                f"Duration: {result.news_analysis_llm.expected_duration_days:.0f}d"
            )

    st.divider()

    # ── Supporting signals ────────────────────────────────────────────────────
    st.subheader("Supporting Signals")
    s1, s3 = st.columns(2)
    s1.metric("Live Weather Severity", f"{result.live_weather_severity:.3f}" if result.live_weather_severity is not None else "N/A")
    if result.simulation_result:
        s3.metric("Stockout Probability (L6 Monte Carlo)", f"{result.simulation_result.stockout_probability_pct:.1f}%")

    if result.forecast_result:
        st.divider()
        _render_demand_forecast(result.forecast_result)
    else:
        # L5 was skipped — surface the reason from agent logs
        l5_logs = [lg for lg in result.agent_logs if lg.startswith("L5:")]
        if l5_logs:
            st.info(f"ℹ️ {l5_logs[-1]}")

    st.divider()
    _render_impact_simulation(result)

    st.divider()

    # ── Mitigation ────────────────────────────────────────────────────────────
    if result.mitigation_action:
        st.subheader("Mitigation Recommendation")
        st.caption("GPT-4o + RAG (L7)" if result.mitigation_llm else "Rule-based fallback (L7)")
        st.write(result.mitigation_action.summary)
        for rec in result.mitigation_action.recommendations:
            st.markdown(f"- {rec}")
        st.caption(f"Cost delta: {result.mitigation_action.cost_delta}")
        if result.mitigation_action.india_sourcing_recommendations:
            st.subheader("🇮🇳 India Sourcing (ISM/PLI)")
            for rec in result.mitigation_action.india_sourcing_recommendations:
                st.write(f"• {rec}")
        if result.mitigation_action.rag_citations:
            with st.expander("RAG Citations (L7)"):
                for c in result.mitigation_action.rag_citations:
                    st.caption(c)
        if result.risk_classification and result.risk_classification.critical_flag:
            st.error("🚨 CRITICAL: Immediate mitigation required.")

    # ── Agent logs ────────────────────────────────────────────────────────────
    with st.expander("Agent Logs"):
        for log in result.agent_logs:
            st.text(log)


def main() -> None:
    st.set_page_config(
        page_title="Supply Chain Disruption Predictor",
        layout="wide",
    )
    page = st.sidebar.radio(
        "Navigate",
        ["Data Ingestion", "Live Data Feed", "RAG Search", "Scenario Analyzer", "Demand Forecasts"],
    )
    if page == "Data Ingestion":
        show_data_loader()
    elif page == "Live Data Feed":
        show_ingestion_dashboard()
    elif page == "RAG Search":
        show_rag_search()
    elif page == "Demand Forecasts":
        show_demand_forecasts()
    else:
        show_scenario_analyzer()


if __name__ == "__main__":
    main()
