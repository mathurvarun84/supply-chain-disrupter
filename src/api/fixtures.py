from src.api.schemas import AgentState

INITIAL_AGENTS = [
    AgentState(id="L1", name="Ingestion", status="Idle"),
    AgentState(id="L2", name="News", status="Idle"),
    AgentState(id="L3", name="Weather", status="Idle"),
    AgentState(id="L4", name="Risk", status="Idle"),
    AgentState(id="L5", name="Forecast", status="Idle"),
    AgentState(id="L6", name="Simulate", status="Idle"),
    AgentState(id="L7", name="Mitigate", status="Idle"),
]
# Day 1's status bar must show a fresh, un-run state (all Idle), unlike the
# mockup's INITIAL_AGENTS which shows a completed run.

RUN_ID_FIXTURE = "a9f2-3b7c-11ef"

NEWS_GROUPS = [
    {
        "group": "Hub City Queries",
        "items": [
            {"headline": "TSMC halts advanced node production after magnitude 7.2 earthquake near Hsinchu",
             "source": "Reuters", "tag": "Hsinchu, TW", "time": "14m ago", "score": 0.94},
            {"headline": "Aftershocks disrupt power grid supply to TSMC Fab 18 and Fab 21",
             "source": "Bloomberg", "tag": "Hsinchu, TW", "time": "22m ago", "score": 0.91},
            {"headline": "ASML chip equipment shipments grounded as Taiwan ports close temporarily",
             "source": "Nikkei Asia", "tag": "Osaka, JP", "time": "38m ago", "score": 0.87},
            {"headline": "Samsung Austin fab activates contingency stock protocols amid Asia supply fears",
             "source": "WSJ", "tag": "Austin, TX", "time": "51m ago", "score": 0.79},
        ],
    },
    {
        "group": "Hub Country Queries",
        "items": [
            {"headline": "Taiwan government declares force majeure on semiconductor exports",
             "source": "FT", "tag": "Taiwan", "time": "1h ago", "score": 0.96},
            {"headline": "South Korean chipmakers prepare contingency sourcing after Taiwan quake",
             "source": "Yonhap", "tag": "South Korea", "time": "1h 12m ago", "score": 0.82},
        ],
    },
    {
        "group": "Supplier Country Queries",
        "items": [
            {"headline": "Indian wafer substrate suppliers receive emergency purchase orders from EU OEMs",
             "source": "Mint", "tag": "India", "time": "1h 44m ago", "score": 0.74},
            {"headline": "Malaysia substrate plants activate 24/7 shifts as Taiwan supply dries up",
             "source": "Star", "tag": "Malaysia", "time": "2h 5m ago", "score": 0.71},
            {"headline": "German chemical suppliers raise force majeure risk assessment to elevated",
             "source": "Handelsblatt", "tag": "Germany", "time": "2h 31m ago", "score": 0.68},
        ],
    },
]

WEATHER_CITIES = [
    {"name": "Hsinchu", "flag": "\U0001F1F9\U0001F1FC", "wind": 62, "precip": 18.4, "temp": 24,
     "icon": "⛈️", "severity": 9.2, "trigger": True},
    {"name": "Osaka", "flag": "\U0001F1EF\U0001F1F5", "wind": 14, "precip": 2.1, "temp": 18,
     "icon": "☁️", "severity": 2.1, "trigger": False},
    {"name": "Austin", "flag": "\U0001F1FA\U0001F1F8", "wind": 22, "precip": 0.3, "temp": 31,
     "icon": "⛅", "severity": 1.4, "trigger": False},
    {"name": "Shanghai", "flag": "\U0001F1E8\U0001F1F3", "wind": 19, "precip": 4.7, "temp": 27,
     "icon": "\U0001F327️", "severity": 3.8, "trigger": False},
    {"name": "Singapore", "flag": "\U0001F1F8\U0001F1EC", "wind": 11, "precip": 6.2, "temp": 29,
     "icon": "\U0001F326️", "severity": 2.3, "trigger": False},
    {"name": "Rotterdam", "flag": "\U0001F1F3\U0001F1F1", "wind": 28, "precip": 1.1, "temp": 16,
     "icon": "☁️", "severity": 1.8, "trigger": False},
]

LOG_LINES = [
    {"level": "L1", "text": "Ingested 9 news rows, 6 weather rows → run_id " + RUN_ID_FIXTURE, "tab": 0},
    {"level": "L2", "text": "News analysis: category=GEOPOLITICAL_CONFLICT, signals=3, geo_component=0.71", "tab": 1},
    {"level": "L3", "text": "Weather risk: Hsinchu severity=9.2/10, is_trigger_hub=true, geo_component=0.52", "tab": 1},
    {"level": "L4", "text": "Ensemble: rule=CRITICAL, distilbert=CRITICAL(94%), llm=CRITICAL, judge=CRITICAL", "tab": 1},
    {"level": "L4", "text": "composite_score=0.847, threshold=0.47 → verdict=CRITICAL, verdict_type=majority_rule", "tab": 1},
    {"level": "L5", "text": "Prophet forecast: -26% expected demand drop, Laptops/Phones most affected", "tab": 2},
    {"level": "L6", "text": "Monte Carlo: P50 stockout=41%, P90=68%, revenue_at_risk_P50=$4.2M, 500 runs", "tab": 2},
    {"level": "L7", "text": "Mitigation plan · 3 ranked actions · Slack: FIRED (critical_flag=True, code-enforced)", "tab": 3},
]

GANTT = [
    {"id": "L1", "start": 0, "dur": 4.2, "color": "#22C55E"},
    {"id": "L2", "start": 4.2, "dur": 3.1, "color": "#22C55E"},
    {"id": "L3", "start": 4.2, "dur": 1.8, "color": "#22C55E"},
    {"id": "L4", "start": 7.3, "dur": 7.2, "color": "#22C55E"},
    {"id": "L5", "start": 14.5, "dur": 5.6, "color": "#818CF8"},
    {"id": "L6", "start": 14.5, "dur": 8.4, "color": "#818CF8"},
    {"id": "L7", "start": 23.1, "dur": 6.3, "color": "#22C55E"},
]

FORECAST_CATEGORIES = ["Laptops", "Phones", "Headphones", "Speakers"]

FORECAST_SERIES = [
    {
        "day": f"D+{i + 1}",
        "baseline": round(1000 + __import__("math").sin(i * 0.3) * 40 + i * 1.5),
        "adjusted": round(max(380, 1000 - (i * 42 if i < 8 else 336) + __import__("math").sin(i * 0.3) * 25 + i * 1.5)),
    }
    for i in range(30)
]

MONTE_CARLO = [
    {"range": "0-10%", "count": 12}, {"range": "10-20%", "count": 28}, {"range": "20-30%", "count": 47},
    {"range": "30-40%", "count": 89}, {"range": "40-50%", "count": 124}, {"range": "50-60%", "count": 96},
    {"range": "60-70%", "count": 71}, {"range": "70-80%", "count": 28}, {"range": "80-90%", "count": 5},
]

COST_DATA = [
    {"agent": "L2", "cost": 0.0034}, {"agent": "L3", "cost": 0.0012},
    {"agent": "L4", "cost": 0.0089}, {"agent": "L7", "cost": 0.0156},
]

VERDICT_DIST = [
    {"name": "Majority Rule", "value": 67, "color": "#3B82F6"},
    {"name": "LLM-Arbitrated", "value": 24, "color": "#8B5CF6"},
    {"name": "Escalated", "value": 9, "color": "#F59E0B"},
]

LATENCY_DATA = [
    {"agent": "L1", "p50": 4.2, "p90": 6.8}, {"agent": "L2", "p50": 3.1, "p90": 5.4},
    {"agent": "L3", "p50": 1.8, "p90": 2.9}, {"agent": "L4", "p50": 7.2, "p90": 11.3},
    {"agent": "L5", "p50": 5.6, "p90": 8.1}, {"agent": "L6", "p50": 8.4, "p90": 14.2},
    {"agent": "L7", "p50": 6.3, "p90": 9.7},
]

PROMPT_LOG = [
    {"ts": "14:32:14", "agent": "L7", "model": "gpt-4o",
     "prompt": "Generate mitigation plan for CRITICAL disruption: Taiwan earthquake M7.2...",
     "resp": '{"urgency":"IMMEDIATE","actions":[{"rank":1,"text":"Reroute Cape of Good Hope...',
     "tokens": 2183, "cost": 0.0156, "latency": 6.3},
    {"ts": "14:32:07", "agent": "L4", "model": "gpt-4o",
     "prompt": "Classify supply chain risk given signals: geo=0.71, supply=0.89, freight=0.54...",
     "resp": '{"verdict":"CRITICAL","confidence":0.94,"rationale":"Seismic event...',
     "tokens": 1247, "cost": 0.0089, "latency": 7.2},
    {"ts": "14:31:58", "agent": "L2", "model": "gpt-4o-mini",
     "prompt": "Analyze 9 news items for supply chain disruption signals...",
     "resp": '{"category":"GEOPOLITICAL_CONFLICT","signals":3,"geo_component":0.71...',
     "tokens": 892, "cost": 0.0034, "latency": 3.1},
    {"ts": "14:31:52", "agent": "L3", "model": "gpt-4o-mini",
     "prompt": "Evaluate weather impact on 6 fab hub cities. Hsinchu: wind=62km/h, precip=18.4mm...",
     "resp": '{"hsinchu_severity":9.2,"is_trigger_hub":true,"geo_component":0.52...',
     "tokens": 412, "cost": 0.0012, "latency": 1.8},
]

GUARDRAIL_TABLE = [
    {"name": "prompt-injection-screen", "dir": "input", "agent": "L2", "pass_count": 142, "fail_count": 3,
     "reason": "Adversarial suffix detected in Red Sea headline seed"},
    {"name": "length-cap-4096", "dir": "input", "agent": "L2/L4", "pass_count": 145, "fail_count": 0, "reason": "—"},
    {"name": "structured-output-schema", "dir": "output", "agent": "L4", "pass_count": 141, "fail_count": 4,
     "reason": "Missing field: verdict_confidence"},
    {"name": "fallback-on-failure", "dir": "output", "agent": "L4", "pass_count": 145, "fail_count": 0, "reason": "—"},
    {"name": "faithfulness-gate", "dir": "output", "agent": "L7", "pass_count": 138, "fail_count": 7,
     "reason": "faithfulness=0.61 < 0.75 → routed to human review"},
    {"name": "slack-critical-flag-guard", "dir": "output", "agent": "L7", "pass_count": 145, "fail_count": 0, "reason": "—"},
]

RAGAS_SCORES = [
    {"metric": "Faithfulness", "score": 0.87, "threshold": 0.75, "passed": True},
    {"metric": "Answer Relevance", "score": 0.91, "threshold": 0.80, "passed": True},
    {"metric": "Context Precision", "score": 0.79, "threshold": 0.70, "passed": True},
    {"metric": "Context Recall", "score": 0.83, "threshold": 0.75, "passed": True},
]

CORPUS = [
    {"name": "historical_precedents", "docs": 847, "real": 612, "synth": 235, "last_ingested_at": "2025-06-28 03:12 UTC"},
    {"name": "export_control_corpus", "docs": 324, "real": 324, "synth": 0, "last_ingested_at": "2025-06-27 18:45 UTC"},
    {"name": "india_sourcing_corpus", "docs": 193, "real": 97, "synth": 96, "last_ingested_at": "2025-06-30 09:22 UTC"},
]

GOLD_QA = [
    # 60/40 agent_pattern/natural_question split (10/6) so the mix is
    # visibly present in the Gold Dataset table, not a single example of
    # each — agent_pattern rows mirror the terse internal RAG query
    # strings from MITIGATION["rag_query_trace"]; natural_question rows
    # are full English questions an evaluator would ask by hand.
    {"question": "Recovery timeline after major Taiwan earthquake for TSMC advanced node output?",
     "ground_truth": "4–6 weeks for advanced nodes; 2–3 weeks for mature nodes (2016 precedent)", "match": True,
     "source_collection": "historical_precedents", "query_style": "natural_question"},
    {"question": "Export control regulations affecting EUV equipment shipments to Taiwan?",
     "ground_truth": "EAR-99 classification, BIS Entity List restrictions, CHIPS Act Section 22 provisions", "match": True,
     "source_collection": "export_control_corpus", "query_style": "natural_question"},
    {"question": "PLI-certified Indian substrate suppliers with emergency capacity?",
     "ground_truth": "Kaynes Technology (Mysuru), Tata Electronics (Dholera), SPEL Semiconductor (Chennai)", "match": True,
     "source_collection": "india_sourcing_corpus", "query_style": "natural_question"},
    {"question": "Red Sea crisis impact on Rotterdam port throughput?",
     "ground_truth": "+14 day avg transit via Cape of Good Hope; Suez Canal volume −42%", "match": False,
     "source_collection": "historical_precedents", "query_style": "natural_question"},
    {"question": "How long did DRAM spot prices stay elevated after the 2016 Taiwan quake?",
     "ground_truth": "DRAM spot prices spiked 15% within 72h and normalized over ~5 weeks", "match": True,
     "source_collection": "historical_precedents", "query_style": "natural_question"},
    {"question": "Which agency administers the BIS Entity List restrictions cited for chip tooling?",
     "ground_truth": "U.S. Department of Commerce, Bureau of Industry and Security (BIS)", "match": False,
     "source_collection": "export_control_corpus", "query_style": "natural_question"},
    {"question": "historical_disruption_lookup: Taiwan earthquake M7.2 recovery_timeline advanced_node",
     "ground_truth": "4–6 weeks advanced nodes / 2–3 weeks mature nodes; DRAM spot +15% within 72h (2016 precedent)",
     "match": True, "source_collection": "historical_precedents", "query_style": "agent_pattern"},
    {"question": "export_control_check: EUV_equipment shipment_route=Taiwan export_control_norm>0.50",
     "ground_truth": "EAR-99 classification; BIS Entity List; CHIPS Act Section 22 provisions apply",
     "match": True, "source_collection": "export_control_corpus", "query_style": "agent_pattern"},
    {"question": "india_sourcing_query: wafer_substrate geo_component>0.40 asia_hub_affected=True",
     "ground_truth": "Kaynes Technology (Mysuru) 45K wafer/mo, Tata Electronics (Dholera) 20K wafer/mo, both PLI-certified",
     "match": True, "source_collection": "india_sourcing_corpus", "query_style": "agent_pattern"},
    {"question": "historical_disruption_lookup: Red Sea Houthi_attacks freight_reroute Suez_Canal",
     "ground_truth": "+14 day avg transit via Cape of Good Hope; Suez Canal volume −42%",
     "match": False, "source_collection": "historical_precedents", "query_style": "agent_pattern"},
    {"question": "export_control_check: EAR-99 tooling_category=lithography destination=Taiwan",
     "ground_truth": "EAR-99 classification, BIS Entity List restrictions, CHIPS Act Section 22 provisions",
     "match": True, "source_collection": "export_control_corpus", "query_style": "agent_pattern"},
    {"question": "india_sourcing_query: substrate_supplier PLI_scheme certified_capacity>=20000",
     "ground_truth": "SPEL Semiconductor (Chennai) and Tata Electronics (Dholera) both PLI Semiconductor Scheme certified",
     "match": True, "source_collection": "india_sourcing_corpus", "query_style": "agent_pattern"},
    {"question": "historical_disruption_lookup: South_Korea contingency_sourcing Taiwan_quake precedent",
     "ground_truth": "South Korean chipmakers activated substrate contingency sourcing within 48h of prior Taiwan quakes",
     "match": True, "source_collection": "historical_precedents", "query_style": "agent_pattern"},
    {"question": "export_control_check: freight_route=Rotterdam export_control_norm<0.50",
     "ground_truth": "No export control restriction applies below the 0.50 norm threshold for Rotterdam-bound freight",
     "match": False, "source_collection": "export_control_corpus", "query_style": "agent_pattern"},
    {"question": "india_sourcing_query: wafer_substrate lead_time<=72h region=Karnataka",
     "ground_truth": "Kaynes Technology (Mysuru, Karnataka) — 45K wafer/mo, 72h confirmed leadtime",
     "match": True, "source_collection": "india_sourcing_corpus", "query_style": "agent_pattern"},
    {"question": "What safety-stock buffer policy applies during a demand trough for Laptops/Phones?",
     "ground_truth": "Liquidate up to 30% of safety stock while preserving a 15% minimum buffer per InventoryPolicy_v4",
     "match": True, "source_collection": "historical_precedents", "query_style": "natural_question"},
]

RISK_CLASSIFICATION = {
    "run_id": RUN_ID_FIXTURE,
    "verdict_type": "majority_rule",
    "composite_score": 0.847,
    "threshold": 0.47,
    "rule_signal": {
        "label": "CRITICAL",
        "detail": {"geo": 0.71, "supply_disruption": 0.89, "freight": 0.54, "defect": 0.23},
        "confidence": None,
        "rationale": "weighted_sum = 0.633",
        "citations": [],
    },
    "distilbert_signal": {
        "label": "CRITICAL",
        "detail": {},
        "confidence": 0.94,
        "rationale": "66M params · local inference · temp=0.0",
        "citations": [],
    },
    "llm_signal": {
        "label": "CRITICAL",
        "detail": {},
        "confidence": None,
        "rationale": (
            "Taiwan M7.2 directly impacts TSMC Fab 18/21 advanced nodes (3nm, 5nm). "
            "Historical precedent: 2016 Taiwan quake caused 15% DRAM spot price spike within 72h. "
            "EAR-99 EUV tooling controls compound recovery timeline. Recommend immediate escalation."
        ),
        "citations": ["TW_2016_Quake_Impact", "EAR-99_EUV_Controls"],
    },
    "judge_text": (
        "All three signals agree: CRITICAL. Judge concurs — seismic impact on leading-edge node "
        "capacity + geopolitical escalation matches historical Taiwan Strait crisis playbook. Confidence: 0.94."
    ),
    "slack_should_fire": True,
}

MITIGATION = {
    "run_id": RUN_ID_FIXTURE,
    "urgency": "IMMEDIATE",
    "ranked_actions": [
        {"rank": 1, "text": ("Reroute via Cape of Good Hope — activate 12h advance booking with Maersk and "
                             "MSC on Rotterdam leg. ETA +3 days, freight premium ~$180K vs Suez baseline."),
         "citations": ["RouteMap_Config_v2", "Maersk_RedSea_2025"]},
        {"rank": 2, "text": ("Emergency PO: 45,000 wafer substrates from Kaynes Technology Mysuru (PLI-certified) "
                             "+ 20,000 from Tata Electronics Dholera — confirmed 72h leadtime."),
         "citations": ["india_sourcing_corpus", "PLI_Semicond_Scheme_2023"]},
        {"rank": 3, "text": ("Liquidate 30% safety stock buffer (Laptops, Phones categories) to prevent stockout "
                             "during demand trough. Preserve 15% minimum buffer per InventoryPolicy_v4."),
         "citations": ["InventoryPolicy_v4"]},
    ],
    "rag_query_trace": [
        "historical_disruption_lookup → historical_precedents (always fired)",
        "export_control_check → export_control_corpus (export_control_norm=0.62 > 0.50)",
        "india_sourcing_query → india_sourcing_corpus (geo_component > 0.40 AND asia_hub_affected=True)",
    ],
    "india_sourcing_recommendations": [
        "Kaynes Technology — Mysuru, Karnataka — 45K wafer/mo — PLI Semiconductor Scheme",
        "Tata Electronics — Dholera SEZ, Gujarat — 20K wafer/mo — ISM Greenfield 2024",
    ],
    "slack_preview": (
        "CRITICAL disruption detected\nRisk: 0.847 | GEOPOLITICAL_CONFLICT\nAffected: TW Fab hubs (TSMC)\n"
        f"Actions: 3 ranked · India sourcing ✓\nrun_id: {RUN_ID_FIXTURE}"
    ),
    "cost_delta_usd": 180000,
}
