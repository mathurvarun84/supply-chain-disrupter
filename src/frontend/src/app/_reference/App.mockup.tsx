import { useState, useRef, useEffect, Fragment } from "react";
import {
  Activity, BarChart2, Shield, Eye, Database, Settings, Play,
  ChevronDown, ChevronRight, Copy, AlertTriangle, Clock, Globe,
  RefreshCw, X, Bell, Zap, Map, Server, Lock, ArrowRight,
  AlertCircle, CheckCircle, Package, ExternalLink,
} from "lucide-react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell, ReferenceLine,
} from "recharts";

// ─────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────

type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
type AgentStatus = "Idle" | "Running" | "Complete" | "Skipped-Optional" | "Failed-Fallback";
type SourceType = "LIVE" | "DEMO-INJECTED" | "REPLAY";

// ─────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────

const RISK_COLORS: Record<RiskLevel, string> = {
  LOW: "#22C55E",
  MEDIUM: "#F59E0B",
  HIGH: "#F97316",
  CRITICAL: "#EF4444",
};

const STATUS_COLORS: Record<AgentStatus, string> = {
  Idle: "#334155",
  Running: "#F59E0B",
  Complete: "#22C55E",
  "Skipped-Optional": "#818CF8",
  "Failed-Fallback": "#EF4444",
};

const BG = "#0B1220";
const PANEL = "#121A2B";
const BORDER = "#1E293B";
const BORDER2 = "#243044";

const DEMOS = [
  { id: "taiwan", label: "Taiwan Earthquake", preview: "Expect: CRITICAL · Slack fires · ~33s", risk: "CRITICAL" as RiskLevel },
  { id: "redsea", label: "Red Sea Crisis", preview: "Expect: HIGH · Slack quiet · ~31s", risk: "HIGH" as RiskLevel },
  { id: "injection", label: "Prompt-Injection Guardrail Demo", preview: "Expect: guardrail block on Tab 5 · classification unaffected · ~28s", risk: "HIGH" as RiskLevel },
  { id: "baseline", label: "Clean / Low-Risk Baseline", preview: "Expect: LOW · all guardrails pass · Slack quiet · ~29s", risk: "LOW" as RiskLevel },
];

// ─────────────────────────────────────────────────────────────
// MOCK DATA
// ─────────────────────────────────────────────────────────────

const INITIAL_AGENTS: Array<{ id: string; name: string; status: AgentStatus }> = [
  { id: "L1", name: "Ingestion", status: "Complete" },
  { id: "L2", name: "News", status: "Complete" },
  { id: "L3", name: "Weather", status: "Complete" },
  { id: "L4", name: "Risk", status: "Complete" },
  { id: "L5", name: "Forecast", status: "Skipped-Optional" },
  { id: "L6", name: "Simulate", status: "Skipped-Optional" },
  { id: "L7", name: "Mitigate", status: "Complete" },
];

const NEWS_GROUPS = [
  {
    group: "Hub City Queries",
    items: [
      { headline: "TSMC halts advanced node production after magnitude 7.2 earthquake near Hsinchu", source: "Reuters", tag: "Hsinchu, TW", time: "14m ago", score: 0.94 },
      { headline: "Aftershocks disrupt power grid supply to TSMC Fab 18 and Fab 21", source: "Bloomberg", tag: "Hsinchu, TW", time: "22m ago", score: 0.91 },
      { headline: "ASML chip equipment shipments grounded as Taiwan ports close temporarily", source: "Nikkei Asia", tag: "Osaka, JP", time: "38m ago", score: 0.87 },
      { headline: "Samsung Austin fab activates contingency stock protocols amid Asia supply fears", source: "WSJ", tag: "Austin, TX", time: "51m ago", score: 0.79 },
    ],
  },
  {
    group: "Hub Country Queries",
    items: [
      { headline: "Taiwan government declares force majeure on semiconductor exports", source: "FT", tag: "Taiwan", time: "1h ago", score: 0.96 },
      { headline: "South Korean chipmakers prepare contingency sourcing after Taiwan quake", source: "Yonhap", tag: "South Korea", time: "1h 12m ago", score: 0.82 },
    ],
  },
  {
    group: "Supplier Country Queries",
    items: [
      { headline: "Indian wafer substrate suppliers receive emergency purchase orders from EU OEMs", source: "Mint", tag: "India", time: "1h 44m ago", score: 0.74 },
      { headline: "Malaysia substrate plants activate 24/7 shifts as Taiwan supply dries up", source: "Star", tag: "Malaysia", time: "2h 5m ago", score: 0.71 },
      { headline: "German chemical suppliers raise force majeure risk assessment to elevated", source: "Handelsblatt", tag: "Germany", time: "2h 31m ago", score: 0.68 },
    ],
  },
];

const WEATHER_CITIES = [
  { name: "Hsinchu", flag: "🇹🇼", wind: 62, precip: 18.4, temp: 24, icon: "⛈️", severity: 9.2, trigger: true },
  { name: "Osaka", flag: "🇯🇵", wind: 14, precip: 2.1, temp: 18, icon: "☁️", severity: 2.1, trigger: false },
  { name: "Austin", flag: "🇺🇸", wind: 22, precip: 0.3, temp: 31, icon: "⛅", severity: 1.4, trigger: false },
  { name: "Shanghai", flag: "🇨🇳", wind: 19, precip: 4.7, temp: 27, icon: "🌧️", severity: 3.8, trigger: false },
  { name: "Singapore", flag: "🇸🇬", wind: 11, precip: 6.2, temp: 29, icon: "🌦️", severity: 2.3, trigger: false },
  { name: "Rotterdam", flag: "🇳🇱", wind: 28, precip: 1.1, temp: 16, icon: "☁️", severity: 1.8, trigger: false },
];

const LOG_LINES = [
  { level: "L1", text: "Ingested 9 news rows, 6 weather rows → run_id a9f2-3b7c-11ef", tab: 0 },
  { level: "L2", text: "News analysis: category=GEOPOLITICAL_CONFLICT, signals=3, geo_component=0.71", tab: 1 },
  { level: "L3", text: "Weather risk: Hsinchu severity=9.2/10, is_trigger_hub=true, geo_component=0.52", tab: 1 },
  { level: "L4", text: "Ensemble: rule=CRITICAL, distilbert=CRITICAL(94%), llm=CRITICAL, judge=CRITICAL", tab: 1 },
  { level: "L4", text: "composite_score=0.847, threshold=0.47 → verdict=CRITICAL, verdict_type=majority_rule", tab: 1 },
  { level: "L5", text: "Prophet forecast: -26% expected demand drop, Laptops/Phones most affected", tab: 2 },
  { level: "L6", text: "Monte Carlo: P50 stockout=41%, P90=68%, revenue_at_risk_P50=$4.2M, 500 runs", tab: 2 },
  { level: "L7", text: "Mitigation plan · 3 ranked actions · Slack: FIRED (critical_flag=True, code-enforced)", tab: 3 },
];

const GANTT = [
  { id: "L1", start: 0, dur: 4.2, color: "#22C55E" },
  { id: "L2", start: 4.2, dur: 3.1, color: "#22C55E" },
  { id: "L3", start: 4.2, dur: 1.8, color: "#22C55E" },
  { id: "L4", start: 7.3, dur: 7.2, color: "#22C55E" },
  { id: "L5", start: 14.5, dur: 5.6, color: "#818CF8" },
  { id: "L6", start: 14.5, dur: 8.4, color: "#818CF8" },
  { id: "L7", start: 23.1, dur: 6.3, color: "#22C55E" },
];

const TOTAL_DURATION = 29.4;

const FORECAST_DATA = Array.from({ length: 30 }, (_, i) => ({
  day: `D+${i + 1}`,
  baseline: Math.round(1000 + Math.sin(i * 0.3) * 40 + i * 1.5),
  adjusted: Math.round(Math.max(380, 1000 - (i < 8 ? i * 42 : 336) + Math.sin(i * 0.3) * 25 + i * 1.5)),
}));

const MONTE_CARLO = [
  { range: "0-10%", count: 12 }, { range: "10-20%", count: 28 }, { range: "20-30%", count: 47 },
  { range: "30-40%", count: 89 }, { range: "40-50%", count: 124 }, { range: "50-60%", count: 96 },
  { range: "60-70%", count: 71 }, { range: "70-80%", count: 28 }, { range: "80-90%", count: 5 },
];

const COST_DATA = [
  { agent: "L2", cost: 0.0034 }, { agent: "L3", cost: 0.0012 },
  { agent: "L4", cost: 0.0089 }, { agent: "L7", cost: 0.0156 },
];

const VERDICT_DIST = [
  { name: "Majority Rule", value: 67, color: "#3B82F6" },
  { name: "LLM-Arbitrated", value: 24, color: "#8B5CF6" },
  { name: "Escalated", value: 9, color: "#F59E0B" },
];

const LATENCY_DATA = [
  { agent: "L1", p50: 4.2, p90: 6.8 }, { agent: "L2", p50: 3.1, p90: 5.4 },
  { agent: "L3", p50: 1.8, p90: 2.9 }, { agent: "L4", p50: 7.2, p90: 11.3 },
  { agent: "L5", p50: 5.6, p90: 8.1 }, { agent: "L6", p50: 8.4, p90: 14.2 },
  { agent: "L7", p50: 6.3, p90: 9.7 },
];

const PROMPT_LOG = [
  { ts: "14:32:14", agent: "L7", model: "gpt-4o", prompt: "Generate mitigation plan for CRITICAL disruption: Taiwan earthquake M7.2...", resp: '{"urgency":"IMMEDIATE","actions":[{"rank":1,"text":"Reroute Cape of Good Hope...', tokens: 2183, cost: 0.0156, latency: 6.3 },
  { ts: "14:32:07", agent: "L4", model: "gpt-4o", prompt: "Classify supply chain risk given signals: geo=0.71, supply=0.89, freight=0.54...", resp: '{"verdict":"CRITICAL","confidence":0.94,"rationale":"Seismic event...', tokens: 1247, cost: 0.0089, latency: 7.2 },
  { ts: "14:31:58", agent: "L2", model: "gpt-4o-mini", prompt: "Analyze 9 news items for supply chain disruption signals...", resp: '{"category":"GEOPOLITICAL_CONFLICT","signals":3,"geo_component":0.71...', tokens: 892, cost: 0.0034, latency: 3.1 },
  { ts: "14:31:52", agent: "L3", model: "gpt-4o-mini", prompt: "Evaluate weather impact on 6 fab hub cities. Hsinchu: wind=62km/h, precip=18.4mm...", resp: '{"hsinchu_severity":9.2,"is_trigger_hub":true,"geo_component":0.52...', tokens: 412, cost: 0.0012, latency: 1.8 },
];

const GUARDRAIL_TABLE = [
  { name: "prompt-injection-screen", dir: "input", agent: "L2", pass: 142, fail: 3, reason: "Adversarial suffix detected in Red Sea headline seed" },
  { name: "length-cap-4096", dir: "input", agent: "L2/L4", pass: 145, fail: 0, reason: "—" },
  { name: "structured-output-schema", dir: "output", agent: "L4", pass: 141, fail: 4, reason: "Missing field: verdict_confidence" },
  { name: "fallback-on-failure", dir: "output", agent: "L4", pass: 145, fail: 0, reason: "—" },
  { name: "faithfulness-gate", dir: "output", agent: "L7", pass: 138, fail: 7, reason: "faithfulness=0.61 < 0.75 → routed to human review" },
  { name: "slack-critical-flag-guard", dir: "output", agent: "L7", pass: 145, fail: 0, reason: "—" },
];

const RAGAS_SCORES = [
  { metric: "Faithfulness", score: 0.87, threshold: 0.75, pass: true },
  { metric: "Answer Relevance", score: 0.91, threshold: 0.80, pass: true },
  { metric: "Context Precision", score: 0.79, threshold: 0.70, pass: true },
  { metric: "Context Recall", score: 0.83, threshold: 0.75, pass: true },
];

const CORPUS = [
  { name: "historical_precedents", docs: 847, real: 612, synth: 235, ts: "2025-06-28 03:12 UTC" },
  { name: "export_control_corpus", docs: 324, real: 324, synth: 0, ts: "2025-06-27 18:45 UTC" },
  { name: "india_sourcing_corpus", docs: 193, real: 97, synth: 96, ts: "2025-06-30 09:22 UTC" },
];

const GOLD_QA = [
  { q: "Recovery timeline after major Taiwan earthquake for TSMC advanced node output?", truth: "4–6 weeks for advanced nodes; 2–3 weeks for mature nodes (2016 precedent)", match: true },
  { q: "Export control regulations affecting EUV equipment shipments to Taiwan?", truth: "EAR-99 classification, BIS Entity List restrictions, CHIPS Act Section 22 provisions", match: true },
  { q: "PLI-certified Indian substrate suppliers with emergency capacity?", truth: "Kaynes Technology (Mysuru), Tata Electronics (Dholera), SPEL Semiconductor (Chennai)", match: true },
  { q: "Red Sea crisis impact on Rotterdam port throughput?", truth: "+14 day avg transit via Cape of Good Hope; Suez Canal volume −42%", match: false },
];

// ─────────────────────────────────────────────────────────────
// SHARED COMPONENTS
// ─────────────────────────────────────────────────────────────

function RiskBadge({ level, pulse = false, size = "md" }: { level: RiskLevel; pulse?: boolean; size?: "sm" | "md" | "lg" }) {
  const sizes = { sm: "text-[10px] px-2 py-0.5", md: "text-xs px-2.5 py-1", lg: "text-xl px-5 py-2 font-bold" };
  const c = RISK_COLORS[level];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-mono font-semibold tracking-widest shrink-0 ${sizes[size]}`}
      style={{ background: c + "18", color: c, border: `1px solid ${c}44` }}
    >
      {pulse && level === "CRITICAL" && <span className="inline-block w-1.5 h-1.5 rounded-full animate-ping" style={{ backgroundColor: c }} />}
      {level}
    </span>
  );
}

function AgentNode({ id, name, status, compact = false }: { id: string; name: string; status: AgentStatus; compact?: boolean }) {
  const c = STATUS_COLORS[status];
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div
        title={`${id}: ${name} — ${status}`}
        className={`flex items-center justify-center rounded font-mono font-bold text-[10px] ${compact ? "w-8 h-7" : "w-10 h-9"} ${status === "Running" ? "animate-pulse" : ""}`}
        style={{ background: c + "20", color: c, border: `1.5px solid ${c}50` }}
      >
        {id}
      </div>
      {!compact && <span className="text-[9px] text-slate-600 text-center leading-none" style={{ maxWidth: 44 }}>{name}</span>}
    </div>
  );
}

function CitationChip({ source, collection }: { source: string; collection: string }) {
  return (
    <span
      title={`${collection}: ${source}`}
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full font-mono cursor-pointer hover:opacity-80 transition-opacity"
      style={{ background: "#3B82F618", color: "#60A5FA", border: "1px solid #3B82F630" }}
    >
      <ExternalLink size={9} />
      {source}
    </span>
  );
}

const TOOLTIP_STYLE = {
  backgroundColor: PANEL, border: `1px solid ${BORDER}`, borderRadius: 6, color: "#CBD5E1", fontSize: 11,
};

// ─────────────────────────────────────────────────────────────
// TAB 1 — LIVE FEED
// ─────────────────────────────────────────────────────────────

function TabLiveFeed({ onTabSwitch }: { onTabSwitch: (t: number) => void }) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 grid gap-3 p-3 overflow-hidden" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>

        {/* News Panel */}
        <div className="flex flex-col overflow-hidden rounded-lg" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="px-3 py-2.5 flex items-center justify-between shrink-0" style={{ borderBottom: `1px solid ${BORDER}` }}>
            <span className="text-xs font-semibold text-slate-200">Google News RSS — 14 parallel queries</span>
            <span className="text-[10px] font-mono text-slate-500">9 new · run_id a9f2…</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-3">
            {NEWS_GROUPS.map((g) => (
              <div key={g.group}>
                <div className="text-[9px] font-semibold text-slate-600 uppercase tracking-wider mb-1.5 px-1">{g.group}</div>
                {g.items.map((item, i) => (
                  <div key={i} className="mb-1.5 p-2 rounded" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-[11px] text-slate-300 leading-snug mb-1.5">{item.headline}</div>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-[9px] text-slate-500">{item.source}</span>
                      <span className="text-[9px] text-slate-700">·</span>
                      <span className="text-[9px] font-mono text-blue-400">{item.tag}</span>
                      <span className="text-[9px] text-slate-700">·</span>
                      <span className="text-[9px] text-slate-600">{item.time}</span>
                      <span className="ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded" style={{
                        background: item.score > 0.85 ? "#EF444420" : item.score > 0.7 ? "#F9731620" : "#F59E0B20",
                        color: item.score > 0.85 ? "#EF4444" : item.score > 0.7 ? "#F97316" : "#F59E0B",
                      }}>{item.score.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Weather Panel */}
        <div className="flex flex-col overflow-hidden rounded-lg" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="px-3 py-2.5 shrink-0" style={{ borderBottom: `1px solid ${BORDER}` }}>
            <span className="text-xs font-semibold text-slate-200">Open-Meteo — 6 Fab-Hub Cities</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 grid grid-cols-2 gap-2 content-start">
            {WEATHER_CITIES.map((city) => (
              <div key={city.name} className="p-2.5 rounded-lg" style={{
                background: BG,
                border: city.trigger ? `1.5px solid ${RISK_COLORS.CRITICAL}50` : `1px solid ${BORDER}`,
                boxShadow: city.trigger ? `0 0 14px ${RISK_COLORS.CRITICAL}18` : "none",
              }}>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-semibold text-slate-200">{city.flag} {city.name}</span>
                  {city.trigger && <span className="text-[9px] font-mono text-red-400 border border-red-500/30 px-1 rounded">TRIGGER</span>}
                </div>
                <div className="text-xl mb-1.5">{city.icon}</div>
                <div className="space-y-0.5 text-[10px]">
                  <div className="flex justify-between"><span className="text-slate-600">Wind</span><span className="font-mono text-slate-400">{city.wind} km/h</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Precip</span><span className="font-mono text-slate-400">{city.precip} mm</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Temp</span><span className="font-mono text-slate-400">{city.temp}°C</span></div>
                </div>
                <div className="mt-2">
                  <div className="flex justify-between text-[9px] mb-1">
                    <span className="text-slate-600">raw_severity</span>
                    <span className="font-mono font-semibold" style={{ color: city.severity > 7 ? RISK_COLORS.CRITICAL : city.severity > 4 ? RISK_COLORS.MEDIUM : RISK_COLORS.LOW }}>{city.severity}/10</span>
                  </div>
                  <div className="h-1 rounded-full overflow-hidden" style={{ background: "#1E293B" }}>
                    <div className="h-full rounded-full" style={{
                      width: `${city.severity * 10}%`,
                      background: city.severity > 7 ? RISK_COLORS.CRITICAL : city.severity > 4 ? RISK_COLORS.MEDIUM : RISK_COLORS.LOW,
                    }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Agent Log */}
        <div className="flex flex-col overflow-hidden rounded-lg" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="px-3 py-2.5 flex items-center gap-2 shrink-0" style={{ borderBottom: `1px solid ${BORDER}` }}>
            <span className="text-xs font-semibold text-slate-200">Running Documentary</span>
            <span className="text-[10px] font-mono text-green-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse inline-block" />LIVE
            </span>
          </div>
          <div ref={logRef} className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono text-[11px]" style={{ background: "#060F1C" }}>
            {LOG_LINES.map((line, i) => (
              <div key={i} onClick={() => onTabSwitch(line.tab)}
                className="flex gap-2 px-1 py-0.5 rounded cursor-pointer hover:bg-slate-800/40 transition-colors">
                <span className="shrink-0 font-bold" style={{ color: STATUS_COLORS.Complete }}>[{line.level}]</span>
                <span className="text-slate-400">{line.text}</span>
              </div>
            ))}
            <div className="text-slate-700 animate-pulse px-1 mt-1">▊</div>
          </div>
        </div>
      </div>

      {/* Gantt Strip */}
      <div className="px-3 pb-3 shrink-0">
        <div className="rounded-lg p-3" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider mb-2">Pipeline Timeline — end-to-end: 29.4s</div>
          <div className="space-y-1">
            {GANTT.map((row) => (
              <div key={row.id} className="flex items-center gap-2">
                <span className="text-[9px] font-mono text-slate-600 w-4 shrink-0">{row.id}</span>
                <div className="flex-1 relative h-3.5 rounded overflow-hidden" style={{ background: "#0B1220" }}>
                  <div className="absolute top-0 h-full rounded flex items-center px-1 text-[9px] font-mono"
                    style={{
                      left: `${(row.start / TOTAL_DURATION) * 100}%`,
                      width: `${(row.dur / TOTAL_DURATION) * 100}%`,
                      background: row.color + "28",
                      border: `1px solid ${row.color}44`,
                      color: row.color,
                      minWidth: 28,
                    }}>{row.dur.toFixed(1)}s</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TAB 2 — RISK CLASSIFICATION
// ─────────────────────────────────────────────────────────────

function TabRiskClassification() {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">

      {/* Verdict Card */}
      <div className="rounded-xl p-5" style={{ background: PANEL, border: `1px solid ${RISK_COLORS.CRITICAL}40`, boxShadow: `0 0 32px ${RISK_COLORS.CRITICAL}0C` }}>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[10px] font-mono text-slate-600 uppercase tracking-widest mb-2">Risk Verdict — run_id a9f2-3b7c-11ef</div>
            <RiskBadge level="CRITICAL" pulse size="lg" />
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <span className="text-[10px] px-2 py-0.5 rounded font-mono" style={{ background: "#3B82F618", color: "#60A5FA", border: "1px solid #3B82F630" }}>verdict_type: majority_rule</span>
              <span className="text-[10px] font-mono text-slate-500">composite_score: <strong className="text-slate-300">0.847</strong></span>
              <span className="text-[10px] font-mono text-slate-600">threshold: 0.47</span>
            </div>
          </div>
          {/* Radial gauge */}
          <div className="flex flex-col items-center shrink-0">
            <div className="relative w-24 h-24">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle cx="50" cy="50" r="38" fill="none" stroke="#1E293B" strokeWidth="9" />
                <circle cx="50" cy="50" r="38" fill="none" stroke={RISK_COLORS.CRITICAL} strokeWidth="9"
                  strokeDasharray={`${0.847 * 238.76} 238.76`} strokeLinecap="round" />
                <circle cx="50" cy="50" r="38" fill="none" stroke="#F59E0B" strokeWidth="3"
                  strokeDasharray={`2 ${238.76 - 2}`} strokeDashoffset={`-${0.47 * 238.76}`} strokeLinecap="round" />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-lg font-mono font-bold text-slate-100">0.847</span>
                <span className="text-[9px] text-slate-600">composite</span>
              </div>
            </div>
            <span className="text-[9px] font-mono text-amber-500">↑ threshold 0.47</span>
          </div>
        </div>
      </div>

      {/* Three-Signal Ensemble */}
      <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider px-1">Three-Signal Ensemble — Independent Witnesses</div>
      <div className="grid grid-cols-3 gap-3">

        {/* Rule-Based */}
        <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="text-xs font-semibold text-blue-400 mb-3">Rule-Based Composite</div>
          {[
            { label: "geo", w: 0.4, v: 0.71 },
            { label: "supply_disruption", w: 0.3, v: 0.89 },
            { label: "freight", w: 0.15, v: 0.54 },
            { label: "defect", w: 0.15, v: 0.23 },
          ].map((c) => (
            <div key={c.label} className="mb-2">
              <div className="flex justify-between text-[10px] mb-1">
                <span className="font-mono text-slate-500">{c.label} ×{c.w}</span>
                <span className="font-mono text-slate-300">{(c.v * c.w).toFixed(3)}</span>
              </div>
              <div className="h-1 rounded-full overflow-hidden" style={{ background: BORDER }}>
                <div className="h-full rounded-full" style={{ width: `${c.v * 100}%`, background: "#3B82F6" }} />
              </div>
            </div>
          ))}
          <div className="mt-3 pt-3 flex justify-between items-center" style={{ borderTop: `1px solid ${BORDER}` }}>
            <span className="text-[10px] font-mono text-slate-600">weighted_sum = 0.633</span>
            <RiskBadge level="CRITICAL" size="sm" />
          </div>
        </div>

        {/* DistilBERT */}
        <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="text-xs font-semibold text-purple-400 mb-3">Fine-Tuned DistilBERT</div>
          <div className="flex flex-col items-center flex-1 justify-center py-2">
            <RiskBadge level="CRITICAL" />
            <div className="mt-3 text-center">
              <div className="text-4xl font-mono font-bold text-slate-100">94%</div>
              <div className="text-[10px] text-slate-500">confidence</div>
            </div>
          </div>
          <div className="h-1.5 rounded-full overflow-hidden mb-3" style={{ background: BORDER }}>
            <div className="h-full rounded-full" style={{ width: "94%", background: "#8B5CF6" }} />
          </div>
          <div className="text-[10px] font-mono text-slate-600 leading-relaxed">66M params · local inference<br />temperature=0.0 · deterministic</div>
        </div>

        {/* GPT-4o + RAG */}
        <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="text-xs font-semibold text-emerald-400 mb-3">GPT-4o + RAG</div>
          <div className="flex items-center gap-2 mb-3">
            <RiskBadge level="CRITICAL" size="sm" />
            <span className="text-[10px] font-mono text-slate-600">llm_verdict</span>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full text-left text-[11px] p-2 rounded mb-3 transition-colors"
            style={{ background: BG, border: `1px solid ${BORDER}` }}
          >
            <div className="flex items-center justify-between text-slate-500">
              <span className="font-semibold">Justification</span>
              <ChevronDown size={11} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
            </div>
            {expanded && (
              <div className="mt-2 text-slate-400 leading-relaxed">
                "Taiwan M7.2 directly impacts TSMC Fab 18/21 advanced nodes (3nm, 5nm). Historical precedent: 2016 Taiwan quake caused 15% DRAM spot price spike within 72h. EAR-99 EUV tooling controls compound recovery timeline. Recommend immediate escalation."
              </div>
            )}
          </button>
          <div className="flex flex-wrap gap-1">
            <CitationChip source="TW_2016_Quake_Impact" collection="historical_precedents" />
            <CitationChip source="EAR-99_EUV_Controls" collection="export_control_corpus" />
          </div>
        </div>
      </div>

      {/* LLM-as-Judge */}
      <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-semibold text-amber-400">LLM-as-Judge Arbitration</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ background: "#22C55E18", color: "#22C55E" }}>CONSENSUS — no arbitration needed</span>
        </div>
        <div className="text-[11px] font-mono text-slate-500 leading-relaxed">
          "All three signals agree: CRITICAL. Judge concurs — seismic impact on leading-edge node capacity + geopolitical escalation matches historical Taiwan Strait crisis playbook. Confidence: 0.94."
        </div>
      </div>

      {/* Guardrail */}
      <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${RISK_COLORS.CRITICAL}35` }}>
        <div className="flex items-center gap-2 mb-2">
          <Lock size={13} className="text-red-400 shrink-0" />
          <span className="text-xs font-semibold text-red-400">Escalation Guard — Code-Enforced, Not LLM-Trusted</span>
        </div>
        <div className="text-[11px] text-slate-500 leading-relaxed font-mono">
          <span className="text-yellow-400">critical_flag = True</span> → <span className="text-red-400">slack_should_fire = True</span> is set by deterministic code (composite_score {'>'} 0.47 AND category ∈ CRITICAL_CATEGORIES). The LLM output is never trusted for this decision — it is overridden by the guard layer regardless of model response.
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TAB 3 — FORECAST & SIMULATION
// ─────────────────────────────────────────────────────────────

function TabForecastSimulation() {
  const [cat, setCat] = useState("Laptops");
  const CATS = ["Laptops", "Phones", "Headphones", "Speakers"];

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="grid grid-cols-2 gap-3 h-full">

        {/* Prophet */}
        <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-semibold text-slate-200">Demand Forecasting — Prophet</span>
            <span className="text-[9px] px-2 py-0.5 rounded font-mono" style={{ background: "#818CF818", color: "#818CF8", border: "1px solid #818CF830" }}>Optional · L5</span>
          </div>
          <div className="flex gap-1.5 mb-3 flex-wrap">
            {CATS.map((c) => (
              <button key={c} onClick={() => setCat(c)} className="text-[10px] px-2 py-0.5 rounded font-mono transition-all"
                style={{ background: cat === c ? "#3B82F620" : BG, color: cat === c ? "#60A5FA" : "#475569", border: `1px solid ${cat === c ? "#3B82F640" : BORDER}` }}>{c}</button>
            ))}
          </div>
          <div className="flex items-baseline gap-2 mb-3">
            <span className="text-3xl font-mono font-bold text-red-400">-26%</span>
            <span className="text-slate-500 text-sm">expected demand drop</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={FORECAST_DATA}>
              <defs>
                <linearGradient id="baseG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="adjG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#EF4444" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" tick={{ fill: "#475569", fontSize: 9 }} interval={4} />
              <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
              <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Area type="monotone" dataKey="baseline" stroke="#3B82F6" fill="url(#baseG)" strokeWidth={2} name="Baseline" dot={false} />
              <Area type="monotone" dataKey="adjusted" stroke="#EF4444" fill="url(#adjG)" strokeWidth={2} name="Disruption-Adjusted" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex gap-1.5 mt-3 flex-wrap">
            {["Chip_Price_Index", "Market_Growth_Rate", "Risk_Score_Composite"].map((r) => (
              <span key={r} className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{ background: "#3B82F610", color: "#60A5FA", border: "1px solid #3B82F625" }}>{r}</span>
            ))}
          </div>
          <div className="mt-1 text-[9px] font-mono text-slate-700">Trained on Ops KPI 2023–2025 · not Lite Master 2015–2018</div>
        </div>

        {/* Monte Carlo */}
        <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-semibold text-slate-200">Monte Carlo Simulation — SimPy</span>
            <span className="text-[9px] px-2 py-0.5 rounded font-mono" style={{ background: "#818CF818", color: "#818CF8", border: "1px solid #818CF830" }}>Optional · L6</span>
          </div>
          <div className="grid grid-cols-3 gap-2 mb-3">
            {[{ label: "P10 Stockout", v: "18%" }, { label: "P50 Stockout", v: "41%" }, { label: "P90 Stockout", v: "68%" }].map((m) => (
              <div key={m.label} className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                <div className="text-xl font-mono font-bold text-slate-100">{m.v}</div>
                <div className="text-[9px] text-slate-600 mt-0.5">{m.label}</div>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={MONTE_CARLO}>
              <XAxis dataKey="range" tick={{ fill: "#475569", fontSize: 9 }} />
              <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
              <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="count" fill="#F97316" radius={[3, 3, 0, 0]} opacity={0.85} name="Runs" />
              <ReferenceLine x="40-50%" stroke="#EF4444" strokeDasharray="4 2"
                label={{ value: "P50", position: "insideTop", fill: "#EF4444", fontSize: 9 }} />
            </BarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-3">
            <div className="rounded p-3" style={{ background: BG, border: `1px solid ${BORDER}` }}>
              <div className="text-[10px] text-slate-600 mb-0.5">Revenue at Risk (P50)</div>
              <div className="text-xl font-mono font-bold text-orange-400">$4.2M</div>
            </div>
            <div className="rounded p-3" style={{ background: BG, border: `1px solid ${BORDER}` }}>
              <div className="text-[10px] text-slate-600 mb-0.5">Alternate Route</div>
              <div className="text-xs font-mono text-blue-400">Cape of Good Hope</div>
              <div className="text-[9px] font-mono text-slate-700">config, not LLM</div>
            </div>
          </div>
          <div className="mt-2 text-right text-[9px] font-mono text-slate-700">500-run Monte Carlo · SimPy discrete-event</div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TAB 4 — MITIGATION PLAN
// ─────────────────────────────────────────────────────────────

function TabMitigationPlan() {
  const [openRag, setOpenRag] = useState<number | null>(null);

  const ACTIONS = [
    { rank: 1, icon: Map, type: "reroute", text: "Reroute via Cape of Good Hope — activate 12h advance booking with Maersk and MSC on Rotterdam leg. ETA +3 days, freight premium ~$180K vs Suez baseline.", cites: ["RouteMap_Config_v2", "Maersk_RedSea_2025"] },
    { rank: 2, icon: Package, type: "sourcing", text: "Emergency PO: 45,000 wafer substrates from Kaynes Technology Mysuru (PLI-certified) + 20,000 from Tata Electronics Dholera — confirmed 72h leadtime.", cites: ["india_sourcing_corpus", "PLI_Semicond_Scheme_2023"] },
    { rank: 3, icon: Server, type: "inventory", text: "Liquidate 30% safety stock buffer (Laptops, Phones categories) to prevent stockout during demand trough. Preserve 15% minimum buffer per InventoryPolicy_v4.", cites: ["InventoryPolicy_v4"] },
  ];

  const RAG_QUERIES = [
    { name: "historical_disruption_lookup", coll: "historical_precedents", cond: "always fired", chunks: [{ t: "2016 Hsinchu quake: DRAM +15% in 72h, recovery 5 wks", s: 0.94 }, { t: "2021 Renesas fab fire: 12-week recovery for MCUs", s: 0.89 }] },
    { name: "export_control_check", coll: "export_control_corpus", cond: "export_control_norm=0.62 > 0.50", chunks: [{ t: "EAR-99 EUV equipment controls active TW→US", s: 0.91 }, { t: "CHIPS Act sec.22 restrictions on advanced nodes", s: 0.87 }] },
    { name: "india_sourcing_query", coll: "india_sourcing_corpus", cond: "geo_component > 0.40 AND asia_hub_affected=True", chunks: [{ t: "Kaynes Technology Mysuru: 45K/mo capacity, PLI-certified", s: 0.96 }, { t: "Tata Electronics Dholera SEZ: greenfield, 2024 operational", s: 0.88 }] },
  ];

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">

      {/* Urgency Banner */}
      <div className="rounded-lg px-4 py-3 flex items-center gap-3" style={{ background: "#EF444410", border: `1px solid ${RISK_COLORS.CRITICAL}40` }}>
        <AlertTriangle size={15} className="text-red-400 shrink-0" />
        <div>
          <span className="text-sm font-bold text-red-400 mr-2">IMMEDIATE ACTION REQUIRED</span>
          <span className="text-[11px] text-red-400/60">Urgency: CRITICAL · Mitigation window: 48h</span>
        </div>
        <div className="ml-auto">
          <span className="text-[10px] font-mono px-2 py-1 rounded flex items-center gap-1.5" style={{ background: "#EF444420", color: "#EF4444", border: "1px solid #EF444440" }}>
            <Bell size={10} />🔔 Slack Alert FIRED
          </span>
        </div>
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: "1fr 280px" }}>
        <div className="space-y-3">
          {/* Ranked Actions */}
          <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">Ranked Mitigation Actions</div>
          {ACTIONS.map((a) => (
            <div key={a.rank} className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full flex items-center justify-center font-mono font-bold text-sm shrink-0" style={{ background: "#3B82F620", color: "#60A5FA" }}>{a.rank}</div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1.5">
                    <a.icon size={11} className="text-slate-500" />
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded text-slate-500" style={{ background: BORDER }}>{a.type}</span>
                  </div>
                  <div className="text-xs text-slate-300 leading-relaxed mb-2">{a.text}</div>
                  <div className="flex gap-1.5 flex-wrap">
                    {a.cites.map((c) => <CitationChip key={c} source={c} collection="RAG" />)}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* RAG Query Trace */}
          <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">RAG Query Trace — 3 ChromaDB Queries Fired</div>
            <div className="space-y-1.5">
              {RAG_QUERIES.map((q, i) => (
                <div key={i}>
                  <button onClick={() => setOpenRag(openRag === i ? null : i)}
                    className="w-full text-left flex items-center gap-2.5 p-2 rounded transition-colors hover:bg-slate-800/30"
                    style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <ChevronRight size={11} className={`text-slate-600 transition-transform shrink-0 ${openRag === i ? "rotate-90" : ""}`} />
                    <span className="text-[10px] font-mono text-blue-400">{q.name}</span>
                    <span className="text-[10px] text-slate-600">{q.coll}</span>
                    <span className="text-[10px] text-slate-600 ml-auto">{q.cond}</span>
                  </button>
                  {openRag === i && (
                    <div className="pl-5 pt-1 space-y-1">
                      {q.chunks.map((c, j) => (
                        <div key={j} className="flex items-center gap-3 text-[10px] font-mono px-2 py-1 rounded" style={{ background: "#070D18" }}>
                          <span className="text-slate-700">#{j + 1}</span>
                          <span className="flex-1 text-slate-500">{c.t}</span>
                          <span className="text-emerald-500">score: {c.s}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {/* India Sourcing */}
          <div className="rounded-lg p-4" style={{ background: PANEL, border: "1px solid #3B82F630" }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">India Sourcing</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded font-mono" style={{ background: "#3B82F618", color: "#60A5FA" }}>Differentiator</span>
            </div>
            {[
              { name: "Kaynes Technology", loc: "Mysuru, Karnataka", cap: "45K wafer/mo", prog: "PLI Semiconductor Scheme" },
              { name: "Tata Electronics", loc: "Dholera SEZ, Gujarat", cap: "20K wafer/mo", prog: "ISM Greenfield 2024" },
            ].map((s) => (
              <div key={s.name} className="mb-2 p-2.5 rounded" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                <div className="text-xs font-semibold text-slate-200">{s.name}</div>
                <div className="text-[10px] text-slate-600">{s.loc}</div>
                <div className="text-[10px] font-mono text-green-400 mt-0.5">{s.cap}</div>
                <div className="mt-1"><CitationChip source={s.prog} collection="india_sourcing_corpus" /></div>
              </div>
            ))}
          </div>

          {/* Slack Preview */}
          <div className="rounded-lg p-3" style={{ background: PANEL, border: "1px solid #22C55E30" }}>
            <div className="text-[10px] font-semibold text-green-400 mb-2 flex items-center gap-1.5">
              <CheckCircle size={11} /> Slack Alert Sent
            </div>
            <div className="rounded p-2 text-[10px] font-mono text-slate-500 leading-relaxed" style={{ background: "#070D18" }}>
              🚨 *CRITICAL* disruption detected<br />
              Risk: 0.847 | GEOPOLITICAL_CONFLICT<br />
              Affected: TW Fab hubs (TSMC)<br />
              Actions: 3 ranked · India sourcing ✓<br />
              run_id: a9f2-3b7c-11ef
            </div>
          </div>

          {/* Cost Delta */}
          <div className="rounded-lg p-3" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="text-[10px] text-slate-600 mb-2">Reroute Cost Delta</div>
            <div className="text-2xl font-mono font-bold text-orange-400">+$180K</div>
            <div className="text-[10px] text-slate-600 mt-0.5">vs Suez baseline · Cape of Good Hope</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TAB 5 — OBSERVABILITY & GUARDRAILS
// ─────────────────────────────────────────────────────────────

function TabObservability() {
  const [sub, setSub] = useState(0);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex gap-4 px-4 pt-3 shrink-0" style={{ borderBottom: `1px solid ${BORDER}`, background: PANEL }}>
        {["Observability", "Guardrails"].map((t, i) => (
          <button key={t} onClick={() => setSub(i)}
            className="text-sm pb-2.5 font-medium transition-colors border-b-2"
            style={{ color: sub === i ? "#60A5FA" : "#475569", borderBottomColor: sub === i ? "#3B82F6" : "transparent" }}>
            {t}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {sub === 0 ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">

              {/* Cost by Agent */}
              <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
                <div className="text-xs font-semibold text-slate-400 mb-0.5">Cost by Agent</div>
                <div className="text-[10px] font-mono text-slate-600 mb-3">Session total: $0.0291</div>
                <ResponsiveContainer width="100%" height={130}>
                  <BarChart data={COST_DATA} layout="vertical">
                    <XAxis type="number" tick={{ fill: "#475569", fontSize: 9 }} tickFormatter={(v) => `$${v.toFixed(4)}`} />
                    <YAxis type="category" dataKey="agent" tick={{ fill: "#94A3B8", fontSize: 10, fontFamily: "JetBrains Mono" }} width={25} />
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`$${v.toFixed(4)}`, "Cost"]} />
                    <Bar dataKey="cost" fill="#3B82F6" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Verdict Distribution */}
              <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
                <div className="text-xs font-semibold text-slate-400 mb-3">Verdict-Type Distribution</div>
                <div className="flex items-center gap-4">
                  <ResponsiveContainer width={110} height={110}>
                    <PieChart>
                      <Pie data={VERDICT_DIST} cx="50%" cy="50%" innerRadius={32} outerRadius={52} paddingAngle={2} dataKey="value">
                        {VERDICT_DIST.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="space-y-2">
                    {VERDICT_DIST.map((d) => (
                      <div key={d.name} className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: d.color }} />
                        <span className="text-[10px] text-slate-400">{d.name}</span>
                        <span className="text-[10px] font-mono text-slate-200 ml-auto pl-3">{d.value}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Latency */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">P50 / P90 Latency per Agent (s)</div>
              <ResponsiveContainer width="100%" height={110}>
                <BarChart data={LATENCY_DATA} layout="vertical">
                  <XAxis type="number" tick={{ fill: "#475569", fontSize: 9 }} />
                  <YAxis type="category" dataKey="agent" tick={{ fill: "#94A3B8", fontSize: 10, fontFamily: "JetBrains Mono" }} width={25} />
                  <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="p50" fill="#3B82F6" name="P50" radius={[0, 2, 2, 0]} barSize={5} />
                  <Bar dataKey="p90" fill="#8B5CF6" name="P90" radius={[0, 2, 2, 0]} barSize={5} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Prompt Inspector */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">Prompt / Response Inspector — llm_call_log</div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="text-slate-600" style={{ borderBottom: `1px solid ${BORDER}` }}>
                      {["Timestamp", "Agent", "Model", "Prompt Preview", "Tokens", "Cost", "Latency"].map((h) => (
                        <th key={h} className="text-left py-1.5 px-2 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PROMPT_LOG.map((row, i) => (
                      <Fragment key={i}>
                        <tr className="cursor-pointer hover:bg-slate-800/25 transition-colors" style={{ borderBottom: `1px solid ${BORDER}` }}
                          onClick={() => setExpandedRow(expandedRow === i ? null : i)}>
                          <td className="py-1.5 px-2 text-slate-600">{row.ts}</td>
                          <td className="py-1.5 px-2 text-blue-400">{row.agent}</td>
                          <td className="py-1.5 px-2 text-slate-500">{row.model}</td>
                          <td className="py-1.5 px-2 text-slate-400 max-w-[200px] truncate">{row.prompt}</td>
                          <td className="py-1.5 px-2 text-slate-300">{row.tokens}</td>
                          <td className="py-1.5 px-2 text-emerald-500">${row.cost.toFixed(4)}</td>
                          <td className="py-1.5 px-2 text-slate-500">{row.latency}s</td>
                        </tr>
                        {expandedRow === i && (
                          <tr style={{ borderBottom: `1px solid ${BORDER}` }}>
                            <td colSpan={7} className="px-4 py-3" style={{ background: "#070D18" }}>
                              <div className="text-[10px] text-slate-500"><span className="text-slate-600">Prompt: </span>{row.prompt}</div>
                              <div className="text-[10px] text-slate-500 mt-1"><span className="text-slate-600">Response: </span>{row.resp}</div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Headline metric */}
            <div className="rounded-lg p-4 flex items-center gap-5" style={{ background: "#EF444410", border: `1px solid ${RISK_COLORS.CRITICAL}30` }}>
              <Bell size={24} className="text-red-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-500 mb-0.5 uppercase tracking-wider">Slack Alerts Suppressed by Guardrail</div>
                <div className="text-4xl font-mono font-bold text-red-400">7</div>
              </div>
              <div className="ml-auto text-[10px] text-slate-600 font-mono max-w-[200px] leading-relaxed">
                faithfulness_gate failures this session — routed to human review
              </div>
            </div>

            {/* Guardrail Table */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">Guardrail Activity</div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="text-slate-600" style={{ borderBottom: `1px solid ${BORDER}` }}>
                      {["Guardrail", "Direction", "Agent", "Pass", "Fail", "Last Reason"].map((h) => (
                        <th key={h} className="text-left py-1.5 px-2 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {GUARDRAIL_TABLE.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-800/20 transition-colors" style={{ borderBottom: `1px solid ${BORDER}` }}>
                        <td className="py-1.5 px-2 text-blue-400">{row.name}</td>
                        <td className="py-1.5 px-2">
                          <span className="px-1.5 py-0.5 rounded" style={{ background: row.dir === "input" ? "#3B82F614" : "#8B5CF614", color: row.dir === "input" ? "#60A5FA" : "#A78BFA" }}>{row.dir}</span>
                        </td>
                        <td className="py-1.5 px-2 text-slate-500">{row.agent}</td>
                        <td className="py-1.5 px-2 text-green-400">{row.pass}</td>
                        <td className="py-1.5 px-2" style={{ color: row.fail > 0 ? RISK_COLORS.HIGH : "#475569" }}>{row.fail}</td>
                        <td className="py-1.5 px-2 text-slate-600 max-w-[220px] truncate">{row.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Guardrail Map */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">Guardrail Map — Pipeline Annotations</div>
              <div className="flex items-start gap-1.5 overflow-x-auto pb-1">
                {[
                  { id: "L1", guards: ["rate-limiter"] },
                  { id: "L2", guards: ["prompt-injection-screen", "length-cap-4096", "structured-output-schema", "fallback-on-failure"] },
                  { id: "L3", guards: ["schema-validation", "severity-clamp-10"] },
                  { id: "L4", guards: ["output-schema", "fallback-on-failure", "critical-flag-override"] },
                  { id: "L5", guards: ["optional-skip"] },
                  { id: "L6", guards: ["optional-skip"] },
                  { id: "L7", guards: ["faithfulness-gate", "slack-critical-flag-guard"] },
                ].map((node, i, arr) => (
                  <div key={node.id} className="flex items-start gap-1 shrink-0">
                    <div className="flex flex-col items-center gap-1">
                      <div className="text-[9px] font-mono font-bold px-2 py-1 rounded" style={{ background: "#22C55E18", color: "#22C55E", border: "1px solid #22C55E35" }}>{node.id}</div>
                      <div className="space-y-0.5">
                        {node.guards.map((g) => (
                          <div key={g} className="text-[8px] font-mono px-1.5 py-0.5 rounded whitespace-nowrap" style={{ background: BG, color: "#64748B", border: `1px solid ${BORDER}` }}>{g}</div>
                        ))}
                      </div>
                    </div>
                    {i < arr.length - 1 && <div className="mt-3 w-3 h-px shrink-0" style={{ background: BORDER2 }} />}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// TAB 6 — RAG / RAGAS EVALUATION
// ─────────────────────────────────────────────────────────────

function TabRAGEval() {
  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">

      {/* RAGAS Scorecard */}
      <div className="grid grid-cols-4 gap-3">
        {RAGAS_SCORES.map((m) => (
          <div key={m.metric} className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${m.pass ? "#22C55E30" : "#EF444430"}` }}>
            <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-2">{m.metric}</div>
            <div className="text-3xl font-mono font-bold mb-1" style={{ color: m.pass ? "#22C55E" : "#EF4444" }}>{m.score.toFixed(2)}</div>
            <div className="h-1.5 rounded-full overflow-hidden mb-2" style={{ background: BORDER }}>
              <div className="h-full rounded-full" style={{ width: `${m.score * 100}%`, background: m.pass ? "#22C55E" : "#EF4444" }} />
            </div>
            <div className="flex items-center justify-between text-[9px]">
              <span className="text-slate-700 font-mono">threshold: {m.threshold}</span>
              <span style={{ color: m.pass ? "#22C55E" : "#EF4444" }}>{m.pass ? "✓ Pass" : "✗ Fail"}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: "1fr 1fr" }}>

        {/* Corpus Health */}
        <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
          <div className="text-xs font-semibold text-slate-400 mb-3">Corpus Health — 3 ChromaDB Collections</div>
          <div className="space-y-2">
            {CORPUS.map((c) => (
              <div key={c.name} className="p-3 rounded" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                <div className="text-[11px] font-mono text-blue-400 mb-1.5">{c.name}</div>
                <div className="flex items-center gap-2 text-[10px] flex-wrap">
                  <span className="font-mono text-slate-500">{c.docs} docs</span>
                  <span className="px-1.5 py-0.5 rounded font-mono" style={{ background: "#22C55E14", color: "#22C55E" }}>REAL: {c.real}</span>
                  {c.synth > 0 && <span className="px-1.5 py-0.5 rounded font-mono" style={{ background: "#8B5CF614", color: "#A78BFA" }}>SYNTH: {c.synth}</span>}
                </div>
                <div className="text-[9px] font-mono text-slate-700 mt-1.5">Last re-ingested: {c.ts}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          {/* Retrieval Pipeline */}
          <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="text-xs font-semibold text-slate-400 mb-3">Retrieval Pipeline</div>
            <div className="flex items-center gap-2 text-[10px] font-mono overflow-x-auto">
              {[
                { label: "bi-encoder", detail: "all-MiniLM-L6-v2", out: "top-10" },
                { label: "cross-encoder", detail: "ms-marco-MiniLM-L-6-v2", out: "top-3/4" },
                { label: "LLM context", detail: "passed via prompt", out: "" },
              ].map((stage, i) => (
                <div key={i} className="flex items-center gap-2">
                  {i > 0 && <ArrowRight size={10} className="text-slate-700 shrink-0" />}
                  <div className="p-2 rounded shrink-0" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-blue-400">{stage.label}</div>
                    <div className="text-slate-600">{stage.detail}</div>
                    {stage.out && <div className="text-emerald-500">{stage.out}</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Faithfulness Gate */}
          <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="text-xs font-semibold text-slate-400 mb-2">Faithfulness Gate Status</div>
            <div className="text-[11px] text-slate-500 leading-relaxed mb-3">
              When <span className="font-mono text-amber-400">faithfulness {"<"} 0.75</span> → mitigation plan routed to human review, Slack suppressed.
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="p-2 rounded text-center text-[10px]" style={{ background: "#22C55E0C", border: "1px solid #22C55E30" }}>
                <div className="font-mono font-bold text-green-400">0.87 ✓</div>
                <div className="text-slate-600 mt-0.5">Current — Slack allowed</div>
              </div>
              <div className="p-2 rounded text-center text-[10px]" style={{ background: "#EF44440C", border: "1px solid #EF444430" }}>
                <div className="font-mono font-bold text-red-400">0.61 ✗</div>
                <div className="text-slate-600 mt-0.5">Example — human review</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Gold Dataset */}
      <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
        <div className="text-xs font-semibold text-slate-400 mb-3">Gold Dataset — Chunk-Grounded QA Test Set</div>
        <div className="space-y-1.5">
          {GOLD_QA.map((item, i) => (
            <div key={i} className="flex gap-3 p-2.5 rounded text-[10px]" style={{ background: BG, border: `1px solid ${BORDER}` }}>
              <div className={`mt-0.5 shrink-0 font-mono font-bold ${item.match ? "text-green-400" : "text-red-400"}`}>{item.match ? "✓" : "✗"}</div>
              <div>
                <div className="text-slate-400 mb-1">Q: {item.q}</div>
                <div className="font-mono text-slate-600">A: {item.truth}</div>
              </div>
              <div className="ml-auto flex items-center gap-1 shrink-0">
                {item.match
                  ? <CheckCircle size={12} className="text-green-500" />
                  : <AlertCircle size={12} className="text-red-500" />}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────

const TABS = [
  { icon: Activity, label: "Live Feed" },
  { icon: Shield, label: "Risk Classification" },
  { icon: BarChart2, label: "Forecast & Simulation" },
  { icon: Map, label: "Mitigation Plan" },
  { icon: Eye, label: "Observability" },
  { icon: Database, label: "RAG / RAGAS" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [showModal, setShowModal] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [agents, setAgents] = useState(INITIAL_AGENTS);
  const [risk, setRisk] = useState<RiskLevel>("CRITICAL");
  const [sourceType, setSourceType] = useState<SourceType>("DEMO-INJECTED");
  const [criticalBanner, setCriticalBanner] = useState(true);
  const [lastIngested, setLastIngested] = useState(42);

  const SOURCE_STYLES: Record<SourceType, { bg: string; color: string; pulse: boolean }> = {
    LIVE: { bg: "#22C55E18", color: "#22C55E", pulse: true },
    "DEMO-INJECTED": { bg: "#3B82F618", color: "#60A5FA", pulse: false },
    REPLAY: { bg: "#47556918", color: "#94A3B8", pulse: false },
  };

  const runPipeline = (demo: typeof DEMOS[0] | null) => {
    setShowModal(false);
    setPipelineRunning(true);
    const targetRisk = demo ? demo.risk : ("HIGH" as RiskLevel);
    setSourceType(demo ? "DEMO-INJECTED" : "LIVE");
    setAgents(INITIAL_AGENTS.map((a) => ({ ...a, status: "Idle" as AgentStatus })));
    setLastIngested(0);

    let delay = 0;
    INITIAL_AGENTS.forEach((agent, i) => {
      setTimeout(() => setAgents((prev) => prev.map((a, idx) => idx === i ? { ...a, status: "Running" } : a)), delay);
      delay += 700;
      setTimeout(() => {
        setAgents((prev) => prev.map((a, idx) => idx === i ? { ...a, status: i >= 4 && i <= 5 ? "Skipped-Optional" : "Complete" } : a));
        if (i === INITIAL_AGENTS.length - 1) {
          setPipelineRunning(false);
          setRisk(targetRisk);
          setCriticalBanner(targetRisk === "CRITICAL");
        }
      }, delay);
      delay += 350;
    });
  };

  const ss = SOURCE_STYLES[sourceType];

  return (
    <div className="flex flex-col h-screen overflow-hidden" style={{ background: BG, color: "#E2E8F0", fontFamily: "'Inter', sans-serif" }}>

      {/* Critical Banner */}
      {criticalBanner && risk === "CRITICAL" && (
        <div className="flex items-center gap-3 px-4 py-1.5 text-xs font-semibold shrink-0" style={{ background: "#EF444418", borderBottom: `1px solid ${RISK_COLORS.CRITICAL}45` }}>
          <span className="w-2 h-2 rounded-full bg-red-500 animate-ping inline-block shrink-0" />
          <AlertTriangle size={13} className="text-red-400 shrink-0" />
          <span className="text-red-300">CRITICAL DISRUPTION DETECTED — composite_score: 0.847 — critical_flag=True enforced in code — Slack alert fired</span>
          <button onClick={() => setCriticalBanner(false)} className="ml-auto text-red-500 hover:text-red-300 transition-colors shrink-0"><X size={13} /></button>
        </div>
      )}

      {/* Top Status Bar */}
      <div className="flex items-center gap-4 px-4 py-2 shrink-0" style={{ background: PANEL, borderBottom: `1px solid ${BORDER}` }}>

        {/* Wordmark */}
        <div className="flex items-center gap-2 shrink-0 mr-1">
          <div className="w-6 h-6 rounded flex items-center justify-center" style={{ background: "linear-gradient(135deg, #1D4ED8, #7C3AED)" }}>
            <Zap size={11} className="text-white" />
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-200 leading-none">Supply Chain</div>
            <div className="text-[9px] text-slate-600 leading-none tracking-wide">Command Center</div>
          </div>
        </div>

        {/* Pipeline Strip */}
        <div className="flex items-center gap-1 flex-1 justify-center min-w-0">
          {agents.map((agent, i) => (
            <div key={agent.id} className="flex items-center gap-1">
              <AgentNode id={agent.id} name={agent.name} status={agent.status} compact />
              {i < agents.length - 1 && (
                <div className="w-4 h-px transition-colors" style={{ background: agent.status === "Complete" ? "#22C55E35" : "#1E293B" }} />
              )}
            </div>
          ))}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-1 cursor-pointer group" title="Click to copy run_id">
            <span className="text-[9px] font-mono text-slate-700">run_id</span>
            <span className="text-[9px] font-mono text-slate-500 group-hover:text-slate-300 transition-colors">a9f2-3b7c-11ef</span>
            <Copy size={8} className="text-slate-700 group-hover:text-slate-500" />
          </div>

          <span className="text-[9px] font-mono px-2 py-0.5 rounded flex items-center gap-1"
            style={{ background: ss.bg, color: ss.color, border: `1px solid ${ss.color}35` }}>
            {ss.pulse && <span className="w-1 h-1 rounded-full inline-block animate-pulse" style={{ background: ss.color }} />}
            {sourceType}
          </span>

          <div className="flex items-center gap-1 text-[9px] font-mono text-slate-600">
            <Clock size={9} />
            <span>{lastIngested === 0 ? "just now" : `${lastIngested}s ago`}</span>
          </div>

          <div className="text-[9px] font-mono flex items-center gap-1 px-1.5 py-0.5 rounded text-green-400" style={{ background: "#22C55E0C", border: "1px solid #22C55E28" }}>
            <Server size={9} />OPENAI: connected
          </div>

          <button
            onClick={() => setShowModal(true)}
            disabled={pipelineRunning}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold text-white transition-all disabled:opacity-50"
            style={{ background: pipelineRunning ? "#1E293B" : "linear-gradient(135deg, #1D4ED8, #4F46E5)", border: "1px solid #3B82F630" }}
          >
            {pipelineRunning ? <RefreshCw size={11} className="animate-spin" /> : <Play size={11} fill="white" />}
            {pipelineRunning ? "Running…" : "Run Pipeline"}
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">

        {/* Icon Rail */}
        <div className="flex flex-col items-center gap-1 py-3 shrink-0" style={{ width: 48, background: PANEL, borderRight: `1px solid ${BORDER}` }}>
          {TABS.map((tab, i) => (
            <button key={i} onClick={() => setActiveTab(i)} title={tab.label}
              className="flex items-center justify-center w-9 h-9 rounded transition-all"
              style={{ background: activeTab === i ? "#3B82F618" : "transparent", color: activeTab === i ? "#60A5FA" : "#334155" }}>
              <tab.icon size={15} />
            </button>
          ))}
          <div className="mt-auto">
            <button title="Settings" className="flex items-center justify-center w-9 h-9 rounded transition-colors hover:text-slate-400" style={{ color: "#334155" }}>
              <Settings size={15} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tab Bar */}
          <div className="flex items-center gap-0.5 px-3 pt-2 shrink-0" style={{ background: PANEL, borderBottom: `1px solid ${BORDER}` }}>
            {TABS.map((tab, i) => (
              <button key={i} onClick={() => setActiveTab(i)}
                className="flex items-center gap-1.5 px-3 pb-2 text-[11px] font-medium transition-colors border-b-2"
                style={{ color: activeTab === i ? "#60A5FA" : "#475569", borderBottomColor: activeTab === i ? "#3B82F6" : "transparent" }}>
                <tab.icon size={11} />
                {tab.label}
              </button>
            ))}
            <div className="ml-auto pb-2 flex items-center gap-2">
              <RiskBadge level={risk} pulse={risk === "CRITICAL"} size="sm" />
            </div>
          </div>

          {/* Tab Body */}
          <div className="flex-1 overflow-hidden" style={{ background: BG }}>
            {activeTab === 0 && <TabLiveFeed onTabSwitch={setActiveTab} />}
            {activeTab === 1 && <TabRiskClassification />}
            {activeTab === 2 && <TabForecastSimulation />}
            {activeTab === 3 && <TabMitigationPlan />}
            {activeTab === 4 && <TabObservability />}
            {activeTab === 5 && <TabRAGEval />}
          </div>
        </div>
      </div>

      {/* Run Pipeline Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "#07101Acc" }}>
          <div className="rounded-xl p-5 w-[460px]" style={{ background: PANEL, border: `1px solid ${BORDER2}`, boxShadow: "0 30px 60px #00000080" }}>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-semibold text-slate-200">Run Pipeline</span>
              <button onClick={() => setShowModal(false)} className="text-slate-600 hover:text-slate-300 transition-colors"><X size={15} /></button>
            </div>

            <button onClick={() => runPipeline(null)}
              className="w-full text-left p-3 rounded-lg mb-4 transition-colors hover:border-slate-600"
              style={{ background: "#1E293B", border: "1px solid #334155" }}>
              <div className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Globe size={13} className="text-green-400" />
                Start Live Ingestion
              </div>
              <div className="text-[10px] font-mono text-slate-600 mt-1">14 Google News RSS + 6 Open-Meteo · writes to live_news_ingest / live_weather_ingest · then runs L2–L7</div>
            </button>

            <div className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider mb-2">Inject Demo Scenario</div>
            <div className="space-y-2">
              {DEMOS.map((demo) => (
                <button key={demo.id} onClick={() => runPipeline(demo)}
                  className="w-full text-left p-3 rounded-lg transition-all hover:border-slate-600"
                  style={{ background: BG, border: `1px solid ${BORDER}` }}>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-slate-300">{demo.label}</span>
                    <RiskBadge level={demo.risk} size="sm" />
                  </div>
                  <div className="text-[10px] font-mono text-slate-600 mt-1">{demo.preview}</div>
                </button>
              ))}
            </div>

            <div className="text-[9px] font-mono text-slate-700 pt-3 mt-3 leading-relaxed" style={{ borderTop: `1px solid ${BORDER}` }}>
              Setup initializes SQLite schema (10 tables), ChromaDB collections (historical_precedents, export_control_corpus, india_sourcing_corpus), and starts the L1 ingestion scheduler.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
