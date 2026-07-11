import { Fragment, useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  PieChart, Pie, Cell, ResponsiveContainer,
} from "recharts";
import { usePipelineStatus } from "./hooks/usePipelineStatus";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

// ── Shared style tokens (matches App.mockup.tsx) ──────────────────────────
const BG     = "#070D18";
const PANEL  = "#0B1629";
const BORDER = "#1E293B";
const TOOLTIP_STYLE = {
  background: "#0B1629",
  border: "1px solid #1E293B",
  borderRadius: 6,
  fontSize: 10,
  color: "#94A3B8",
};

// ── API types ─────────────────────────────────────────────────────────────
interface CostByAgent    { agent: string; cost: number }
interface VerdictSlice   { name: string; value: number; color: string }
interface LatencyByAgent { agent: string; p50: number; p90: number }
interface PromptLogRow   {
  ts: string; agent: string; model: string;
  prompt: string; resp: string; full_prompt?: string;
  tokens: number; cost: number; latency: number;
}

// ── Custom fetch hook ─────────────────────────────────────────────────────
function useObservabilityData() {
  const [costData,    setCostData]    = useState<CostByAgent[]>([]);
  const [verdictData, setVerdictData] = useState<VerdictSlice[]>([]);
  const [latencyData, setLatencyData] = useState<LatencyByAgent[]>([]);
  const [promptLog,   setPromptLog]   = useState<PromptLogRow[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const [c, v, l, p] = await Promise.all([
          fetch(`${API_BASE_URL}/api/observability/cost`).then(r => r.json()),
          fetch(`${API_BASE_URL}/api/observability/verdicts`).then(r => r.json()),
          fetch(`${API_BASE_URL}/api/observability/latency`).then(r => r.json()),
          fetch(`${API_BASE_URL}/api/observability/prompt-log`).then(r => r.json()),
        ]);
        setCostData(c);
        setVerdictData(v);
        setLatencyData(l);
        setPromptLog(p);
      } catch (_) {
        // Silent — dashboard degrades gracefully to empty charts
      }
    };
    load();
  }, []);

  return { costData, verdictData, latencyData, promptLog };
}

// ── Component ──────────────────────────────────────────────────────────────
export function TabObservability() {
  const [sub, setSub]               = useState(0);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const { data: pipeline }                              = usePipelineStatus();
  const { costData, verdictData, latencyData, promptLog } = useObservabilityData();

  const sessionTotal = costData.reduce((s, r) => s + r.cost, 0);
  const traceUrl     = pipeline?.langfuse_trace_url ?? null;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Sub-tab bar */}
      <div
        className="flex gap-4 px-4 pt-3 shrink-0"
        style={{ borderBottom: `1px solid ${BORDER}`, background: PANEL }}
      >
        {["Observability", "Guardrails"].map((t, i) => (
          <button
            key={t}
            onClick={() => setSub(i)}
            className="text-sm pb-2.5 font-medium transition-colors border-b-2"
            style={{
              color: sub === i ? "#60A5FA" : "#475569",
              borderBottomColor: sub === i ? "#3B82F6" : "transparent",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {sub === 0 ? (
          <div className="space-y-3">
            {/* Row 1: Cost + Verdict */}
            <div className="grid grid-cols-2 gap-3">

              {/* Cost by Agent */}
              <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
                <div className="text-xs font-semibold text-slate-400 mb-0.5">Cost by Agent</div>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-[10px] font-mono text-slate-600">
                    Session total: <span className="text-slate-300">${sessionTotal.toFixed(4)}</span>
                  </span>
                  {traceUrl && (
                    <a
                      href={traceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-[9px] font-mono text-blue-400 hover:text-blue-300 transition-colors ml-auto"
                    >
                      View trace in Langfuse
                      <ExternalLink size={9} />
                    </a>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={130}>
                  <BarChart data={costData} layout="vertical">
                    <XAxis
                      type="number"
                      tick={{ fill: "#475569", fontSize: 9 }}
                      tickFormatter={(v: number) => `$${v.toFixed(4)}`}
                    />
                    <YAxis
                      type="category"
                      dataKey="agent"
                      tick={{ fill: "#94A3B8", fontSize: 10, fontFamily: "JetBrains Mono" }}
                      width={80}
                    />
                    <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={(v: number) => [`$${v.toFixed(4)}`, "Cost"]}
                    />
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
                      <Pie
                        data={verdictData}
                        cx="50%" cy="50%"
                        innerRadius={32} outerRadius={52}
                        paddingAngle={2}
                        dataKey="value"
                      >
                        {verdictData.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="space-y-2">
                    {verdictData.map((d) => (
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

            {/* P50 / P90 Latency */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">P50 / P90 Latency per Agent (s)</div>
              <ResponsiveContainer width="100%" height={140}>
                <BarChart data={latencyData} layout="vertical">
                  <XAxis type="number" tick={{ fill: "#475569", fontSize: 9 }} />
                  <YAxis
                    type="category"
                    dataKey="agent"
                    tick={{ fill: "#94A3B8", fontSize: 10, fontFamily: "JetBrains Mono" }}
                    width={100}
                  />
                  <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" horizontal={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="p50" fill="#3B82F6" name="P50" radius={[0, 2, 2, 0]} barSize={5} />
                  <Bar dataKey="p90" fill="#8B5CF6" name="P90" radius={[0, 2, 2, 0]} barSize={5} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Prompt / Response Inspector */}
            <div className="rounded-lg p-4" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
              <div className="text-xs font-semibold text-slate-400 mb-3">
                Prompt / Response Inspector — llm_call_log
              </div>
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
                    {promptLog.map((row, i) => (
                      <Fragment key={i}>
                        <tr
                          className="cursor-pointer hover:bg-slate-800/25 transition-colors"
                          style={{ borderBottom: `1px solid ${BORDER}` }}
                          onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                        >
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
                            <td colSpan={7} className="px-4 py-3" style={{ background: BG }}>
                              <div className="text-[10px] text-slate-500">
                                <span className="text-slate-600 block mb-1">Full Prompt:</span>
                                <pre className="whitespace-pre-wrap break-words text-slate-400 max-h-64 overflow-y-auto leading-relaxed">
                                  {row.full_prompt ?? row.prompt}
                                </pre>
                              </div>
                              <div className="text-[10px] text-slate-500 mt-3">
                                <span className="text-slate-600 block mb-1">Full Response:</span>
                                <pre className="whitespace-pre-wrap break-words text-slate-400 max-h-64 overflow-y-auto leading-relaxed">
                                  {row.resp}
                                </pre>
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    ))}
                    {promptLog.length === 0 && (
                      <tr>
                        <td colSpan={7} className="py-6 text-center text-slate-600 text-[10px]">
                          No LLM calls recorded yet — run the pipeline to populate this table.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <GuardrailsSubTab />
        )}
      </div>
    </div>
  );
}

// ── Guardrails sub-tab (reads from /api/guardrails/events) ────────────────
interface GuardrailRow {
  name: string; dir: "input" | "output"; agent: string;
  pass_count: number; fail_count: number; last_reason: string;
}

const RISK_COLORS = { CRITICAL: "#EF4444", HIGH: "#F97316" };

function GuardrailsSubTab() {
  const [rows, setRows] = useState<GuardrailRow[]>([]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/guardrails/events`)
      .then(r => r.json())
      .then(setRows)
      .catch(() => {});
  }, []);

  const suppressed = rows.reduce((s, r) => s + (r.fail_count || 0), 0);

  return (
    <div className="space-y-3">
      {/* Headline */}
      <div
        className="rounded-lg p-4 flex items-center gap-5"
        style={{ background: "#EF444410", border: `1px solid ${RISK_COLORS.CRITICAL}30` }}
      >
        <div>
          <div className="text-[10px] text-slate-500 mb-0.5 uppercase tracking-wider">
            Slack Alerts Suppressed by Guardrail
          </div>
          <div className="text-4xl font-mono font-bold text-red-400">{suppressed}</div>
        </div>
        <div className="ml-auto text-[10px] text-slate-600 font-mono max-w-[200px] leading-relaxed">
          faithfulness_gate failures this session — routed to human review
        </div>
      </div>

      {/* Guardrail table */}
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
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className="hover:bg-slate-800/20 transition-colors"
                  style={{ borderBottom: `1px solid ${BORDER}` }}
                >
                  <td className="py-1.5 px-2 text-blue-400">{row.name}</td>
                  <td className="py-1.5 px-2">
                    <span
                      className="px-1.5 py-0.5 rounded"
                      style={{
                        background: row.dir === "input" ? "#3B82F614" : "#8B5CF614",
                        color: row.dir === "input" ? "#60A5FA" : "#A78BFA",
                      }}
                    >
                      {row.dir}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-slate-500">{row.agent}</td>
                  <td className="py-1.5 px-2 text-green-400">{row.pass_count}</td>
                  <td
                    className="py-1.5 px-2"
                    style={{ color: row.fail_count > 0 ? RISK_COLORS.HIGH : "#475569" }}
                  >
                    {row.fail_count}
                  </td>
                  <td className="py-1.5 px-2 text-slate-600 max-w-[220px] truncate">{row.last_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Guardrail map */}
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
                <div
                  className="text-[9px] font-mono font-bold px-2 py-1 rounded"
                  style={{ background: "#22C55E18", color: "#22C55E", border: "1px solid #22C55E35" }}
                >
                  {node.id}
                </div>
                <div className="space-y-0.5">
                  {node.guards.map((g) => (
                    <div
                      key={g}
                      className="text-[8px] font-mono px-1.5 py-0.5 rounded whitespace-nowrap"
                      style={{ background: BG, color: "#64748B", border: `1px solid ${BORDER}` }}
                    >
                      {g}
                    </div>
                  ))}
                </div>
              </div>
              {i < arr.length - 1 && (
                <div className="mt-3 w-3 h-px shrink-0" style={{ background: BORDER }} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
