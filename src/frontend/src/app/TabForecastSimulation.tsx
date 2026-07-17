/**
 * Screen 3 (Forecast & Simulation) tab body — Prophet dual-area chart +
 * category selector (left) and Monte Carlo histogram + P10/P50/P90 tiles
 * (right), matching the layout from _reference/App.mockup.tsx's
 * TabForecastSimulation. Wires to real L5/L6 output via
 * GET /api/forecast/{run_id}?category= and GET /api/simulation/{run_id}
 * (see src/api/routers/forecast.py, simulation.py; src/agents/pipeline_bridge.py
 * writes the snapshot rows these read).
 *
 * "category" here is the real winning ops_kpi SKU_id for this run, not one
 * of a fixed Laptops/Phones/Headphones/Speakers set — ops_kpi has no
 * product-category dimension (see pipeline_bridge.py's module docstring).
 * The selector still renders whatever `available categories` the backend
 * actually returns for this run_id, so it never shows a category with no
 * real data behind it.
 */
import { useEffect, useState } from "react";
import { ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from "recharts";
import { API_BASE_URL } from "./api/config";

const BG = "#070D18";
const PANEL = "#0B1629";
const BORDER = "#1E293B";
const TOOLTIP_STYLE = {
  background: "#0B1629",
  border: "1px solid #1E293B",
  borderRadius: 6,
  fontSize: 10,
  color: "#94A3B8",
};

interface ForecastPoint { day: string; baseline: number; adjusted: number }
interface ForecastResponse {
  run_id: string;
  category: string;
  categories: string[];
  series: ForecastPoint[];
}

interface SimulationBucket { range: string; count: number }
interface SimulationResponse {
  run_id: string;
  p10: number;
  p50: number;
  p90: number;
  revenue_at_risk_usd: number;
  alternate_route: string;
  histogram: SimulationBucket[];
  revenue_at_risk_p10_usd: number | null;
  revenue_at_risk_p90_usd: number | null;
  days_to_stockout_p10: number | null;
  days_to_stockout_p50: number | null;
  days_to_stockout_p90: number | null;
}

// GET /api/forecast/sku/{sku_id} — the existing full L5 DemandForecastingAgent
// v3 response (model selection, MAPE, regressors). Reused here rather than
// re-plumbing this detail into forecast_output, which only stores the
// run-level baseline/adjusted series (see module docstring).
interface SkuForecastDetail {
  sku_id: string;
  model_selected: string;
  regressors_used: string[];
  regressor_selection_method: string;
  stockout_prob: number | null;
  mape_prophet_trend_only: number | null;
  mape_prophet_selected: number | null;
  mape_dataset_baseline_avg: number | null;
  mape_dataset_ai_avg: number | null;
  mape_improvement_pct_vs_dataset_baseline: number | null;
}

function useForecast(runId: string | undefined, category: string | undefined) {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "missing" | "error">("loading");

  useEffect(() => {
    if (!runId) {
      setStatus("missing");
      return;
    }
    setStatus("loading");
    const qs = category ? `?category=${encodeURIComponent(category)}` : "";
    fetch(`${API_BASE_URL}/api/forecast/${runId}${qs}`)
      .then((r) => {
        if (r.status === 404) { setStatus("missing"); return null; }
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((json) => {
        if (json) { setData(json); setStatus("ready"); }
      })
      .catch(() => setStatus("error"));
  }, [runId, category]);

  return { data, status };
}

function useSkuForecastDetail(skuId: string | undefined) {
  const [data, setData] = useState<SkuForecastDetail | null>(null);

  useEffect(() => {
    setData(null);
    // "Skipped-Optional" is the fallback category persist_forecast_output()
    // writes when L5 didn't run — there's no real SKU_id to look up detail
    // for in that case.
    if (!skuId || skuId === "Skipped-Optional") return;
    fetch(`${API_BASE_URL}/api/forecast/sku/${skuId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((json) => { if (json) setData(json); })
      .catch(() => {});
  }, [skuId]);

  return data;
}

function useSimulation(runId: string | undefined) {
  const [data, setData] = useState<SimulationResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "missing" | "error">("loading");

  useEffect(() => {
    if (!runId) {
      setStatus("missing");
      return;
    }
    setStatus("loading");
    fetch(`${API_BASE_URL}/api/simulation/${runId}`)
      .then((r) => {
        if (r.status === 404) { setStatus("missing"); return null; }
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((json) => {
        if (json) { setData(json); setStatus("ready"); }
      })
      .catch(() => setStatus("error"));
  }, [runId]);

  return { data, status };
}

function expectedDropPct(series: ForecastPoint[]): number {
  const totalBaseline = series.reduce((s, p) => s + p.baseline, 0);
  const totalAdjusted = series.reduce((s, p) => s + p.adjusted, 0);
  if (totalBaseline <= 0) return 0;
  return Math.round(((totalBaseline - totalAdjusted) / totalBaseline) * 100);
}

function EmptyPanel({ title, badge, message }: { title: string; badge: string; message: string }) {
  return (
    <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-slate-200">{title}</span>
        <span
          className="text-[9px] px-2 py-0.5 rounded font-mono"
          style={{ background: "#818CF818", color: "#818CF8", border: "1px solid #818CF830" }}
        >
          {badge}
        </span>
      </div>
      <div className="flex-1 flex items-center justify-center text-[11px] text-slate-600 text-center px-6">
        {message}
      </div>
    </div>
  );
}

export function TabForecastSimulation({ runId }: { runId?: string }) {
  const [category, setCategory] = useState<string | undefined>(undefined);
  const { data: forecast, status: forecastStatus } = useForecast(runId, category);
  const skuDetail = useSkuForecastDetail(forecast?.category);
  const { data: simulation, status: simulationStatus } = useSimulation(runId);

  // The backend decides the actual selected category (defaults to the
  // first real one it has); adopt it once loaded so the selector reflects
  // what's really being shown rather than an unconfirmed guess.
  useEffect(() => {
    if (forecast && category === undefined) setCategory(forecast.category);
  }, [forecast, category]);

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="grid grid-cols-2 gap-3 h-full">
        {/* Prophet */}
        {forecastStatus === "ready" && forecast ? (
          <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-semibold text-slate-200">Demand Forecasting — Prophet</span>
              <span
                className="text-[9px] px-2 py-0.5 rounded font-mono"
                style={{ background: "#818CF818", color: "#818CF8", border: "1px solid #818CF830" }}
              >
                Optional · L5
              </span>
            </div>
            <div className="flex gap-1.5 mb-3 flex-wrap">
              {forecast.categories.map((c) => (
                <button
                  key={c}
                  onClick={() => setCategory(c)}
                  className="text-[10px] px-2 py-0.5 rounded font-mono transition-all"
                  style={{
                    background: category === c ? "#3B82F620" : BG,
                    color: category === c ? "#60A5FA" : "#475569",
                    border: `1px solid ${category === c ? "#3B82F640" : BORDER}`,
                  }}
                >
                  {c}
                </button>
              ))}
            </div>
            <div className="flex items-baseline gap-2 mb-3">
              <span className="text-3xl font-mono font-bold text-red-400">
                {expectedDropPct(forecast.series)}%
              </span>
              <span className="text-slate-500 text-sm">expected demand drop</span>
            </div>

            {skuDetail && (
              <div className="grid grid-cols-4 gap-2 mb-3">
                <div className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                  <div className="text-xs font-mono font-bold text-slate-200 capitalize">{skuDetail.model_selected}</div>
                  <div className="text-[9px] text-slate-600 mt-0.5">Model Selected</div>
                </div>
                {skuDetail.stockout_prob !== null && (
                  <div className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-xs font-mono font-bold text-slate-200">{(skuDetail.stockout_prob * 100).toFixed(1)}%</div>
                    <div className="text-[9px] text-slate-600 mt-0.5">Stockout Prob (L5)</div>
                  </div>
                )}
                {skuDetail.mape_prophet_selected !== null && (
                  <div className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-xs font-mono font-bold text-slate-200">{skuDetail.mape_prophet_selected.toFixed(1)}%</div>
                    <div className="text-[9px] text-slate-600 mt-0.5">MAPE (selected)</div>
                  </div>
                )}
                {skuDetail.mape_improvement_pct_vs_dataset_baseline !== null && (
                  <div className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-xs font-mono font-bold text-emerald-400">
                      {skuDetail.mape_improvement_pct_vs_dataset_baseline.toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-slate-600 mt-0.5">MAPE Improvement</div>
                  </div>
                )}
              </div>
            )}

            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={forecast.series}>
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

            {skuDetail && (
              <div className="flex gap-1.5 mt-3 flex-wrap">
                {skuDetail.regressors_used.length > 0 ? (
                  skuDetail.regressors_used.map((r) => (
                    <span
                      key={r}
                      className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                      style={{ background: "#3B82F610", color: "#60A5FA", border: "1px solid #3B82F625" }}
                    >
                      {r}
                    </span>
                  ))
                ) : (
                  <span className="text-[10px] px-1.5 py-0.5 rounded font-mono text-slate-600" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    trend-only ({skuDetail.regressor_selection_method})
                  </span>
                )}
              </div>
            )}

            <div className="mt-1 text-[9px] font-mono text-slate-700">
              {forecast.category === "Skipped-Optional"
                ? "L5 skipped for this run (insufficient ops_kpi history) — fallback series shown"
                : `SKU ${forecast.category} · Ops KPI 2023–2025`}
            </div>
          </div>
        ) : forecastStatus === "missing" ? (
          <EmptyPanel
            title="Demand Forecasting — Prophet"
            badge="Optional · L5"
            message="No forecast snapshot for this run yet. Run the pipeline first."
          />
        ) : forecastStatus === "error" ? (
          <EmptyPanel title="Demand Forecasting — Prophet" badge="Optional · L5" message="Could not load forecast data." />
        ) : (
          <EmptyPanel title="Demand Forecasting — Prophet" badge="Optional · L5" message="Loading…" />
        )}

        {/* Monte Carlo */}
        {simulationStatus === "ready" && simulation ? (
          <div className="rounded-lg p-4 flex flex-col" style={{ background: PANEL, border: `1px solid ${BORDER}` }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-semibold text-slate-200">Monte Carlo Simulation</span>
              <span
                className="text-[9px] px-2 py-0.5 rounded font-mono"
                style={{ background: "#818CF818", color: "#818CF8", border: "1px solid #818CF830" }}
              >
                Optional · L6
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[
                { label: "P10 Stockout", v: `${simulation.p10}%` },
                { label: "P50 Stockout", v: `${simulation.p50}%` },
                { label: "P90 Stockout", v: `${simulation.p90}%` },
              ].map((m) => (
                <div key={m.label} className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                  <div className="text-xl font-mono font-bold text-slate-100">{m.v}</div>
                  <div className="text-[9px] text-slate-600 mt-0.5">{m.label}</div>
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={simulation.histogram}>
                <XAxis dataKey="range" tick={{ fill: "#475569", fontSize: 9 }} />
                <YAxis tick={{ fill: "#475569", fontSize: 9 }} />
                <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="#F97316" radius={[3, 3, 0, 0]} opacity={0.85} name="Runs" />
                <ReferenceLine
                  x={simulation.histogram.reduce((closest, b) => {
                    const mid = parseInt(b.range, 10) + 5;
                    const closestMid = parseInt(closest.range, 10) + 5;
                    return Math.abs(mid - simulation.p50) < Math.abs(closestMid - simulation.p50) ? b : closest;
                  }, simulation.histogram[0])?.range}
                  stroke="#EF4444"
                  strokeDasharray="4 2"
                  label={{ value: "P50", position: "insideTop", fill: "#EF4444", fontSize: 9 }}
                />
              </BarChart>
            </ResponsiveContainer>
            {simulation.revenue_at_risk_p10_usd !== null && simulation.revenue_at_risk_p90_usd !== null ? (
              <div className="grid grid-cols-3 gap-2 mt-3">
                {[
                  { label: "Revenue at Risk P10", v: simulation.revenue_at_risk_p10_usd },
                  { label: "Revenue at Risk P50", v: simulation.revenue_at_risk_usd },
                  { label: "Revenue at Risk P90", v: simulation.revenue_at_risk_p90_usd },
                ].map((m) => (
                  <div key={m.label} className="rounded p-3 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                    <div className="text-lg font-mono font-bold text-orange-400">${(m.v / 1_000_000).toFixed(1)}M</div>
                    <div className="text-[9px] text-slate-600 mt-0.5">{m.label}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded p-3 mt-3" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                <div className="text-[10px] text-slate-600 mb-0.5">Revenue at Risk (P50)</div>
                <div className="text-xl font-mono font-bold text-orange-400">
                  ${(simulation.revenue_at_risk_usd / 1_000_000).toFixed(1)}M
                </div>
              </div>
            )}

            {simulation.days_to_stockout_p50 !== null && (
              <div className="grid grid-cols-3 gap-2 mt-2">
                {[
                  { label: "Days to Stockout P10", v: simulation.days_to_stockout_p10 },
                  { label: "Days to Stockout P50", v: simulation.days_to_stockout_p50 },
                  { label: "Days to Stockout P90", v: simulation.days_to_stockout_p90 },
                ]
                  .filter((m) => m.v !== null)
                  .map((m) => (
                    <div key={m.label} className="rounded p-2 text-center" style={{ background: BG, border: `1px solid ${BORDER}` }}>
                      <div className="text-lg font-mono font-bold text-slate-200">{m.v!.toFixed(0)}</div>
                      <div className="text-[9px] text-slate-600 mt-0.5">{m.label}</div>
                    </div>
                  ))}
              </div>
            )}

            <div className="rounded p-3 mt-2" style={{ background: BG, border: `1px solid ${BORDER}` }}>
              <div className="text-[10px] text-slate-600 mb-0.5">Alternate Route</div>
              <div className="text-xs font-mono text-blue-400">{simulation.alternate_route}</div>
              <div className="text-[9px] font-mono text-slate-700">config, not LLM</div>
            </div>
          </div>
        ) : simulationStatus === "missing" ? (
          <EmptyPanel
            title="Monte Carlo Simulation"
            badge="Optional · L6"
            message="No simulation snapshot for this run yet. Run the pipeline first."
          />
        ) : simulationStatus === "error" ? (
          <EmptyPanel title="Monte Carlo Simulation" badge="Optional · L6" message="Could not load simulation data." />
        ) : (
          <EmptyPanel title="Monte Carlo Simulation" badge="Optional · L6" message="Loading…" />
        )}
      </div>
    </div>
  );
}
