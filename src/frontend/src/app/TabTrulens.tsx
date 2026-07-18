/**
 * TruLens tab — triggers one demo-scenario pipeline run instrumented with
 * TruLens (src/api/routers/trulens.py -> run_with_trulens()), completely
 * separate from the live "Run Pipeline" control in the top status bar.
 * Links out to TruLens's own Streamlit dashboard (port 8502, launched
 * separately via `python -m src.evaluation.trulens_integration.cli
 * dashboard`) rather than rebuilding a view of it here.
 */
import { useState } from "react";
import { ExternalLink, Play, RefreshCw, Sparkles } from "lucide-react";
import { RiskBadge } from "./components/risk-classification/RiskBadge";
import { useTrulensMetrics, useTrulensRun, useTrulensStatus } from "./hooks/useTrulens";
import { DEMO_SCENARIOS, type DemoScenarioId } from "./types/pipeline";
import type { RiskLevel } from "./types/riskClassification";

const TRULENS_DASHBOARD_URL = "http://localhost:8502";

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-btn border border-border bg-background px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-mono text-foreground">{value}</div>
    </div>
  );
}

export function TabTrulens() {
  const [scenarioId, setScenarioId] = useState<DemoScenarioId>(DEMO_SCENARIOS[0].id);
  const [runId, setRunId] = useState<string | undefined>(undefined);

  const { mutate, isPending: isSubmitting } = useTrulensRun(setRunId);
  const { data: status } = useTrulensStatus(runId);
  const { data: metrics } = useTrulensMetrics();

  const isCapturing = isSubmitting || status?.status === "pending" || status?.status === "running";

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <section className="rounded-panel bg-card border border-border p-4">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          <Sparkles size={12} />
          TruLens Capture
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground max-w-2xl">
          Runs one demo scenario through the L1-L7 pipeline with TruLens instrumentation
          attached (per-node latency, LLM cost/tokens). This is a separate, manually-triggered
          capture — it does not affect or read from the live pipeline's Run history.
        </p>

        <div className="mt-3 flex items-center gap-2">
          <select
            value={scenarioId}
            onChange={(e) => setScenarioId(e.target.value as DemoScenarioId)}
            disabled={isCapturing}
            className="text-xs font-mono px-2 py-1.5 rounded bg-background border border-border text-muted-strong disabled:opacity-50"
          >
            {DEMO_SCENARIOS.map((demo) => (
              <option key={demo.id} value={demo.id}>
                {demo.label}
              </option>
            ))}
          </select>
          <button
            onClick={() => mutate({ demo_scenario_id: scenarioId })}
            disabled={isCapturing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-white transition-opacity disabled:opacity-50 disabled:cursor-not-allowed bg-secondary border border-primary/20"
          >
            {isCapturing ? <RefreshCw size={11} className="animate-spin" /> : <Play size={11} fill="white" />}
            {isCapturing ? "Capturing…" : "Run Capture"}
          </button>
          <a
            href={TRULENS_DASHBOARD_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-muted-strong border border-border hover:border-muted-strong transition-colors"
          >
            <ExternalLink size={11} />
            Open TruLens Dashboard
          </a>
        </div>
        <div className="mt-2 text-[10px] text-muted-foreground">
          Dashboard not loading? Launch it with{" "}
          <code className="font-mono text-muted-strong">
            python -m src.evaluation.trulens_integration.cli dashboard
          </code>
          .
        </div>
      </section>

      {status && (
        <section className="rounded-panel bg-card border border-border p-4">
          <div className="flex items-center justify-between">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Last Capture Result
            </div>
            <span className="text-[10px] font-mono text-muted-foreground">run_id {status.run_id}</span>
          </div>

          {status.status === "failed" ? (
            <div className="mt-3 text-xs text-risk-critical">{status.error ?? "Capture failed."}</div>
          ) : status.status !== "complete" ? (
            <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
              <RefreshCw size={11} className="animate-spin" />
              {status.status === "pending" ? "Queued…" : "Running pipeline with TruLens instrumentation…"}
            </div>
          ) : (
            <div className="mt-3 space-y-3">
              <div className="flex items-center gap-2">
                {status.risk_label && <RiskBadge level={status.risk_label as RiskLevel | "N/A"} size="sm" />}
                <span className="text-[10px] text-muted-foreground">{status.completed_at}</span>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatCard label="Cost (USD)" value={`$${(status.cost_summary?.cost_usd ?? 0).toFixed(4)}`} />
                <StatCard
                  label="Tokens (in/out)"
                  value={`${status.cost_summary?.prompt_tokens ?? 0} / ${status.cost_summary?.completion_tokens ?? 0}`}
                />
                <StatCard label="Latency Check" value={`${Math.round((status.node_latency_check ?? 0) * 100)}%`} />
                <StatCard label="Models" value={status.cost_summary?.models.join(", ") || "—"} />
              </div>
              {status.node_latencies_ms && Object.keys(status.node_latencies_ms).length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
                    Node Latencies
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(status.node_latencies_ms).map(([node, ms]) => (
                      <span
                        key={node}
                        className="text-[10px] font-mono rounded-pill border border-border bg-background px-2 py-1 text-muted-strong"
                      >
                        {node}: {ms.toFixed(0)}ms
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      <section className="rounded-panel bg-card border border-border p-4">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Risk Score Stability (last {metrics?.days ?? 30} days)
        </div>
        {metrics ? (
          <div className="mt-3 flex items-center gap-3">
            <div className="text-2xl font-semibold text-foreground">{metrics.risk_score_stability}</div>
            <div className="text-[11px] text-muted-foreground">
              1.0 - coefficient of variation across {metrics.n_runs} recent composite risk scores.
              Target &gt; 0.70.
            </div>
          </div>
        ) : (
          <div className="mt-3 text-xs text-muted-foreground">Loading…</div>
        )}
      </section>
    </div>
  );
}
