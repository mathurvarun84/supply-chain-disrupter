/**
 * Screen 4 (Mitigation Plan) tab body — fetches the real mitigation response
 * for the active pipeline run and renders the urgency banner, three ranked
 * action cards, the expandable RAG trace, and the right-side summary rail.
 */
import { AlertTriangle } from "lucide-react";
import { RiskBadge } from "./components/risk-classification/RiskBadge";
import { useMitigation } from "./hooks/useMitigation";
import { MitigationActionCard } from "./components/mitigation-plan/MitigationActionCard";
import { MitigationSidebar } from "./components/mitigation-plan/MitigationSidebar";
import { MitigationTraceDetails } from "./components/mitigation-plan/MitigationTraceDetails";
import type { MitigationUrgency } from "./types/mitigation";
import type { RiskLevel } from "./types/riskClassification";

const URGENCY_LABEL: Record<MitigationUrgency, string> = {
  LOW: "Low urgency",
  MEDIUM: "Moderate urgency",
  HIGH: "High urgency",
  IMMEDIATE: "Immediate action required",
};

const URGENCY_STYLES: Record<MitigationUrgency, string> = {
  LOW: "border-risk-low/25 bg-risk-low/10 text-risk-low",
  MEDIUM: "border-risk-medium/25 bg-risk-medium/10 text-risk-medium",
  HIGH: "border-risk-high/25 bg-risk-high/10 text-risk-high",
  IMMEDIATE: "border-risk-critical/25 bg-risk-critical/10 text-risk-critical",
};

function statusCopy(errorMessage: string | null, hasRunId: boolean) {
  if (!hasRunId) {
    return "Run the pipeline to load the mitigation plan.";
  }
  if (errorMessage) {
    return "Mitigation data is still being generated for this run.";
  }
  return "Loading mitigation plan…";
}

export function TabMitigationPlan({ runId }: { runId?: string }) {
  const { data, isLoading, isError, error } = useMitigation(runId);
  const errorMessage = error instanceof Error ? error.message : null;
  const topThree = data?.ranked_actions.slice(0, 3) ?? [];
  const riskLevel = (data?.risk_level ?? "LOW") as RiskLevel;
  const urgency = (data?.urgency ?? "LOW") as MitigationUrgency;
  const bannerClass = URGENCY_STYLES[urgency];

  if (isLoading || (!data && !isError)) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        {statusCopy(errorMessage, Boolean(runId))}
      </div>
    );
  }

  if (isError && !data) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-risk-critical">
        {statusCopy(errorMessage, Boolean(runId))}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        {statusCopy(null, Boolean(runId))}
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <section className={`rounded-panel border p-4 ${bannerClass}`}>
        <div className="flex flex-wrap items-center gap-3">
          <AlertTriangle size={16} />
          <div className="min-w-0">
            <div className="text-sm font-semibold uppercase tracking-wider">{URGENCY_LABEL[urgency]}</div>
            <div className="text-[11px] text-muted-foreground">
              Risk level is sourced from the L4 risk snapshot for this run.
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <RiskBadge level={riskLevel} pulse={riskLevel === "CRITICAL"} size="sm" />
            <span className="text-[10px] font-mono rounded-pill border border-border bg-background px-2 py-1 text-muted-strong">
              run_id {data.run_id}
            </span>
          </div>
        </div>
        {data.summary && (
          <p className="mt-3 max-w-4xl text-sm leading-6 text-foreground">{data.summary}</p>
        )}
      </section>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <main className="space-y-3">
          <section className="rounded-panel bg-card border border-border p-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Ranked Mitigation Actions</div>
            <div className="mt-3 space-y-3">
              {topThree.map((action) => (
                <MitigationActionCard key={action.rank} action={action} />
              ))}
            </div>
          </section>

          <MitigationTraceDetails trace={data.rag_query_trace} />
        </main>

        <MitigationSidebar data={data} />
      </div>
    </div>
  );
}
