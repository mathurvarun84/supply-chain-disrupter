/**
 * Screen 4 mitigation sidebar sections for India sourcing, Slack preview,
 * and cost delta. It renders only real values from the API and shows an
 * explicit empty state when the backend has no persisted content yet.
 */
import type { MitigationResponse } from "../../types/mitigation";

function EmptyValue({ label }: { label: string }) {
  return <div className="text-xs text-muted-foreground">No persisted {label.toLowerCase()} for this run.</div>;
}

export function MitigationSidebar({ data }: { data: MitigationResponse }) {
  return (
    <aside className="space-y-3">
      <section className="rounded-panel p-4 bg-card border border-border">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">India Sourcing Recommendations</div>
        <div className="mt-3 space-y-2">
          {data.india_sourcing_recommendations.length > 0 ? (
            data.india_sourcing_recommendations.map((item) => (
              <div key={item} className="rounded-btn border border-border bg-background px-3 py-2 text-xs text-foreground">
                {item}
              </div>
            ))
          ) : (
            <EmptyValue label="India sourcing recommendation" />
          )}
        </div>
      </section>

      <section className="rounded-panel p-4 bg-card border border-border">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Slack Message Preview</div>
        <div className="mt-3 rounded-btn border border-border bg-background px-3 py-2 text-xs leading-5 text-foreground whitespace-pre-wrap font-mono">
          {data.slack_alert_fired && data.slack_preview ? (
            data.slack_preview
          ) : (
            <span className="font-sans text-muted-foreground">No alert fired for this run.</span>
          )}
        </div>
      </section>

      <section className="rounded-panel p-4 bg-card border border-border">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Cost Delta</div>
        <div className="mt-3 flex items-end justify-between gap-3 rounded-btn border border-border bg-background px-3 py-2">
          <div className="text-lg font-semibold text-foreground">
            {data.cost_delta_usd != null ? `$${data.cost_delta_usd.toLocaleString()}` : "—"}
          </div>
          <div className="text-[10px] text-muted-foreground">{data.cost_delta ?? "No persisted cost delta for this run."}</div>
        </div>
      </section>
    </aside>
  );
}
