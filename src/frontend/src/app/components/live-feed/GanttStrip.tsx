/**
 * Screen 1 (Live Feed) — Pipeline Timeline Gantt strip, below the 3-column grid.
 * Real data source: GET /api/live-feed/gantt. Only the L1 bar in this
 * component's data is real (from ingestion_run_log durations); the L2-L7
 * bars are the unmodified Day-1 fixture data (source="stub") and will
 * become real on Day 9 — do not edit the L2-L7 fixture values from this
 * component. Stub bars render dimmed so the scope cut stays visible.
 */
import { useLiveFeedGantt } from "../../hooks/useLiveFeed";

export function GanttStrip() {
  const { data, isLoading, isError } = useLiveFeedGantt();
  const bars = data?.bars ?? [];
  const totalDuration = bars.length > 0 ? Math.max(...bars.map((b) => b.start + b.dur)) : 0;

  return (
    <div className="px-3 pb-3 shrink-0">
      <div className="rounded-lg p-3 bg-card border border-border">
        <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Pipeline Timeline — end-to-end: {totalDuration.toFixed(1)}s
        </div>
        {isLoading && (
          <div className="text-xs text-muted-foreground">Loading timeline…</div>
        )}
        {isError && (
          <div className="text-xs text-risk-critical">Could not load Gantt data.</div>
        )}
        <div className="space-y-1">
          {bars.map((row) => (
            <div
              key={row.id}
              className={`flex items-center gap-2 transition-opacity duration-300 motion-reduce:transition-none ${row.source === "stub" ? "opacity-60" : ""}`}
            >
              <span className="text-[9px] font-mono text-muted-foreground w-4 shrink-0">
                {row.id}
              </span>
              <div className="flex-1 relative h-3.5 rounded overflow-hidden bg-background">
                <div
                  className="absolute top-0 h-full rounded flex items-center px-1 text-[9px] font-mono transition-all duration-300 ease-out motion-reduce:transition-none"
                  style={{
                    left: totalDuration > 0 ? `${(row.start / totalDuration) * 100}%` : "0%",
                    width: totalDuration > 0 ? `${(row.dur / totalDuration) * 100}%` : "0%",
                    background: row.color + "28",
                    border: `1px solid ${row.color}44`,
                    color: row.color,
                    minWidth: 28,
                  }}
                >
                  {row.dur.toFixed(1)}s
                </div>
              </div>
              {row.source === "stub" && (
                <span className="text-[9px] text-muted-foreground italic shrink-0">stub</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
