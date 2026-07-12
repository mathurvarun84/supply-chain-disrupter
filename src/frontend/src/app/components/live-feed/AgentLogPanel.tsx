/**
 * Screen 1 (Live Feed) — "Running Documentary" agent log stream.
 * Real data source: GET /api/live-feed/logs?run_id=. L2-L7 lines are real
 * once a run_id is active (fetch_run_logs() reads agent_execution_log);
 * without an active run_id the endpoint falls back to Day-1 fixture stub
 * lines (source="stub"), rendered dimmed with a "stub" caption so the
 * fallback stays visible to an evaluator rather than being hidden. When a
 * run_id IS active but agent_execution_log has no rows for it yet (L1's
 * pre-flight ingestion sweep, or the first second before L1 writes its
 * first row), the same stub fallback fires — a "pipeline running, waiting
 * for data…" banner distinguishes that from "no run has ever started" so
 * it doesn't read as broken.
 */
import { useRef, useEffect } from "react";
import { RefreshCw } from "lucide-react";
import { useLiveFeedLogs } from "../../hooks/useLiveFeed";
import { prefersReducedMotion } from "../../utils/animation";

export function AgentLogPanel({
  runId,
  onTabSwitch,
}: {
  runId?: string;
  onTabSwitch: (t: number) => void;
}) {
  const logRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Log lines only ever append (see header comment) — anything past the
  // previously-seen length is a genuinely new line, not a background
  // refetch echoing the same data.
  const seenCount = useRef(0);
  const { data, isLoading, isError } = useLiveFeedLogs(runId);

  const waitingForRun = Boolean(runId) && data?.lines.every((l) => l.source === "stub");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "end",
    });
  }, [data]);

  useEffect(() => {
    if (data) seenCount.current = data.lines.length;
  }, [data]);

  return (
    <div className="flex flex-col overflow-hidden rounded-lg bg-card border border-border">
      <div className="px-3 py-2.5 flex items-center gap-2 shrink-0 border-b border-border">
        <span className="text-xs font-semibold text-foreground">Running Documentary</span>
        <span className="text-[10px] font-mono text-status-complete flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-status-complete animate-pulse inline-block" />
          LIVE
        </span>
      </div>
      <div
        ref={logRef}
        className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono text-[11px] bg-background"
      >
        {isLoading && (
          <div className="text-muted-foreground px-1">Loading agent log…</div>
        )}
        {isError && (
          <div className="text-risk-critical px-1">Could not load agent log.</div>
        )}
        {waitingForRun && (
          <div className="flex items-center gap-1.5 px-1 py-1 mb-1 text-status-running">
            <RefreshCw size={10} className="animate-spin shrink-0" />
            Pipeline run {runId?.slice(0, 8)}… is running — waiting for L1-L7 data (polling every 2s).
          </div>
        )}
        {data?.lines.map((line, i) => (
          <div
            key={i}
            onClick={() => onTabSwitch(line.tab)}
            className={`flex gap-2 px-1 py-0.5 rounded cursor-pointer hover:bg-secondary/40 transition-colors ${
              line.source === "stub" ? "opacity-60" : ""
            } ${i >= seenCount.current ? "animate-slide-in-top motion-reduce:animate-none" : ""}`}
          >
            <span className="shrink-0 font-bold text-status-complete">[{line.level}]</span>
            <span className="text-muted-strong">{line.text}</span>
            {line.source === "stub" && (
              <span className="ml-auto shrink-0 text-[9px] text-muted-foreground italic">
                stub
              </span>
            )}
          </div>
        ))}
        <div className="text-border2 animate-pulse px-1 mt-1">▊</div>
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
