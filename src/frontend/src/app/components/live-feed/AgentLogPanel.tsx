/**
 * Screen 1 (Live Feed) — "Running Documentary" agent log stream.
 * Real data source: GET /api/live-feed/logs. Only the L1 line in this
 * component's data is real (from ingestion_run_log); the L2-L7 lines are
 * the unmodified Day-1 fixture data (source="stub") and will become real on
 * Day 9 — do not edit the L2-L7 fixture values from this component. Stub
 * lines render dimmed with a "stub" caption so the scope cut stays visible
 * to an evaluator rather than being hidden.
 */
import { useRef, useEffect } from "react";
import { useLiveFeedLogs } from "../../hooks/useLiveFeed";
import { prefersReducedMotion } from "../../utils/animation";

export function AgentLogPanel({ onTabSwitch }: { onTabSwitch: (t: number) => void }) {
  const logRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Log lines only ever append (see header comment) — anything past the
  // previously-seen length is a genuinely new line, not a background
  // refetch echoing the same data.
  const seenCount = useRef(0);
  const { data, isLoading, isError } = useLiveFeedLogs();

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
