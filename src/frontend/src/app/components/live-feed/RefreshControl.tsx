/**
 * Screen 1 (Live Feed) — manual ingestion trigger.
 * New in this task (not part of the original mockup). Calls
 * POST /api/live-feed/refresh, which runs DataIngestionAgent.run_batch() as
 * a backend BackgroundTask — this is the only user action on Screen 1 that
 * causes outbound network calls. Styled to match the existing "Run Pipeline"
 * button in the top status bar (App.tsx) — same classNames, no new tokens.
 */
import { useState, useEffect } from "react";
import { RefreshCw } from "lucide-react";
import { useTriggerRefresh, useIngestStatus } from "../../hooks/useLiveFeed";

export function RefreshControl() {
  const { mutate, isPending } = useTriggerRefresh();
  const [pollFast, setPollFast] = useState(false);
  const { data: status } = useIngestStatus(isPending || pollFast);

  // Keep fast-polling ingest-status until the backend itself reports the
  // lock is free, so the button doesn't flip back on prematurely between
  // the POST returning and the background task actually starting.
  useEffect(() => {
    if (status?.is_running) {
      setPollFast(true);
    } else if (!isPending) {
      setPollFast(false);
    }
  }, [status?.is_running, isPending]);

  const running = isPending || Boolean(status?.is_running);

  return (
    <div className="flex items-center justify-between px-3 pt-3 pb-1 shrink-0">
      <button
        onClick={() => mutate()}
        disabled={running}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-white bg-secondary border border-primary/20 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
      >
        <RefreshCw size={11} className={running ? "animate-spin" : ""} />
        {running ? "Fetching live signals…" : "Refresh Live Data"}
      </button>
      {status?.last_run && (
        <span className="text-[10px] font-mono text-muted-foreground">
          Last run: {status.last_run.run_ts_utc} · {status.last_run.rows_inserted} inserted
        </span>
      )}
    </div>
  );
}
