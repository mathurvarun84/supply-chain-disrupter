/**
 * Run Pipeline modal — opened from the top status bar's "Run Pipeline"
 * button (App.tsx). Markup/structure follows _reference/App.mockup.tsx's
 * Run Pipeline Modal (lines ~1232-1271) — Start Live Ingestion button +
 * Inject Demo Scenario card list — extended with a Replay section (the
 * mockup predates the mode/replay_run_id split added by this task).
 *
 * Writes: calls usePipelineRun() on submit, which POSTs /api/pipeline/run
 * and hands the resulting run_id to App.tsx's activeRunId state via
 * onRunStarted — the same shared value TabLiveFeed/etc. would read from if
 * they were wired to a specific run_id (Day 9 does not extend that wiring
 * beyond the status bar — see ARCHITECTURE.md scope notes).
 */
import { useState } from "react";
import { Globe, X } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { RiskBadge } from "../risk-classification/RiskBadge";
import { usePipelineRun } from "../../hooks/usePipelineRun";
import { fetchRecentRuns, type RecentRun } from "../../api/pipeline";
import { DEMO_SCENARIOS } from "../../types/pipeline";

export function DemoScenarioInjector({
  onClose,
  onRunStarted,
}: {
  onClose: () => void;
  onRunStarted: (runId: string) => void;
}) {
  const [replayRunId, setReplayRunId] = useState<string>("");

  const { data: recentRuns } = useQuery<RecentRun[]>({
    queryKey: ["pipeline-recent-runs"],
    queryFn: fetchRecentRuns,
  });

  const { mutate, isPending } = usePipelineRun((runId) => {
    onRunStarted(runId);
    onClose();
  });

  const runLive = () => mutate({ mode: "live" });

  const runDemo = (scenarioId: (typeof DEMO_SCENARIOS)[number]["id"]) => {
    mutate({ mode: "demo", demo_scenario_id: scenarioId });
  };

  const runReplay = () => {
    if (!replayRunId) return;
    mutate({ mode: "replay", replay_run_id: replayRunId });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70"
      onClick={onClose}
    >
      <div
        className="rounded-xl p-5 w-[460px] bg-panel border border-border shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm font-semibold text-foreground">Run Pipeline</span>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
            <X size={15} />
          </button>
        </div>

        <button
          onClick={runLive}
          disabled={isPending}
          className="w-full text-left p-3 rounded-lg mb-4 transition-colors hover:border-muted-strong bg-secondary border border-border disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Globe size={13} className="text-risk-low" />
            Start Live Ingestion
          </div>
          <div className="text-[10px] font-mono text-muted-foreground mt-1">
            14 Google News RSS + 6 Open-Meteo · writes to live_news_ingest / live_weather_ingest · then runs L2-L7
          </div>
        </button>

        <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Inject Demo Scenario
        </div>
        <div className="space-y-2 mb-4">
          {DEMO_SCENARIOS.map((demo) => (
            <button
              key={demo.id}
              onClick={() => runDemo(demo.id)}
              disabled={isPending}
              className="w-full text-left p-3 rounded-lg transition-all hover:border-muted-strong bg-background border border-border disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-muted-strong">{demo.label}</span>
                <RiskBadge level={demo.severity} size="sm" />
              </div>
            </button>
          ))}
        </div>

        <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Replay a Past Run
        </div>
        <div className="flex items-center gap-2">
          <select
            value={replayRunId}
            onChange={(e) => setReplayRunId(e.target.value)}
            className="flex-1 text-xs font-mono px-2 py-1.5 rounded bg-background border border-border text-muted-strong"
          >
            <option value="">Select a completed run…</option>
            {(recentRuns ?? []).map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
          <button
            onClick={runReplay}
            disabled={isPending || !replayRunId}
            className="px-3 py-1.5 rounded-btn text-xs font-semibold text-white bg-secondary border border-primary/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Replay
          </button>
        </div>
      </div>
    </div>
  );
}
