/**
 * Polls GET /api/pipeline/status for one run_id (or the most recently
 * active run when runId is undefined — the app's initial-load default).
 * Polls every 2s while the run is still in progress, and stops once
 * is_complete flips true so a finished run doesn't keep hitting the API.
 */
import { useQuery } from "@tanstack/react-query";
import { fetchPipelineStatus } from "../api/pipeline";

export type { PipelineStatus, AgentState } from "../types/pipeline";

const STATUS_POLL_INTERVAL_MS = 2_000;

export function usePipelineStatus(runId?: string) {
  return useQuery({
    queryKey: ["pipeline-status", runId ?? "latest"],
    queryFn: () => fetchPipelineStatus(runId),
    refetchInterval: (query) => (query.state.data?.is_complete ? false : STATUS_POLL_INTERVAL_MS),
  });
}
