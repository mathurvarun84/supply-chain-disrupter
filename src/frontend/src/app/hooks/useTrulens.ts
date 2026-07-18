/**
 * Hooks for the TruLens tab: a mutation to start a capture run and a
 * polling query for its status (mirrors usePipelineRun/usePipelineStatus'
 * split), plus a query for the historical risk_score_stability metric.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchTrulensMetrics, fetchTrulensStatus, postTrulensRun } from "../api/trulens";

export function useTrulensRun(onStarted: (runId: string) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: postTrulensRun,
    onSuccess: (data) => {
      onStarted(data.run_id);
      qc.invalidateQueries({ queryKey: ["trulens-status", data.run_id] });
    },
  });
}

export function useTrulensStatus(runId?: string) {
  return useQuery({
    queryKey: ["trulens-status", runId],
    queryFn: () => fetchTrulensStatus(runId ?? ""),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "complete" || status === "failed" ? false : 1500;
    },
  });
}

export function useTrulensMetrics() {
  return useQuery({
    queryKey: ["trulens-metrics"],
    queryFn: fetchTrulensMetrics,
  });
}
