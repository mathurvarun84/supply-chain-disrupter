/**
 * Mutation hook for POST /api/pipeline/run. On success, hands the new
 * run_id to onStarted() (App.tsx's activeRunId state — the single shared
 * "which run is the status bar/GET endpoints pointed at" value) and
 * invalidates pipeline-status so usePipelineStatus starts polling it
 * immediately instead of waiting for its next interval tick.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { postPipelineRun } from "../api/pipeline";

export function usePipelineRun(onStarted: (runId: string) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: postPipelineRun,
    onSuccess: (data) => {
      onStarted(data.run_id);
      qc.invalidateQueries({ queryKey: ["pipeline-status"] });
    },
  });
}
