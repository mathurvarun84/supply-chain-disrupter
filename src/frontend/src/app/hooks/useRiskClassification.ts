/**
 * React Query hook for Screen 2 (Risk Classification). Follows the active
 * pipeline run_id, same as useForecast/useSimulation/useMitigation — GET
 * /api/risk-classification/{run_id} now reads the risk_classification_output
 * snapshot written as soon as L4 finishes inside run_pipeline() (see
 * src/api/routers/risk.py), not "whatever order was last classified
 * anywhere." Polls every 3s while the pipeline is still running (no run_id
 * yet, or a 404 meaning L4 hasn't finished) so the tab lights up on its own
 * instead of requiring a manual refresh; stops polling once data lands.
 * Falls back to GET /api/risk-classification/latest only when no run has
 * ever been started (runId undefined and nothing cached yet).
 */

import { useQuery } from "@tanstack/react-query";
import { fetchLatestRiskClassification, fetchRiskClassification } from "../api/riskClassification";

export function useRiskClassification(runId?: string) {
  return useQuery({
    queryKey: ["risk-classification", runId ?? "latest"],
    queryFn: () =>
      runId ? fetchRiskClassification(runId) : fetchLatestRiskClassification(),
    staleTime: Infinity,
    retry: false,
    refetchInterval: (query) => (runId && !query.state.data ? 2000 : false),
  });
}
