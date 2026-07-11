/**
 * React Query hook for Screen 2 (Risk Classification). Fetch-once per
 * run_id (staleTime: Infinity) — unlike Live Feed's news/weather, a
 * classification result doesn't change mid-run, so this never polls.
 * Defaults to GET /api/risk-classification/latest until a demo-scenario/
 * record picker exists (Day 9) to supply an explicit run_id.
 */

import { useQuery } from "@tanstack/react-query";
import { fetchLatestRiskClassification, fetchRiskClassification } from "../api/riskClassification";

export function useRiskClassification(runId?: string) {
  return useQuery({
    queryKey: ["risk-classification", runId ?? "latest"],
    queryFn: () =>
      runId ? fetchRiskClassification(runId) : fetchLatestRiskClassification(),
    staleTime: Infinity,
  });
}
