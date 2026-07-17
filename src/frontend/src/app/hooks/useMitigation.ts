/**
 * React Query hook for Screen 4 Mitigation Plan. Polls until the run's real
 * mitigation row exists, then stays stable because mitigation output is
 * immutable for a completed run.
 */
import { useQuery } from "@tanstack/react-query";
import { fetchMitigation } from "../api/mitigation";

export function useMitigation(runId?: string) {
  return useQuery({
    queryKey: ["mitigation", runId],
    queryFn: () => fetchMitigation(runId ?? ""),
    enabled: Boolean(runId),
    retry: false,
    staleTime: Infinity,
    refetchInterval: (query) => (query.state.data ? false : 2000),
  });
}
