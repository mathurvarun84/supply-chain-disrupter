/**
 * React Query hooks for Screen 1 (Live Feed).
 * News/weather/logs/gantt poll every 15s; ingest-status polls at 2s while a
 * manual refresh is in flight (else 30s) so RefreshControl can flip the
 * button back on promptly once DataIngestionAgent.run_batch() completes.
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchLiveFeedNews,
  fetchLiveFeedWeather,
  fetchIngestStatus,
  fetchLiveFeedLogs,
  fetchLiveFeedGantt,
  triggerLiveFeedRefresh,
} from "../api/liveFeed";

export function useLiveFeedNews() {
  return useQuery({
    queryKey: ["live-feed", "news"],
    queryFn: fetchLiveFeedNews,
    refetchInterval: 15_000,
  });
}

export function useLiveFeedWeather() {
  return useQuery({
    queryKey: ["live-feed", "weather"],
    queryFn: fetchLiveFeedWeather,
    refetchInterval: 15_000,
  });
}

export function useIngestStatus(pollFast: boolean) {
  return useQuery({
    queryKey: ["live-feed", "ingest-status"],
    queryFn: fetchIngestStatus,
    refetchInterval: pollFast ? 2_000 : 30_000,
  });
}

export function useLiveFeedLogs(runId?: string) {
  return useQuery({
    queryKey: ["live-feed", "logs", runId],
    queryFn: () => fetchLiveFeedLogs(runId),
    refetchInterval: 15_000,
  });
}

export function useLiveFeedGantt(runId?: string) {
  return useQuery({
    queryKey: ["live-feed", "gantt", runId],
    queryFn: () => fetchLiveFeedGantt(runId),
    refetchInterval: 15_000,
  });
}

export function useTriggerRefresh() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: triggerLiveFeedRefresh,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["live-feed"] });
    },
  });
}
