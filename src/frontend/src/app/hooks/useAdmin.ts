/**
 * React Query hooks for the Admin page. useAdminStatus polls GET
 * /api/admin/status every 3s while either build job is running (so the
 * status card animates live), otherwise every 15s.
 *
 * The two mutation hooks optimistically write {status: "running"} into the
 * admin-status cache the instant POST /db/build or /rag/build returns,
 * rather than waiting for the next poll to catch up: POST returns as soon
 * as the BackgroundTask is *queued*, and invalidateQueries alone can race
 * the background task actually starting, leaving the UI showing "idle" for
 * up to one poll interval right after the click — the exact "did anything
 * happen?" gap this is fixing. A real poll (fired by the invalidate) still
 * follows immediately after to pick up the server-authoritative state.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchAdminStatus, postDatabaseBuild, postRagBuild } from "../api/admin";
import type { AdminStatusResponse } from "../types/admin";

const ADMIN_STATUS_KEY = ["admin-status"];

export function useAdminStatus() {
  return useQuery({
    queryKey: ADMIN_STATUS_KEY,
    queryFn: fetchAdminStatus,
    refetchInterval: (query) => {
      const data = query.state.data;
      const anyRunning = data?.db_job.status === "running" || data?.rag_job.status === "running";
      return anyRunning ? 3_000 : 15_000;
    },
  });
}

function markJobRunning(
  qc: ReturnType<typeof useQueryClient>,
  job: "db_job" | "rag_job",
  triggeredAt: string,
) {
  qc.setQueryData<AdminStatusResponse>(ADMIN_STATUS_KEY, (old) =>
    old
      ? { ...old, [job]: { status: "running", started_at: triggeredAt, finished_at: null, error: null, result: null } }
      : old,
  );
  qc.invalidateQueries({ queryKey: ADMIN_STATUS_KEY });
}

export function useBuildDatabase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: postDatabaseBuild,
    onSuccess: (resp) => markJobRunning(qc, "db_job", resp.triggered_at),
  });
}

export function useBuildRag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (flush: boolean) => postRagBuild(flush),
    onSuccess: (resp) => markJobRunning(qc, "rag_job", resp.triggered_at),
  });
}
