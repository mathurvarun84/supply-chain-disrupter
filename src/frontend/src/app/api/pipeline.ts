/**
 * Typed fetch functions for the Run Pipeline control — GET/POST
 * /api/pipeline/*. Backed by src/api/routers/pipeline.py.
 */
import type {
  PipelineRunRequest,
  PipelineRunResponse,
  PipelineStatus,
} from "../types/pipeline";
import { API_BASE_URL } from "./config";

export const fetchPipelineStatus = async (runId?: string): Promise<PipelineStatus> => {
  const res = await fetch(
    `${API_BASE_URL}/api/pipeline/status${runId ? `?run_id=${runId}` : ""}`,
  );
  if (!res.ok) throw new Error(`pipeline/status failed: ${res.status}`);
  return res.json();
};

export const postPipelineRun = async (
  body: PipelineRunRequest,
): Promise<PipelineRunResponse> => {
  const res = await fetch(`${API_BASE_URL}/api/pipeline/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`pipeline/run failed: ${res.status}`);
  return res.json();
};

export interface RecentRun {
  run_id: string;
  last_started_at: string | null;
}

export const fetchRecentRuns = async (): Promise<RecentRun[]> => {
  const res = await fetch(`${API_BASE_URL}/api/pipeline/runs`);
  if (!res.ok) throw new Error(`pipeline/runs failed: ${res.status}`);
  const body = await res.json();
  return body.runs;
};
