/**
 * Typed fetch functions for the TruLens tab — GET/POST /api/trulens/*.
 * Backed by src/api/routers/trulens.py.
 */
import type {
  TruLensMetrics,
  TruLensRunRequest,
  TruLensRunResponse,
  TruLensStatus,
} from "../types/trulens";
import { API_BASE_URL } from "./config";

export const postTrulensRun = async (
  body: TruLensRunRequest,
): Promise<TruLensRunResponse> => {
  const res = await fetch(`${API_BASE_URL}/api/trulens/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`trulens/run failed: ${res.status}`);
  return res.json();
};

export const fetchTrulensStatus = async (runId: string): Promise<TruLensStatus> => {
  const res = await fetch(`${API_BASE_URL}/api/trulens/status/${runId}`);
  if (!res.ok) throw new Error(`trulens/status failed: ${res.status}`);
  return res.json();
};

export const fetchTrulensMetrics = async (): Promise<TruLensMetrics> => {
  const res = await fetch(`${API_BASE_URL}/api/trulens/metrics`);
  if (!res.ok) throw new Error(`trulens/metrics failed: ${res.status}`);
  return res.json();
};
