/**
 * Typed fetch helper for Screen 4 Mitigation Plan — GET /api/mitigation/{run_id}.
 */
import type { MitigationResponse } from "../types/mitigation";
import { API_BASE_URL } from "./config";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status}`);
  }
  return res.json();
}

export const fetchMitigation = (runId: string) =>
  getJSON<MitigationResponse>(`/api/mitigation/${runId}`);
