/**
 * Typed fetch functions for Screen 2 (Risk Classification) — GET
 * /api/risk-classification/{run_id} and /latest. Backed by
 * src/api/routers/risk.py (run_id is an order_id — see that file's
 * module docstring for why ingestion_run_id doesn't apply here).
 */

import type { RiskClassification } from "../types/riskClassification";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const fetchLatestRiskClassification = () =>
  getJSON<RiskClassification>("/api/risk-classification/latest");

export const fetchRiskClassification = (runId: string) =>
  getJSON<RiskClassification>(`/api/risk-classification/${runId}`);
