/**
 * Types for the TruLens tab — POST/GET /api/trulens/*. Backed by
 * src/api/routers/trulens.py, which wraps
 * src.evaluation.trulens_integration.wrapper.run_with_trulens() as a
 * one-off capture run, separate from the live "Run Pipeline" control.
 */
import type { DemoScenarioId } from "./pipeline";

export type TruLensRunStatus = "pending" | "running" | "complete" | "failed";

export interface TruLensRunRequest {
  demo_scenario_id: DemoScenarioId;
}

export interface TruLensRunResponse {
  run_id: string;
  accepted_at: string;
}

export interface TruLensStatus {
  run_id: string;
  status: TruLensRunStatus;
  demo_scenario_id: DemoScenarioId;
  risk_label?: string;
  node_latencies_ms?: Record<string, number>;
  cost_summary?: {
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number;
    models: string[];
  };
  node_latency_check?: number;
  completed_at?: string;
  error?: string;
}

export interface TruLensMetrics {
  days: number;
  n_runs: number;
  risk_score_stability: number;
}
