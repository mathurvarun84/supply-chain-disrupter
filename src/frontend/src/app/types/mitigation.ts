/**
 * Screen 4 mitigation response types for GET /api/mitigation/{run_id}.
 * Mirrors the backend schema and keeps nullable fields visible so the UI
 * can show missing Slack/cost data instead of inventing values.
 */
import type { RiskLevel } from "./riskClassification";

export type MitigationUrgency = "LOW" | "MEDIUM" | "HIGH" | "IMMEDIATE";

export interface MitigationRankedAction {
  rank: number;
  text: string;
  citations: string[];
}

export interface MitigationResponse {
  run_id: string;
  risk_level: RiskLevel;
  summary: string | null;
  urgency: MitigationUrgency;
  ranked_actions: MitigationRankedAction[];
  rag_citations: string[];
  rag_query_trace: string[];
  india_sourcing_recommendations: string[];
  slack_preview: string | null;
  cost_delta: string | null;
  cost_delta_usd: number | null;
}
