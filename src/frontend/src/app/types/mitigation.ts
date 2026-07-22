/**
 * Screen 4 mitigation response types for GET /api/mitigation/{run_id}.
 * Mirrors the backend schema (src/api/schemas.py) and keeps nullable fields
 * visible so the UI can show missing Slack/cost data instead of inventing
 * values.
 */
import type { RiskLevel } from "./riskClassification";

export type MitigationUrgency = "LOW" | "MEDIUM" | "HIGH" | "IMMEDIATE";

export type MitigationActionType =
  | "INVENTORY"
  | "ROUTING"
  | "SOURCING"
  | "INDIA-SOURCING"
  | "MONITOR"
  | "FINANCIAL";

export interface MitigationCitation {
  source_file: string;
  collection: string;
}

export interface MitigationRankedAction {
  rank: number;
  text: string;
  action_type: MitigationActionType;
  citations: MitigationCitation[];
}

export interface RagTraceChunk {
  source_file: string | null;
  collection: string | null;
  similarity_score: number | null;
  snippet: string | null;
}

export type RagQueryName = "historical_disruption_lookup" | "export_control_check" | "india_sourcing_query";

export interface RagTraceQuery {
  query_name: RagQueryName;
  query_text: string;
  fired: boolean;
  fire_condition: string;
  retrieved_chunks: RagTraceChunk[];
}

export interface MitigationResponse {
  run_id: string;
  risk_level: RiskLevel;
  summary: string | null;
  urgency: MitigationUrgency;
  mitigation_window: string | null;
  ranked_actions: MitigationRankedAction[];
  rag_query_trace: RagTraceQuery[];
  india_sourcing_recommendations: string[];
  slack_alert_fired: boolean;
  slack_preview: string | null;
  cost_delta: string | null;
  cost_delta_usd: number | null;
  sku_id: string | null;
  impact_duration_days: number | null;
}
