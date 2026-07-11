/**
 * TypeScript types for Screen 2 (Risk Classification) — mirrors
 * src/api/risk_classification_schemas.py 1:1. Keep both files in sync
 * when adding/changing fields.
 */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface RuleSignal {
  composite_score: number;
  geo_component: number;
  supply_component: number;
  freight_component: number;
  defect_component: number;
  base_label: RiskLevel;
  escalated_label: RiskLevel;
  escalated: boolean;
  duration_days: number | null;
  delivery_status_override: string | null;
}

export interface DistilBertSignal {
  predicted_label: RiskLevel | "N/A" | null;
  confidence: number | null;
  probability_distribution: Record<string, number>;
  model_source: string;
  inference_ms: number | null;
}

export interface LlmSignal {
  predicted_label: RiskLevel | null;
  rationale: string | null;
  rag_citations: string[];
  rag_chunks_used: number;
  confidence_level: "high" | "medium" | "low" | null;
  primary_driver: "geo" | "supply" | "freight" | "defect" | "delivery_status" | null;
}

export interface JudgeVerdict {
  final_label: RiskLevel | null;
  verdict_type:
    | "unanimous"
    | "majority_rule"
    | "override_distilbert"
    | "override_llm"
    | "defer_to_rules"
    | null;
  reasoning: string | null;
  signals_agreed: boolean | null;
  disagreement_explanation: string | null;
}

export interface RiskClassification {
  run_id: string;
  order_id: number;
  mode: "live" | "replay";
  rule_signal: RuleSignal;
  distilbert_signal: DistilBertSignal;
  llm_signal: LlmSignal;
  judge_verdict: JudgeVerdict | null;
  final_label: RiskLevel;
  final_critical_flag: boolean;
  slack_should_fire: boolean;
  threshold: number;
  from_cache: boolean;
}
