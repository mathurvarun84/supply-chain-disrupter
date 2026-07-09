/**
 * TypeScript types for Screen 6 (RAG / RAGAS Evaluation) — mirrors
 * RagasScore / CorpusHealth / GoldQARow in src/api/schemas.py 1:1. Keep
 * both files in sync when adding/changing fields.
 */

export interface RagasScore {
  metric: string;
  score: number;
  threshold: number;
  passed: boolean;
}

export interface CorpusHealth {
  name: string;
  docs: number;
  real: number;
  synth: number;
  last_ingested_at: string;
}

export type QueryStyle = "agent_pattern" | "natural_question";

export interface GoldQARow {
  question: string;
  ground_truth: string;
  match: boolean;
  source_collection: string | null;
  source_chunk_id: string | null;
  query_style: QueryStyle;
}
