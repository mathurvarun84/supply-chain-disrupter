/**
 * React Query hooks for Screen 6 (RAG / RAGAS Evaluation). One hook per
 * endpoint — each panel owns its own query, same rule as Screen 5's
 * Observability sub-tabs. Fixture data doesn't change mid-run, so these
 * fetch once (staleTime: Infinity), same pattern as useRiskClassification.
 */

import { useQuery } from "@tanstack/react-query";
import { fetchGoldDataset, fetchRagCorpus, fetchRagScorecard } from "../api/ragEval";

export function useRagScorecard() {
  return useQuery({
    queryKey: ["rag-scorecard"],
    queryFn: fetchRagScorecard,
    staleTime: Infinity,
  });
}

export function useRagCorpus() {
  return useQuery({
    queryKey: ["rag-corpus"],
    queryFn: fetchRagCorpus,
    staleTime: Infinity,
  });
}

export function useGoldDataset() {
  return useQuery({
    queryKey: ["rag-gold-dataset"],
    queryFn: fetchGoldDataset,
    staleTime: Infinity,
  });
}
