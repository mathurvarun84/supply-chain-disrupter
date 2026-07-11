/**
 * Typed fetch functions for Screen 6 (RAG / RAGAS Evaluation) — GET
 * /api/rag/scorecard, /api/rag/corpus, /api/rag/gold-dataset. Backed by
 * src/api/routers/rag.py; still fixture JSON in this cut (Day 8 swaps in
 * real RAGAS + live ChromaDB collection stats, no route signatures change).
 */

import type { CorpusHealth, GoldQARow, RagasScore } from "../types/ragEval";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const fetchRagScorecard = () => getJSON<RagasScore[]>("/api/rag/scorecard");

export const fetchRagCorpus = () => getJSON<CorpusHealth[]>("/api/rag/corpus");

export const fetchGoldDataset = () => getJSON<GoldQARow[]>("/api/rag/gold-dataset");
