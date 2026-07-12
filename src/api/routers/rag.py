"""RAG / RAGAS (Screen 6) — reads ChromaDB health + persisted RAGAS outputs."""

from typing import List

from fastapi import APIRouter

from src.api.schemas import CorpusHealth, GoldQARow, RagasScore
from src.rag.utils import fetch_corpus_health, fetch_gold_dataset, fetch_ragas_scorecard

router = APIRouter()


@router.get("/scorecard", response_model=List[RagasScore])
def get_scorecard():
    """Reads persisted RAGAS Phase 1-4 output (evaluation/ragas/ragas_scores_full.json).

    Was: return FIXTURE_RAGAS_SCORES. Returns [] when no evaluation file exists
    (documented scope cut — corpus may postdate last RAGAS run).
    Consumed by: Screen 6 RAGAS scorecard tiles."""
    return fetch_ragas_scorecard()


@router.get("/corpus", response_model=List[CorpusHealth])
def get_corpus():
    """Reads live collection.count() for all 3 ChromaDB named collections.

    Was: return FIXTURE_CORPUS with hardcoded chunk counts.
    Consumed by: Screen 6 corpus health cards."""
    return fetch_corpus_health()


@router.get("/gold-dataset", response_model=List[GoldQARow])
def get_gold_dataset():
    """Reads chunk-grounded gold QA from evaluation/ragas/test_dataset.json.

    Was: return FIXTURE_GOLD_QA. Consumed by: Screen 6 Gold Dataset table."""
    return fetch_gold_dataset()
