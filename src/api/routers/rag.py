from typing import List
from fastapi import APIRouter
from src.api.schemas import RagasScore, CorpusHealth, GoldQARow
from src.api.fixtures import RAGAS_SCORES, CORPUS, GOLD_QA

router = APIRouter()


@router.get("/scorecard", response_model=List[RagasScore])
def get_scorecard():
    return RAGAS_SCORES


@router.get("/corpus", response_model=List[CorpusHealth])
def get_corpus():
    return CORPUS


@router.get("/gold-dataset", response_model=List[GoldQARow])
def get_gold_dataset():
    return GOLD_QA
