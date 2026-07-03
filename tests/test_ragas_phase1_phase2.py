"""
test_ragas_phase1_phase2.py — Tests for RAGAS Phase 1 (gold dataset generator)
and Phase 2 (trace interceptor).

All tests run WITHOUT an API key and WITHOUT a populated ChromaDB — every
external dependency (Anthropic, OpenAI, retrieval) is mocked.

Run: python -m pytest tests/test_ragas_phase1_phase2.py -v --tb=short
"""

from __future__ import annotations

import hashlib
import random
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import evaluation.ragas.generate_test_dataset as gen
from evaluation.ragas.generate_test_dataset import (
    GoldQAPair,
    TARGET_TOTAL_PAIRS,
    compute_allocation,
    dedup_records,
    load_excluded_chunk_ids,
    sample_chunks,
)
from evaluation.ragas.rag_tracer import (
    RAGTraceCollector,
    _extract_answer_text,
    trace_retrieval_only,
)


def _chunk(chunk_id: str, text: str = None) -> dict:
    return {
        "id": chunk_id,
        "text": text or f"Chunk {chunk_id} " + "content words " * 40,
        "metadata": {"source_file": f"{chunk_id}.docx"},
    }


# ---------------------------------------------------------------------------
# 1. Allocation floor / ceiling
# ---------------------------------------------------------------------------

def test_allocation_respects_floor_and_ceiling():
    allocation = compute_allocation({"a": 500, "b": 10, "c": 8})
    assert all(v >= 6 for v in allocation.values()), allocation
    assert all(v <= 18 for v in allocation.values()), allocation
    assert sum(allocation.values()) == TARGET_TOTAL_PAIRS, allocation


# ---------------------------------------------------------------------------
# 2. Unusable chunk triggers replacement
# ---------------------------------------------------------------------------

def test_unusable_chunk_triggers_replacement(monkeypatch):
    unusable = GoldQAPair(
        question="",
        ground_truth="",
        chunk_is_usable=False,
        unusable_reason="fragmentary table",
    )
    valid = GoldQAPair(
        question="What impact did the export controls have on chip supply?",
        ground_truth="The controls restricted advanced chip exports which tightened global supply significantly.",
        chunk_is_usable=True,
    )
    mock_llm = MagicMock(side_effect=[unusable, valid])
    monkeypatch.setattr(gen, "call_anthropic_structured", mock_llm)

    slots = [(_chunk("bad_chunk"), "natural_question")]
    pool = [_chunk("good_chunk")]
    records, rejected = gen.generate_qa_pairs("export_control_corpus", slots, pool)

    assert mock_llm.call_count == 2
    assert len(records) == 1
    assert records[0]["source_chunk_id"] == "good_chunk"
    assert rejected == ["bad_chunk"]  # rejected chunk id is logged/recorded


# ---------------------------------------------------------------------------
# 3. Dedup drops near-duplicates
# ---------------------------------------------------------------------------

def test_dedup_drops_near_duplicates():
    base_q = "Q" * 50
    base_t = "T" * 30
    records = [
        {"question": base_q + " variant one", "ground_truth": base_t + " extra"},
        {"question": base_q + " variant two", "ground_truth": base_t + " other"},
    ]
    unique, dropped = dedup_records(records)
    assert len(unique) == 1
    assert dropped == 1


# ---------------------------------------------------------------------------
# 4. Excluded chunk ids are never sampled
# ---------------------------------------------------------------------------

def test_excluded_chunk_ids_never_sampled(tmp_path):
    excluded_file = tmp_path / "excluded_chunk_ids.txt"
    excluded_file.write_text(
        "# manually rejected during review\nexcluded_id_123\n", encoding="utf-8"
    )
    excluded = load_excluded_chunk_ids(excluded_file)
    assert excluded == {"excluded_id_123"}

    chunks = [_chunk("excluded_id_123")] + [_chunk(f"ok_{i}") for i in range(10)]
    rng = random.Random(42)
    for _ in range(20):  # sample repeatedly — the excluded id must never appear
        sampled = sample_chunks(chunks, 5, rng, excluded)
        assert all(c["id"] != "excluded_id_123" for c in sampled)


# ---------------------------------------------------------------------------
# 5. Tracer restores originals even on exception
# ---------------------------------------------------------------------------

def test_tracer_restores_functions_on_exception():
    import src.rag.retriever as retriever_mod
    import src.rag.utils as rag_utils_mod
    import src.utils.openai_utils as openai_mod

    orig_retrieve = retriever_mod.retrieve_and_rerank
    orig_query = rag_utils_mod.query_chroma_rag
    orig_llm = openai_mod.call_openai_structured

    with pytest.raises(ValueError, match="boom"):
        with RAGTraceCollector(integration_point="test"):
            # confirm the patch is actually in place inside the with-block
            assert retriever_mod.retrieve_and_rerank is not orig_retrieve
            raise ValueError("boom")

    assert retriever_mod.retrieve_and_rerank is orig_retrieve
    assert rag_utils_mod.query_chroma_rag is orig_query
    assert openai_mod.call_openai_structured is orig_llm


# ---------------------------------------------------------------------------
# 6. _extract_answer_text branch coverage
# ---------------------------------------------------------------------------

def test_extract_answer_text_branches():
    class FakeSignal:
        rationale = "geo risk elevated because of export restrictions"

    class FakeMitigation:
        ranked_actions = ["Qualify Dixon as second source", "Raise safety stock"]
        urgency = "HIGH"

    class FakeOther(BaseModel):
        verdict: str = "CRITICAL"

    assert (
        _extract_answer_text(FakeSignal())
        == "geo risk elevated because of export restrictions"
    )
    assert (
        _extract_answer_text(FakeMitigation())
        == "HIGH: Qualify Dixon as second source; Raise safety stock"
    )
    assert _extract_answer_text(FakeOther()) == FakeOther().model_dump_json()


# ---------------------------------------------------------------------------
# 7. trace_retrieval_only makes zero LLM calls
# ---------------------------------------------------------------------------

def test_trace_retrieval_only_no_llm(monkeypatch):
    import src.rag.retriever as retriever_mod
    import src.utils.openai_utils as openai_mod

    gold_text = "gold chunk text about BIS export controls"
    fake_hits = [
        {"text": "some other chunk", "metadata": {}},
        {"text": gold_text, "metadata": {}},
    ]
    monkeypatch.setattr(
        retriever_mod, "retrieve_and_rerank", lambda *a, **k: fake_hits
    )

    def _llm_must_not_be_called(*args, **kwargs):
        raise AssertionError("call_openai_structured was invoked in retrieval-only mode")

    monkeypatch.setattr(openai_mod, "call_openai_structured", _llm_must_not_be_called)

    case = {
        "question": "What do the BIS rules restrict?",
        "ground_truth": "They restrict advanced semiconductor exports.",
        "source_collection": "export_control_corpus",
        "source_chunk_id": hashlib.sha256(gold_text.encode()).hexdigest(),
        "query_style": "natural_question",
    }
    records = trace_retrieval_only(
        [case], use_two_stage=True, chunk_id_convention="sha256_fallback"
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["answer"] is None
    assert rec["gold_chunk_retrieved"] is True
    assert rec["gold_chunk_rank"] == 2  # gold chunk was the second hit
    assert len(rec["contexts"]) == 2


# ---------------------------------------------------------------------------
# 8. Dry run without API key writes nothing, exits 0
# ---------------------------------------------------------------------------

def test_dry_run_without_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(gen, "has_anthropic_api_key", lambda: False)
    monkeypatch.setattr(gen, "run_preflight", lambda strict=True: True)

    fake_chunks = {
        "historical_precedents": [_chunk(f"hp_{i}") for i in range(20)],
        "export_control_corpus": [_chunk(f"ec_{i}") for i in range(20)],
        "india_sourcing_corpus": [_chunk(f"is_{i}") for i in range(20)],
    }
    monkeypatch.setattr(gen, "fetch_eligible_chunks", lambda excluded: fake_chunks)

    output_path = tmp_path / "test_dataset.json"
    review_path = tmp_path / "test_dataset_review.md"
    monkeypatch.setattr(gen, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(gen, "REVIEW_PATH", review_path)
    monkeypatch.setattr(gen, "EXCLUDED_IDS_PATH", tmp_path / "excluded_chunk_ids.txt")

    exit_code = gen.main()

    assert exit_code == 0
    assert not output_path.exists()
    assert not review_path.exists()
    out = capsys.readouterr().out
    assert "Dry run complete" in out
