import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ragas.rag_tracer import RAGTraceCollector
from src.evaluation.patch_registry import claim_patch, release_patch


def test_rag_trace_collector_claims_call_openai_structured():
    release_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")

    with RAGTraceCollector(integration_point="test") as collector:
        assert "call_openai_structured" in collector._claimed

    # released on exit
    assert claim_patch("call_openai_structured", "trulens") is True
    release_patch("call_openai_structured", "trulens")


def test_rag_trace_collector_skips_call_openai_structured_when_already_claimed_by_trulens():
    release_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")
    assert claim_patch("call_openai_structured", "trulens") is True

    with RAGTraceCollector(integration_point="test") as collector:
        assert "call_openai_structured" not in collector._claimed
        # retrieval-only targets aren't registry-gated at all (only
        # call_openai_structured is), so they're still patched normally —
        # confirmed via _patches (what actually got monkey-patched), not
        # _claimed (which only tracks registry-gated attrs).
        patched_attrs = {attr for _, attr, _ in collector._patches}
        assert "retrieve_and_rerank" in patched_attrs
        assert "call_openai_structured" not in patched_attrs

    release_patch("call_openai_structured", "trulens")
