import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.patch_registry import claim_patch, release_patch


def test_claim_grants_when_unclaimed():
    release_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")
    assert claim_patch("call_openai_structured", "trulens") is True
    release_patch("call_openai_structured", "trulens")


def test_claim_rejects_second_owner():
    release_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")
    assert claim_patch("call_openai_structured", "trulens") is True
    assert claim_patch("call_openai_structured", "ragas") is False
    release_patch("call_openai_structured", "trulens")


def test_same_owner_can_reclaim():
    release_patch("call_openai_structured", "trulens")
    assert claim_patch("call_openai_structured", "trulens") is True
    assert claim_patch("call_openai_structured", "trulens") is True
    release_patch("call_openai_structured", "trulens")


def test_release_by_non_owner_is_noop():
    release_patch("call_openai_structured", "trulens")
    claim_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")
    assert claim_patch("call_openai_structured", "ragas") is False
    release_patch("call_openai_structured", "trulens")
