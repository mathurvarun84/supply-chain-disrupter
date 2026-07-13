import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.patch_registry import _active_patches, claim_patch, release_patch


@pytest.fixture(autouse=True)
def _reset_registry():
    # Direct reset rather than relying on tests calling release_patch a
    # precisely-balanced number of times — the reentrant/depth-tracked
    # design added below means an unbalanced release leaves a leftover
    # claim that would otherwise leak into the next test.
    _active_patches.clear()
    yield
    _active_patches.clear()


def test_claim_grants_when_unclaimed():
    assert claim_patch("call_openai_structured", "trulens") is True


def test_claim_rejects_second_owner():
    assert claim_patch("call_openai_structured", "trulens") is True
    assert claim_patch("call_openai_structured", "ragas") is False


def test_same_owner_can_reclaim():
    assert claim_patch("call_openai_structured", "trulens") is True
    assert claim_patch("call_openai_structured", "trulens") is True


def test_release_by_non_owner_is_noop():
    claim_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "ragas")
    assert claim_patch("call_openai_structured", "ragas") is False


def test_nested_same_owner_release_does_not_release_outer_claim():
    # Regression test for the bug flagged in PR review: nested
    # claim/claim/release from the SAME owner must not release the outer
    # claim. Before the fix, _active_patches was a flat {target: owner}
    # map with no depth tracking, so the inner release deleted the entry
    # outright — letting a completely different owner claim the target
    # while the outer scope was still logically active.
    claim_patch("call_openai_structured", "trulens")  # outer enters
    claim_patch("call_openai_structured", "trulens")  # inner enters (same owner)
    release_patch("call_openai_structured", "trulens")  # inner exits

    assert claim_patch("call_openai_structured", "ragas") is False  # outer still owns it

    release_patch("call_openai_structured", "trulens")  # outer exits
    assert claim_patch("call_openai_structured", "ragas") is True  # now genuinely free


def test_release_below_zero_depth_is_noop():
    claim_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "trulens")
    release_patch("call_openai_structured", "trulens")  # already fully released
    assert claim_patch("call_openai_structured", "ragas") is True
