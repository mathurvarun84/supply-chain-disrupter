import sys
import uuid
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.openai_patch import LLMCallRecord, patch_openai_calls


class _FakeOutput(BaseModel):
    label: str


def test_patch_intercepts_all_four_agent_module_references():
    import src.agents.news_agent.agent as news_mod
    import src.agents.weather_agent.agent as weather_mod
    import src.agents.risk_classifier_agent.llm_signal as llm_signal_mod
    import src.agents.risk_classifier_agent.judge_agent as judge_mod
    import src.utils.openai_utils as openai_utils_mod

    original = openai_utils_mod.call_openai_structured

    with patch_openai_calls(on_call=lambda rec: None):
        assert news_mod.call_openai_structured is not original
        assert weather_mod.call_openai_structured is not original
        assert llm_signal_mod.call_openai_structured is not original
        assert judge_mod.call_openai_structured is not original
        assert news_mod.call_openai_structured is weather_mod.call_openai_structured

    assert news_mod.call_openai_structured is original
    assert weather_mod.call_openai_structured is original
    assert llm_signal_mod.call_openai_structured is original
    assert judge_mod.call_openai_structured is original


def test_patch_captures_call_record_and_reads_tokens_from_sqlite():
    from src.utils.db_utils import ensure_schema, insert_llm_call_log

    ensure_schema()
    run_id = str(uuid.uuid4())
    captured = []

    def fake_original(system_prompt, user_message, response_model, model="gpt-4o",
                       max_tokens=1024, *, run_id=None, agent_name=None, trace=None, span=None):
        # call_openai_structured already writes its own llm_call_log row as a
        # side effect (via record_llm_generation) — the fake reproduces that
        # side effect so the patch's SQLite read-back can be tested in isolation.
        insert_llm_call_log(
            run_id=run_id, agent_name=agent_name, model=model,
            prompt_preview=user_message[:200], full_prompt=f"{system_prompt}\n{user_message}",
            full_response='{"label": "HIGH"}', input_tokens=123, output_tokens=45,
            total_tokens=168, cost_usd=0.00042, latency_ms=250.0, status="success",
            retry_count=0, error_message=None, langfuse_trace_id=None, langfuse_generation_id=None,
        )
        return _FakeOutput(label="HIGH")

    # Patched directly on the canonical module (not via a by-value-imported
    # reference) — cross-module propagation of *this* fake is a separate
    # concern already covered by test_patch_intercepts_all_four_agent_module_
    # references, which exercises the real original.
    with mock_patch("src.utils.openai_utils.call_openai_structured", fake_original):
        with patch_openai_calls(on_call=captured.append):
            import src.utils.openai_utils as openai_utils_mod
            result = openai_utils_mod.call_openai_structured(
                system_prompt="sys", user_message="msg", response_model=_FakeOutput,
                model="gpt-4.1-mini", run_id=run_id, agent_name="L2_news",
            )

    assert result.label == "HIGH"
    assert len(captured) == 1
    record: LLMCallRecord = captured[0]
    assert record.run_id == run_id
    assert record.agent_name == "L2_news"
    assert record.model == "gpt-4.1-mini"
    assert record.input_tokens == 123
    assert record.output_tokens == 45
    assert record.status == "success"


def test_second_claimant_is_skipped_not_double_patched():
    # patch_openai_calls always claims as owner "trulens" — the real conflict
    # scenario is a *different* owner (e.g. RAGAS's tracer) already holding
    # the claim, not a nested call from the same owner (which is allowed by
    # design, per test_same_owner_can_reclaim in test_trulens_patch_registry.py).
    from src.evaluation.patch_registry import claim_patch, release_patch

    assert claim_patch("call_openai_structured", "ragas") is True
    with patch_openai_calls(on_call=lambda rec: None) as granted:
        assert granted is False
    release_patch("call_openai_structured", "ragas")
