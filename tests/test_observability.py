"""
Tests for src/utils/observability.py and its instrumentation hooks.

All tests are fully mocked — no real network calls to Langfuse or OpenAI.
Run: python -m pytest tests/test_observability.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 1. Kill switch
# ---------------------------------------------------------------------------

def test_disabled_by_default(monkeypatch):
    """LANGFUSE_ENABLED unset → observability_enabled() is False, client is None."""
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    # Clear lru_cache so env changes take effect
    from src.utils import observability as obs
    obs._get_langfuse_client.cache_clear()

    assert obs.observability_enabled() is False
    assert obs._get_langfuse_client() is None


# ---------------------------------------------------------------------------
# 2. pipeline_trace no-op when disabled
# ---------------------------------------------------------------------------

def test_pipeline_trace_noop_when_disabled(monkeypatch):
    """With client=None, pipeline_trace() yields None and does not raise."""
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)

    from src.utils import observability as obs
    obs._get_langfuse_client.cache_clear()

    with patch.object(obs, "_get_langfuse_client", return_value=None):
        with obs.pipeline_trace("run-123", mode="demo") as trace:
            assert trace is None


# ---------------------------------------------------------------------------
# 3. agent_span writes SQLite even when Langfuse is disabled
# ---------------------------------------------------------------------------

def test_agent_span_writes_sqlite_even_when_langfuse_disabled():
    """agent_span writes insert + update to agent_execution_log regardless of Langfuse."""
    from src.utils import observability as obs

    with (
        patch.object(obs, "_get_langfuse_client", return_value=None),
        patch("src.utils.db_utils.insert_agent_execution") as mock_insert,
        patch("src.utils.db_utils.update_agent_execution") as mock_update,
    ):
        with obs.agent_span(None, "run-abc", "L2_news"):
            pass

    mock_insert.assert_called_once()
    insert_kwargs = mock_insert.call_args.kwargs
    assert insert_kwargs["run_id"] == "run-abc"
    assert insert_kwargs["agent_name"] == "L2_news"
    assert insert_kwargs["status"] == "Running"

    mock_update.assert_called_once()
    update_kwargs = mock_update.call_args.kwargs
    assert update_kwargs["status"] == "Complete"


# ---------------------------------------------------------------------------
# 4. agent_span re-raises real agent exceptions
# ---------------------------------------------------------------------------

def test_agent_span_reraises_on_agent_exception():
    """agent_span does not swallow exceptions; update is called with Failed-Fallback."""
    from src.utils import observability as obs

    with (
        patch.object(obs, "_get_langfuse_client", return_value=None),
        patch("src.utils.db_utils.insert_agent_execution"),
        patch("src.utils.db_utils.update_agent_execution") as mock_update,
    ):
        with pytest.raises(ValueError, match="boom"):
            with obs.agent_span(None, "run-xyz", "L4_risk_classifier"):
                raise ValueError("boom")

    update_kwargs = mock_update.call_args.kwargs
    assert update_kwargs["status"] == "Failed-Fallback"
    assert "boom" in (update_kwargs["error_message"] or "")


# ---------------------------------------------------------------------------
# 5. record_llm_generation writes llm_call_log
# ---------------------------------------------------------------------------

def test_record_llm_generation_writes_llm_call_log():
    """record_llm_generation() always calls insert_llm_call_log with correct cost."""
    from src.utils import observability as obs

    with patch("src.utils.db_utils.insert_llm_call_log") as mock_insert:
        obs.record_llm_generation(
            None, None,
            run_id="run-001",
            agent_name="L2_news",
            model="gpt-4.1-mini",
            system_prompt="sys",
            user_message="user",
            parsed_output={"result": "ok"},
            input_tokens=500,
            output_tokens=200,
            latency_ms=320.0,
            status="success",
        )

    mock_insert.assert_called_once()
    kwargs = mock_insert.call_args.kwargs
    assert kwargs["run_id"] == "run-001"
    assert kwargs["agent_name"] == "L2_news"
    assert kwargs["status"] == "success"
    # cost = (500/1000)*0.0004 + (200/1000)*0.0016 = 0.0002 + 0.00032 = 0.00052
    assert abs(kwargs["cost_usd"] - 0.00052) < 1e-8


# ---------------------------------------------------------------------------
# 6. calculate_cost_usd — known model
# ---------------------------------------------------------------------------

def test_calculate_cost_usd_known_model():
    """gpt-4o cost: (1000/1000)*0.0025 + (500/1000)*0.010 = 0.0025 + 0.005 = 0.0075"""
    from src.utils.observability import calculate_cost_usd

    cost = calculate_cost_usd("gpt-4o", 1000, 500)
    assert abs(cost - 0.0075) < 1e-8


# ---------------------------------------------------------------------------
# 7. calculate_cost_usd — unknown model returns 0.0, no raise
# ---------------------------------------------------------------------------

def test_calculate_cost_usd_unknown_model_returns_zero():
    """Unknown model string → 0.0, logs a warning, does not raise."""
    from src.utils.observability import calculate_cost_usd

    cost = calculate_cost_usd("gpt-99-turbo-ultra", 1000, 500)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# 8. call_openai_structured records generation on success
# ---------------------------------------------------------------------------

def test_call_openai_structured_records_generation_on_success():
    """Successful parse → record_llm_generation called once with status='success'."""
    from pydantic import BaseModel

    class FakeOutput(BaseModel):
        label: str

    fake_message = SimpleNamespace(
        parsed=FakeOutput(label="HIGH"),
        refusal=None,
    )
    fake_choice = SimpleNamespace(message=fake_message)
    fake_usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    fake_completion = SimpleNamespace(choices=[fake_choice], usage=fake_usage)

    with (
        patch("src.utils.openai_utils._get_client") as mock_client,
        patch("src.utils.observability.record_llm_generation") as mock_record,
    ):
        mock_client.return_value.beta.chat.completions.parse.return_value = fake_completion

        from src.utils.openai_utils import call_openai_structured
        result = call_openai_structured(
            system_prompt="sys",
            user_message="user",
            response_model=FakeOutput,
            run_id="run-999",
            agent_name="L2_news",
        )

    assert result.label == "HIGH"
    mock_record.assert_called_once()
    _, kwargs = mock_record.call_args
    assert kwargs["status"] == "success"
    assert kwargs["input_tokens"] == 100
    assert kwargs["output_tokens"] == 50


# ---------------------------------------------------------------------------
# 9. call_openai_structured records generation on failure
# ---------------------------------------------------------------------------

def test_call_openai_structured_records_generation_on_failure():
    """Parse failure (result=None → refusal) → record_llm_generation with status='failed_fallback'."""
    from pydantic import BaseModel

    class FakeOutput(BaseModel):
        label: str

    fake_message = SimpleNamespace(parsed=None, refusal="content policy")
    fake_choice = SimpleNamespace(message=fake_message)
    fake_usage = SimpleNamespace(prompt_tokens=80, completion_tokens=0)
    fake_completion = SimpleNamespace(choices=[fake_choice], usage=fake_usage)

    with (
        patch("src.utils.openai_utils._get_client") as mock_client,
        patch("src.utils.observability.record_llm_generation") as mock_record,
    ):
        mock_client.return_value.beta.chat.completions.parse.return_value = fake_completion

        from src.utils.openai_utils import call_openai_structured
        with pytest.raises(RuntimeError, match="no parsed result"):
            call_openai_structured(
                system_prompt="sys",
                user_message="user",
                response_model=FakeOutput,
                run_id="run-999",
                agent_name="L2_news",
            )

    mock_record.assert_called_once()
    _, kwargs = mock_record.call_args
    assert kwargs["status"] == "failed_fallback"
