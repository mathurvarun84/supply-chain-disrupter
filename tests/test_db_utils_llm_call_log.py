import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.db_utils import ensure_schema, fetch_latest_llm_call_log, insert_llm_call_log


def test_fetch_latest_llm_call_log_returns_most_recent_row():
    ensure_schema()
    run_id = str(uuid.uuid4())
    insert_llm_call_log(
        run_id=run_id, agent_name="L2_news", model="gpt-4.1-mini",
        prompt_preview="p1", full_prompt="P1", full_response="R1",
        input_tokens=100, output_tokens=50, total_tokens=150,
        cost_usd=0.0001, latency_ms=200.0, status="success",
        retry_count=0, error_message=None,
        langfuse_trace_id=None, langfuse_generation_id=None,
    )
    insert_llm_call_log(
        run_id=run_id, agent_name="L2_news", model="gpt-4.1-mini",
        prompt_preview="p2", full_prompt="P2", full_response="R2",
        input_tokens=200, output_tokens=80, total_tokens=280,
        cost_usd=0.0002, latency_ms=300.0, status="success",
        retry_count=0, error_message=None,
        langfuse_trace_id=None, langfuse_generation_id=None,
    )

    row = fetch_latest_llm_call_log(run_id, "L2_news")

    assert row is not None
    assert row["input_tokens"] == 200
    assert row["output_tokens"] == 80
    assert row["full_response"] == "R2"


def test_fetch_latest_llm_call_log_returns_none_when_missing():
    ensure_schema()
    assert fetch_latest_llm_call_log("no-such-run-id", "L2_news") is None
