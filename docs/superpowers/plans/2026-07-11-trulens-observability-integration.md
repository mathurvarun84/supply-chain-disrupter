# TruLens Observability Integration (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TruLens tracing (per-node latency, token/cost capture) and four custom evaluation metrics (risk drift, ensemble agreement, latency thresholds, forecast accuracy) to the L1–L7 LangGraph pipeline, viewable in a local TruLens dashboard, with zero changes to `src/agents/`.

**Architecture:** An external wrapper package (`src/evaluation/trulens_integration/`) provides `run_with_trulens(payload)` as a drop-in replacement for `run_agent_graph(payload)`. It builds the existing compiled graph via `build_agent_graph(payload)` (unmodified), drives it with `app.stream(..., stream_mode="updates")` to observe per-node completions without touching `langgraph_engine.py`, and monkey-patches `call_openai_structured` across every module holding a reference to it (mirroring the existing, proven pattern in `evaluation/ragas/rag_tracer.py`) to capture per-call model/latency/tokens/cost. A small `patch_registry` prevents this patch from colliding with RAGAS's own tracer if both run in the same process.

**Tech Stack:** `trulens-core`, `trulens-dashboard`, `trulens-feedback`, `trulens-providers-openai` (2.8.x line), Python's `sqlite3`, `pytest` + `unittest.mock`.

## Global Constraints

- Reference spec: `docs/specs/2026-07-06-trulens-integration-design.md` (as amended 2026-07-11).
- Zero changes to any file under `src/agents/` — all instrumentation is external.
- Dependency pins: `trulens-core>=2.8,<3`, `trulens-dashboard>=2.8,<3`, `trulens-feedback>=2.8,<3`, `trulens-providers-openai>=2.8,<3`. Never `trulens-eval` (deprecated, unmaintained since 2025-12-01).
- Session API is `TruSession()`; dashboard launch is `trulens.dashboard.run_dashboard(session, port=...)`. Never `Tru()` / `tru.run_dashboard()`.
- TruLens SQLite storage: `data/trulens/trulens.db`.
- Dashboard port: `8502` (the Streamlit app itself stays on `8501`).
- Exceptions raised by L2, L3, L4, or L7 (all `_critical_node`-wrapped in `langgraph_engine.py:148-150,159`) must propagate out of `run_with_trulens()` uncaught, exactly as they do out of `run_agent_graph()` today. Only tracing/telemetry-capture code may be wrapped in its own try/except-and-log.
- `call_openai_structured` (`src/utils/openai_utils.py`) is imported by value into four agent modules — `src/agents/news_agent/agent.py`, `src/agents/weather_agent/agent.py`, `src/agents/risk_classifier_agent/llm_signal.py`, `src/agents/risk_classifier_agent/judge_agent.py` — so any patch must be applied to each of those modules' own attribute, not just `src.utils.openai_utils`.
- `run_agent_graph()` calls `app.invoke(GlobalState())` with no `run_id` — production graph runs today have `run_id=None` throughout. `run_with_trulens()` must mint its own `run_id` and seed `GlobalState(run_id=run_id)` so `llm_call_log`/`agent_execution_log` rows for a TruLens-wrapped run are attributable (mirrors what `run_agent_sequence()` already does at `langgraph_engine.py:204,209`).
- Correction vs. the spec text: `evaluation/ragas/test_dataset.json` holds RAG gold Q&A pairs (`question`/`ground_truth`/`source_collection` — no `port`/`sku`/`event_date`), so it cannot drive `run_with_trulens(payload)`. Task 12 below uses a small local list of pipeline scenario payloads shaped like the real caller at `src/dashboard/dashboard.py:216-229` instead.

---

## Task 1: Pin TruLens dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: importable `trulens.core`, `trulens.dashboard`, `trulens.feedback`, `trulens.providers.openai` packages for all later tasks.

- [ ] **Step 1: Add the dependency lines**

Append to `requirements.txt` (after the existing `langfuse>=2.36.0` line):

```
trulens-core>=2.8,<3
trulens-dashboard>=2.8,<3
trulens-feedback>=2.8,<3
trulens-providers-openai>=2.8,<3
```

- [ ] **Step 2: Install and smoke-check the import surface**

Run:
```bash
python3 -m pip install -r requirements.txt
python3 -c "from trulens.core import TruSession, Feedback; from trulens.dashboard import run_dashboard; from trulens.providers.openai import OpenAI as TruOpenAIProvider; print('ok')"
```
Expected: `ok` with no `ImportError`/`ModuleNotFoundError`. If any of these four names don't exist in the installed version, stop and check `python3 -c "import trulens.core; help(trulens.core)"` before continuing — the exact class/module names are the one part of this plan sourced from external docs rather than this repo, and the installed version is the source of truth.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: pin trulens-core/dashboard/feedback/providers-openai 2.8.x"
```

---

## Task 2: Patch registry

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/patch_registry.py`
- Test: `tests/test_trulens_patch_registry.py`

**Interfaces:**
- Produces: `claim_patch(target: str, owner: str) -> bool`, `release_patch(target: str, owner: str) -> None`, both importable from `src.evaluation.patch_registry`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_patch_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_patch_registry.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.patch_registry'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/__init__.py
```
(empty file — marks the package)

```python
# src/evaluation/patch_registry.py
"""
patch_registry.py — Coordinates monkey-patch ownership of shared functions
(currently `call_openai_structured`) between the TruLens wrapper and the
RAGAS tracer so a run that starts both in the same process doesn't
double-patch or restore the wrong original.

Not thread-safe by design: both callers are single-threaded CLI/script
invocations in this codebase's actual usage pattern.
"""

from __future__ import annotations

_active_patches: dict[str, str] = {}


def claim_patch(target: str, owner: str) -> bool:
    """Return True if `owner` now holds the patch on `target`.

    Granted when `target` is unclaimed or already held by `owner`.
    Rejected when a different owner currently holds it.
    """
    current = _active_patches.get(target)
    if current is not None and current != owner:
        return False
    _active_patches[target] = owner
    return True


def release_patch(target: str, owner: str) -> None:
    """Release `owner`'s claim on `target`. No-op if `owner` doesn't hold it."""
    if _active_patches.get(target) == owner:
        del _active_patches[target]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_patch_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/__init__.py src/evaluation/patch_registry.py tests/test_trulens_patch_registry.py
git commit -m "feat: add TruLens/RAGAS patch registry"
```

---

## Task 3: `fetch_latest_llm_call_log` read helper

**Files:**
- Modify: `src/utils/db_utils.py`
- Test: `tests/test_db_utils_llm_call_log.py`

**Interfaces:**
- Consumes: existing `insert_llm_call_log(**kwargs)` (`src/utils/db_utils.py:309`), existing `ensure_schema()` (creates `llm_call_log` table, `src/utils/db_utils.py:84-104`).
- Produces: `fetch_latest_llm_call_log(run_id: str, agent_name: str) -> Optional[Dict[str, Any]]`, importable from `src.utils.db_utils`. Returned dict has keys matching the `llm_call_log` columns (`model`, `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, `latency_ms`, `status`, `ts`, ...).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_utils_llm_call_log.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_db_utils_llm_call_log.py -v`
Expected: `ImportError: cannot import name 'fetch_latest_llm_call_log'`

- [ ] **Step 3: Add the function**

Add to `src/utils/db_utils.py`, directly below `fetch_latest_weather_signal` (around line 191), following its exact pattern:

```python
def fetch_latest_llm_call_log(run_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
    """Return the most recent llm_call_log row for (run_id, agent_name), or None."""
    ensure_schema()
    rows = execute_query(
        """
        SELECT * FROM llm_call_log
        WHERE run_id = ? AND agent_name = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (run_id, agent_name),
    )
    return dict(rows[0]) if rows else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_db_utils_llm_call_log.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/db_utils.py tests/test_db_utils_llm_call_log.py
git commit -m "feat: add fetch_latest_llm_call_log read helper"
```

---

## Task 4: OpenAI call patcher

**Files:**
- Create: `src/evaluation/trulens_integration/__init__.py`
- Create: `src/evaluation/trulens_integration/openai_patch.py`
- Test: `tests/test_trulens_openai_patch.py`

**Interfaces:**
- Consumes: `claim_patch`/`release_patch` (Task 2), `fetch_latest_llm_call_log` (Task 3), `call_openai_structured` (`src/utils/openai_utils.py`).
- Produces:
  ```python
  @dataclass
  class LLMCallRecord:
      run_id: Optional[str]
      agent_name: Optional[str]
      model: Optional[str]
      system_prompt: str
      user_message: str
      latency_ms: float
      input_tokens: Optional[int]
      output_tokens: Optional[int]
      cost_usd: Optional[float]
      status: str
      parsed_output: Any

  @contextmanager
  def patch_openai_calls(on_call: Callable[[LLMCallRecord], None]) -> Generator[None, None, None]:
      ...
  ```
  `on_call` is invoked once per intercepted `call_openai_structured` call, after it returns or raises. Later tasks (wrapper.py) pass a callback that forwards to TruLens/telemetry storage.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_openai_patch.py
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
    captured = []

    with patch_openai_calls(on_call=captured.append):
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

    with mock_patch("src.utils.openai_utils.call_openai_structured", fake_original):
        with patch_openai_calls(on_call=captured.append):
            import src.agents.news_agent.agent as news_mod
            result = news_mod.call_openai_structured(
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
    from src.evaluation.patch_registry import release_patch

    with patch_openai_calls(on_call=lambda rec: None):
        with patch_openai_calls(on_call=lambda rec: None) as inner_granted:
            assert inner_granted is False
    release_patch("call_openai_structured", "trulens")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_openai_patch.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/trulens_integration/__init__.py
```
(empty file — marks the package; populated with real exports in Task 9)

```python
# src/evaluation/trulens_integration/openai_patch.py
"""
openai_patch.py — Intercepts every call_openai_structured() invocation for
TruLens telemetry, without modifying src/agents/.

call_openai_structured is imported BY VALUE in four agent modules, so
patching only src.utils.openai_utils would miss all of them. This mirrors
the sys.modules scan-and-restore pattern already proven in
evaluation/ragas/rag_tracer.py (RAGTraceCollector.__enter__/__exit__).

Token/cost figures are not available from call_openai_structured's return
value (it returns only the parsed Pydantic object) — they are read back
from the llm_call_log row that call_openai_structured already writes via
its own record_llm_generation() hook, keyed by (run_id, agent_name).
"""

from __future__ import annotations

import functools
import importlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

from src.evaluation.patch_registry import claim_patch, release_patch

_TARGET_MODULE = "src.utils.openai_utils"
_TARGET_ATTR = "call_openai_structured"
_OWNER = "trulens"


@dataclass
class LLMCallRecord:
    run_id: Optional[str]
    agent_name: Optional[str]
    model: Optional[str]
    system_prompt: str
    user_message: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    status: str
    parsed_output: Any


def _make_wrapper(original: Callable, on_call: Callable[[LLMCallRecord], None]) -> Callable:
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        run_id = kwargs.get("run_id")
        agent_name = kwargs.get("agent_name")
        system_prompt = kwargs.get("system_prompt", args[0] if args else "")
        user_message = kwargs.get("user_message", args[1] if len(args) > 1 else "")
        model = kwargs.get("model")

        try:
            result = original(*args, **kwargs)
        except Exception:
            latency_ms = (time.monotonic() - t0) * 1000.0
            try:
                on_call(LLMCallRecord(
                    run_id=run_id, agent_name=agent_name, model=model,
                    system_prompt=system_prompt, user_message=user_message,
                    latency_ms=latency_ms, input_tokens=None, output_tokens=None,
                    cost_usd=None, status="failed_fallback", parsed_output=None,
                ))
            except Exception:
                pass  # telemetry capture must never mask the real exception
            raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        input_tokens = output_tokens = cost_usd = None
        status = "success"
        if run_id and agent_name:
            from src.utils.db_utils import fetch_latest_llm_call_log
            row = fetch_latest_llm_call_log(run_id, agent_name)
            if row is not None:
                input_tokens = row.get("input_tokens")
                output_tokens = row.get("output_tokens")
                cost_usd = row.get("cost_usd")
                status = row.get("status", status)

        try:
            on_call(LLMCallRecord(
                run_id=run_id, agent_name=agent_name, model=model,
                system_prompt=system_prompt, user_message=user_message,
                latency_ms=latency_ms, input_tokens=input_tokens,
                output_tokens=output_tokens, cost_usd=cost_usd,
                status=status, parsed_output=result,
            ))
        except Exception:
            pass  # non-blocking: tracing failures never affect the pipeline

        return result

    return wrapper


@contextmanager
def patch_openai_calls(on_call: Callable[[LLMCallRecord], None]) -> Generator[bool, None, None]:
    """Patch call_openai_structured everywhere it's referenced.

    Yields True if this call actually holds the patch, False if another
    owner (e.g. a concurrent RAGAS tracer) already claimed it — in which
    case no patching happens and agent calls run unpatched for this scope.
    """
    if not claim_patch(_TARGET_ATTR, _OWNER):
        yield False
        return

    module = importlib.import_module(_TARGET_MODULE)
    original = getattr(module, _TARGET_ATTR)
    wrapper = _make_wrapper(original, on_call)

    patched: list[tuple] = []
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, _TARGET_ATTR, None) is original:
                setattr(mod, _TARGET_ATTR, wrapper)
                patched.append((mod, _TARGET_ATTR, original))
        except Exception:
            continue

    try:
        yield True
    finally:
        for mod, attr, orig in patched:
            try:
                setattr(mod, attr, orig)
            except Exception:
                pass
        release_patch(_TARGET_ATTR, _OWNER)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_openai_patch.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/trulens_integration/__init__.py src/evaluation/trulens_integration/openai_patch.py tests/test_trulens_openai_patch.py
git commit -m "feat: add OpenAI call patcher with sys.modules-safe patching"
```

---

## Task 5: Node metric extractors

**Files:**
- Create: `src/evaluation/trulens_integration/node_extractors.py`
- Test: `tests/test_trulens_node_extractors.py`

**Interfaces:**
- Consumes: `GlobalState`, `RiskClassificationResult`, `ForecastResult` (`src/agents/state.py`).
- Produces: `extract_l4_signals(state: GlobalState) -> Optional[dict]` (keys: `composite_score`, `rule_label`, `distilbert_label`, `llm_label`), `extract_l5_forecast(state: GlobalState) -> Optional[dict]` (key: `expected_drop_pct`), `NODE_LATENCY_LABELS: dict[str, str]` mapping graph node names to short labels used by the latency feedback function.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_node_extractors.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import (
    DistilBERTSignal, ForecastResult, GlobalState, LLMSignal,
    RiskClassificationResult, RuleBasedSignal,
)
from src.evaluation.trulens_integration.node_extractors import (
    NODE_LATENCY_LABELS, extract_l4_signals, extract_l5_forecast,
)


def _rule_signal(label: str) -> RuleBasedSignal:
    return RuleBasedSignal(
        composite_score=0.8, geo_component=0.4, supply_component=0.2,
        freight_component=0.1, defect_component=0.1, base_label=label,
        escalated_label=label, escalated=False, duration_days=None,
    )


def test_extract_l4_signals_returns_none_without_classification():
    state = GlobalState()
    assert extract_l4_signals(state) is None


def test_extract_l4_signals_pulls_all_three_labels():
    state = GlobalState(
        risk_classification=RiskClassificationResult(
            mode="live", composite_score=0.82, geo_component=0.4,
            supply_component=0.2, freight_component=0.1, defect_component=0.1,
            duration_days=None, base_label="CRITICAL", final_label="CRITICAL",
            escalated=False, rationale="r", critical_flag=True,
            rule_signal=_rule_signal("CRITICAL"),
            distilbert_signal=DistilBERTSignal(
                predicted_label="HIGH", confidence=0.7,
                probability_distribution={"HIGH": 0.7}, model_source="ft",
                inference_ms=20.0,
            ),
            llm_signal=LLMSignal(
                predicted_label="CRITICAL", rationale="r", rag_citations=[],
                rag_chunks_used=2, confidence_level="high", primary_driver="geo",
            ),
        )
    )

    result = extract_l4_signals(state)

    assert result == {
        "composite_score": 0.82,
        "rule_label": "CRITICAL",
        "distilbert_label": "HIGH",
        "llm_label": "CRITICAL",
    }


def test_extract_l5_forecast_returns_none_without_result():
    assert extract_l5_forecast(GlobalState()) is None


def test_extract_l5_forecast_pulls_expected_drop():
    state = GlobalState(
        forecast_result=ForecastResult(prophet_forecast=[], expected_drop_pct=12.5)
    )
    assert extract_l5_forecast(state) == {"expected_drop_pct": 12.5}


def test_node_latency_labels_cover_critical_path():
    assert NODE_LATENCY_LABELS["l2_news_analysis"] == "L2"
    assert NODE_LATENCY_LABELS["l3_weather_monitoring"] == "L3"
    assert NODE_LATENCY_LABELS["l4_risk_classifier"] == "L4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_node_extractors.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration.node_extractors'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/trulens_integration/node_extractors.py
"""
node_extractors.py — Pure functions that pull TruLens-relevant metrics out
of GlobalState after a node completes. No TruLens dependency; independently
testable.
"""

from __future__ import annotations

from typing import Optional

from src.agents.state import GlobalState

NODE_LATENCY_LABELS: dict[str, str] = {
    "l1_data_ingestion": "L1",
    "l2_news_analysis": "L2",
    "l3_weather_monitoring": "L3",
    "l4_risk_classifier": "L4",
    "l5_demand_forecast": "L5",
    "l6_simulation": "L6",
    "l7_mitigation": "L7",
}


def extract_l4_signals(state: GlobalState) -> Optional[dict]:
    """Pull the three L4 ensemble labels + composite score, or None if L4 hasn't run."""
    rc = state.risk_classification
    if rc is None:
        return None
    return {
        "composite_score": rc.composite_score,
        "rule_label": rc.rule_signal.escalated_label if rc.rule_signal else None,
        "distilbert_label": rc.distilbert_signal.predicted_label if rc.distilbert_signal else None,
        "llm_label": rc.llm_signal.predicted_label if rc.llm_signal else None,
    }


def extract_l5_forecast(state: GlobalState) -> Optional[dict]:
    """Pull the L5 forecast's expected demand drop, or None if L5 was skipped."""
    fr = state.forecast_result
    if fr is None:
        return None
    return {"expected_drop_pct": fr.expected_drop_pct}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_node_extractors.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/trulens_integration/node_extractors.py tests/test_trulens_node_extractors.py
git commit -m "feat: add GlobalState node metric extractors"
```

---

## Task 6: Custom feedback functions

**Files:**
- Create: `src/evaluation/trulens_integration/feedback_functions.py`
- Test: `tests/test_trulens_feedback_functions.py`

**Interfaces:**
- Consumes: nothing beyond stdlib (pure functions over plain Python values, so they're testable without TruLens, SQLite, or GlobalState).
- Produces:
  ```python
  def risk_score_stability(composite_scores: list[float]) -> float: ...
  def ensemble_agreement(label_triples: list[tuple[str, str, str]]) -> float: ...
  def node_latency_check(latencies_ms: dict[str, float]) -> float: ...
  def forecast_accuracy(predicted_drop_pct: float, actual_drop_pct: float) -> float: ...
  ```
  All four return a float in `[0.0, 1.0]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_feedback_functions.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.feedback_functions import (
    ensemble_agreement, forecast_accuracy, node_latency_check, risk_score_stability,
)


def test_risk_score_stability_is_1_when_scores_identical():
    assert risk_score_stability([0.5, 0.5, 0.5]) == 1.0


def test_risk_score_stability_drops_with_high_variance():
    score = risk_score_stability([0.1, 0.9, 0.2, 0.8])
    assert 0.0 <= score < 0.7


def test_risk_score_stability_handles_fewer_than_two_scores():
    assert risk_score_stability([]) == 1.0
    assert risk_score_stability([0.5]) == 1.0


def test_ensemble_agreement_all_agree():
    triples = [("HIGH", "HIGH", "HIGH"), ("LOW", "LOW", "LOW")]
    assert ensemble_agreement(triples) == 1.0


def test_ensemble_agreement_majority_counts_as_agreement():
    # 2 of 3 match -> counted as agreement
    triples = [("HIGH", "HIGH", "MEDIUM")]
    assert ensemble_agreement(triples) == 1.0


def test_ensemble_agreement_no_majority_counts_as_disagreement():
    triples = [("HIGH", "MEDIUM", "LOW")]
    assert ensemble_agreement(triples) == 0.0


def test_ensemble_agreement_empty_list_is_perfect_by_convention():
    assert ensemble_agreement([]) == 1.0


def test_node_latency_check_all_pass():
    latencies = {"L2": 1000.0, "L3": 1500.0, "L4": 4000.0, "total": 10000.0}
    assert node_latency_check(latencies) == 1.0


def test_node_latency_check_partial_pass():
    latencies = {"L2": 3000.0, "L3": 1500.0, "L4": 4000.0, "total": 10000.0}
    score = node_latency_check(latencies)
    assert 0.0 < score < 1.0


def test_node_latency_check_empty_dict_is_perfect_by_convention():
    assert node_latency_check({}) == 1.0


def test_forecast_accuracy_perfect_prediction():
    assert forecast_accuracy(predicted_drop_pct=10.0, actual_drop_pct=10.0) == 1.0


def test_forecast_accuracy_worst_case_clips_to_zero():
    score = forecast_accuracy(predicted_drop_pct=0.0, actual_drop_pct=100.0)
    assert score == 0.0

def test_forecast_accuracy_both_zero_is_perfect_by_convention():
    assert forecast_accuracy(predicted_drop_pct=0.0, actual_drop_pct=0.0) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_feedback_functions.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration.feedback_functions'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/trulens_integration/feedback_functions.py
"""
feedback_functions.py — Four domain-specific TruLens metrics for the L1-L7
pipeline. Pure functions over plain values (no TruLens/SQLite dependency)
so they're unit-testable in isolation; wrapper.py and cli.py are
responsible for sourcing the historical data these take as input and for
recording the resulting score into TruLens.

Targets (from docs/specs/2026-07-06-trulens-integration-design.md):
  risk_score_stability: coefficient of variation across last 30 runs, target CV < 0.30
  ensemble_agreement:   fraction of runs where >=2 of 3 signals agree, target > 0.66
  node_latency_check:   binary pass/fail per node vs. threshold, target: all pass
  forecast_accuracy:    1 - relative forecast error, target > 0.80
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

# node label -> latency threshold in ms (from the spec's "Custom Feedback Functions" table)
_LATENCY_THRESHOLDS_MS: dict[str, float] = {
    "L2": 2000.0,
    "L3": 2000.0,
    "L4": 5000.0,
    "total": 15000.0,
}


def risk_score_stability(composite_scores: list[float]) -> float:
    """1.0 - coefficient_of_variation, clipped to [0, 1]. CV < 0.30 => score > 0.70."""
    if len(composite_scores) < 2:
        return 1.0
    avg = mean(composite_scores)
    if avg == 0:
        return 1.0
    cv = pstdev(composite_scores) / avg
    return max(0.0, min(1.0, 1.0 - cv))


def ensemble_agreement(label_triples: list[tuple[str, str, str]]) -> float:
    """Fraction of (rule, distilbert, llm) triples where >=2 labels match."""
    if not label_triples:
        return 1.0
    agreements = 0
    for triple in label_triples:
        counts = Counter(triple)
        if counts.most_common(1)[0][1] >= 2:
            agreements += 1
    return agreements / len(label_triples)


def node_latency_check(latencies_ms: dict[str, float]) -> float:
    """Fraction of measured nodes whose latency is within threshold. 1.0 = all pass."""
    checked = {k: v for k, v in latencies_ms.items() if k in _LATENCY_THRESHOLDS_MS}
    if not checked:
        return 1.0
    passing = sum(1 for label, ms in checked.items() if ms <= _LATENCY_THRESHOLDS_MS[label])
    return passing / len(checked)


def forecast_accuracy(predicted_drop_pct: float, actual_drop_pct: float) -> float:
    """1 - abs(predicted - actual) / max(predicted, actual), clipped to [0, 1]."""
    denom = max(predicted_drop_pct, actual_drop_pct)
    if denom == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(predicted_drop_pct - actual_drop_pct) / denom))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_feedback_functions.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/trulens_integration/feedback_functions.py tests/test_trulens_feedback_functions.py
git commit -m "feat: add 4 custom TruLens feedback functions"
```

---

## Task 7: TruLens session config + dashboard launcher

**Files:**
- Create: `src/evaluation/trulens_integration/config.py`
- Test: `tests/test_trulens_config.py`

**Interfaces:**
- Consumes: `TruSession`, `run_dashboard` from the packages pinned in Task 1.
- Produces: `get_session() -> TruSession` (cached singleton, mirrors the `@lru_cache` pattern in `src/utils/openai_utils.py:56` and `src/utils/observability.py:44`), `launch_dashboard(port: int = 8502) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_config.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.config import get_session, launch_dashboard


def test_get_session_returns_same_instance_on_repeated_calls():
    s1 = get_session()
    s2 = get_session()
    assert s1 is s2


def test_get_session_uses_data_trulens_sqlite_path():
    session = get_session()
    assert "data/trulens/trulens.db" in str(session.connector.db.engine.url)


def test_launch_dashboard_calls_run_dashboard_with_port_8502():
    with patch("src.evaluation.trulens_integration.config.run_dashboard") as mock_run:
        launch_dashboard()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("port") == 8502
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_config.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration.config'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/trulens_integration/config.py
"""
config.py — TruLens session + dashboard initialization.

Uses the current (2.8.x) TruSession API — NOT the deprecated Tru() class
from trulens-eval (removed from maintenance 2025-12-01). If TruSession's
constructor signature has changed since this was written, this is the one
file to check against `python3 -c "import trulens.core; help(trulens.core.TruSession)"`
for the currently-installed version before trusting the code below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from trulens.core import TruSession
from trulens.dashboard import run_dashboard

DB_PATH = Path("data/trulens/trulens.db")


@lru_cache(maxsize=1)
def get_session() -> TruSession:
    """Process-lifetime TruSession backed by SQLite at data/trulens/trulens.db."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return TruSession(database_url=f"sqlite:///{DB_PATH}")


def launch_dashboard(port: int = 8502) -> None:
    """Launch the TruLens Streamlit dashboard as its own process on `port`."""
    run_dashboard(get_session(), port=port)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_config.py -v`
Expected: 3 passed. If `test_get_session_uses_data_trulens_sqlite_path` fails because `session.connector.db.engine.url` isn't the right attribute path on the installed version, inspect `get_session().connector.__dict__` interactively and adjust both the assertion and, if needed, this note — this is the second (and last) place in the plan touching TruLens-version-specific internals.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/trulens_integration/config.py tests/test_trulens_config.py
git commit -m "feat: add TruSession config and dashboard launcher"
```

---

## Task 8: Pipeline wrapper (`run_with_trulens`)

**Files:**
- Create: `src/evaluation/trulens_integration/wrapper.py`
- Test: `tests/test_trulens_wrapper.py`

**Interfaces:**
- Consumes: `build_agent_graph` (`src/agents/langgraph_engine.py:137`), `GlobalState` (`src/agents/state.py`), `patch_openai_calls`/`LLMCallRecord` (Task 4), `NODE_LATENCY_LABELS`/`extract_l4_signals`/`extract_l5_forecast` (Task 5), `get_session` (Task 7).
- Produces: `run_with_trulens(payload: dict) -> GlobalState` — drop-in replacement for `run_agent_graph(payload)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_wrapper.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import GlobalState
from src.evaluation.trulens_integration.wrapper import run_with_trulens


class _FakeCompiledGraph:
    """Stands in for build_agent_graph(payload)'s return value."""

    def stream(self, initial_state, stream_mode):
        assert stream_mode == "updates"
        assert isinstance(initial_state, GlobalState)
        assert initial_state.run_id is not None
        yield {"l1_data_ingestion": {"agent_logs": ["L1: ok"]}}
        yield {"l2_news_analysis": {"agent_logs": ["L1: ok", "L2: ok"]}}
        yield {"l7_mitigation": {"agent_logs": ["L1: ok", "L2: ok", "L7: ok"]}}


def test_run_with_trulens_returns_merged_final_state():
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        result = run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert isinstance(result, GlobalState)
    assert result.agent_logs == ["L1: ok", "L2: ok", "L7: ok"]
    assert result.run_id is not None


def test_run_with_trulens_mints_run_id_when_absent_from_payload():
    captured_state = {}

    class _CapturingGraph(_FakeCompiledGraph):
        def stream(self, initial_state, stream_mode):
            captured_state["run_id"] = initial_state.run_id
            return super().stream(initial_state, stream_mode)

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_CapturingGraph(),
    ):
        run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert captured_state["run_id"]


def test_run_with_trulens_uses_run_id_from_payload_when_present():
    class _CapturingGraph(_FakeCompiledGraph):
        captured = {}

        def stream(self, initial_state, stream_mode):
            self.captured["run_id"] = initial_state.run_id
            return super().stream(initial_state, stream_mode)

    graph = _CapturingGraph()
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=graph,
    ):
        run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03", "run_id": "fixed-id-123"})

    assert graph.captured["run_id"] == "fixed-id-123"


def test_run_with_trulens_records_per_node_latency():
    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_FakeCompiledGraph(),
    ):
        with patch(
            "src.evaluation.trulens_integration.wrapper._record_pipeline_run"
        ) as mock_record:
            run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})

    assert mock_record.called
    _, kwargs = mock_record.call_args
    latencies = kwargs["node_latencies_ms"]
    assert set(latencies.keys()) == {"l1_data_ingestion", "l2_news_analysis", "l7_mitigation"}
    assert all(v >= 0.0 for v in latencies.values())


def test_run_with_trulens_lets_node_exceptions_propagate():
    class _RaisingGraph:
        def stream(self, initial_state, stream_mode):
            yield {"l1_data_ingestion": {"agent_logs": ["L1: ok"]}}
            raise ValueError("Risk label is required for mitigation")

    with patch(
        "src.evaluation.trulens_integration.wrapper.build_agent_graph",
        return_value=_RaisingGraph(),
    ):
        try:
            run_with_trulens({"sku": "CHIP_AP", "event_date": "2024-04-03"})
            assert False, "expected ValueError to propagate"
        except ValueError as exc:
            assert "Risk label is required" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_wrapper.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration.wrapper'`

- [ ] **Step 3: Write the implementation**

```python
# src/evaluation/trulens_integration/wrapper.py
"""
wrapper.py — run_with_trulens(): drop-in replacement for
langgraph_engine.run_agent_graph() that adds TruLens tracing without
touching langgraph_engine.py or any file under src/agents/.

Per-node latency comes from LangGraph's native app.stream(stream_mode=
"updates") rather than monkey-patching langgraph_engine internals — a node
exception still propagates out of the stream iterator exactly as
app.invoke() would raise it (see docs/specs/2026-07-06-trulens-integration-
design.md, "Non-Interference Guarantees"). Only telemetry capture around
each yielded update is wrapped in its own try/except.

run_agent_graph() itself calls app.invoke(GlobalState()) with no run_id, so
production graph runs have run_id=None throughout and their llm_call_log
rows are unattributable. This wrapper mints (or reuses payload["run_id"])
and seeds GlobalState(run_id=...) so the OpenAI patch's SQLite lookups in
openai_patch.py work correctly.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from src.agents.langgraph_engine import build_agent_graph
from src.agents.state import GlobalState
from src.evaluation.trulens_integration.node_extractors import (
    NODE_LATENCY_LABELS, extract_l4_signals, extract_l5_forecast,
)
from src.evaluation.trulens_integration.openai_patch import LLMCallRecord, patch_openai_calls

logger = logging.getLogger(__name__)


def _record_pipeline_run(
    *,
    run_id: str,
    node_latencies_ms: Dict[str, float],
    llm_calls: list[LLMCallRecord],
    l4_signals: Dict[str, Any] | None,
    l5_forecast: Dict[str, Any] | None,
) -> None:
    """Best-effort: write this run's telemetry into the TruLens session.

    Isolated from the streaming/merge loop in run_with_trulens() so a
    TruLens-side failure here can never affect the returned GlobalState.
    """
    try:
        from src.evaluation.trulens_integration.config import get_session

        session = get_session()
        session.connector.db.insert_record(  # placeholder call shape — see Step 3 note below
            run_id=run_id,
            node_latencies_ms=node_latencies_ms,
            llm_call_count=len(llm_calls),
            l4_signals=l4_signals,
            l5_forecast=l5_forecast,
        )
    except Exception as exc:
        logger.warning("TruLens record write failed (non-blocking) for run_id=%s: %s", run_id, exc)


def run_with_trulens(payload: Dict[str, Any]) -> GlobalState:
    """Drop-in replacement for run_agent_graph(payload) with TruLens tracing."""
    run_id = payload.get("run_id") or str(uuid.uuid4())
    app = build_agent_graph(payload)

    node_latencies_ms: Dict[str, float] = {}
    llm_calls: list[LLMCallRecord] = []

    with patch_openai_calls(on_call=llm_calls.append):
        state = GlobalState(run_id=run_id)
        node_start = time.monotonic()
        for update in app.stream(GlobalState(run_id=run_id), stream_mode="updates"):
            for node_name, delta in update.items():
                elapsed_ms = (time.monotonic() - node_start) * 1000.0
                try:
                    node_latencies_ms[node_name] = elapsed_ms
                except Exception as exc:
                    logger.warning("Latency capture failed for %s (non-blocking): %s", node_name, exc)
                state = state.model_copy(update=delta)
                node_start = time.monotonic()

    labeled_latencies = {
        NODE_LATENCY_LABELS.get(name, name): ms for name, ms in node_latencies_ms.items()
    }
    labeled_latencies["total"] = sum(node_latencies_ms.values())

    _record_pipeline_run(
        run_id=run_id,
        node_latencies_ms=labeled_latencies,
        llm_calls=llm_calls,
        l4_signals=extract_l4_signals(state),
        l5_forecast=extract_l5_forecast(state),
    )

    return state
```

**Note for the implementer:** `_record_pipeline_run`'s call to `session.connector.db.insert_record(...)` is a placeholder call shape, not a verified TruLens API — this is intentional and different from every other piece of code in this plan. TruLens's low-level API for writing an arbitrary, non-`Feedback`-bound telemetry record (as opposed to recording a `Feedback` score against a `Record` produced by a `TruApp`-wrapped call) was not confirmed against the installed 2.8.x package during planning. Before this step is considered done: run `python3 -c "from trulens.core import TruSession; help(TruSession)"` (and `help(TruSession().connector)` if that resolves) against the version installed in Task 1, find the actual supported way to persist a custom record/event, and replace this call accordingly. Do not mark this task's tests as sufficient proof this works end-to-end with a real TruLens dashboard — that's what Task 13's manual checklist is for.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_wrapper.py -v`
Expected: 5 passed (these tests patch `_record_pipeline_run` directly or don't require it to succeed against a real TruLens session, so they pass regardless of the placeholder noted above)

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/trulens_integration/wrapper.py tests/test_trulens_wrapper.py
git commit -m "feat: add run_with_trulens pipeline wrapper with streamed per-node latency"
```

---

## Task 9: CLI + package exports

**Files:**
- Create: `src/evaluation/trulens_integration/cli.py`
- Modify: `src/evaluation/trulens_integration/__init__.py`
- Test: `tests/test_trulens_cli.py`

**Interfaces:**
- Consumes: `run_with_trulens` (Task 8), `launch_dashboard` (Task 7), `risk_score_stability` (Task 6).
- Produces: module-level `main(argv: list[str] | None = None) -> int`, package exports `from src.evaluation.trulens_integration import run_with_trulens, launch_dashboard`, and `fetch_recent_composite_scores(days: int) -> list[float]` added to `src/utils/db_utils.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_cli.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.cli import main


def test_run_command_calls_run_with_trulens_with_scenario_payload():
    with patch("src.evaluation.trulens_integration.cli.run_with_trulens") as mock_run:
        mock_run.return_value = MagicMock(risk_label="HIGH")
        exit_code = main([
            "run", "--port", "Chennai", "--sku", "CHIP-001", "--event-date", "2024-03-15",
        ])

    assert exit_code == 0
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    payload = kwargs["payload"] if "payload" in kwargs else mock_run.call_args[0][0]
    assert payload["affected_port"] == "Chennai"
    assert payload["sku"] == "CHIP-001"
    assert payload["event_date"] == "2024-03-15"


def test_dashboard_command_calls_launch_dashboard():
    with patch("src.evaluation.trulens_integration.cli.launch_dashboard") as mock_launch:
        exit_code = main(["dashboard"])

    assert exit_code == 0
    mock_launch.assert_called_once_with(port=8502)


def test_no_command_prints_help_and_returns_nonzero():
    exit_code = main([])
    assert exit_code != 0


def test_query_command_prints_risk_drift_score():
    with patch(
        "src.evaluation.trulens_integration.cli.fetch_recent_composite_scores",
        return_value=[0.5, 0.5, 0.5],
    ) as mock_fetch:
        exit_code = main(["query", "--metric", "risk_drift", "--days", "14"])

    assert exit_code == 0
    mock_fetch.assert_called_once_with(14)


def test_query_command_rejects_unknown_metric():
    exit_code = main(["query", "--metric", "not_a_real_metric", "--days", "30"])
    assert exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_cli.py -v`
Expected: `ModuleNotFoundError: No module named 'src.evaluation.trulens_integration.cli'`

- [ ] **Step 3: Add `fetch_recent_composite_scores` to `db_utils.py`**

Add to `src/utils/db_utils.py`, directly below `fetch_latest_llm_call_log` (added in Task 3):

```python
def fetch_recent_composite_scores(days: int) -> List[float]:
    """Return composite_score values from risk_classifications in the last `days` days."""
    ensure_schema()
    rows = execute_query(
        """
        SELECT composite_score FROM risk_classifications
        WHERE run_ts >= datetime('now', ?)
        ORDER BY run_ts DESC
        """,
        (f"-{int(days)} days",),
    )
    return [row["composite_score"] for row in rows]
```

- [ ] **Step 4: Write the CLI implementation**

```python
# src/evaluation/trulens_integration/cli.py
"""
cli.py — python -m src.evaluation.trulens_integration.cli {run,dashboard,query}
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from src.evaluation.trulens_integration.config import launch_dashboard
from src.evaluation.trulens_integration.feedback_functions import risk_score_stability
from src.evaluation.trulens_integration.wrapper import run_with_trulens
from src.utils.db_utils import fetch_recent_composite_scores

_SUPPORTED_METRICS = {"risk_drift"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trulens_integration")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run one scenario with TruLens instrumentation")
    run_parser.add_argument("--port", required=True)
    run_parser.add_argument("--sku", required=True)
    run_parser.add_argument("--event-date", required=True)

    sub.add_parser("dashboard", help="Launch the TruLens dashboard on port 8502")

    query_parser = sub.add_parser("query", help="Query historical TruLens metrics")
    query_parser.add_argument("--metric", required=True)
    query_parser.add_argument("--days", type=int, default=30)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        payload = {
            "affected_port": args.port,
            "sku": args.sku,
            "event_date": args.event_date,
        }
        result = run_with_trulens(payload)
        print(f"risk_label={result.risk_label}")
        return 0

    if args.command == "dashboard":
        launch_dashboard(port=8502)
        return 0

    if args.command == "query":
        if args.metric not in _SUPPORTED_METRICS:
            print(f"Unknown metric '{args.metric}'. Supported: {sorted(_SUPPORTED_METRICS)}")
            return 1
        scores = fetch_recent_composite_scores(args.days)
        stability = risk_score_stability(scores)
        print(f"metric=risk_drift days={args.days} n_runs={len(scores)} stability_score={stability:.3f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

```python
# src/evaluation/trulens_integration/__init__.py
from src.evaluation.trulens_integration.config import launch_dashboard
from src.evaluation.trulens_integration.wrapper import run_with_trulens

__all__ = ["run_with_trulens", "launch_dashboard"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_cli.py -v`
Expected: 5 passed. If `test_run_command_calls_run_with_trulens_with_scenario_payload` fails on how `payload` is extracted from `mock_run.call_args`, adjust `main()`'s call to `run_with_trulens` to pass `payload` as a plain positional dict (`run_with_trulens(payload)`), matching Task 8's signature, and fix the test's extraction to `mock_run.call_args[0][0]` only.

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/trulens_integration/cli.py src/evaluation/trulens_integration/__init__.py tests/test_trulens_cli.py
git commit -m "feat: add TruLens CLI (run/dashboard/query) and package exports"
```

---

## Task 10: RAGAS tracer patch-registry integration

**Files:**
- Modify: `evaluation/ragas/rag_tracer.py`
- Test: `tests/test_ragas_trulens_patch_coexistence.py`

**Interfaces:**
- Consumes: `claim_patch`/`release_patch` (Task 2), existing `RAGTraceCollector.__enter__`/`__exit__` (`evaluation/ragas/rag_tracer.py:138-171`).
- Produces: `RAGTraceCollector` now records which of its `PATCH_SPECS` targets it actually acquired (`self._claimed: set[str]`), so `__exit__` only restores originals for targets it holds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragas_trulens_patch_coexistence.py
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
        # retrieval-only targets are unaffected — they're not patch-registry-gated
        assert "retrieve_and_rerank" in collector._claimed

    release_patch("call_openai_structured", "trulens")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ragas_trulens_patch_coexistence.py -v`
Expected: `AttributeError: 'RAGTraceCollector' object has no attribute '_claimed'`

- [ ] **Step 3: Modify `evaluation/ragas/rag_tracer.py`**

Add the import near the top (after the existing stdlib imports, before `logger = ...`):

```python
from src.evaluation.patch_registry import claim_patch, release_patch
```

Add a registry-gated target set right after `PATCH_SPECS` (line 51):

```python
# Targets that must be coordinated with other patchers (currently only
# TruLens — see docs/specs/2026-07-06-trulens-integration-design.md,
# "RAGAS Coexistence"). Retrieval targets aren't patched by anything else,
# so they're always claimed unconditionally.
_REGISTRY_GATED_ATTRS = {"call_openai_structured"}
```

Replace `__init__` (lines 129-134) to add `self._claimed`:

```python
    def __init__(self, integration_point: str, traces_dir: Path = TRACES_DIR):
        self.integration_point = integration_point
        self.traces_dir = traces_dir
        self.records: List[dict] = []
        self._pending: Optional[dict] = None
        self._patches: List[tuple] = []  # (module, attr, original)
        self._claimed: set[str] = set()
```

Replace `__enter__` (lines 138-157) to check the registry before patching a gated attribute:

```python
    def __enter__(self) -> "RAGTraceCollector":
        for module_path, attr, kind in PATCH_SPECS:
            if attr in _REGISTRY_GATED_ATTRS:
                if not claim_patch(attr, "ragas"):
                    logger.warning(
                        "Skipping patch of %s — already claimed by another tracer", attr
                    )
                    continue
                self._claimed.add(attr)

            module = importlib.import_module(module_path)
            original = getattr(module, attr)
            wrapper = self._make_wrapper(original, attr, kind)
            # Patch the canonical module AND every already-imported module
            # holding its own reference to the same function object (e.g.
            # `from src.rag.retriever import retrieve_and_rerank` inside an
            # agent module) — otherwise that agent would keep calling the
            # unpatched original and never get traced.
            for mod in list(sys.modules.values()):
                if mod is None:
                    continue
                try:
                    if getattr(mod, attr, None) is original:
                        setattr(mod, attr, wrapper)
                        self._patches.append((mod, attr, original))
                except Exception:
                    continue
        return self
```

Replace `__exit__` (lines 159-170) to release claimed targets:

```python
    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self._finalize_pending()
        finally:
            # ALWAYS restore originals, even if the wrapped code raised.
            for module, attr, original in self._patches:
                try:
                    setattr(module, attr, original)
                except Exception:
                    logger.error("Failed to restore %s.%s", module, attr)
            self._patches.clear()
            for attr in self._claimed:
                release_patch(attr, "ragas")
            self._claimed.clear()
        return False  # re-raise any exception after restore
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_ragas_trulens_patch_coexistence.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the existing RAGAS smoke test to confirm no regression**

Run: `python3 -m evaluation.ragas.rag_tracer --smoke`
Expected: same output shape as before this change (gold-chunk hit rate summary) — confirms the registry gating didn't break the existing retrieval-tracing behavior.

- [ ] **Step 6: Commit**

```bash
git add evaluation/ragas/rag_tracer.py tests/test_ragas_trulens_patch_coexistence.py
git commit -m "feat: gate RAGTraceCollector's call_openai_structured patch behind patch_registry"
```

---

## Task 11: Batch evaluation runner

**Files:**
- Create: `evaluation/trulens_runner.py`
- Test: `tests/test_trulens_runner.py`

**Interfaces:**
- Consumes: `run_with_trulens` (Task 8).
- Produces: `SCENARIOS: list[dict]`, `main() -> int` writing `evaluation/trulens_scores.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trulens_runner.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.trulens_runner import SCENARIOS, main


def test_scenarios_have_required_payload_keys():
    required = {"disruption_type", "affected_port", "affected_route", "severity",
                "shock_duration_days", "recovery_window_days", "synthetic_ratio",
                "simulation_trials", "sku", "event_date"}
    assert len(SCENARIOS) >= 2
    for scenario in SCENARIOS:
        assert required.issubset(scenario["payload"].keys())


def test_main_writes_trulens_scores_json(tmp_path):
    output_path = tmp_path / "trulens_scores.json"
    fake_result = MagicMock()
    fake_result.risk_label = "CRITICAL"
    fake_result.risk_score_composite = 0.9

    with patch("evaluation.trulens_runner.run_with_trulens", return_value=fake_result):
        with patch("evaluation.trulens_runner.OUTPUT_PATH", output_path):
            exit_code = main()

    assert exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert len(data) == len(SCENARIOS)
    assert data[0]["risk_label"] == "CRITICAL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_trulens_runner.py -v`
Expected: `ModuleNotFoundError: No module named 'evaluation.trulens_runner'`

- [ ] **Step 3: Write the implementation**

`evaluation/ragas/test_dataset.json` holds RAG gold Q&A pairs (`question`/`ground_truth`/`source_collection`), not `port`/`sku`/`event_date` pipeline scenarios, so it can't drive `run_with_trulens`. These two scenarios mirror the real payload shape used by the Streamlit Scenario Analyzer at `src/dashboard/dashboard.py:216-229`.

```python
# evaluation/trulens_runner.py
"""
trulens_runner.py — Runs a small set of pipeline scenarios through
run_with_trulens() and writes their risk outcomes to
evaluation/trulens_scores.json.

Usage: python -m evaluation.trulens_runner
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.wrapper import run_with_trulens

OUTPUT_PATH = Path(__file__).parent / "trulens_scores.json"

SCENARIOS: list[dict] = [
    {
        "name": "taiwan_earthquake",
        "payload": {
            "disruption_type": "earthquake",
            "affected_port": "Eastern Asia",
            "affected_route": "Hsinchu to Singapore",
            "severity": 0.95,
            "shock_duration_days": 6,
            "recovery_window_days": 90,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "CHIP_AP",
            "event_date": "2024-04-03",
        },
    },
    {
        "name": "red_sea_crisis",
        "payload": {
            "disruption_type": "geopolitical",
            "affected_port": "Western Europe",
            "affected_route": "Suez Canal to Rotterdam",
            "severity": 0.85,
            "shock_duration_days": 14,
            "recovery_window_days": 120,
            "synthetic_ratio": 0.0,
            "simulation_trials": 500,
            "sku": "ELECTRONICS_EU",
            "event_date": "2024-01-15",
        },
    },
]


def main() -> int:
    results = []
    for scenario in SCENARIOS:
        state = run_with_trulens(scenario["payload"])
        results.append({
            "scenario": scenario["name"],
            "risk_label": state.risk_label,
            "composite_score": state.risk_score_composite,
        })

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} scenario result(s) to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_trulens_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add evaluation/trulens_runner.py tests/test_trulens_runner.py
git commit -m "feat: add TruLens batch scenario runner"
```

---

## Task 12: Full test suite regression check

**Files:** none (verification-only task)

- [ ] **Step 1: Run the entire existing test suite to confirm no regressions**

Run:
```bash
python3 -m pytest tests/ -v
```
Expected: every previously-passing test still passes, plus all TruLens tests added in Tasks 2-11. Pay particular attention to `tests/test_observability.py`, `tests/test_llm_agents.py`, `tests/test_risk_classifier_agent.py`, and `tests/test_news_weather_agents_v2.py` — these exercise `call_openai_structured` call sites directly and are the fastest signal if `openai_patch.py`'s restore-on-exit (Task 4) ever leaves a module permanently patched.

- [ ] **Step 2: Run the RAGAS QA scripts to confirm the rag_tracer.py edit didn't break them**

Run:
```bash
python -m evaluation.qa_09_l2_l3_real_ingest_smoke
python -m evaluation.qa_11_langgraph_l1_l2_l3_integration
```
Expected: both scripts report PASS on all assertions, matching their pre-existing behavior (these don't call `RAGTraceCollector` directly but exercise the same `src.agents` import graph that Task 4's patch touches).

- [ ] **Step 3: No commit** — this task only verifies Tasks 1-11; nothing to stage.

---

## Task 13: Manual dashboard verification (not automatable)

**Files:** none

This task requires `OPENAI_API_KEY` set (real LLM calls) and is run by hand, matching the spec's Manual Verification Checklist.

- [ ] **Step 1: Run one instrumented scenario**

```bash
python -m src.evaluation.trulens_integration.cli run --port Chennai --sku CHIP-001 --event-date 2024-03-15
```
Expected: prints `risk_label=<LOW|MEDIUM|HIGH|CRITICAL>` with no traceback.

- [ ] **Step 2: Cross-check token counts against the existing `[LLM]` log line**

The console output from Step 1 includes lines like `[LLM] tool=NewsAnalysisLLMOutput model=gpt-4.1-mini in=<N> out=<M> latency=<S>s` (from `src/utils/openai_utils.py:134-137`, unaffected by this integration). Separately query:
```bash
python3 -c "
from src.utils.db_utils import execute_query
rows = execute_query('SELECT agent_name, input_tokens, output_tokens FROM llm_call_log ORDER BY id DESC LIMIT 5')
for r in rows: print(dict(r))
"
```
Expected: the `input_tokens`/`output_tokens` in these rows match the `in=`/`out=` values from the `[LLM]` console lines for the same run. If they don't match, the patch in Task 4 is not correctly intercepting the real call sites — this is the single most important manual check, since it's the one thing the unit tests (which use fakes) can't prove.

- [ ] **Step 3: Launch the dashboard**

```bash
python -m src.evaluation.trulens_integration.cli dashboard
```
Then open `http://localhost:8502` in a browser. Expected: dashboard loads without error; the Streamlit app itself (if also running via `python -m streamlit run src/main.py`) remains reachable at `http://localhost:8501` simultaneously.

- [ ] **Step 4: Verify the L7/mitigation non-interference invariant**

```bash
python3 -c "
from src.evaluation.trulens_integration.wrapper import run_with_trulens
try:
    run_with_trulens({'affected_port': 'Chennai', 'sku': 'DOES_NOT_EXIST', 'event_date': '1999-01-01'})
except Exception as exc:
    print(f'Exception propagated as expected: {type(exc).__name__}: {exc}')
else:
    print('FAIL: expected an exception to propagate from an invalid scenario')
"
```
Expected: `Exception propagated as expected: ...` — confirms a real pipeline failure isn't silently swallowed by the wrapper's telemetry capture.

- [ ] **Step 5: Run the batch scenario runner**

```bash
python -m evaluation.trulens_runner
cat evaluation/trulens_scores.json
```
Expected: valid JSON with 2 entries (`taiwan_earthquake`, `red_sea_crisis`), each with a `risk_label` and `composite_score`.

- [ ] **Step 6: No commit** — this is a verification pass; report results back before proceeding to any Phase 2 (guardrails) work.
