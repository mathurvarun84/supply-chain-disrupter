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
