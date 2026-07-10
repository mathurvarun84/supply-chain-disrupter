"""
openai_utils.py — Shared OpenAI structured-output helpers for all LLM agents.

Provides:
  - call_openai_structured()  — Pydantic-validated responses via beta.parse API
  - build_rag_context()       — ChromaDB query + dedup + formatted chunk string
  - format_sqlite_record()    — labeled two-column block for LLM user messages
  - format_semiconductor_signals() — year-grouped semiconductor_signals rows
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar


def _load_project_env() -> None:
    """Load .env from project root so Streamlit and scripts pick up OPENAI_API_KEY."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


_load_project_env()

from openai import BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Model constants — do not substitute without updating agent assignments.
MODEL_FAST = "gpt-4.1-mini"
MODEL_REASONING = "gpt-4o"
TEMPERATURE = 0.0

T = TypeVar("T", bound=BaseModel)

# Keys always shown in format_sqlite_record even when value is None.
CRITICAL_COLUMNS = {
    "order_id", "order_date", "order_region", "delivery_status",
    "disruption_event_label", "risk_score_composite", "supply_disruption_index",
    "export_control_level", "disruption_news_count", "alternate_supplier_available",
    "lead_time_variance_days", "defect_rate_pct", "safety_stock_units",
    "stockout_probability_pct", "chip_price_index", "latitude", "longitude",
}


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Return a process-lifetime OpenAI client. Raises if OPENAI_API_KEY is unset."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add to .env.")
    from src.utils.hf_utils import insecure_ssl_enabled

    if insecure_ssl_enabled():
        import httpx

        return OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))
    return OpenAI(api_key=api_key)


def call_openai_structured(
    system_prompt: str,
    user_message: str,
    response_model: Type[T],
    model: str = MODEL_REASONING,
    max_tokens: int = 1024,
    *,
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    trace=None,
    span=None,
) -> T:
    """
    Call OpenAI structured output API and return a validated Pydantic instance.

    Uses client.beta.chat.completions.parse() exclusively.
    temperature is always 0.0 (deterministic). Retries on RateLimitError only.

    run_id / agent_name / trace / span are optional keyword-only args for
    observability. All existing call sites continue to work without them.
    When provided, each call writes to llm_call_log (always) and a nested
    Langfuse generation under the owning agent span (best-effort).
    """
    retry_count_ref = [0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    def _call() -> T:
        client = _get_client()
        t0 = time.monotonic()
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format=response_model,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )
        except BadRequestError as exc:
            logger.error("OpenAI BadRequestError for model=%s: %s", model, exc)
            raise
        except RateLimitError as exc:
            logger.warning("OpenAI RateLimitError for model=%s — retrying: %s", model, exc)
            retry_count_ref[0] += 1
            raise

        elapsed = time.monotonic() - t0
        message = completion.choices[0].message
        if message.parsed is None:
            refusal = getattr(message, "refusal", None) or "unknown refusal"
            raise RuntimeError(f"OpenAI returned no parsed result: {refusal}")

        usage = completion.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        latency_ms = elapsed * 1000
        logger.info(
            "[LLM] tool=%s model=%s in=%d out=%d latency=%.2fs",
            response_model.__name__, model, in_tok, out_tok, elapsed,
        )

        if run_id and agent_name:
            try:
                from src.utils.observability import record_llm_generation
                record_llm_generation(
                    trace, span,
                    run_id=run_id, agent_name=agent_name, model=model,
                    system_prompt=system_prompt, user_message=user_message,
                    parsed_output=message.parsed,
                    input_tokens=in_tok, output_tokens=out_tok,
                    latency_ms=latency_ms, status="success",
                    retry_count=retry_count_ref[0],
                )
            except Exception as obs_exc:
                logger.warning("Observability record failed (non-blocking): %s", obs_exc)

        return message.parsed

    try:
        return _call()
    except Exception as exc:
        if run_id and agent_name:
            try:
                from src.utils.observability import record_llm_generation
                record_llm_generation(
                    trace, span,
                    run_id=run_id, agent_name=agent_name, model=model,
                    system_prompt=system_prompt, user_message=user_message,
                    parsed_output=None, input_tokens=0, output_tokens=0,
                    latency_ms=0.0, status="failed_fallback",
                    retry_count=retry_count_ref[0], error_message=str(exc),
                )
            except Exception as obs_exc:
                logger.warning("Observability failure record failed (non-blocking): %s", obs_exc)
        raise


def build_rag_context(
    queries: List[Optional[Tuple[str, int]]],
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Issue multiple ChromaDB queries, deduplicate hits, and format as numbered chunks.

    Each query item is (query_text, n_results) or None (skipped silently).
    Never raises — returns empty string on any exception.
    """
    try:
        from src.rag.utils import query_chroma_rag
    except ImportError:
        return ""

    try:
        seen: set = set()
        sections: List[str] = []
        chunk_idx = 0

        for item in queries:
            if item is None:
                continue
            query_text, n_results = item
            hits = query_chroma_rag(query_text, n_results=n_results)
            if not hits:
                sections.append(f"[No results for: {query_text[:60]}]")
                continue

            section_parts: List[str] = []
            for hit in hits:
                meta = hit.get("metadata", {})
                dedup_key = meta.get("source", "") + hit.get("text", "")[:40]
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                chunk_idx += 1
                source = meta.get("source", "unknown")
                doc_type = meta.get("type", "document")
                distance = hit.get("distance", 0.0)
                text = hit.get("text", "")[:450]
                section_parts.append(
                    f"[{chunk_idx}] Source: {source} ({doc_type}) | "
                    f"cosine-distance: {distance:.4f}\n    {text}"
                )
            if section_parts:
                sections.append("\n".join(section_parts))

        return separator.join(sections)
    except Exception as exc:
        logger.warning("build_rag_context failed (non-blocking): %s", exc)
        return ""


def format_sqlite_record(record: dict, table_name: str = "lite_master") -> str:
    """
    Format a SQLite record dict as a labeled two-column block for LLM context.

    Omits None-valued keys unless they are in CRITICAL_COLUMNS.
    """
    lines = [f"## SQLite Record ({table_name})"]
    for key, value in record.items():
        if value is None and key not in CRITICAL_COLUMNS:
            continue
        display = value if value is not None else "NULL"
        lines.append(f"  {key:<35}: {display}")
    return "\n".join(lines)


def format_semiconductor_signals(rows: list) -> str:
    """
    Format semiconductor_signals rows grouped by year for LLM context.

    Each row shows year, company, SDI, ECL, known event, and known severity.
    """
    if not rows:
        return "## Semiconductor Signals (from semiconductor_signals table)\n  (no rows found)"
    lines = ["## Semiconductor Signals (from semiconductor_signals table)"]
    for r in rows:
        row = dict(r) if not isinstance(r, dict) else r
        lines.append(
            f"  Year={row.get('year', '?')} | {row.get('company', '?')} "
            f"| SDI={float(row.get('supply_disruption_index') or 0):.2f} "
            f"| ECL={float(row.get('export_control_level') or 0):.2f} "
            f"| Known={row.get('known_disruption_event', '—')} "
            f"({row.get('known_severity', '—')})"
        )
    return "\n".join(lines)


def has_openai_api_key() -> bool:
    """Return True when OPENAI_API_KEY is set in the environment."""
    _load_project_env()
    return bool(os.getenv("OPENAI_API_KEY"))
