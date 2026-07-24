"""
Input, output, and execution guardrails for the L1-L7 supply-chain
disruption pipeline.

Three function families — validate_input_*, validate_output_*,
validate_execution_* — each returning a uniform GuardrailResult, plus one
log_guardrail_event() writer used by all three, so every checkpoint
produces one consistent guardrail_events row regardless of what it checks.

Design reference: Safety_and_Guardrails_Module_CapstoneP8.docx §3-§5.
The 14 functions in the two doc-named families cite their exact doc §3/§4
row in their docstring. The 2 execution guardrails (§2.4 in the build
prompt) were added after the design doc was written and cite this
conversation instead.

Ground-truth notes from the Step 1 codebase check (see session report):
  - "Label-enum enforcement" (doc §4) assumes predicted_label/final_label
    are already Pydantic-enum-restricted. They are not — RiskClassificationResult
    .final_label, LLMSignal.predicted_label, and JudgeVerdict.final_label are
    all plain `str` in src/agents/state.py. validate_output_label_enum()
    below is therefore real detection logic, not a logging-only wrapper.
  - "Locked-formula tamper check" (doc §4) assumes an LLM enhancement prompt
    echoes back composite_score/final_label for tamper-checking. No current
    call site does this (JudgeVerdict carries no composite_score field to
    echo). validate_output_locked_formula() is implemented and tested but
    has no production call site today — see session scope note.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, Iterable, Optional

from pydantic import BaseModel

from src.utils.db_utils import insert_guardrail_event

logger = logging.getLogger(__name__)


class GuardrailResult(BaseModel):
    passed: bool
    reason: str
    guardrail_name: str


def log_guardrail_event(
    agent_name: str,
    guardrail_name: str,
    direction: str,
    passed: bool,
    reason: str,
    record_id: Optional[str] = None,
) -> None:
    """Single writer for every guardrail checkpoint — input, output, or
    execution. Doc §5.1: 'so every checkpoint... produces one consistent
    row.' Delegates to db_utils.insert_guardrail_event() (parameterised
    INSERT) — never opens its own connection or builds SQL itself."""
    try:
        insert_guardrail_event(
            event_id=str(uuid.uuid4()),
            agent_name=agent_name,
            guardrail_name=guardrail_name,
            direction=direction,
            passed=passed,
            reason=reason,
            record_id=record_id,
        )
    except Exception as exc:
        # A guardrail's own logging must never break the pipeline it's protecting.
        logger.warning("log_guardrail_event failed (non-blocking): %s", exc)


# ---------------------------------------------------------------------------
# Input guardrails (doc §3)
# ---------------------------------------------------------------------------

def validate_input_schema(payload: dict, required_fields: Iterable[str]) -> GuardrailResult:
    """Doc §3 — Schema/type validation on ingest. Hook: L1, before writing
    to live_news_ingest / live_weather_ingest. Malformed Open-Meteo or RSS
    payloads must not propagate into L2/L3 classifiers."""
    missing = [f for f in required_fields if payload.get(f) is None]
    if missing:
        return GuardrailResult(
            passed=False,
            reason=f"Missing/null required field(s): {', '.join(missing)}",
            guardrail_name="schema_validation",
        )
    return GuardrailResult(passed=True, reason="All required fields present.", guardrail_name="schema_validation")


_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?previous\s+instructions",
    r"ignore\s+(all\s+|the\s+)?prior\s+(instructions|context)",
    r"disregard\s+(the\s+)?(above|previous|prior)",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s+prompt",
    r"\bclassify\s+(this\s+)?as\s+(low|medium|high|critical)\b",
    r"\bmark\s+(this\s+)?(as\s+)?critical\b",
    r"override\s+(the\s+)?(classification|verdict|label)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def validate_input_prompt_injection(text: str, source: str = "") -> GuardrailResult:
    """Doc §3 — Prompt-injection screen. Hook: before news text enters any
    LLM prompt (L2's build_news_signals input, L4 Signal 3's rag_context
    assembly). Flags instruction-like patterns embedded in scraped headlines
    (e.g. 'ignore previous instructions', 'classify as', imperative verbs
    directed at a classifier rather than describing an event). On failure:
    caller excludes the offending text from the prompt and continues —
    matches doc §7's expected behaviour exactly, this function only detects
    and logs."""
    match = _INJECTION_RE.search(text or "")
    if match:
        attempted = (text or "").strip()
        if len(attempted) > 200:
            attempted = attempted[:200] + "…"
        return GuardrailResult(
            passed=False,
            reason=f"Adversarial instruction-like pattern detected in {source or 'text'}: "
                   f"matched {match.group(0)!r} in attempted input {attempted!r}. "
                   f"Mitigation: content excluded from the LLM prompt; pipeline continued with "
                   f"sanitized/placeholder input instead of the injected text.",
            guardrail_name="prompt_injection_screen",
        )
    return GuardrailResult(passed=True, reason="No injection pattern detected.", guardrail_name="prompt_injection_screen")


def validate_input_length(text: str, max_chars: int = 8000) -> GuardrailResult:
    """Doc §3 — Input length/token capping. Hook: before build_rag_context()
    and format_sqlite_record() calls. Enforces the budget llm_call_log
    already tracks, rather than only observing it. Uses a character budget
    (not a tokenizer) to avoid a new dependency; max_chars is a conservative
    proxy for token count."""
    length = len(text or "")
    if length > max_chars:
        return GuardrailResult(
            passed=False,
            reason=f"Input length {length} exceeds cap of {max_chars} chars.",
            guardrail_name="input_length_cap",
        )
    return GuardrailResult(passed=True, reason=f"Input length {length} within cap.", guardrail_name="input_length_cap")


def validate_input_null_fields(record: dict, required_fields: Iterable[str]) -> GuardrailResult:
    """Doc §3 — Null/missing critical-field gate. Hook: before the L4
    rule-based composite calculation. If supply_disruption_index or
    export_control_level is null, caller routes to a deterministic fallback
    label instead of computing on incomplete data — this function only
    detects and logs, the fallback routing stays in the calling agent."""
    missing = [f for f in required_fields if record.get(f) is None]
    if missing:
        return GuardrailResult(
            passed=False,
            reason=f"Critical field(s) null: {', '.join(missing)}",
            guardrail_name="null_field_gate",
        )
    return GuardrailResult(passed=True, reason="All critical fields present.", guardrail_name="null_field_gate")


def validate_input_rate_limit(source_name: str, trip_counter: Dict[str, int], threshold: int = 3) -> GuardrailResult:
    """Doc §3 — Rate limit/circuit breaker. Hook: api_clients.py / L1
    ingestion, around the 20 parallel calls. Formalises the existing
    fallback paths with a trip counter, surfaced on the dashboard as 'N
    calls degraded to fallback.' trip_counter is caller-owned (per-run dict
    keyed by source_name) so this function stays stateless."""
    trip_counter[source_name] = trip_counter.get(source_name, 0) + 1
    count = trip_counter[source_name]
    if count > threshold:
        return GuardrailResult(
            passed=False,
            reason=f"{source_name} degraded to fallback {count} time(s) — exceeds threshold {threshold}.",
            guardrail_name="rate_limit_circuit_breaker",
        )
    return GuardrailResult(
        passed=True,
        reason=f"{source_name} fallback count {count} within threshold {threshold}.",
        guardrail_name="rate_limit_circuit_breaker",
    )


_SUSPICIOUS_SQL_RE = re.compile(r"%s|f\"|f'|\+\s*str\(|\{.*\}.*(select|insert|update|delete)", re.IGNORECASE)


def validate_input_sql_params(query: str, params: tuple) -> GuardrailResult:
    """Doc §3 — Parameterised-query guardrail. Hook: db_utils.execute_query().
    Asserts the query string contains no interpolated values (no f-string/%
    formatting) and params is always a tuple — closes the SQL-injection
    surface on any record_id/region input. Given the Step 1 codebase check
    already confirmed execute_query()/execute_non_query() are the sole write
    paths and use only '?' placeholders, this is a confirming assertion, not
    new defensive logic."""
    if not isinstance(params, tuple):
        return GuardrailResult(
            passed=False,
            reason=f"params must be a tuple, got {type(params).__name__}.",
            guardrail_name="parameterised_query_guardrail",
        )
    if _SUSPICIOUS_SQL_RE.search(query):
        return GuardrailResult(
            passed=False,
            reason="Query string contains string-formatting artifacts — possible SQL injection surface.",
            guardrail_name="parameterised_query_guardrail",
        )
    placeholder_count = query.count("?")
    if params and placeholder_count != len(params):
        return GuardrailResult(
            passed=False,
            reason=f"Placeholder count ({placeholder_count}) does not match params length ({len(params)}).",
            guardrail_name="parameterised_query_guardrail",
        )
    return GuardrailResult(passed=True, reason="Query is parameterised.", guardrail_name="parameterised_query_guardrail")


# ---------------------------------------------------------------------------
# Output guardrails (doc §4)
# ---------------------------------------------------------------------------

def validate_output_schema(result: BaseModel) -> GuardrailResult:
    """Doc §4 — Structured-output schema enforcement. Already implemented
    via Pydantic response_format on every call_openai_structured() call — a
    malformed completion never reaches here because the OpenAI SDK itself
    rejects it (call_openai_structured raises before returning). This
    function logs a pass event after the fact; 'passed' is True on every
    call this executes for."""
    return GuardrailResult(
        passed=isinstance(result, BaseModel),
        reason=f"{type(result).__name__} parsed and validated by response_format.",
        guardrail_name="structured_output_schema",
    )


def validate_output_hard_business_rule(final_label: str, critical_flag: bool) -> GuardrailResult:
    """Doc §4 — Hard business-rule override, the strongest demo example.
    Confirms critical_flag == (final_label == "CRITICAL") and logs the
    check. This WRAPS the existing enforcement — judge_agent.py's
    Shipping-canceled override forces final_label before this ever runs,
    and src/api/routers/risk.py / src/agents/pipeline_bridge.py both derive
    critical_flag/slack_should_fire from that already-corrected final_label
    server-side — it does not recompute or duplicate that logic."""
    expected = final_label == "CRITICAL"
    if critical_flag != expected:
        return GuardrailResult(
            passed=False,
            reason=f"critical_flag={critical_flag} disagrees with final_label={final_label!r} "
                   f"(expected {expected}).",
            guardrail_name="hard_business_rule_override",
        )
    return GuardrailResult(
        passed=True,
        reason=f"critical_flag={critical_flag} matches final_label={final_label!r}.",
        guardrail_name="hard_business_rule_override",
    )


def validate_output_numeric_bounds(value: Optional[float], field_name: str, lo: float = 0.0, hi: float = 1.0) -> GuardrailResult:
    """Doc §4 — Numeric bounds check. Hook: L4 signal outputs (DistilBERTSignal
    .confidence has no Pydantic bound today), L4 LLM enhancement.
    composite_score and confidence must lie in [lo,hi]; out-of-range values
    are clipped by the CALLER after this returns passed=False — this
    function detects and logs, doesn't mutate."""
    if value is None or not (lo <= value <= hi):
        return GuardrailResult(
            passed=False,
            reason=f"{field_name}={value} outside bounds [{lo}, {hi}].",
            guardrail_name="numeric_bounds_check",
        )
    return GuardrailResult(passed=True, reason=f"{field_name}={value} within [{lo}, {hi}].", guardrail_name="numeric_bounds_check")


def validate_output_citation_groundedness(rag_citations: Iterable[str], known_sources: Iterable[str]) -> GuardrailResult:
    """Doc §4 — Citation groundedness check. Hook: L4 Signal 3
    (run_llm_signal's rag_citations — no existing validation, this is real
    detection there), L7 mitigation synthesis (mitigation_agent.py's
    _validate_citations()/_extract_known_sources() already sanitize
    citations against known_sources — this call logs that outcome rather
    than re-deriving it). Every citation must match a source actually
    retrieved for that call — catches fabricated citations before they
    reach evaluator-facing rationale. Matching is case-insensitive substring,
    same rule mitigation_agent.py's own _validate_citations() uses."""
    citations = list(rag_citations)
    known_lower = [s.lower() for s in known_sources]
    fabricated = [c for c in citations if not any(src in c.lower() for src in known_lower)]
    if fabricated:
        return GuardrailResult(
            passed=False,
            reason=f"Fabricated citation(s) not in retrieved sources: {fabricated}",
            guardrail_name="citation_groundedness_check",
        )
    return GuardrailResult(
        passed=True,
        reason=f"All {len(citations)} citation(s) grounded in retrieved sources: {citations}.",
        guardrail_name="citation_groundedness_check",
    )


_VALID_LABELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def validate_output_label_enum(label: str) -> GuardrailResult:
    """Doc §4 — Label-enum enforcement. NOTE: the design doc assumes this is
    already implemented via a Pydantic enum on predicted_label/final_label.
    The Step 1 codebase check found this is NOT the case —
    RiskClassificationResult.final_label, LLMSignal.predicted_label, and
    JudgeVerdict.final_label are all plain `str` in src/agents/state.py.
    This function is therefore real detection logic, not a logging-only
    wrapper like validate_output_schema()."""
    if label not in _VALID_LABELS:
        return GuardrailResult(
            passed=False,
            reason=f"Label {label!r} not in allowed set {_VALID_LABELS}.",
            guardrail_name="label_enum_enforcement",
        )
    return GuardrailResult(passed=True, reason=f"Label {label!r} is valid.", guardrail_name="label_enum_enforcement")


def validate_output_locked_formula(
    echoed_value: Optional[float], source_value: float, field_name: str, tolerance: float = 1e-6
) -> GuardrailResult:
    """Doc §4 — Locked-formula tamper check. Hook: an L4 enhancement prompt
    that echoes composite_score/final_label back for tamper-checking. NOTE:
    the Step 1 codebase check found no current call site actually echoes a
    locked value this way — JudgeVerdict carries no composite_score field.
    This function is implemented and unit-tested per spec but has no
    production call site today; see session scope note. echoed_value is
    None-able for exactly that reason — a None echo is treated as
    'nothing to tamper-check', not a failure."""
    if echoed_value is None:
        return GuardrailResult(
            passed=True,
            reason=f"{field_name} not echoed by this call — nothing to tamper-check.",
            guardrail_name="locked_formula_tamper_check",
        )
    if abs(echoed_value - source_value) > tolerance:
        return GuardrailResult(
            passed=False,
            reason=f"Echoed {field_name}={echoed_value} diverges from locked value {source_value} "
                   f"by more than {tolerance}.",
            guardrail_name="locked_formula_tamper_check",
        )
    return GuardrailResult(
        passed=True,
        reason=f"Echoed {field_name}={echoed_value} matches locked value.",
        guardrail_name="locked_formula_tamper_check",
    )


def _classify_llm_failure(exception: Exception) -> str:
    """Best-effort human-readable category for an LLM-call failure, derived
    from the exception's class name + message (no hard `openai` import here
    to keep this module dependency-light — openai_utils.py already imports
    the SDK and lets its real exception classes/messages flow through
    unchanged). Used only to label *what kind* of limit/failure tripped the
    fallback; the raw exception text is always included alongside it."""
    type_name = type(exception).__name__
    msg = str(exception).lower()
    if "ratelimit" in type_name.lower() or "rate limit" in msg or "rate_limit" in msg:
        return "OpenAI rate limit (requests-per-minute or tokens-per-minute cap) exceeded"
    if "quota" in msg:
        return "OpenAI billing quota exceeded"
    if "timeout" in type_name.lower() or "timed out" in msg or "timeout" in msg:
        return "call exceeded the execution timeout budget"
    if "badrequest" in type_name.lower():
        return "OpenAI rejected the request (bad request)"
    if "authenticat" in msg or "api key" in msg or "api_key" in msg:
        return "OpenAI authentication failed (missing/invalid API key)"
    return f"unclassified error ({type_name})"


def validate_output_fallback_triggered(agent_name: str, exception: Exception) -> GuardrailResult:
    """Doc §4 — Fallback-on-failure guardrail. Hook: all 4 LLM-calling
    agents (L2 news_agent, L3 weather_agent, L4 llm_signal/judge_agent, L7
    mitigation_agent) — each already has a try/except-return-None(or
    rule-based) fallback path. This makes each fallback trigger a LOGGED
    event instead of a silent default. Reason names the failure category
    (rate limit / quota / timeout / auth / unclassified) plus the raw
    exception text and the mitigation applied. Call from inside each
    agent's existing except block."""
    category = _classify_llm_failure(exception)
    return GuardrailResult(
        passed=False,
        reason=f"{agent_name} LLM call failed — {category}: {exception}. "
               f"Mitigation: routed to the deterministic rule-based fallback; "
               f"pipeline continued without the LLM output.",
        guardrail_name="fallback_on_failure",
    )


def validate_output_ragas_faithfulness_gate(faithfulness_score: Optional[float], threshold: float = 0.75) -> GuardrailResult:
    """Doc §4 — RAGAS faithfulness threshold gate. Hook: post-hoc on L4/L7
    outputs, pre-Slack for CRITICAL alerts. Branch B (see session report):
    no inline, per-run faithfulness scorer exists in this codebase today —
    everything under evaluation/ragas/ is a batch runner over a gold
    dataset. faithfulness_score is therefore always None in production
    right now, and this is a no-op pass-through (passed=True) so the gate
    is dashboard-visible without blocking Slack alerts on a capability that
    isn't built. When a score IS available (e.g. once an inline scorer
    exists), a below-threshold score fails the gate and the caller routes
    the mitigation plan to human review instead of auto-firing Slack."""
    if faithfulness_score is None:
        return GuardrailResult(
            passed=True,
            reason="faithfulness scoring not available inline — see scope note",
            guardrail_name="ragas_faithfulness_gate",
        )
    if faithfulness_score < threshold:
        return GuardrailResult(
            passed=False,
            reason=f"faithfulness_score={faithfulness_score} below threshold {threshold} — routed to human review.",
            guardrail_name="ragas_faithfulness_gate",
        )
    return GuardrailResult(
        passed=True,
        reason=f"faithfulness_score={faithfulness_score} meets threshold {threshold}.",
        guardrail_name="ragas_faithfulness_gate",
    )


# ---------------------------------------------------------------------------
# Execution/resource guardrails (added post-doc — see module docstring)
# ---------------------------------------------------------------------------

def validate_execution_timeout(
    agent_name: str, elapsed_seconds: float, timeout_seconds: float, retry_count: int, max_retries: int
) -> GuardrailResult:
    """Resource/execution guardrail (added post-doc, not in doc §3/§4) —
    per-call timeout + bounded retry. Hook: wraps call_openai_structured()
    itself, the single choke point all 4 LLM-calling agents already route
    through — wire ONCE here, not per-agent. Reuses the project's existing
    tenacity retry/backoff mechanics inside call_openai_structured(); this
    function only checks the outcome against the timeout/retry budget and
    logs it. On exhausting max_retries: passed=False, caller routes to the
    agent's existing fallback-on-failure path rather than raising further."""
    if retry_count >= max_retries:
        return GuardrailResult(
            passed=False,
            reason=f"{agent_name} exhausted {retry_count}/{max_retries} retries "
                   f"(elapsed={elapsed_seconds:.1f}s, budget={timeout_seconds:.1f}s).",
            guardrail_name="execution_timeout_retry",
        )
    if elapsed_seconds > timeout_seconds:
        return GuardrailResult(
            passed=False,
            reason=f"{agent_name} call took {elapsed_seconds:.1f}s, exceeding budget {timeout_seconds:.1f}s.",
            guardrail_name="execution_timeout_retry",
        )
    return GuardrailResult(
        passed=True,
        reason=f"{agent_name} call completed in {elapsed_seconds:.1f}s (budget {timeout_seconds:.1f}s).",
        guardrail_name="execution_timeout_retry",
    )


def validate_execution_cost_breaker(run_id: str, cumulative_cost_usd: float, per_run_cap_usd: float) -> GuardrailResult:
    """Resource/execution guardrail (added post-doc, not in doc §3/§4) —
    per-run cost circuit breaker. Hook: before each call_openai_structured()
    invocation, check cumulative cost already logged in llm_call_log for
    this run_id (db_utils.fetch_cost_by_run() reuses the cost-by-agent
    panel's existing SUM(cost_usd) pattern, filtered by run_id instead of
    agent_name — no duplicate cost-tracking path). On exceeding
    per_run_cap_usd: passed=False, the calling agent skips the LLM call and
    routes to its existing fallback path — treated the same way agents
    already treat a missing OPENAI_API_KEY, not as a new failure mode."""
    if cumulative_cost_usd > per_run_cap_usd:
        return GuardrailResult(
            passed=False,
            reason=f"run_id={run_id} cumulative cost ${cumulative_cost_usd:.4f} exceeds "
                   f"per-run cap ${per_run_cap_usd:.4f}.",
            guardrail_name="per_run_cost_breaker",
        )
    return GuardrailResult(
        passed=True,
        reason=f"run_id={run_id} cumulative cost ${cumulative_cost_usd:.4f} within cap ${per_run_cap_usd:.4f}.",
        guardrail_name="per_run_cost_breaker",
    )
