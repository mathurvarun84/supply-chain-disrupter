# TruLens Guardrails Design Spec (Phase 2)

**Date:** 2026-07-08  
**Status:** Approved  
**Author:** Claude + User collaboration  
**Dependency:** Phase 1 — TruLens Observability (2026-07-06)

## Overview

Add runtime guardrails to validate LLM agent outputs in the LangGraph pipeline. Guardrails run synchronously after each LLM agent, log violations to the TruLens dashboard, and allow the pipeline to continue (soft enforcement).

## Goals

1. **Validate LLM outputs** — Catch out-of-bounds values, invalid categories, missing citations
2. **Soft enforcement** — Log violations to dashboard, don't halt execution
3. **Coverage** — All LLM agents (L2 News, L3 Weather, L4 Risk Classifier)
4. **Visibility** — Violations appear in TruLens dashboard with details

## Non-Goals (Out of Scope)

- Hard enforcement (halting pipeline on failure)
- Automatic retry/fallback mechanisms
- Non-LLM agents (L1, L5, L6, L7)
- Custom guardrails UI beyond TruLens dashboard

## Requirements

| Requirement | Detail |
|-------------|--------|
| Dependency | Phase 1 TruLens integration must be implemented first |
| Failure mode | Log and continue (soft enforcement) |
| Agents covered | L2 News, L3 Weather, L4 Risk Classifier |
| Guardrail count | 8 total (2 for L2, 2 for L3, 4 for L4) |

## Guardrail Definitions

### L2 News Agent Guardrails

| Guardrail | Check | Pass Condition |
|-----------|-------|----------------|
| `l2_severity_bounds` | `news_analysis_llm.news_severity_component` | Value in [0.0, 1.0] |
| `l2_category_valid` | `news_analysis_llm.category` | One of: `weather`, `geopolitical`, `logistics`, `demand`, `supplier`, `regulatory`, `other` |

### L3 Weather Agent Guardrails

| Guardrail | Check | Pass Condition |
|-----------|-------|----------------|
| `l3_geo_risk_bounds` | `live_weather_severity` | Value in [0.0, 1.0] and not None |
| `l3_hub_coverage` | Weather data exists for active hub | `active_record["port"]` has corresponding weather signal |

### L4 Risk Classifier Guardrails

| Guardrail | Check | Pass Condition |
|-----------|-------|----------------|
| `l4_composite_bounds` | `risk_classification.composite_score` | Value in [0.0, 1.0] |
| `l4_label_score_alignment` | Label matches score tier | LOW: <0.3, MEDIUM: 0.3-0.5, HIGH: 0.5-0.7, CRITICAL: >0.7 |
| `l4_ensemble_coherence` | Signal agreement | ≥2 of 3 signals (rule, distilbert, llm) agree on tier |
| `l4_citation_required` | RAG grounding for high-risk | HIGH/CRITICAL labels have ≥1 entry in `rag_citations` |

### Guardrail Output Format

Each guardrail returns:

```python
{
    "passed": bool,
    "score": float,      # 1.0 if passed, 0.0 if failed
    "message": str,      # Human-readable explanation
    "details": dict      # Actual values for debugging
}
```

## Architecture

### Pipeline Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Pipeline Execution                              │
│                                                                         │
│   L1 ──→ L2 ──→ L3 ──→ L4 ──→ L5 ──→ L6 ──→ L7                        │
│          │      │      │                                                │
│          ▼      ▼      ▼                                                │
│       ┌─────┐┌─────┐┌─────┐                                            │
│       │Guard││Guard││Guard│  ← Guardrails run AFTER each LLM agent     │
│       │rails││rails││rails│    but BEFORE next node starts             │
│       └──┬──┘└──┬──┘└──┬──┘                                            │
│          │      │      │                                                │
└──────────┼──────┼──────┼────────────────────────────────────────────────┘
           │      │      │
           ▼      ▼      ▼
    ┌─────────────────────────┐
    │   TruLens Dashboard     │
    │   ┌───────────────────┐ │
    │   │ Phase 1: Traces   │ │  ← Latency, tokens, metrics
    │   ├───────────────────┤ │
    │   │ Phase 2: Violations│ │  ← Guardrail failures
    │   └───────────────────┘ │
    └─────────────────────────┘
```

### Execution Flow

```
Phase 1 flow:          Phase 1 + Phase 2 flow:
─────────────          ──────────────────────
node_start()           node_start()
  ↓                      ↓
agent_fn()             agent_fn()
  ↓                      ↓
node_end()             run_guardrails()    ← NEW
  ↓                      ↓
(next node)            node_end()
                         ↓
                       (next node)
```

### File Structure

```
src/evaluation/trulens_integration/
├── __init__.py              # Add guardrail exports
├── config.py                # (unchanged)
├── wrapper.py               # Add guardrail hook point
├── openai_patch.py          # (unchanged)
├── feedback_functions.py    # (unchanged - Phase 1 async metrics)
├── node_extractors.py       # (unchanged)
├── cli.py                   # (unchanged)
├── guardrails.py            # NEW: Pure guardrail check functions
└── guardrail_runner.py      # NEW: TruLens feedback integration
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `guardrails.py` | Pure Python validation logic — no TruLens dependency |
| `guardrail_runner.py` | Wraps guardrails as TruLens Feedback, handles logging |
| `wrapper.py` | Calls `guardrail_runner` after each LLM node |

This separation allows guardrails to be tested independently and reused outside TruLens if needed.

## Soft Enforcement Mechanism

### Behavior on Failure

When a guardrail fails:

1. **Log the violation** — Record to TruLens with full context
2. **Continue execution** — Pipeline proceeds to next node unchanged
3. **No exception raised** — Agent output passes through as-is
4. **Dashboard visibility** — Violation appears as low-score feedback entry

### Rationale for Soft Enforcement

| Hard Enforcement | Soft Enforcement (chosen) |
|------------------|---------------------------|
| Halts on failure | Logs and continues |
| Risk of blocking valid edge cases | Captures data for analysis |
| Requires fallback logic | Simple to implement |
| Production-critical | Development/monitoring focus |

For a capstone project, soft enforcement provides visibility without the complexity of fallback handling. Can be upgraded to hard enforcement later if needed.

### Violation Record Structure

Each violation logged to TruLens:

```python
{
    "record_id": "uuid",
    "app_id": "supply-chain-disrupter",
    "agent": "l4_risk_classifier",
    "guardrail": "l4_label_score_alignment",
    "passed": False,
    "score": 0.0,
    "message": "Label CRITICAL does not match score 0.45 (expected HIGH)",
    "details": {
        "composite_score": 0.45,
        "final_label": "CRITICAL",
        "expected_label": "HIGH"
    },
    "timestamp": "2026-07-08T14:32:01Z"
}
```

### Dashboard View

In TruLens dashboard, violations appear as:

- **Feedback entries** with score 0.0 (failed) or 1.0 (passed)
- **Filterable** by guardrail name, agent, time range
- **Drillable** to see full violation details and agent context

## Testing & Verification

### Unit Tests

| Test | Verifies |
|------|----------|
| `test_l2_severity_bounds_valid` | Accepts 0.0, 0.5, 1.0 |
| `test_l2_severity_bounds_invalid` | Rejects -0.1, 1.5, None |
| `test_l2_category_valid` | Accepts all enum values |
| `test_l2_category_invalid` | Rejects "unknown", "", None |
| `test_l3_geo_risk_bounds` | Accepts valid range, rejects out-of-bounds |
| `test_l4_composite_bounds` | Accepts valid range, rejects out-of-bounds |
| `test_l4_label_score_alignment` | Correct tier mapping for all thresholds |
| `test_l4_ensemble_coherence` | Passes with 2/3 agreement, fails with 0/3 |
| `test_l4_citation_required` | HIGH/CRITICAL need citations, LOW/MEDIUM don't |

### Integration Tests

| Test | Verifies |
|------|----------|
| `test_guardrails_run_after_node` | Guardrails execute between L2→L3, L3→L4 |
| `test_violation_logged_to_trulens` | Failed guardrail appears in TruLens DB |
| `test_pipeline_continues_on_failure` | Violation doesn't halt execution |
| `test_multiple_violations_single_run` | All failures captured, not just first |

### Manual Verification Checklist

1. Run pipeline with intentionally bad L4 output (mock `composite_score = 1.5`)
2. Verify violation appears in TruLens dashboard at `http://localhost:8502`
3. Confirm pipeline completed despite violation
4. Check violation record has correct `agent`, `guardrail`, `message`, `details`
5. Run clean scenario — verify all guardrails show score 1.0 (passed)

### Test Data Scenarios

| Scenario | Expected Violations |
|----------|---------------------|
| Taiwan earthquake (normal) | 0 — all guardrails pass |
| Synthetic: severity=1.5 | 1 — `l2_severity_bounds` fails |
| Synthetic: label mismatch | 1 — `l4_label_score_alignment` fails |
| Synthetic: no citations on CRITICAL | 1 — `l4_citation_required` fails |
| Synthetic: all signals disagree | 1 — `l4_ensemble_coherence` fails |

## Dependencies

- Phase 1 TruLens Observability must be implemented first
- No additional Python packages required beyond Phase 1

## Future Enhancements (Out of Scope)

- Hard enforcement mode (halt on failure)
- Configurable thresholds per guardrail
- Automatic retry with prompt modification
- Guardrails for L5, L6, L7 agents
- Custom alerting (Slack, email) on violations
