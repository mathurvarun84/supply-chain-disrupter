# TruLens Observability & Evaluation Design Spec (Phase 1)

**Date:** 2026-07-06 (amended 2026-07-11)  
**Status:** Approved  
**Author:** Claude + User collaboration

**Amendment (2026-07-11):** `trulens-eval` was deprecated in 2024 and removed
from maintenance by 2025-12-01. This revision repins to the current
`trulens-core` / `trulens-dashboard` / `trulens-feedback` /
`trulens-providers-openai` package line (2.8.x) and its `TruSession()` API,
and corrects the OpenAI patching strategy to account for by-value imports
in the agent modules. See "Dependencies," "OpenAI Patching Strategy," and
"RAGAS Coexistence" below.

## Overview

Add TruLens observability and evaluation to the LangGraph agent pipeline (L1-L7) for development debugging, production monitoring, and capstone demonstration. The integration is fully non-invasive — zero changes to agent code in `src/agents/`.

**This spec covers two capabilities:**
- **Observability** — Per-node latency, token usage, input/output tracing
- **Evaluation** — Async feedback functions for risk drift, ensemble agreement, latency thresholds, forecast accuracy

## Requirements

| Requirement | Detail |
|-------------|--------|
| Primary goals | Development debugging, production monitoring, capstone demo |
| Agent code changes | Zero — all instrumentation external |
| RAGAS coexistence | Critical — must not break existing evaluation |
| Pipeline coverage | Full (L1-L7), no gaps |
| Storage | SQLite at `data/trulens/trulens.db` |
| Key metrics | Risk drift, ensemble agreement, latency, tokens, forecast accuracy |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Production Code (UNCHANGED)                      │
│  src/agents/langgraph_engine.py                                 │
│  └── run_agent_graph(payload) → L1→L2→L3→L4→L5→L6→L7           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │   TruLens Wrapper Layer       │
              │   (src/evaluation/trulens/)   │
              │                               │
              │  ┌─────────────────────────┐  │
              │  │ run_with_trulens()      │  │  ← New entry point
              │  │  - Patches nodes        │  │
              │  │  - Patches OpenAI calls │  │
              │  │  - Records to TruLens   │  │
              │  └─────────────────────────┘  │
              │                               │
              │  ┌─────────────────────────┐  │
              │  │ Feedback Functions      │  │  ← Custom metrics
              │  │  - risk_score_stability │  │
              │  │  - ensemble_agreement   │  │
              │  │  - node_latency         │  │
              │  │  - forecast_accuracy    │  │
              │  └─────────────────────────┘  │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  data/trulens/trulens.db      │  ← SQLite persistence
              └───────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  TruLens Dashboard (:8502)    │  ← Separate from Streamlit
              └───────────────────────────────┘
```

Production code never imports from `src/evaluation/`. The wrapper imports production code and wraps it externally.

## File Structure

```
src/evaluation/trulens_integration/
├── __init__.py              # Exports run_with_trulens, launch_dashboard
├── config.py                # TruLens session + SQLite backend init
├── wrapper.py               # Core instrumentation (patches graph + OpenAI)
├── openai_patch.py          # Monkey-patch call_openai_structured
├── feedback_functions.py    # 4 custom metrics
├── node_extractors.py       # Extract metrics from GlobalState per node
└── cli.py                   # CLI: run, dashboard, query commands

src/evaluation/
└── patch_registry.py        # Coordinates TruLens/RAGAS patching

evaluation/
└── trulens_runner.py        # Batch evaluation over test scenarios
```

**Modified (existing) files:**

| File | Change |
|------|--------|
| `evaluation/ragas/rag_tracer.py` | `RAGTraceCollector.__enter__`/`__exit__` call `patch_registry.claim_patch("call_openai_structured", "ragas")` / `release_patch(...)` around its existing `sys.modules` patch loop, so a concurrent TruLens run doesn't double-patch or restore the wrong original. No change to its patch mechanism itself — it already does this correctly (see below). |

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Initialize `TruSession()` (formerly `Tru()` in the deprecated `trulens-eval` API) with SQLite at `data/trulens/trulens.db`, configure the OpenAI feedback provider via `trulens.providers.openai` |
| `wrapper.py` | Context manager that patches `run_agent_graph`, captures per-node telemetry, registers with TruLens recorder |
| `openai_patch.py` | Patches `call_openai_structured` to capture model, tokens, latency, prompt/response — see "OpenAI Patching Strategy" below for the by-value-import handling |
| `feedback_functions.py` | Implements `risk_score_stability`, `ensemble_agreement`, `node_latency_check`, `forecast_accuracy` |
| `node_extractors.py` | Functions to pull metrics from `GlobalState` after each agent (e.g., L4 → composite_score, signals) |
| `cli.py` | Entry points: `python -m src.evaluation.trulens_integration.cli run/dashboard/query` |
| `patch_registry.py` | Prevents TruLens and RAGAS from patching same function simultaneously |
| `trulens_runner.py` | Loads test cases from `evaluation/ragas/test_dataset.json`, runs each through instrumented pipeline |

## OpenAI Patching Strategy

`call_openai_structured` (`src/utils/openai_utils.py`) is imported **by value**
— `from src.utils.openai_utils import call_openai_structured` — in four agent
modules, each binding its own local reference at import time:

- `src/agents/news_agent/agent.py`
- `src/agents/weather_agent/agent.py`
- `src/agents/risk_classifier_agent/llm_signal.py`
- `src/agents/risk_classifier_agent/judge_agent.py`

Patching only the `src.utils.openai_utils` module attribute does **not**
affect these four call sites — they never look the function up through the
module again. `openai_patch.py` must instead scan `sys.modules` for every
already-imported module whose `call_openai_structured` attribute `is` the
original function object, patch each one, and restore all of them on exit.

This is not new work: `evaluation/ragas/rag_tracer.py:138-157`
(`RAGTraceCollector.__enter__`/`__exit__`) already implements exactly this
pattern for the same function, plus `retrieve_and_rerank` and
`query_chroma_rag`. `openai_patch.py` should mirror that implementation
rather than a single-attribute patch.

## Non-Interference Guarantees

`wrapper.py`'s "patches `run_agent_graph`, captures per-node telemetry" was
previously unspecified on *mechanism* — this is the one place a naive
implementation could actually change pipeline behavior, since L2, L3, L4,
and L7 are registered as `_critical_node(...)` (`langgraph_engine.py:148-150,159`),
meaning an exception raised inside any of them (e.g. `mitigation_agent.py:23`'s
`ValueError` when `risk_label` is `None`) is expected to propagate uncaught
out of `run_agent_graph()` today — no retry, no fallback, no swallowing.
Only L5/L6 are `_optional_node(...)`-wrapped with skip-and-continue.

`wrapper.py` must capture per-node timing via LangGraph's native
`app.stream(GlobalState(), stream_mode="updates")` on the already-compiled
graph, consuming the update for each node as it completes, rather than
monkey-patching `_critical_node`/`_optional_node`/`_l1_node` internals in
`langgraph_engine.py`. This only changes the outermost call in the wrapper's
own copy of `run_agent_graph()`'s body (`.invoke()` → iterate `.stream()`
and accumulate the final state the same way `.invoke()` does internally) —
it never touches `langgraph_engine.py` itself, and an exception raised by a
node still propagates out of the stream iterator exactly as `.invoke()`
would raise it. Telemetry capture around each yielded update must be
wrapped in its own try/except that logs and continues (mirroring
`observability.py`'s `agent_span()` pattern) so a *tracing* bug can never
turn into a pipeline failure, but a genuine *agent* exception (like L7's
`ValueError`) must never be caught or suppressed by the wrapper.

## Custom Feedback Functions

Four domain-specific metrics that run asynchronously after each pipeline execution:

| Function | What it Measures | How it's Computed | Target |
|----------|------------------|-------------------|--------|
| `risk_score_stability` | Risk score drift over time | Coefficient of variation of `composite_score` across last 30 runs | CV < 0.30 |
| `ensemble_agreement` | L4 classifier confidence | Fraction of runs where Rule/DistilBERT/LLM signals agree (2+ match) | > 0.66 |
| `node_latency_check` | Per-node performance | Binary pass/fail per node against thresholds (L2/L3 < 2s, L4 < 5s, total < 15s) | All pass |
| `forecast_accuracy` | Prophet prediction quality | 1 - abs(predicted_drop - actual_drop) / max(predicted, actual) when actuals available | > 0.80 |

### Data Sources

- `risk_score_stability` → queries historical `risk_classification.composite_score` from TruLens DB
- `ensemble_agreement` → extracts `rule_signal`, `distilbert_signal`, `llm_signal` from L4 output
- `node_latency_check` → captured by wrapper during execution (start/end timestamps per node)
- `forecast_accuracy` → compares `forecast_result.expected_drop_pct` vs ground truth (when available in test dataset)

## RAGAS Coexistence

Both TruLens and RAGAS monkey-patch `call_openai_structured`. A patch registry prevents conflicts.

### Patch Registry (`src/evaluation/patch_registry.py`)

```python
_active_patches = {}  # {"call_openai_structured": "trulens" | "ragas"}

def claim_patch(target: str, owner: str) -> bool:
    """Returns True if patch granted, False if already claimed by another."""
    if target in _active_patches and _active_patches[target] != owner:
        return False
    _active_patches[target] = owner
    return True

def release_patch(target: str, owner: str):
    if _active_patches.get(target) == owner:
        del _active_patches[target]
```

### Usage Pattern

- TruLens wrapper calls `claim_patch("call_openai_structured", "trulens")` before patching
- RAGAS tracer (`RAGTraceCollector.__enter__` in `evaluation/ragas/rag_tracer.py`,
  which requires a small edit to add the `claim_patch`/`release_patch` calls
  around its existing `sys.modules` loop — see "Modified (existing) files"
  above) calls `claim_patch("call_openai_structured", "ragas")` before patching
- If claim fails, skip patching that target (log warning, continue with partial instrumentation)
- Both release on context manager exit
- First one wins; they rarely run simultaneously in this codebase's actual
  usage pattern (RAGAS evaluation is a standalone script invocation, not run
  inside a live TruLens-instrumented pipeline process) — the registry is a
  safety net for the rare case they overlap, not the primary defense

## Entry Points

### 1. CLI Commands

```bash
# Run a single scenario with instrumentation
python -m src.evaluation.trulens_integration.cli run \
  --port "Chennai" --sku "CHIP-001" --event-date "2024-03-15"

# Launch dashboard (port 8502)
python -m src.evaluation.trulens_integration.cli dashboard

# Query historical metrics
python -m src.evaluation.trulens_integration.cli query \
  --metric risk_drift --days 30
```

### 2. Batch Evaluation Runner

```bash
# Run all test scenarios from RAGAS test dataset
python -m evaluation.trulens_runner

# Output: evaluation/trulens_scores.json
```

### 3. Programmatic API

```python
from src.evaluation.trulens_integration import run_with_trulens

# Drop-in replacement for run_agent_graph
result = run_with_trulens(payload)
# Returns same GlobalState, but execution is traced
```

### Dashboard

- Launched via the current API: `from trulens.dashboard import run_dashboard; run_dashboard(session, port=8502)` (the deprecated `tru.run_dashboard()` no longer exists in `trulens-eval`)
- Runs on port 8502 as its own Streamlit process (the app's own Streamlit UI runs separately on 8501 — two browser tabs, two processes)
- Two built-in pages: **Leaderboard** (aggregate feedback scores per app version) and **Evaluations** (per-trace/per-span drill-down)
- Shows execution timeline with per-node breakdown
- Displays feedback scores over time
- Provides trace explorer for drill-down
- Tracks token usage and cost

## Testing & Verification

### Unit Tests

| Test | Verifies |
|------|----------|
| `test_wrapper_captures_latency` | Wrapper records start/end time for each node |
| `test_openai_patch_captures_tokens` | Patch extracts prompt_tokens, completion_tokens from response |
| `test_feedback_functions_compute` | Each of 4 feedback functions returns valid 0-1 score |
| `test_patch_registry_prevents_conflicts` | Second patcher gets rejected, first one works |

### Integration Tests

| Test | Verifies |
|------|----------|
| `test_full_pipeline_with_trulens` | Run Taiwan earthquake scenario, verify trace in SQLite |
| `test_ragas_coexistence` | Run RAGAS evaluation, then TruLens — both work independently |
| `test_dashboard_launches` | CLI `dashboard` command starts server on 8502 |

### Manual Verification Checklist

1. Run `python -m src.evaluation.trulens_integration.cli run --port Chennai --sku CHIP-001`
2. Open `http://localhost:8502` — verify trace appears
3. Check L1-L7 latency breakdown visible in dashboard
4. Compare token counts with existing `[LLM]` log lines — should match
5. Run `python -m evaluation.trulens_runner` — verify `trulens_scores.json` created

## Dependencies

`trulens-eval` is deprecated (deprecation warning period started
2024-09-01, hard-error period 2024-10-14, removed from maintenance
2025-12-01) and must not be used. Add to `requirements.txt` instead:

```
trulens-core>=2.8,<3
trulens-dashboard>=2.8,<3
trulens-feedback>=2.8,<3
trulens-providers-openai>=2.8,<3
```

API note: session initialization is `TruSession()`, not the deprecated
`Tru()`. Dashboard launch is `trulens.dashboard.run_dashboard(session,
port=...)`, not `tru.run_dashboard()`. Current TruLens (2.x) instruments
via OpenTelemetry spans under the hood; the external monkey-patch wrapper
described in this spec remains valid as a "custom app" integration style,
but `config.py` and `wrapper.py` must target the 2.x API surface, not the
0.x API implied by earlier drafts of this spec.

## Out of Scope

- Distributed tracing for batch ingestion jobs
- A/B testing different LLM models
- Automated retraining triggers
- ML-based drift prediction
- Multi-user cloud dashboard
