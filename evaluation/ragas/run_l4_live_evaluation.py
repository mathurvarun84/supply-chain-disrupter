"""
run_l4_live_evaluation.py — RAGAS evaluation of L4 Risk Classifier Signal 3
against REAL historical orders (RAGAS Phase 3 extension, L4-specific).

Runs the actual `risk_classifier_agent()` (src/agents/risk_classifier_agent)
against real rows from outputs/supply_chain.db (replay mode — stored
composite/label already present, rows are never overwritten), captures
Signal 3's (RAG query, retrieved chunks, LLMSignal.rationale) triple via a
narrow monkeypatch, and scores it with RAGAS Faithfulness + Answer Relevancy.

SCOPE — Faithfulness + Answer Relevancy only, no Context Precision/Recall:
Live classifications have no gold reference rationale to compare against, so
RAGAS's reference-dependent metrics (Context Precision, Context Recall)
cannot be computed here — there is no ground_truth column. Faithfulness (is
the rationale grounded in what Signal 3 was actually shown) and Answer
Relevancy (does the rationale address the RAG query) are both reference-free
and are precisely the two metrics that speak to "did the model hallucinate
beyond the retrieved evidence" for this GPT-4o call.

SCOPE — why only Signal 3, not `_gather_rag_citations` or the Judge:
`_gather_rag_citations` (src/agents/risk_classifier_agent/agent.py) produces
a deterministic, non-LLM rationale from a SEPARATE `query_chroma_rag` call
that Signal 3's LLM never sees — it cannot hallucinate, so RAGAS doesn't
apply to it. The Judge (`run_judge`) makes its own `call_openai_structured`
call with no retrieval step of its own; it is patched to a no-op here
(saves an unrelated GPT-4o call per order) since only Signal 3's grounding
is in scope. The agent's judge -> llm_signal -> rule fallback chain means
disabling the judge simply makes `final_label` fall through to
`llm_signal.predicted_label` for this script's run — it does not affect
what is being scored (LLMSignal.rationale itself, captured directly).

Capture mechanism is deliberately NOT the general-purpose RAGTraceCollector
(evaluation/ragas/rag_tracer.py): that tracer patches both
`retrieve_and_rerank` AND `query_chroma_rag` globally, which would blend
`_gather_rag_citations`'s chunks (never shown to the LLM) into Signal 3's
context and would also capture the Judge's context-less call as a second,
noisy trace record. Instead this script patches only
`src.rag.retriever.retrieve_and_rerank` (the two-stage retrieval Signal 3's
prompt is actually built from) and
`src.agents.risk_classifier_agent.llm_signal.call_openai_structured` (Signal
3's own LLM call only), restoring both after every order.

Sampling: stratified by delivery_status (Shipping canceled / Late delivery /
Shipping on time / Advance shipping) so Signal 3 gets exercised across the
full LOW..CRITICAL label range, not just one easy bucket.

Usage:
    python -m evaluation.ragas.run_l4_live_evaluation [--n-per-bucket N] [--yes]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from evaluation.ragas.preflight import run_preflight
from evaluation.ragas.run_evaluation import (
    JUDGE_EMBEDDINGS,
    JUDGE_MODEL,
    TARGET_METRICS,
    _build_ragas_llm_and_embeddings,
    write_trace_jsonl,
)

logger = logging.getLogger(__name__)

_RAGAS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = _RAGAS_DIR / "ragas_scores_l4_live.json"

DELIVERY_STATUS_BUCKETS = [
    "Shipping canceled",
    "Late delivery",
    "Shipping on time",
    "Advance shipping",
]
DEFAULT_N_PER_BUCKET = 5

# Faithfulness + Answer Relevancy targets reuse the same locked values as
# Phase 3's full mode (see run_evaluation.TARGET_METRICS) — no new thresholds.
L4_TARGET_METRICS = {
    "faithfulness": TARGET_METRICS["faithfulness"],
    "answer_relevancy": TARGET_METRICS["answer_relevancy"],
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGAS live evaluation of L4 Risk Classifier Signal 3 (Faithfulness + Answer Relevancy)"
    )
    parser.add_argument("--n-per-bucket", type=int, default=DEFAULT_N_PER_BUCKET)
    parser.add_argument("--yes", action="store_true", help="Skip the cost-confirmation prompt")
    return parser


# ---------------------------------------------------------------------------
# Order sampling — stratified by delivery_status
# ---------------------------------------------------------------------------

def fetch_stratified_orders(n_per_bucket: int) -> List[dict]:
    from src.utils.db_utils import execute_query

    orders: List[dict] = []
    for status in DELIVERY_STATUS_BUCKETS:
        rows = execute_query(
            "SELECT * FROM daily_records WHERE delivery_status = ? ORDER BY record_id LIMIT ?",
            (status, n_per_bucket),
        )
        orders.extend(dict(r) for r in rows)
    return orders


# ---------------------------------------------------------------------------
# Narrow Signal-3 capture — see module docstring for why this isn't RAGTraceCollector
# ---------------------------------------------------------------------------

class Signal3Capture:
    """Captures Signal 3's RAG query text, retrieved chunk texts, and the
    resulting LLMSignal for exactly one risk_classifier_agent() call."""

    def __init__(self) -> None:
        self.retrieval_calls: List[dict] = []
        self.llm_signal_result: Any = None
        self._orig_retrieve = None
        self._orig_call_llm = None
        self._retriever_mod = None
        self._llm_signal_mod = None

    def __enter__(self) -> "Signal3Capture":
        import src.agents.risk_classifier_agent.llm_signal as llm_signal_mod
        import src.rag.retriever as retriever_mod

        self._retriever_mod = retriever_mod
        self._llm_signal_mod = llm_signal_mod
        self._orig_retrieve = retriever_mod.retrieve_and_rerank
        self._orig_call_llm = llm_signal_mod.call_openai_structured

        def retrieve_wrapper(*args, **kwargs):
            hits = self._orig_retrieve(*args, **kwargs)
            query = kwargs.get("query", args[0] if args else "")
            self.retrieval_calls.append({"query": query, "hits": list(hits or [])})
            return hits

        def call_llm_wrapper(*args, **kwargs):
            result = self._orig_call_llm(*args, **kwargs)
            self.llm_signal_result = result
            return result

        retriever_mod.retrieve_and_rerank = retrieve_wrapper
        llm_signal_mod.call_openai_structured = call_llm_wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._retriever_mod.retrieve_and_rerank = self._orig_retrieve
        self._llm_signal_mod.call_openai_structured = self._orig_call_llm
        return False


# ---------------------------------------------------------------------------
# Run the real agent for one order and score-ready-ify the result
# ---------------------------------------------------------------------------

def run_order_and_capture(record: dict) -> dict:
    """
    Run the real risk_classifier_agent() against one historical order,
    capturing Signal 3's (question, contexts, answer) without touching
    production code. Never raises — returns evaluated=False on any failure.
    """
    from src.agents.risk_classifier_agent import risk_classifier_agent
    from src.agents.state import EventMetadata, GlobalState

    order_id = record.get("order_id") or record.get("record_id")
    delivery_status = record.get("delivery_status")

    base = {
        "order_id": order_id,
        "delivery_status": delivery_status,
        "stored_label": record.get("disruption_event_label"),
    }

    state = GlobalState(
        event_metadata=EventMetadata(
            disruption_type="test",
            affected_port=str(record.get("port", "")),
            affected_route="test",
            severity=0.5,
            shock_duration_days=0,
            recovery_window_days=30,
            synthetic_ratio=0.0,
        ),
        active_record=record,
        news_signals=[],
        live_weather_severity=0.3,
    )

    capture = Signal3Capture()
    try:
        with capture:
            with patch("src.agents.risk_classifier_agent.agent.insert_risk_classification"):
                with patch("src.agents.risk_classifier_agent.agent.update_risk_label"):
                    with patch("src.agents.risk_classifier_agent.agent.run_judge", return_value=None):
                        risk_classifier_agent(state)
    except Exception as exc:
        logger.warning("risk_classifier_agent failed for order_id=%s: %s", order_id, exc)
        return {**base, "question": "", "contexts": [], "answer": None, "evaluated": False, "skipped_reason": f"agent_call_failed: {exc}"}

    llm_signal = capture.llm_signal_result
    if llm_signal is None:
        return {**base, "question": "", "contexts": [], "answer": None, "evaluated": False, "skipped_reason": "llm_signal_none"}

    question = capture.retrieval_calls[0]["query"] if capture.retrieval_calls else ""
    contexts = [
        hit.get("text", "")
        for call in capture.retrieval_calls
        for hit in call["hits"]
    ]

    if not question or not contexts:
        return {
            **base,
            "question": question,
            "contexts": contexts,
            "answer": llm_signal.rationale,
            "evaluated": False,
            "skipped_reason": "no_context_retrieved",
        }

    return {
        **base,
        "predicted_label": llm_signal.predicted_label,
        "confidence_level": llm_signal.confidence_level,
        "primary_driver": llm_signal.primary_driver,
        "question": question,
        "contexts": contexts,
        "answer": llm_signal.rationale,
        "evaluated": True,
        "skipped_reason": None,
    }


# ---------------------------------------------------------------------------
# Cost guard
# ---------------------------------------------------------------------------

def cost_guard(n_orders: int, skip_confirmation: bool) -> bool:
    generation_calls = n_orders  # Signal 3 gpt-4o call — unavoidable side effect of running the real agent
    ragas_internal_calls = n_orders * 4  # faithfulness (~1/row) + answer_relevancy (~3/row) via JUDGE_MODEL
    total = generation_calls + ragas_internal_calls
    print(
        f"\nCost estimate: {generation_calls} Signal-3 call(s) (gpt-4o, from running the real agent) + "
        f"~{ragas_internal_calls} ragas internal judge call(s) ({JUDGE_MODEL}) = ~{total} call(s) total."
    )
    if n_orders > 20 and not skip_confirmation:
        answer = input(f"This will make ~{total} calls across {n_orders} real orders. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return False
    return True


# ---------------------------------------------------------------------------
# RAGAS scoring — Faithfulness + Answer Relevancy only, no ground_truth
# ---------------------------------------------------------------------------

def run_ragas_l4_metrics(evaluated_records: List[dict]):
    from datasets import Dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in evaluated_records],
            "contexts": [r["contexts"] for r in evaluated_records],
            "answer": [r["answer"] for r in evaluated_records],
        }
    )
    llm, embeddings = _build_ragas_llm_and_embeddings()
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
    return result.to_pandas()


def _aggregate(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {**{m: 0.0 for m in L4_TARGET_METRICS}, "n_cases": 0}
    agg = {m: round(sum(r[m] for r in rows) / n, 4) for m in L4_TARGET_METRICS}
    agg["n_cases"] = n
    return agg


def _group_by(rows: List[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {name: _aggregate(recs) for name, recs in groups.items()}


def flag_weak(by_bucket: Dict[str, dict]) -> List[dict]:
    flagged = []
    for bucket, scores in by_bucket.items():
        for metric, target in L4_TARGET_METRICS.items():
            score = scores.get(metric, 0.0)
            if score < target:
                flagged.append(
                    {"delivery_status": bucket, "metric": metric, "score": score, "target": target, "gap": round(target - score, 4)}
                )
    flagged.sort(key=lambda f: f["gap"], reverse=True)
    return flagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("run_l4_live_evaluation requires OPENAI_API_KEY — Signal 3 cannot run without it.")
        return 1

    if not run_preflight(strict=True):
        print("Preflight failed — fix ChromaDB ingestion before evaluating.")
        return 1

    orders = fetch_stratified_orders(args.n_per_bucket)
    if not orders:
        print("No orders found in outputs/supply_chain.db — run the ETL build first.")
        return 1

    print(f"\n{len(orders)} order(s) sampled ({args.n_per_bucket} per delivery_status bucket).")
    if not cost_guard(len(orders), args.yes):
        return 1

    records = [run_order_and_capture(record) for record in orders]
    evaluated = [r for r in records if r["evaluated"]]
    skipped = [r for r in records if not r["evaluated"]]

    per_case: List[dict] = []
    if evaluated:
        df = run_ragas_l4_metrics(evaluated)
        for i, rec in enumerate(evaluated):
            row = df.iloc[i]
            per_case.append(
                {
                    "order_id": rec["order_id"],
                    "delivery_status": rec["delivery_status"],
                    "predicted_label": rec.get("predicted_label"),
                    "faithfulness": round(float(row["faithfulness"]), 4),
                    "answer_relevancy": round(float(row["answer_relevancy"]), 4),
                    "answer": rec["answer"],
                    "evaluated": True,
                }
            )
    for rec in skipped:
        per_case.append(
            {
                "order_id": rec["order_id"],
                "delivery_status": rec["delivery_status"],
                "evaluated": False,
                "skipped_reason": rec["skipped_reason"],
            }
        )

    scored = [r for r in per_case if r["evaluated"]]
    overall = _aggregate(scored)
    by_delivery_status = _group_by(scored, "delivery_status")
    flagged = flag_weak(by_delivery_status)

    payload = {
        "mode": "l4_live",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "judge_embeddings": JUDGE_EMBEDDINGS,
        "n_per_bucket": args.n_per_bucket,
        "n_orders_total": len(orders),
        "n_orders_evaluated": len(evaluated),
        "n_orders_skipped": len(skipped),
        "overall": overall,
        "by_delivery_status": by_delivery_status,
        "flagged": flagged,
        "per_case": per_case,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")

    trace_path = write_trace_jsonl("l4_live", per_case)
    print(f"Wrote trace {trace_path}")

    print(f"\nBy delivery_status: {by_delivery_status}")
    print(f"Overall: {overall}")
    if flagged:
        print("\nWEAK BUCKETS/METRICS — Day 26 candidates")
        for f in flagged:
            print(f"  {f['delivery_status']:20s} {f['metric']:20s} score={f['score']:.4f} target={f['target']:.4f} gap={f['gap']:.4f}")
    else:
        print("\nNo buckets/metrics below target.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
