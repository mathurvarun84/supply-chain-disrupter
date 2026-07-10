"""
run_evaluation.py — RAG evaluation runner (RAGAS Phase 3).

Two modes, selected by --mode:

  retrieval-only  — ZERO LLM judge cost. Runs real production retrieval
    (retrieve_and_rerank) for every gold test case and scores it with
    embedding-based (non-LLM) proxies: Hit Rate@k, Mean Reciprocal Rank, and
    cosine-similarity context relevance/recall, all using the SAME bi-encoder
    that powers production retrieval (get_embedding_model() from
    src/utils/rag_utils.py). Safe to re-run constantly, works with NO
    OPENAI_API_KEY at all.

  full            — Full RAGAS metric suite (Faithfulness, Answer Relevancy,
    Context Precision, Context Recall) computed via the `ragas` library
    (v0.4.3 in this environment — ragas.evaluate() plus the classic
    ragas.metrics.{faithfulness,answer_relevancy,context_precision,
    context_recall} objects; these names are deprecated in favor of
    ragas.metrics.collections in ragas>=0.4 but remain functional and are
    used here deliberately for a stable, documented evaluate() call shape).
    Requires OPENAI_API_KEY. Makes real gpt-4.1-mini judge calls (MODEL_FAST —
    chosen over gpt-4o to keep repeated Day-26 error-analysis runs cheap;
    trade-off is a somewhat less consistent LLM-as-judge than gpt-4o on
    borderline cases) plus embedding calls (nontrivial API cost — the script
    prints a cost estimate and requires --yes above a size threshold).

Both modes write per-case AND per-collection/per-style aggregate scores and
explicitly flag weak collections against the project's locked target
metrics (Faithfulness > 0.85, Answer Relevancy > 0.80, Context Precision
> 0.75, Context Recall > 0.75 — project convention: Context Recall shares
Context Precision's bar since no separate target was specified).

KNOWN SCOPE LIMITATION — documented, not hidden:

Gold test cases have no accompanying live SQLite order record (Phase 1
generates them from ChromaDB chunks alone). Full-pipeline mode therefore
cannot replay the EXACT production prompts of build_risk_classifier_context
-> LLMSignal or build_mitigation_context -> MitigationLLMOutput, both of
which require a real lite_master row. Instead, full mode:
  1. Calls the REAL production retrieval function (retrieve_and_rerank)
     with the gold question as the query and the gold source_collection
     as the target — this is the actual retrieval layer, unmodified.
  2. Generates an answer with a minimal, RAG-only "answer strictly from
     context" prompt via call_openai_structured — NOT the full agent
     system prompt (which needs SQLite fields we don't have per-question).
This means full-pipeline Faithfulness/Answer Relevancy scores measure the
RAG layer's retrieval-and-grounding quality, not the complete agent
pipeline's behavior end to end. Same convention as other documented scope
cuts in this project (e.g. drift detection, SHAP).

NOT IN SCOPE: Phase 4 (evaluate_ragas() in fine_tuning/evaluate_all.py).

Usage:
    python -m evaluation.ragas.run_evaluation --mode retrieval-only
    python -m evaluation.ragas.run_evaluation --mode full [--yes] [--limit N] \\
        [--collections export_control_corpus,india_sourcing_corpus]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from evaluation.ragas.preflight import run_preflight
from evaluation.ragas.rag_tracer import _chunk_id_for_hit
from src.utils.openai_utils import MODEL_FAST

logger = logging.getLogger(__name__)

_RAGAS_DIR = Path(__file__).resolve().parent
TEST_DATASET_PATH = _RAGAS_DIR / "test_dataset.json"
TRACES_DIR = _RAGAS_DIR / "traces"
RETRIEVAL_ONLY_OUTPUT_PATH = _RAGAS_DIR / "ragas_scores_retrieval_only.json"
FULL_OUTPUT_PATH = _RAGAS_DIR / "ragas_scores_full.json"

# Locked target metrics — do not invent new thresholds (project convention:
# Context Recall shares Context Precision's bar, see module docstring).
TARGET_METRICS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

JUDGE_MODEL = MODEL_FAST  # gpt-4.1-mini — cheaper than gpt-4o for repeated judge runs
JUDGE_EMBEDDINGS = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGAS evaluation runner (Phase 3)")
    parser.add_argument("--mode", choices=["retrieval-only", "full"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases (post-filter)")
    parser.add_argument("--collections", type=str, default=None, help="CSV of source_collection values to restrict to")
    parser.add_argument("--styles", type=str, default=None, help="CSV of query_style values to restrict to")
    parser.add_argument("--yes", action="store_true", help="Skip the cost-confirmation prompt in full mode")
    parser.add_argument("--top-k", type=int, default=3, dest="rerank_top_k", help="Passed through to retrieve_and_rerank(rerank_top_k=...)")
    parser.add_argument("--bi-encoder-top-n", type=int, default=10, help="Passed through to retrieve_and_rerank(bi_encoder_top_n=...)")
    return parser


# ---------------------------------------------------------------------------
# Load + filter test cases
# ---------------------------------------------------------------------------

def load_test_cases(path: Path = TEST_DATASET_PATH) -> Tuple[List[dict], str]:
    """Load test_dataset.json. Returns (cases, chunk_id_convention)."""
    if not path.exists():
        print(f"Missing {path} — run generate_test_dataset.py first (Phase 1).")
        sys.exit(1)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    convention = dataset.get("metadata", {}).get("chunk_id_convention", "chromadb_native")
    cases = dataset.get("test_cases", [])
    return cases, convention


def apply_filters(
    cases: List[dict],
    collections: Optional[str],
    styles: Optional[str],
    limit: Optional[int],
) -> List[dict]:
    filtered = cases
    if collections:
        wanted = {c.strip() for c in collections.split(",") if c.strip()}
        filtered = [c for c in filtered if c.get("source_collection") in wanted]
    if styles:
        wanted = {s.strip() for s in styles.split(",") if s.strip()}
        filtered = [c for c in filtered if c.get("query_style") in wanted]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def print_case_breakdown(cases: List[dict]) -> None:
    by_collection: Dict[str, int] = {}
    by_style: Dict[str, int] = {}
    for c in cases:
        by_collection[c.get("source_collection", "?")] = by_collection.get(c.get("source_collection", "?"), 0) + 1
        by_style[c.get("query_style", "?")] = by_style.get(c.get("query_style", "?"), 0) + 1
    print(f"\n{len(cases)} test case(s) selected.")
    print(f"  by_collection: {by_collection}")
    print(f"  by_style:      {by_style}")


# ---------------------------------------------------------------------------
# Shared retrieval helper — the ONE call site of retrieve_and_rerank
# ---------------------------------------------------------------------------

def run_retrieval(
    question: str,
    source_collection: str,
    bi_encoder_top_n: int,
    rerank_top_k: int,
) -> List[dict]:
    """
    Real production retrieval — identical call shape to what
    build_risk_classifier_context / build_mitigation_context issue.
    Never raises — returns [] on any exception, with a logged warning.
    """
    try:
        from src.rag.retriever import retrieve_and_rerank

        return retrieve_and_rerank(
            question,
            [source_collection],
            bi_encoder_top_n=bi_encoder_top_n,
            rerank_top_k=rerank_top_k,
        )
    except Exception as exc:
        logger.warning("run_retrieval failed for %r (%s): %s", question[:60], source_collection, exc)
        return []


# ---------------------------------------------------------------------------
# Retrieval-only mode
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np

    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def score_case_retrieval_only(
    case: dict,
    chunks: List[dict],
    chunk_id_convention: str,
    embedding_fn,
) -> dict:
    """Score one gold case against its retrieved chunks (embedding proxies only)."""
    source_collection = case.get("source_collection", "")
    gold_id = case.get("source_chunk_id", "")

    # Hit/rank: did retrieval surface the EXACT chunk this QA pair was generated from?
    retrieved_ids = [_chunk_id_for_hit(hit, source_collection, chunk_id_convention) for hit in chunks]
    hit = gold_id in retrieved_ids
    rank: Optional[int] = retrieved_ids.index(gold_id) + 1 if hit else None

    context_relevance = 0.0
    context_recall_proxy = 0.0
    if chunks:
        texts = [c.get("text", "") for c in chunks]
        question_vec = embedding_fn([case["question"]])[0]
        chunk_vecs = embedding_fn(texts)
        # context_relevance: how semantically close is each retrieved chunk to the
        # question, on average — a cheap, LLM-free proxy for RAGAS's Context Precision.
        sims = [_cosine_similarity(question_vec, v) for v in chunk_vecs]
        context_relevance = sum(sims) / len(sims)

        ground_truth = case.get("ground_truth", "")
        if ground_truth:
            # context_recall_proxy: does the single BEST-matching retrieved chunk
            # cover the gold answer's content — a cheap proxy for Context Recall.
            gt_vec = embedding_fn([ground_truth])[0]
            best_idx = max(range(len(sims)), key=lambda i: sims[i])
            context_recall_proxy = _cosine_similarity(gt_vec, chunk_vecs[best_idx])

    return {
        "question": case["question"],
        "source_collection": source_collection,
        "query_style": case.get("query_style", ""),
        "hit": hit,
        "rank": rank,
        "context_relevance": round(context_relevance, 4),
        "context_recall_proxy": round(context_recall_proxy, 4),
    }


def _aggregate_group(records: List[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"hit_rate_at_k": 0.0, "mrr": 0.0, "mean_context_relevance": 0.0, "mean_context_recall_proxy": 0.0, "n_cases": 0}
    hits = sum(1 for r in records if r["hit"])
    mrr = sum((1.0 / r["rank"]) if r["hit"] else 0.0 for r in records) / n
    return {
        "hit_rate_at_k": round(hits / n, 4),
        "mrr": round(mrr, 4),
        "mean_context_relevance": round(sum(r["context_relevance"] for r in records) / n, 4),
        "mean_context_recall_proxy": round(sum(r["context_recall_proxy"] for r in records) / n, 4),
        "n_cases": n,
    }


def _group_by(records: List[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return {name: _aggregate_group(recs) for name, recs in groups.items()}


def _print_table(title: str, by_group: Dict[str, dict], metric_keys: List[str]) -> None:
    print(f"\n{title}")
    header = f"  {'group':30s}" + "".join(f"{k:>22s}" for k in metric_keys) + f"{'n_cases':>10s}"
    print(header)
    for name, agg in by_group.items():
        row = f"  {name:30s}" + "".join(f"{agg[k]:>22.4f}" for k in metric_keys) + f"{agg['n_cases']:>10d}"
        print(row)


def write_trace_jsonl(mode: str, per_case: List[dict]) -> Path:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = TRACES_DIR / f"eval_run_{mode}_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in per_case:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def run_retrieval_only_mode(cases: List[dict], chunk_id_convention: str, args: argparse.Namespace) -> int:
    from src.utils.rag_utils import get_embedding_model

    embedding_fn = get_embedding_model()

    per_case: List[dict] = []
    for case in cases:
        chunks = run_retrieval(case["question"], case.get("source_collection", ""), args.bi_encoder_top_n, args.rerank_top_k)
        per_case.append(score_case_retrieval_only(case, chunks, chunk_id_convention, embedding_fn))

    overall = _aggregate_group(per_case)
    by_collection = _group_by(per_case, "source_collection")
    by_style = _group_by(per_case, "query_style")

    payload = {
        "mode": "retrieval-only",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {"bi_encoder_top_n": args.bi_encoder_top_n, "rerank_top_k": args.rerank_top_k},
        "overall": overall,
        "by_collection": by_collection,
        "by_style": by_style,
        "per_case": per_case,
    }
    RETRIEVAL_ONLY_OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RETRIEVAL_ONLY_OUTPUT_PATH}")

    trace_path = write_trace_jsonl("retrieval_only", per_case)
    print(f"Wrote trace {trace_path}")

    _print_table("By collection:", by_collection, ["hit_rate_at_k", "mrr", "mean_context_relevance", "mean_context_recall_proxy"])
    _print_table("By style:", by_style, ["hit_rate_at_k", "mrr", "mean_context_relevance", "mean_context_recall_proxy"])
    print(f"\nOverall: {overall}")
    return 0


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------

class RAGASAnswerOutput(BaseModel):
    """Minimal grounded-answer output for RAGAS full-mode evaluation."""

    answer: str = Field(
        ...,
        description=(
            "Answer to the question, using ONLY the provided context. If the "
            "context does not contain the answer, say so explicitly."
        ),
    )


FULL_MODE_SYSTEM_PROMPT = """You answer questions about semiconductor/electronics supply chain risk
using ONLY the provided context chunks. Do not use outside knowledge.
If the context is insufficient, say so explicitly rather than guessing."""


def _build_answer_user_message(question: str, chunks: List[dict]) -> str:
    lines = [f"QUESTION: {question}", "", "CONTEXT:"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata") or {}
        source = meta.get("source_file", meta.get("source", "unknown"))
        lines.append(f"[{i}] ({source}) {chunk.get('text', '')}")
    return "\n".join(lines)


def cost_guard(n_cases: int, skip_confirmation: bool) -> bool:
    """Print the cost estimate and, above threshold, block on user confirmation."""
    generation_calls = n_cases  # one gpt-4o answer-generation call per case
    ragas_internal_calls = n_cases * 5  # ~5 judge calls/case across the 4 RAGAS metrics
    total = generation_calls + ragas_internal_calls
    print(
        f"\nCost estimate: {generation_calls} answer-generation call(s) (gpt-4o) + "
        f"~{ragas_internal_calls} ragas internal judge call(s) ({JUDGE_MODEL}) = ~{total} call(s) total."
    )
    if n_cases > 20 and not skip_confirmation:
        answer = input(f"This will make ~{total} GPT-4o/{JUDGE_MODEL} calls. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return False
    return True


def generate_grounded_answers(cases: List[dict], args: argparse.Namespace) -> List[dict]:
    """
    For each case, run real retrieval + real (RAG-only) generation.

    Returns one record per case:
      {question, source_collection, query_style, source_chunk_id, ground_truth,
       contexts: list[str], answer: str|None, evaluated: bool, skipped_reason: str|None}
    """
    from src.utils.openai_utils import MODEL_REASONING, call_openai_structured

    records: List[dict] = []
    for case in cases:
        chunks = run_retrieval(case["question"], case.get("source_collection", ""), args.bi_encoder_top_n, args.rerank_top_k)
        contexts = [c.get("text", "") for c in chunks]

        # No chunks retrieved → nothing to ground an answer in; skip rather than
        # score a context-less generation as if it were a real RAG failure mode.
        if not chunks:
            records.append(
                {
                    "question": case["question"],
                    "source_collection": case.get("source_collection", ""),
                    "query_style": case.get("query_style", ""),
                    "source_chunk_id": case.get("source_chunk_id", ""),
                    "ground_truth": case.get("ground_truth", ""),
                    "contexts": [],
                    "answer": None,
                    "evaluated": False,
                    "skipped_reason": "no_context_retrieved",
                }
            )
            continue

        try:
            parsed = call_openai_structured(
                FULL_MODE_SYSTEM_PROMPT,
                _build_answer_user_message(case["question"], chunks),
                response_model=RAGASAnswerOutput,
                model=MODEL_REASONING,
            )
            answer = parsed.answer
        except Exception as exc:
            logger.error("Answer generation failed for %r: %s", case["question"][:60], exc)
            records.append(
                {
                    "question": case["question"],
                    "source_collection": case.get("source_collection", ""),
                    "query_style": case.get("query_style", ""),
                    "source_chunk_id": case.get("source_chunk_id", ""),
                    "ground_truth": case.get("ground_truth", ""),
                    "contexts": contexts,
                    "answer": None,
                    "evaluated": False,
                    "skipped_reason": f"generation_failed: {exc}",
                }
            )
            continue

        records.append(
            {
                "question": case["question"],
                "source_collection": case.get("source_collection", ""),
                "query_style": case.get("query_style", ""),
                "source_chunk_id": case.get("source_chunk_id", ""),
                "ground_truth": case.get("ground_truth", ""),
                "contexts": contexts,
                "answer": answer,
                "evaluated": True,
                "skipped_reason": None,
            }
        )
    return records


def _build_ragas_llm_and_embeddings():
    """
    Configure the ragas judge LLM and embeddings explicitly (not ragas's
    default global config, which silently reads env vars).

    NOTE: text-embedding-3-small (OpenAI) here is a DIFFERENT embedding model
    from the domain retrieval bi-encoder (all-MiniLM-L6-v2) used in
    retrieval-only mode. These ragas judge embeddings score semantic
    relevance of the LLM's answer — they are not part of the production
    retrieval path.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0.0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBEDDINGS))
    return llm, embeddings


def run_ragas_evaluate(evaluated_records: List[dict]):
    """Build the ragas Dataset from evaluated (non-skipped) records and run evaluate().

    Deprecation warnings are suppressed only around the ragas/metrics import and the
    evaluate() call itself — ragas 0.4.3 still uses the classic `ragas.metrics.*`
    objects internally, which warn about the newer `ragas.metrics.collections` API;
    the call shape used here is deliberately kept stable rather than migrated.
    """
    from datasets import Dataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in evaluated_records],
            "contexts": [r["contexts"] for r in evaluated_records],
            "answer": [r["answer"] for r in evaluated_records],
            "ground_truth": [r["ground_truth"] for r in evaluated_records],
        }
    )
    llm, embeddings = _build_ragas_llm_and_embeddings()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


def _aggregate_full_group(rows: List[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {**{m: 0.0 for m in TARGET_METRICS}, "n_cases": 0}
    agg = {m: round(sum(r[m] for r in rows) / n, 4) for m in TARGET_METRICS}
    agg["n_cases"] = n
    return agg


def _group_by_full(rows: List[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {name: _aggregate_full_group(recs) for name, recs in groups.items()}


def flag_weak_collections(by_collection: Dict[str, dict]) -> List[dict]:
    """Compare each collection's per-metric average against TARGET_METRICS; return sub-target pairs, worst gap first."""
    flagged = []
    for collection, scores in by_collection.items():
        for metric, target in TARGET_METRICS.items():
            score = scores.get(metric, 0.0)
            if score < target:
                flagged.append(
                    {
                        "source_collection": collection,
                        "metric": metric,
                        "score": score,
                        "target": target,
                        "gap": round(target - score, 4),
                    }
                )
    flagged.sort(key=lambda f: f["gap"], reverse=True)
    return flagged


def run_full_mode(cases: List[dict], args: argparse.Namespace) -> int:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("full mode requires OPENAI_API_KEY — run retrieval-only mode instead, or set the key.")
        return 1

    if not cost_guard(len(cases), args.yes):
        return 1

    records = generate_grounded_answers(cases, args)
    evaluated = [r for r in records if r["evaluated"]]
    skipped = [r for r in records if not r["evaluated"]]
    n_skipped_no_context = sum(1 for r in skipped if r.get("skipped_reason") == "no_context_retrieved")

    per_case: List[dict] = []
    if evaluated:
        df = run_ragas_evaluate(evaluated)
        for i, rec in enumerate(evaluated):
            row = df.iloc[i]
            per_case.append(
                {
                    "question": rec["question"],
                    "source_collection": rec["source_collection"],
                    "query_style": rec["query_style"],
                    "evaluated": True,
                    "faithfulness": round(float(row["faithfulness"]), 4),
                    "answer_relevancy": round(float(row["answer_relevancy"]), 4),
                    "context_precision": round(float(row["context_precision"]), 4),
                    "context_recall": round(float(row["context_recall"]), 4),
                    "answer": rec["answer"],
                }
            )
    for rec in skipped:
        per_case.append(
            {
                "question": rec["question"],
                "source_collection": rec["source_collection"],
                "query_style": rec["query_style"],
                "evaluated": False,
                "skipped_reason": rec["skipped_reason"],
            }
        )

    scored = [r for r in per_case if r["evaluated"]]
    overall = _aggregate_full_group(scored)
    by_collection = _group_by_full(scored, "source_collection")
    by_style = _group_by_full(scored, "query_style")
    flagged = flag_weak_collections(by_collection)

    from ragas import __version__ as ragas_version

    payload = {
        "mode": "full",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "ragas_version": ragas_version,
        "judge_model": JUDGE_MODEL,
        "judge_embeddings": JUDGE_EMBEDDINGS,
        "params": {"bi_encoder_top_n": args.bi_encoder_top_n, "rerank_top_k": args.rerank_top_k},
        "n_cases_total": len(cases),
        "n_cases_evaluated": len(evaluated),
        "n_cases_skipped_no_context": n_skipped_no_context,
        "overall": overall,
        "by_collection": by_collection,
        "by_style": by_style,
        "flagged": flagged,
        "per_case": per_case,
    }
    FULL_OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {FULL_OUTPUT_PATH}")

    trace_path = write_trace_jsonl("full", per_case)
    print(f"Wrote trace {trace_path}")

    metric_keys = list(TARGET_METRICS.keys())
    _print_table("By collection:", by_collection, metric_keys)
    _print_table("By style:", by_style, metric_keys)
    print(f"\nOverall: {overall}")

    if flagged:
        print("\nWEAK COLLECTIONS/METRICS — Day 26 candidates")
        for f in flagged:
            print(f"  {f['source_collection']:30s} {f['metric']:20s} score={f['score']:.4f} target={f['target']:.4f} gap={f['gap']:.4f}")
    else:
        print("\nNo collections/metrics below target.")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if not run_preflight(strict=True):
        print("Preflight failed — fix ChromaDB ingestion before evaluating.")
        return 1

    cases, chunk_id_convention = load_test_cases()
    cases = apply_filters(cases, args.collections, args.styles, args.limit)
    print_case_breakdown(cases)

    if not cases:
        print("No test cases matched the given filters.")
        return 1

    if args.mode == "retrieval-only":
        return run_retrieval_only_mode(cases, chunk_id_convention, args)
    return run_full_mode(cases, args)


if __name__ == "__main__":
    sys.exit(main())
