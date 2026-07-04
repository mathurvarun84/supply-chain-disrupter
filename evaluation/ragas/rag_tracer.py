"""
rag_tracer.py — Non-invasive RAG trace interceptor (RAGAS Phase 2).

Wraps the RAG context builders and the LLM call so (question, contexts,
answer) triples are captured at inference time WITHOUT modifying any
production agent code. Wrapping is done via monkey-patching inside an
explicit context manager — originals are ALWAYS restored on exit, even
when the wrapped code raises.

Patched functions (module attribute paths verified against the codebase):
    src.rag.retriever.retrieve_and_rerank      (two-stage retrieval)
    src.rag.utils.query_chroma_rag             (dashboard search, Stage-1 only)
    src.utils.openai_utils.call_openai_structured  (all agent LLM calls)

Because agent modules may hold their own imported reference to these
functions, the collector also patches every already-imported module whose
attribute `is` the original function object, and restores all of them.

Output: evaluation/ragas/traces/trace_{YYYYMMDD_HHMMSS}.jsonl — the raw
material Phase 3 (run_evaluation.py) will consume.

CLI smoke mode (zero API cost):
    python -m evaluation.ragas.rag_tracer --smoke
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RAGAS_DIR = Path(__file__).resolve().parent
TRACES_DIR = _RAGAS_DIR / "traces"
TEST_DATASET_PATH = _RAGAS_DIR / "test_dataset.json"

# (module path, attribute, kind) — kinds: "retrieval" | "llm"
PATCH_SPECS = [
    ("src.rag.retriever", "retrieve_and_rerank", "retrieval"),
    ("src.rag.utils", "query_chroma_rag", "retrieval"),
    ("src.utils.openai_utils", "call_openai_structured", "llm"),
]


def _extract_answer_text(parsed: Any) -> str:
    """
    Derive the RAGAS 'answer' string from a parsed Pydantic result.

    Exactly three branches (no other heuristics):
      1. `rationale` field (LLMSignal)          → the rationale text.
      2. `ranked_actions` (MitigationLLMOutput) → actions joined with "; ",
         prefixed by urgency.
      3. Anything else                          → parsed.model_dump_json().
    """
    rationale = getattr(parsed, "rationale", None)
    if rationale is not None:
        return str(rationale)

    ranked_actions = getattr(parsed, "ranked_actions", None)
    if ranked_actions is not None:
        urgency = getattr(parsed, "urgency", "")
        joined = "; ".join(str(a) for a in ranked_actions)
        return f"{urgency}: {joined}" if urgency else joined

    return parsed.model_dump_json()


def _chunk_id_for_hit(
    hit: Dict[str, Any], collection: str, convention: str
) -> str:
    """
    Recover a comparable chunk id from a retrieval hit.

    ChromaDB retrieval results don't carry ids, but native ids are
    deterministic (sha256 of "collection|source_file|chunk_index", first 32
    hex chars — see src/rag/collections.py::_doc_id), so they can be
    reconstructed from metadata. Falls back to sha256 of the chunk text.
    """
    meta = hit.get("metadata") or {}
    if convention == "chromadb_native":
        source_file = meta.get("source_file")
        chunk_index = meta.get("chunk_index")
        if source_file is not None and chunk_index is not None:
            key = f"{collection}|{source_file}|{chunk_index}".encode()
            return hashlib.sha256(key).hexdigest()[:32]
    return hashlib.sha256((hit.get("text") or "").encode()).hexdigest()


def write_trace_jsonl(records: List[dict], traces_dir: Path = TRACES_DIR) -> Path:
    """Write one JSON object per line to traces/trace_{YYYYMMDD_HHMMSS}.jsonl."""
    traces_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = traces_dir / f"trace_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d trace record(s) to %s", len(records), path)
    return path


class RAGTraceCollector:
    """
    Context manager that intercepts RAG retrieval + LLM generation for
    RAGAS evaluation, without modifying production code.

    Usage:
        with RAGTraceCollector(integration_point="risk_classifier") as tracer:
            tracer.mark_question("Taiwan earthquake semiconductor impact")
            ctx = build_risk_classifier_context(disruption_type=..., ...)
            # ... agent runs its LLM call ...
        records = tracer.records   # list of trace record dicts
        tracer.write_jsonl()       # persist to traces/

    One trace record groups ALL retrieval calls plus the next LLM call made
    after each mark_question() invocation (one agent call can issue 1-3
    retrieval queries). If mark_question is never called, the first retrieval
    query string is used as the question.
    """

    def __init__(self, integration_point: str, traces_dir: Path = TRACES_DIR):
        self.integration_point = integration_point
        self.traces_dir = traces_dir
        self.records: List[dict] = []
        self._pending: Optional[dict] = None
        self._patches: List[tuple] = []  # (module, attr, original)

    # -- patching ------------------------------------------------------------

    def __enter__(self) -> "RAGTraceCollector":
        for module_path, attr, kind in PATCH_SPECS:
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
        return False  # re-raise any exception after restore

    def _make_wrapper(self, original, attr: str, kind: str):
        if kind == "retrieval":

            @functools.wraps(original)
            def retrieval_wrapper(*args, **kwargs):
                results = original(*args, **kwargs)
                try:
                    self._on_retrieval(attr, args, kwargs, results)
                except Exception as trace_exc:
                    logger.warning("Trace capture failed (non-blocking): %s", trace_exc)
                return results

            return retrieval_wrapper

        @functools.wraps(original)
        def llm_wrapper(*args, **kwargs):
            t0 = time.monotonic()
            parsed = original(*args, **kwargs)
            latency_ms = (time.monotonic() - t0) * 1000.0
            try:
                self._on_llm(args, kwargs, parsed, latency_ms)
            except Exception as trace_exc:
                logger.warning("Trace capture failed (non-blocking): %s", trace_exc)
            return parsed

        return llm_wrapper

    # -- question / record lifecycle ------------------------------------------

    def mark_question(self, question_text: str) -> None:
        """Start a new trace record; finalizes any record in progress."""
        self._finalize_pending()
        self._pending = self._new_pending(question_text)

    def _new_pending(self, question: Optional[str]) -> dict:
        return {
            "question": question,
            "contexts": [],
            "answer": None,
            "integration_point": self.integration_point,
            "collections_queried": [],
            "retrieval_meta": [],
            "retrieval_queries": [],
            "model": None,
            "latency_ms": None,
            "traced_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _ensure_pending(self) -> dict:
        if self._pending is None:
            self._pending = self._new_pending(None)
        return self._pending

    def _finalize_pending(self) -> None:
        """Close out the in-progress record (called on mark_question(), __exit__, or after the LLM call)."""
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        if pending["question"] is None:
            # Fallback: first retrieval query string acts as the question.
            queries = pending.get("retrieval_queries") or []
            pending["question"] = queries[0] if queries else ""
        if not pending["contexts"] and pending["answer"] is None:
            return  # empty record — nothing was captured
        self.records.append(pending)

    # -- capture handlers ------------------------------------------------------

    def _on_retrieval(self, attr: str, args, kwargs, results) -> None:
        """Record one retrieval call's query, collection(s), and hit texts onto the pending record."""
        pending = self._ensure_pending()

        query = args[0] if args else kwargs.get("query", kwargs.get("query_text", ""))
        pending["retrieval_queries"].append(str(query))

        # retrieve_and_rerank and query_chroma_rag pass the collection(s) differently
        # (positional list vs single named kwarg/positional arg) — normalise both to a list.
        if attr == "retrieve_and_rerank":
            collections = args[1] if len(args) > 1 else kwargs.get("collections", [])
        else:  # query_chroma_rag
            collection_name = kwargs.get("collection_name")
            if collection_name is None and len(args) > 2:
                collection_name = args[2]
            if collection_name is None:
                try:
                    from src.rag.utils import DEFAULT_COLLECTION_NAME
                    collection_name = DEFAULT_COLLECTION_NAME
                except ImportError:
                    collection_name = "unknown"
            collections = [collection_name]

        for cname in collections:
            if cname not in pending["collections_queried"]:
                pending["collections_queried"].append(cname)

        for hit in results or []:
            meta = hit.get("metadata") or {}
            pending["contexts"].append(hit.get("text", ""))
            score_entry: Dict[str, Any] = {
                "source_file": meta.get("source_file", meta.get("source", "unknown")),
            }
            # Cross-encoder-reranked hits carry a rerank score; Stage-1-only hits
            # (query_chroma_rag) only have the raw ANN distance — record whichever applies.
            if "cross_encoder_score" in hit:
                score_entry["cross_encoder_score"] = hit["cross_encoder_score"]
            else:
                score_entry["distance"] = hit.get("distance")
            pending["retrieval_meta"].append(score_entry)

    def _on_llm(self, args, kwargs, parsed, latency_ms: float) -> None:
        pending = self._ensure_pending()
        pending["answer"] = _extract_answer_text(parsed)
        pending["model"] = kwargs.get("model") or (args[3] if len(args) > 3 else None)
        pending["latency_ms"] = round(latency_ms, 1)
        # The LLM call closes the (retrievals → generation) group.
        self._finalize_pending()

    # -- persistence -----------------------------------------------------------

    def write_jsonl(self) -> Path:
        return write_trace_jsonl(self.records, self.traces_dir)


# ---------------------------------------------------------------------------
# Retrieval-only tracing (dashboard-search integration point + Phase 3's
# no-LLM-cost mode). Makes ZERO LLM calls — runs with no API key.
# ---------------------------------------------------------------------------

def trace_retrieval_only(
    test_cases: List[dict],
    use_two_stage: bool = True,
    chunk_id_convention: str = "chromadb_native",
    n_results: int = 5,
) -> List[dict]:
    """
    For each gold test case, run retrieval (retrieve_and_rerank against the
    case's source_collection if use_two_stage else query_chroma_rag) and
    return RAGAS-ready records:
      {question, contexts, ground_truth, answer: None,
       source_collection, source_chunk_id, query_style,
       retrieved_chunk_ids, gold_chunk_retrieved: bool, gold_chunk_rank: int|None}

    gold_chunk_retrieved / gold_chunk_rank compare retrieved chunk ids
    (reconstructed from metadata, or sha256 of retrieved texts — matching the
    dataset's chunk_id_convention) against source_chunk_id: a free hit-rate
    metric that needs no LLM at all.
    """
    # Resolve via module attributes at call time so RAGTraceCollector patches
    # (and test monkey-patches) are honoured.
    retriever_mod = importlib.import_module("src.rag.retriever")
    rag_utils_mod = importlib.import_module("src.rag.utils")

    records: List[dict] = []
    for case in test_cases:
        question = case["question"]
        collection = case.get("source_collection", "")

        if use_two_stage:
            hits = retriever_mod.retrieve_and_rerank(
                question,
                [collection],
                bi_encoder_top_n=10,
                rerank_top_k=n_results,
            )
        else:
            hits = rag_utils_mod.query_chroma_rag(
                question, n_results=n_results, collection_name=collection
            )

        # Reconstruct each retrieved chunk's id (ChromaDB hits don't carry it back)
        # so it can be compared against the gold source_chunk_id below.
        retrieved_ids = [
            _chunk_id_for_hit(hit, collection, chunk_id_convention)
            for hit in hits or []
        ]
        gold_id = case.get("source_chunk_id", "")
        gold_rank: Optional[int] = None
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid == gold_id:
                gold_rank = rank
                break

        records.append(
            {
                "question": question,
                "contexts": [hit.get("text", "") for hit in hits or []],
                "ground_truth": case.get("ground_truth", ""),
                "answer": None,
                "source_collection": collection,
                "source_chunk_id": gold_id,
                "query_style": case.get("query_style", ""),
                "retrieved_chunk_ids": retrieved_ids,
                "gold_chunk_retrieved": gold_rank is not None,
                "gold_chunk_rank": gold_rank,
            }
        )
    return records


# ---------------------------------------------------------------------------
# CLI smoke mode
# ---------------------------------------------------------------------------

def _smoke(n_cases: int = 5) -> int:
    if not TEST_DATASET_PATH.exists():
        print(
            f"Missing {TEST_DATASET_PATH} — run generate_test_dataset.py first."
        )
        return 1

    dataset = json.loads(TEST_DATASET_PATH.read_text(encoding="utf-8"))
    convention = dataset.get("metadata", {}).get(
        "chunk_id_convention", "chromadb_native"
    )
    cases = dataset.get("test_cases", [])[:n_cases]
    if not cases:
        print("test_dataset.json contains no test cases.")
        return 1

    print(f"Smoke: retrieval-only tracing on {len(cases)} case(s), zero API cost.\n")
    records = trace_retrieval_only(cases, chunk_id_convention=convention)

    for rec in records:
        print(
            f"  q={rec['question'][:60]!r:64s} n_contexts={len(rec['contexts'])} "
            f"gold_retrieved={rec['gold_chunk_retrieved']} "
            f"gold_rank={rec['gold_chunk_rank']}"
        )

    path = write_trace_jsonl(records)
    hits = sum(1 for r in records if r["gold_chunk_retrieved"])
    print(f"\nGold-chunk hit rate: {hits}/{len(records)}")
    print(f"Trace written: {path}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RAGAS trace interceptor (Phase 2)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run retrieval-only tracing on the first 5 gold test cases.",
    )
    args = parser.parse_args()
    if args.smoke:
        return _smoke()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
