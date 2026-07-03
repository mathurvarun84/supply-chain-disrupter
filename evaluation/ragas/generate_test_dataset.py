"""
generate_test_dataset.py — Chunk-grounded gold QA dataset generator (RAGAS Phase 1).

Pulls REAL indexed chunks from ChromaDB, generates one QA pair per selected
chunk via Claude structured output (call_anthropic_structured), dedups, and
writes evaluation/ragas/test_dataset.json plus a human-review markdown table.

DESIGN DECISIONS (locked):
  1. Chunk-grounded — every QA pair is generated from one specific chunk that
     exists in ChromaDB. Never topic-grounded.
  2. Both query styles, tagged — ~60% "agent_pattern" (keyword-dense retrieval
     strings mimicking build_risk_classifier_context / build_mitigation_context
     queries) and ~40% "natural_question" (evaluator-style questions).

Usage:
    python -m evaluation.ragas.generate_test_dataset

Without ANTHROPIC_API_KEY this is a dry run: preflight + allocation table +
sampled chunk ids are printed, nothing is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from evaluation.ragas.preflight import EXPECTED_COLLECTIONS, run_preflight
from src.utils.anthropic_utils import (
    MODEL_CLAUDE_HAIKU,
    call_anthropic_structured,
    has_anthropic_api_key,
)

logger = logging.getLogger(__name__)

_RAGAS_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = _RAGAS_DIR / "test_dataset.json"
REVIEW_PATH = _RAGAS_DIR / "test_dataset_review.md"
EXCLUDED_IDS_PATH = _RAGAS_DIR / "excluded_chunk_ids.txt"

MIN_CHUNK_CHARS = 200          # skip stub/near-empty chunks
TARGET_TOTAL_PAIRS = 36
AGENT_PATTERN_RATIO = 0.60     # DECISION 2
RANDOM_SEED = 42               # reproducible chunk sampling
ALLOCATION_FLOOR = 6           # min pairs per collection
ALLOCATION_CEILING = 18        # max pairs per collection
MAX_REPLACEMENTS_PER_SLOT = 2  # replacement attempts after an unusable chunk

MIN_QUESTION_WORDS = 4         # degenerate-output guard
MIN_GROUND_TRUTH_WORDS = 8


class GoldQAPair(BaseModel):
    """One chunk-grounded gold QA pair for RAGAS evaluation."""

    question: str = Field(
        ..., description="Question answerable ONLY from the provided chunk"
    )
    ground_truth: str = Field(
        ...,
        description=(
            "Answer faithful to the chunk content, 1-3 sentences, "
            "no outside knowledge"
        ),
    )
    chunk_is_usable: bool = Field(
        ...,
        description=(
            "False if the chunk is too fragmentary/tabular to support a "
            "meaningful QA pair"
        ),
    )
    unusable_reason: Optional[str] = Field(
        None, description="Why the chunk was rejected, if chunk_is_usable is False"
    )


SYSTEM_PROMPT = """You are Claude, generating evaluation QA pairs for a RAG system covering semiconductor /
electronics supply chain risk. You will be given ONE document chunk and a
required question style.

HARD RULES:
1. The question must be answerable using ONLY the provided chunk. If a
   reader had only this chunk, they could answer fully.
2. ground_truth must be derived ONLY from the chunk. Do not add facts,
   dates, numbers, or entity names that are not in the chunk — even if you
   know them to be true.
3. If the chunk is too fragmentary, tabular, or content-poor to support a
   meaningful question, set chunk_is_usable=false and explain why.
4. Question style "agent_pattern": produce a keyword-dense retrieval query
   (no question mark, 6-12 content words), e.g.
   "BIS export control semiconductor restriction entity list alternate sourcing".
   Question style "natural_question": produce a natural evaluator-style
   question, e.g. "What restrictions did the October 2022 BIS rules place
   on advanced semiconductor exports?".
5. Never reference "the chunk", "the document", or "the text" in the
   question or the ground_truth."""


# ---------------------------------------------------------------------------
# Excluded chunk ids (manual review gate — STEP 4)
# ---------------------------------------------------------------------------

def load_excluded_chunk_ids(path: Path = EXCLUDED_IDS_PATH) -> set:
    """
    Read excluded_chunk_ids.txt (one id per line, '#' comments allowed).

    Chunks listed here were flagged during manual review of
    test_dataset_review.md and must never be sampled again.
    """
    if not path.exists():
        return set()
    excluded: set = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            excluded.add(stripped)
    if excluded:
        logger.info("Loaded %d excluded chunk ids from %s", len(excluded), path)
    return excluded


# ---------------------------------------------------------------------------
# Chunk fetch + stratified sampling
# ---------------------------------------------------------------------------

def resolve_chunk_id(chunk_id: Optional[str], text: str) -> Tuple[str, str]:
    """Return (chunk_id, convention) — ChromaDB native id or sha256 fallback."""
    if chunk_id:
        return chunk_id, "chromadb_native"
    return hashlib.sha256(text.encode()).hexdigest(), "sha256_fallback"


def fetch_eligible_chunks(excluded_ids: set) -> Dict[str, List[dict]]:
    """
    Pull ALL chunks from each expected collection, filtering out chunks
    shorter than MIN_CHUNK_CHARS and any manually excluded ids.

    Returns {collection_name: [{"id", "text", "metadata"}, ...]}.
    """
    from src.rag.utils import get_chroma_client

    client = get_chroma_client()
    out: Dict[str, List[dict]] = {}
    for cname in EXPECTED_COLLECTIONS:
        collection = client.get_collection(cname)
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or [{}] * len(documents)

        chunks: List[dict] = []
        for cid, doc, meta in zip(ids, documents, metadatas):
            text = doc or ""
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunk_id, _ = resolve_chunk_id(cid, text)
            if chunk_id in excluded_ids:
                continue
            chunks.append({"id": chunk_id, "text": text, "metadata": meta or {}})
        out[cname] = chunks
        logger.info("%s: %d eligible chunks (>= %d chars)", cname, len(chunks), MIN_CHUNK_CHARS)
    return out


def compute_allocation(
    chunk_counts: Dict[str, int],
    total: int = TARGET_TOTAL_PAIRS,
    floor: int = ALLOCATION_FLOOR,
    ceiling: int = ALLOCATION_CEILING,
) -> Dict[str, int]:
    """
    Allocate QA-pair slots proportionally to eligible chunk counts, with a
    floor and ceiling per collection so none is starved or dominant.

    The ceiling is additionally capped at the collection's own chunk count
    (sampling is without replacement). Rounding drift is distributed to the
    collections with the most chunks first, respecting caps. If total
    capacity is below `total`, the shortfall is accepted and logged.
    """
    caps = {c: min(ceiling, n) for c, n in chunk_counts.items()}
    floors = {c: min(floor, caps[c]) for c in chunk_counts}
    total_chunks = sum(chunk_counts.values())

    allocation: Dict[str, int] = {}
    for cname, n in chunk_counts.items():
        raw = round(total * n / total_chunks) if total_chunks else 0
        allocation[cname] = max(floors[cname], min(caps[cname], raw))

    diff = total - sum(allocation.values())
    order = sorted(chunk_counts, key=lambda c: chunk_counts[c], reverse=True)
    while diff != 0:
        moved = False
        for cname in order:
            if diff > 0 and allocation[cname] < caps[cname]:
                allocation[cname] += 1
                diff -= 1
                moved = True
            elif diff < 0 and allocation[cname] > floors[cname]:
                allocation[cname] -= 1
                diff += 1
                moved = True
            if diff == 0:
                break
        if not moved:
            logger.warning(
                "Allocation capacity exhausted — %d slots short of target %d",
                diff, total,
            )
            break
    return allocation


def sample_chunks(
    chunks: List[dict],
    k: int,
    rng: random.Random,
    excluded_ids: Optional[set] = None,
) -> List[dict]:
    """Sample k chunks WITHOUT replacement, never returning an excluded id."""
    excluded_ids = excluded_ids or set()
    pool = [c for c in chunks if c["id"] not in excluded_ids]
    return rng.sample(pool, min(k, len(pool)))


def assign_styles(
    sampled: List[dict], rng: random.Random
) -> List[Tuple[dict, str]]:
    """
    Shuffle sampled chunks and assign the first round(n * AGENT_PATTERN_RATIO)
    slots "agent_pattern", the rest "natural_question".
    """
    shuffled = list(sampled)
    rng.shuffle(shuffled)
    n_agent = round(len(shuffled) * AGENT_PATTERN_RATIO)
    return [
        (chunk, "agent_pattern" if i < n_agent else "natural_question")
        for i, chunk in enumerate(shuffled)
    ]


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _build_user_message(collection_name: str, chunk: dict, query_style: str) -> str:
    return (
        f"COLLECTION: {collection_name}\n"
        f"SOURCE FILE: {chunk['metadata'].get('source_file', chunk['metadata'].get('source', 'unknown'))}\n"
        f"REQUIRED QUESTION STYLE: {query_style}\n\n"
        f"CHUNK:\n{chunk['text']}"
    )


def _generate_one(
    collection_name: str, chunk: dict, query_style: str
) -> Optional[GoldQAPair]:
    """One structured Claude call for one chunk. None on hard failure."""
    try:
        return call_anthropic_structured(
            SYSTEM_PROMPT,
            _build_user_message(collection_name, chunk, query_style),
            response_model=GoldQAPair,
            model=MODEL_CLAUDE_HAIKU,
        )
    except Exception as exc:
        logger.error(
            "LLM generation failed for chunk %s (%s): %s",
            chunk["id"], collection_name, exc,
        )
        return None


def generate_qa_pairs(
    collection_name: str,
    slots: List[Tuple[dict, str]],
    replacement_pool: List[dict],
) -> Tuple[List[dict], List[str]]:
    """
    Generate one QA record per slot. If Claude marks a chunk unusable (or the
    call fails), sample a replacement chunk from the same collection's pool —
    at most MAX_REPLACEMENTS_PER_SLOT times per slot, then the slot is dropped.

    Returns (records, rejected_chunk_ids).
    """
    records: List[dict] = []
    rejected: List[str] = []
    pool = list(replacement_pool)

    for chunk, query_style in slots:
        current = chunk
        replacements_used = 0
        while True:
            pair = _generate_one(collection_name, current, query_style)
            if pair is not None and pair.chunk_is_usable:
                records.append(
                    {
                        "question": pair.question.strip(),
                        "ground_truth": pair.ground_truth.strip(),
                        "source_collection": collection_name,
                        "source_chunk_id": current["id"],
                        "source_file": current["metadata"].get(
                            "source_file", current["metadata"].get("source", "unknown")
                        ),
                        "query_style": query_style,
                    }
                )
                break

            reason = pair.unusable_reason if pair is not None else "LLM call failed"
            rejected.append(current["id"])
            logger.warning(
                "Rejected chunk %s in %s: %s", current["id"], collection_name, reason
            )
            if replacements_used >= MAX_REPLACEMENTS_PER_SLOT or not pool:
                logger.warning(
                    "Slot dropped in %s after %d replacement attempt(s)",
                    collection_name, replacements_used,
                )
                break
            current = pool.pop(0)
            replacements_used += 1

    return records, rejected


# ---------------------------------------------------------------------------
# Dedup + degenerate guards (mirrors save_all_qa_pairs in
# fine_tuning/generate_training_data.py)
# ---------------------------------------------------------------------------

def dedup_records(records: List[dict]) -> Tuple[List[dict], int]:
    """Drop near-duplicates keyed on question[:50] + ground_truth[:30]."""
    seen: set = set()
    unique: List[dict] = []
    for rec in records:
        key = rec["question"][:50] + rec["ground_truth"][:30]
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    dropped = len(records) - len(unique)
    if dropped:
        logger.info("Dedup dropped %d duplicate pair(s)", dropped)
    return unique, dropped


def filter_degenerate(records: List[dict]) -> Tuple[List[dict], int]:
    """Drop pairs with question < 4 words or ground_truth < 8 words."""
    kept = [
        rec
        for rec in records
        if len(rec["question"].split()) >= MIN_QUESTION_WORDS
        and len(rec["ground_truth"].split()) >= MIN_GROUND_TRUTH_WORDS
    ]
    dropped = len(records) - len(kept)
    if dropped:
        logger.info("Degenerate guard dropped %d pair(s)", dropped)
    return kept, dropped


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_dataset(
    records: List[dict],
    allocation: Dict[str, int],
    chunk_id_convention: str,
    rejected_chunks: int,
    dropped_duplicates: int,
    dropped_degenerate: int,
    output_path: Path,
) -> None:
    style_counts: Dict[str, int] = {"agent_pattern": 0, "natural_question": 0}
    for rec in records:
        style_counts[rec["query_style"]] = style_counts.get(rec["query_style"], 0) + 1

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_CLAUDE_HAIKU,
            "target_pairs": TARGET_TOTAL_PAIRS,
            "final_pairs": len(records),
            "chunk_id_convention": chunk_id_convention,
            "allocation": allocation,
            "style_counts": style_counts,
            "rejected_chunks": rejected_chunks,
            "dropped_duplicates": dropped_duplicates,
            "dropped_degenerate": dropped_degenerate,
        },
        "test_cases": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} test cases to {output_path}")


def write_review_markdown(records: List[dict], review_path: Path) -> None:
    """
    Human-review table (STEP 4). Varun reviews this before Phase 3; bad pairs
    get their source_chunk_id added to excluded_chunk_ids.txt.
    """
    lines = [
        "# RAGAS Gold Test Dataset — Manual Review",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ·  {len(records)} pairs",
        "",
        "To reject a pair, add its `source_chunk_id` (shown in test_dataset.json)",
        f"to `{EXCLUDED_IDS_PATH.name}` (one id per line, `#` comments allowed)",
        "and re-run the generator.",
        "",
        "| # | Collection | Style | Question | Ground Truth | Source File |",
        "|---|-----------|-------|----------|--------------|-------------|",
    ]
    for i, rec in enumerate(records, 1):
        question = rec["question"].replace("|", "\\|")
        truth = rec["ground_truth"][:160].replace("|", "\\|")
        lines.append(
            f"| {i} | {rec['source_collection']} | {rec['query_style']} "
            f"| {question} | {truth} | {rec['source_file']} |"
        )
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote review table to {review_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_allocation_table(
    counts: Dict[str, int], allocation: Dict[str, int]
) -> None:
    print("\nAllocation (proportional, floor=%d, ceiling=%d, total=%d):"
          % (ALLOCATION_FLOOR, ALLOCATION_CEILING, TARGET_TOTAL_PAIRS))
    print(f"  {'collection':35s} {'eligible':>9s} {'allocated':>10s}")
    for cname in counts:
        print(f"  {cname:35s} {counts[cname]:9d} {allocation[cname]:10d}")
    print(f"  {'TOTAL':35s} {sum(counts.values()):9d} {sum(allocation.values()):10d}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Preflight gate — never sample from a broken store.
    if not run_preflight(strict=True):
        print("Preflight failed — fix ChromaDB ingestion before generating.")
        return 1

    excluded_ids = load_excluded_chunk_ids(EXCLUDED_IDS_PATH)
    chunks_by_collection = fetch_eligible_chunks(excluded_ids)

    counts = {c: len(chunks) for c, chunks in chunks_by_collection.items()}
    allocation = compute_allocation(counts)
    _print_allocation_table(counts, allocation)

    rng = random.Random(RANDOM_SEED)
    sampled_by_collection: Dict[str, List[dict]] = {}
    pools_by_collection: Dict[str, List[dict]] = {}
    convention = "chromadb_native"
    for cname, chunks in chunks_by_collection.items():
        sampled = sample_chunks(chunks, allocation[cname], rng, excluded_ids)
        sampled_by_collection[cname] = sampled
        sampled_ids = {c["id"] for c in sampled}
        pool = [c for c in chunks if c["id"] not in sampled_ids]
        rng.shuffle(pool)
        pools_by_collection[cname] = pool

    if not has_anthropic_api_key():
        print("\nSampled chunk ids (dry run):")
        for cname, sampled in sampled_by_collection.items():
            for chunk in sampled:
                print(f"  {cname}: {chunk['id']}")
        print("\nDry run complete — set ANTHROPIC_API_KEY to generate QA pairs.")
        return 0

    all_records: List[dict] = []
    all_rejected: List[str] = []
    for cname, sampled in sampled_by_collection.items():
        slots = assign_styles(sampled, rng)
        print(f"\nGenerating {len(slots)} pairs for {cname} ...")
        records, rejected = generate_qa_pairs(cname, slots, pools_by_collection[cname])
        all_records.extend(records)
        all_rejected.extend(rejected)

    unique, dropped_duplicates = dedup_records(all_records)
    final, dropped_degenerate = filter_degenerate(unique)

    style_final = {
        "agent_pattern": sum(1 for r in final if r["query_style"] == "agent_pattern"),
        "natural_question": sum(
            1 for r in final if r["query_style"] == "natural_question"
        ),
    }
    print(
        f"\nFinal: {len(final)} pairs "
        f"(rejected chunks: {len(all_rejected)}, duplicates: {dropped_duplicates}, "
        f"degenerate: {dropped_degenerate}) | styles: {style_final}"
    )

    write_dataset(
        final,
        allocation=allocation,
        chunk_id_convention=convention,
        rejected_chunks=len(all_rejected),
        dropped_duplicates=dropped_duplicates,
        dropped_degenerate=dropped_degenerate,
        output_path=OUTPUT_PATH,
    )
    write_review_markdown(final, REVIEW_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
