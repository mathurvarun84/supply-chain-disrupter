"""
preflight.py — Verify ChromaDB is in the expected state before generating
the gold test dataset. Run this FIRST. If it fails, fix ingestion before
running generate_test_dataset.py.

Usage:
    python -m evaluation.ragas.preflight

Exit code 0 only if all three expected collections exist and each has
>= MIN_CHUNKS_PER_COLLECTION chunks. Otherwise exit 1 with a precise
message naming what is missing.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)

EXPECTED_COLLECTIONS = [
    "historical_precedents",
    "export_control_corpus",
    "india_sourcing_corpus",
]
MIN_CHUNKS_PER_COLLECTION = 5

REAL_TAG = "[SOURCE: REAL]"
SYNTH_TAG = "[SOURCE: SYNTHESIZED]"


def _count_source_tags(collection) -> Dict[str, int]:
    """Substring-count [SOURCE: REAL] / [SOURCE: SYNTHESIZED] across all documents."""
    result = collection.get(include=["documents"])
    documents = result.get("documents") or []
    real = sum(1 for doc in documents if doc and REAL_TAG in doc)
    synth = sum(1 for doc in documents if doc and SYNTH_TAG in doc)
    return {"real": real, "synthesized": synth, "total": len(documents)}


def run_preflight(strict: bool = True) -> bool:
    """
    Verify the three expected RAG collections exist and hold real content.

    Prints a full inventory of ALL collections, samples 3 chunks per expected
    collection for a human sanity check, and reports [SOURCE: REAL] vs
    [SOURCE: SYNTHESIZED] tag counts per collection.

    Returns True when every expected collection exists with
    >= MIN_CHUNKS_PER_COLLECTION chunks. When strict is False, failures are
    reported but the return value is still accurate — callers decide.
    """
    try:
        from src.rag.utils import get_chroma_client
    except ImportError as exc:
        print(f"[PREFLIGHT] FAIL — cannot import src.rag.utils: {exc}")
        return False

    try:
        client = get_chroma_client()
        all_collections = client.list_collections()
    except Exception as exc:
        print(f"[PREFLIGHT] FAIL — cannot open ChromaDB: {exc}")
        return False

    print("=" * 70)
    print("  RAGAS PREFLIGHT — ChromaDB state check")
    print("=" * 70)

    # 1. Inventory of ALL collections with counts
    print("\nAll collections in store:")
    existing: Dict[str, object] = {}
    for col in all_collections:
        # chromadb >= 0.6 returns Collection objects; older versions may
        # return names. Normalise to (name, collection).
        name = col if isinstance(col, str) else col.name
        collection = client.get_collection(name) if isinstance(col, str) else col
        existing[name] = collection
        print(f"  {name:45s}  {collection.count():6d} chunks")
    if not existing:
        print("  (none)")

    problems: List[str] = []  # collected across all expected collections, reported together at the end

    # 2. Existence + minimum-size check for each collection Phase 1 will sample from
    for cname in EXPECTED_COLLECTIONS:
        print(f"\n--- {cname} ---")
        if cname not in existing:
            problems.append(f"collection '{cname}' does not exist")
            print("  MISSING")
            continue

        collection = existing[cname]
        count = collection.count()
        if count < MIN_CHUNKS_PER_COLLECTION:
            problems.append(
                f"collection '{cname}' has only {count} chunks "
                f"(need >= {MIN_CHUNKS_PER_COLLECTION})"
            )

        # 3. Sample 3 chunks — human sanity check that real content is indexed
        sample = collection.get(limit=3, include=["documents", "metadatas"])
        for doc, meta in zip(
            sample.get("documents") or [], sample.get("metadatas") or []
        ):
            source = (meta or {}).get("source_file", (meta or {}).get("source", "?"))
            preview = (doc or "").replace("\n", " ")[:120]
            print(f"  [{source}] {preview}")

        # 4. [SOURCE: REAL] vs [SOURCE: SYNTHESIZED] tag counts
        tags = _count_source_tags(collection)
        print(
            f"  Source tags: {tags['real']} {REAL_TAG} | "
            f"{tags['synthesized']} {SYNTH_TAG} | {tags['total']} total chunks"
        )
        if cname == "export_control_corpus" and tags["real"] == 0:
            print(
                "  *** WARNING ***  export_control_corpus has ZERO [SOURCE: REAL] "
                "chunks — the 11 new real docs have NOT been re-ingested. "
                "Run build_chroma_complete(flush_existing=False) / "
                "python scripts/build_rag_collections.py first."
            )

    print("\n" + "=" * 70)
    if problems:
        print("[PREFLIGHT] FAIL:")
        for p in problems:
            print(f"  - {p}")
        print("Fix ingestion (python scripts/build_rag_collections.py) and re-run.")
        return False

    print("[PREFLIGHT] OK — all expected collections present and populated.")
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return 0 if run_preflight(strict=True) else 1


if __name__ == "__main__":
    sys.exit(main())
