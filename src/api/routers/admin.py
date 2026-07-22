"""Admin page — GET /status reports SQLite + ChromaDB build state; POST
/db/build and POST /rag/build trigger (re)builds as FastAPI BackgroundTasks,
same fire-and-poll pattern as pipeline.py's POST /run and live_feed.py's
POST /refresh. Job state lives in-process only (module-level dicts), same
scope cut as pipeline.py's _RUN_PHASE — a server restart mid-build loses the
job's status, though the build itself (writing to outputs/supply_chain.db /
outputs/chromadb) is unaffected.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.api.schemas import (
    AdminJobStatus,
    AdminJobTriggerResponse,
    AdminStatusResponse,
    CorpusHealth,
    DatabaseStatus,
    TableListResponse,
    TableRowsResponse,
    TableSummary,
)
from src.utils import db_utils
from src.utils.db_utils import ensure_schema
from src.utils.etl_loader import get_sqlite_stats, load_excel_into_sqlite
from src.utils.ingestion_schema import ensure_ingestion_schema
from src.rag.utils import fetch_corpus_health

logger = logging.getLogger(__name__)
router = APIRouter()

_DB_LOCK = threading.Lock()
_RAG_LOCK = threading.Lock()

_DB_JOB_STATE: Dict[str, Any] = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "result": None}
_RAG_JOB_STATE: Dict[str, Any] = {"status": "idle", "started_at": None, "finished_at": None, "error": None, "result": None}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_db_build() -> None:
    with _DB_LOCK:
        _DB_JOB_STATE.update(status="running", started_at=_utcnow_iso(), finished_at=None, error=None, result=None)
        try:
            rows_loaded = load_excel_into_sqlite()
            ensure_schema()
            ensure_ingestion_schema()
            stats = get_sqlite_stats()
            _DB_JOB_STATE.update(
                status="complete",
                finished_at=_utcnow_iso(),
                result={"rows_loaded": rows_loaded, "stats": stats},
            )
        except Exception as exc:
            logger.exception("Admin DB build failed")
            _DB_JOB_STATE.update(status="failed", finished_at=_utcnow_iso(), error=str(exc))


def _run_rag_build(flush: bool) -> None:
    with _RAG_LOCK:
        _RAG_JOB_STATE.update(status="running", started_at=_utcnow_iso(), finished_at=None, error=None, result=None)
        try:
            from src.rag.utils import (
                CHROMA_DIR,
                build_chroma_complete,
                get_chroma_client,
                get_embedding_model,
                reset_chroma_client,
                reset_embedding_model_cache,
            )
            from src.rag.collections import (
                COLLECTION_CHUNK_CONFIG,
                COLLECTION_NAMES,
                RAG_DATA_ROOT,
                build_collection,
            )

            # A rebuild is the one place we must not trust the cached
            # embedding function: if new weights were pushed to the same
            # Hugging Face repo id since the process started, the
            # name-keyed cache in rag/utils.py has no way to detect that on
            # its own — only an explicit reset here (or a process restart)
            # picks up the update in time for this rebuild to re-embed with it.
            reset_embedding_model_cache()

            if flush and CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)
                reset_chroma_client()
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)

            monolithic_summary = build_chroma_complete(flush_existing=False)

            client = get_chroma_client()
            embed_fn = get_embedding_model()
            named_summary = []
            for cname in COLLECTION_NAMES:
                source_dir = RAG_DATA_ROOT / cname
                chunk_cfg = COLLECTION_CHUNK_CONFIG[cname]
                named_summary.append(
                    build_collection(
                        collection_name=cname,
                        source_dir=source_dir,
                        chroma_client=client,
                        embedding_fn=embed_fn,
                        chunk_cfg=chunk_cfg,
                    )
                )

            _RAG_JOB_STATE.update(
                status="complete",
                finished_at=_utcnow_iso(),
                result={"monolithic": monolithic_summary, "named_collections": named_summary},
            )
        except Exception as exc:
            logger.exception("Admin RAG build failed")
            _RAG_JOB_STATE.update(status="failed", finished_at=_utcnow_iso(), error=str(exc))


def _safe_sqlite_stats() -> Dict[str, Any]:
    """get_sqlite_stats() assumes the ETL tables (lite_master, ops_kpi, ...)
    exist once db_path.exists() — true after load_excel_into_sqlite(), but
    outputs/supply_chain.db may already exist with only the live-ingestion
    tables (ensure_ingestion_schema()) if the ETL step has never run. Treat
    that the same as "no database yet" so the Admin page prompts a build
    instead of 500ing."""
    try:
        return get_sqlite_stats()
    except Exception:
        logger.warning("get_sqlite_stats() failed — DB exists but ETL tables are missing", exc_info=True)
        return {"database_exists": False}


@router.get("/status", response_model=AdminStatusResponse)
def get_admin_status() -> AdminStatusResponse:
    """SQLite table counts/size (get_sqlite_stats), live ChromaDB named-collection
    counts (fetch_corpus_health), and the in-process status of the most recent
    build job for each — poll this after POST /db/build or POST /rag/build."""
    return AdminStatusResponse(
        database=DatabaseStatus(**_safe_sqlite_stats()),
        db_job=AdminJobStatus(**_DB_JOB_STATE),
        rag_job=AdminJobStatus(**_RAG_JOB_STATE),
        corpus=[CorpusHealth(**c) for c in fetch_corpus_health()],
    )


@router.post("/db/build", response_model=AdminJobTriggerResponse)
def build_database(background_tasks: BackgroundTasks) -> AdminJobTriggerResponse:
    """Rebuilds outputs/supply_chain.db from data/raw/supply_chain_lite_master.xlsx
    (load_excel_into_sqlite), then ensures the agent-output and live-ingestion
    tables exist (ensure_schema/ensure_ingestion_schema). Runs as a
    BackgroundTask — poll GET /status for progress."""
    if _DB_LOCK.locked():
        return AdminJobTriggerResponse(status="skipped_already_running", triggered_at=_utcnow_iso())
    background_tasks.add_task(_run_db_build)
    return AdminJobTriggerResponse(status="started", triggered_at=_utcnow_iso())


@router.post("/rag/build", response_model=AdminJobTriggerResponse)
def build_rag(background_tasks: BackgroundTasks, flush: bool = False) -> AdminJobTriggerResponse:
    """Builds both RAG stores: the monolithic electronics_supply_chain_knowledge
    collection (Excel + playbooks + static reports, src/rag/utils.py) and the
    three named collections (historical_precedents / export_control_corpus /
    india_sourcing_corpus, src/rag/collections.py) that Screen 6's corpus
    health cards read. flush=True wipes outputs/chromadb first for a full
    rebuild; default is an incremental upsert. Runs as a BackgroundTask —
    poll GET /status for progress."""
    if _RAG_LOCK.locked():
        return AdminJobTriggerResponse(status="skipped_already_running", triggered_at=_utcnow_iso())
    background_tasks.add_task(_run_rag_build, flush)
    return AdminJobTriggerResponse(status="started", triggered_at=_utcnow_iso())


# ── Data Explorer (read-only table browser) ──────────────────────────────
# Two GET-only endpoints so the Admin page can inspect outputs/supply_chain.db
# without a terminal sqlite3 session. table_name is a path segment, so it
# can't be parameterized like a value — every request re-checks it against a
# fresh sqlite_master read before it touches a SELECT string, and the
# connection itself is opened read-only (mode=ro) as a second, independent
# line of defense.

_INTERNAL_TABLE_PREFIXES = ("sqlite_",)


def _readonly_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_utils.DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/tables", response_model=TableListResponse)
def list_admin_tables() -> TableListResponse:
    """Every real table in supply_chain.db (via sqlite_master, never a
    hardcoded list) with row/column counts. Feeds DataExplorer.tsx's
    table dropdown via useAdminTables()."""
    try:
        conn = _readonly_connection()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    with conn:
        table_names = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not row["name"].startswith(_INTERNAL_TABLE_PREFIXES)
        ]
        summaries = []
        for name in table_names:
            # name comes only from sqlite_master itself here, never from
            # client input, so this interpolation is safe.
            row_count = conn.execute(f'SELECT COUNT(*) AS c FROM "{name}"').fetchone()["c"]
            column_count = len(conn.execute(f'PRAGMA table_info("{name}")').fetchall())
            summaries.append(TableSummary(name=name, row_count=row_count, column_count=column_count))

    return TableListResponse(tables=summaries)


@router.get("/tables/{table_name}", response_model=TableRowsResponse)
def get_admin_table_rows(
    table_name: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> TableRowsResponse:
    """One page of rows for a table, validated on every call against a
    freshly-queried sqlite_master list — never a cached whitelist that could
    go stale after a Recreate DB run. Unknown tables 404, not 500. Feeds
    DataExplorer.tsx's grid via useAdminTableRows()."""
    try:
        conn = _readonly_connection()
    except sqlite3.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    with conn:
        valid_names = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if table_name not in valid_names:
            raise HTTPException(status_code=404, detail=f"Unknown table: {table_name}")

        total_rows = conn.execute(f'SELECT COUNT(*) AS c FROM "{table_name}"').fetchone()["c"]
        columns = [r["name"] for r in conn.execute(f'PRAGMA table_info("{table_name}")')]

        offset = (page - 1) * page_size
        cursor = conn.execute(
            f'SELECT * FROM "{table_name}" LIMIT ? OFFSET ?', (page_size, offset)
        )
        rows = [dict(r) for r in cursor.fetchall()]

    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    return TableRowsResponse(
        table_name=table_name,
        columns=columns,
        rows=rows,
        total_rows=total_rows,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
