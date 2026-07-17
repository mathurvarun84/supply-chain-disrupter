import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path("outputs/supply_chain.db")


def get_connection(timeout: int = 30) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(query: str, params: Tuple = ()) -> List[sqlite3.Row]:
    try:
        with get_connection() as conn:
            return conn.execute(query, params).fetchall()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite query failed: {exc}") from exc


def execute_non_query(query: str, params: Tuple = ()) -> None:
    try:
        with get_connection() as conn:
            conn.execute(query, params)
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite write failed: {exc}") from exc


def execute_many(query: str, params_list: List[Tuple]) -> int:
    """Batch-insert helper. Returns the number of rows affected."""
    if not params_list:
        return 0
    try:
        with get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite batch write failed: {exc}") from exc


def ensure_schema() -> None:
    """Create only the writable agent-output tables.

    The complete source schema is created by etl_loader.load_excel_into_sqlite().
    """
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mitigation_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                event_date TEXT,
                port TEXT,
                sku TEXT,
                risk_label TEXT,
                summary TEXT,
                recommendation TEXT,
                urgency TEXT,
                cost_delta TEXT,
                rag_citations_json TEXT,
                india_sourcing_json TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_mitigation_actions_columns(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mitigation_actions_run_id ON mitigation_actions(run_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_execution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_ms REAL,
                error_message TEXT,
                langfuse_trace_id TEXT,
                langfuse_span_id TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_call_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_preview TEXT,
                full_prompt TEXT,
                full_response TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                latency_ms REAL,
                status TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                langfuse_trace_id TEXT,
                langfuse_generation_id TEXT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_call_log_run_id ON llm_call_log(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_call_log_agent ON llm_call_log(agent_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_execution_log_run_id ON agent_execution_log(run_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'Laptops',
                categories_json TEXT NOT NULL,
                series_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecast_output_run_id ON forecast_output(run_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                p10 REAL NOT NULL,
                p50 REAL NOT NULL,
                p90 REAL NOT NULL,
                revenue_at_risk_usd REAL NOT NULL,
                alternate_route TEXT NOT NULL,
                histogram_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mitigation_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                urgency TEXT NOT NULL,
                ranked_actions_json TEXT NOT NULL,
                rag_query_trace_json TEXT NOT NULL,
                india_sourcing_json TEXT NOT NULL,
                slack_preview TEXT NOT NULL,
                cost_delta_usd REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_classification_output (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                final_label TEXT NOT NULL,
                composite_score REAL,
                critical_flag INTEGER NOT NULL DEFAULT 0,
                full_result_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS guardrail_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dir TEXT NOT NULL,
                agent TEXT NOT NULL,
                pass_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT 'ΓÇö',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    ensure_simulation_schema()
    ensure_forecast_schema()


def _ensure_mitigation_actions_columns(conn: sqlite3.Connection) -> None:
    """Backfill newer mitigation_actions columns on older local databases."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(mitigation_actions)")}
    additions = [
        ("run_id", "TEXT"),
        ("summary", "TEXT"),
        ("urgency", "TEXT"),
        ("rag_citations_json", "TEXT"),
        ("india_sourcing_json", "TEXT"),
    ]
    for column_name, ddl in additions:
        if column_name not in columns:
            conn.execute(f"ALTER TABLE mitigation_actions ADD COLUMN {column_name} {ddl}")


def _normalize_mitigation_urgency(urgency: Optional[str]) -> str:
    if not urgency:
        return "LOW"
    urgency_upper = urgency.upper()
    if urgency_upper == "CRITICAL":
        return "IMMEDIATE"
    if urgency_upper in {"IMMEDIATE", "HIGH", "MEDIUM", "LOW"}:
        return urgency_upper
    if urgency_upper == "ROUTINE":
        return "LOW"
    return "LOW"


def ensure_sku_id_columns() -> None:
    """
    Idempotent migration: adds sku_id to lite_master and risk_classifications
    if either table predates the SKU_Product_Mapping crosswalk work, and adds
    the region/date index used by candidate-gathering. Safe to call on every
    app startup -- checks PRAGMA table_info before altering, so it's a no-op
    on databases already built by the updated etl_loader.py.
    """
    with get_connection() as conn:
        for table in ("lite_master", "risk_classifications"):
            cols = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
            if cols and "sku_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN sku_id TEXT")
                conn.commit()
        # lite_master is guaranteed to exist by this point (etl_loader runs first);
        # guard anyway so this function is safe on a completely fresh environment.
        lm_cols = [row["name"] for row in conn.execute("PRAGMA table_info(lite_master)")]
        if lm_cols:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lite_master_sku_id ON lite_master(sku_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lite_master_region_date ON lite_master(order_region, order_date)")
        conn.commit()


def ensure_ingestion_schema() -> None:
    """Create all live-ingestion tables (delegates to ingestion_schema)."""
    from src.utils.ingestion_schema import ensure_ingestion_schema as _ensure_all

    _ensure_all()


def upsert_weather_signal(signal: Dict[str, Any]) -> None:
    """Insert or refresh one hub's weather signal for a given observation date."""
    execute_non_query(
        """
        INSERT INTO weather_signals (
            hub, latitude, longitude, observation_date, severity,
            wind_score, precipitation_score, weather_code_score,
            max_wind_speed, max_precipitation, weather_summary,
            source_type, ingestion_ts
        ) VALUES (
            :hub, :latitude, :longitude, :observation_date, :severity,
            :wind_score, :precipitation_score, :weather_code_score,
            :max_wind_speed, :max_precipitation, :weather_summary,
            :source_type, :ingestion_ts
        )
        ON CONFLICT(hub, observation_date) DO UPDATE SET
            severity = excluded.severity,
            wind_score = excluded.wind_score,
            precipitation_score = excluded.precipitation_score,
            weather_code_score = excluded.weather_code_score,
            max_wind_speed = excluded.max_wind_speed,
            max_precipitation = excluded.max_precipitation,
            weather_summary = excluded.weather_summary,
            ingestion_ts = excluded.ingestion_ts
        """,
        signal,
    )


def insert_news_signals(articles: List[Dict[str, Any]]) -> int:
    """Insert news rows, skipping duplicates by content_hash. Returns rows added."""
    if not articles:
        return 0
    inserted = 0
    with get_connection() as conn:
        for article in articles:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO news_signals (
                    title, summary, url, publisher, published_at,
                    detected_region, detected_category, query_tag,
                    content_hash, source_type, ingestion_ts
                ) VALUES (
                    :title, :summary, :url, :publisher, :published_at,
                    :detected_region, :detected_category, :query_tag,
                    :content_hash, :source_type, :ingestion_ts
                )
                """,
                article,
            )
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def fetch_latest_weather_signal(hub: str) -> Optional[Dict[str, Any]]:
    ensure_ingestion_schema()
    rows = execute_query(
        """
        SELECT * FROM weather_signals
        WHERE hub = ?
        ORDER BY observation_date DESC, ingestion_ts DESC
        LIMIT 1
        """,
        (hub,),
    )
    return dict(rows[0]) if rows else None


def fetch_latest_llm_call_log(run_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
    """Return the most recent llm_call_log row for (run_id, agent_name), or None."""
    ensure_schema()
    rows = execute_query(
        """
        SELECT * FROM llm_call_log
        WHERE run_id = ? AND agent_name = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (run_id, agent_name),
    )
    return dict(rows[0]) if rows else None


def fetch_recent_composite_scores(days: int) -> List[float]:
    """Return composite_score values from risk_classifications in the last `days` days."""
    # ensure_schema() does NOT create risk_classifications ΓÇö that table is
    # created separately by ensure_risk_classification_table(), normally
    # invoked lazily inside insert_risk_classification(). On a freshly
    # built database where no classification has ever been written yet,
    # calling ensure_schema() here left the table missing and this query
    # crashed with "no such table: risk_classifications" (PR review).
    ensure_risk_classification_table()
    rows = execute_query(
        """
        SELECT composite_score FROM risk_classifications
        WHERE run_ts >= datetime('now', ?)
        ORDER BY run_ts DESC
        """,
        (f"-{int(days)} days",),
    )
    return [row["composite_score"] for row in rows]


def fetch_recent_news(
    region: Optional[str] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    ensure_ingestion_schema()
    if region:
        rows = execute_query(
            """
            SELECT * FROM news_signals
            WHERE detected_region = ?
            ORDER BY published_at DESC, ingestion_ts DESC
            LIMIT ?
            """,
            (region, limit),
        )
    else:
        rows = execute_query(
            """
            SELECT * FROM news_signals
            ORDER BY published_at DESC, ingestion_ts DESC
            LIMIT ?
            """,
            (limit,),
        )
    return [dict(row) for row in rows]


def get_table_count(table: str) -> int:
    rows = execute_query(f"SELECT COUNT(*) AS n FROM {table}")
    return int(rows[0]["n"]) if rows else 0


def fetch_daily_record(
    event_date: str, port: str, sku: str
) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT * FROM daily_records
        WHERE event_date = ? AND port = ? AND sku = ?
        ORDER BY record_id
        LIMIT 1
        """,
        (event_date, port, sku),
    )
    return dict(rows[0]) if rows else None


def fetch_time_series(port: str, sku: str) -> List[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT event_date, SUM(demand) AS demand,
               SUM(import_volume) AS import_volume,
               AVG(price_index) AS price_index
        FROM daily_records
        WHERE port = ? AND sku = ?
        GROUP BY event_date
        ORDER BY event_date
        """,
        (port, sku),
    )
    return [dict(row) for row in rows]


def fetch_inventory_snapshot(port: str, sku: str) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT inventory_level, incoming_supply, lead_time_days
        FROM daily_records
        WHERE port = ? AND sku = ?
        ORDER BY event_date DESC, record_id DESC
        LIMIT 1
        """,
        (port, sku),
    )
    return dict(rows[0]) if rows else None


def update_risk_label(
    event_date: str,
    port: str,
    sku: str,
    composite_score: float,
    label: str,
) -> None:
    execute_non_query(
        """
        UPDATE lite_master
        SET risk_score_composite = ?, disruption_event_label = ?
        WHERE record_id = (
            SELECT record_id FROM daily_records
            WHERE event_date = ? AND port = ? AND sku = ?
            ORDER BY record_id LIMIT 1
        )
        """,
        (composite_score, label, event_date, port, sku),
    )


def insert_mitigation_action(
    *,
    run_id: Optional[str],
    event_date: str,
    port: str,
    sku: str,
    risk_label: str,
    summary: str,
    recommendations: List[str],
    urgency: str,
    cost_delta: str,
    rag_citations: List[str],
    india_sourcing_recommendations: List[str],
) -> None:
    """Persist the agent's real mitigation output keyed by pipeline run_id.

    `recommendations` stays JSON-encoded in the legacy recommendation column
    for backwards compatibility. The new columns keep the run-specific
    summary and provenance available to the API without recreating any values.
    """
    execute_non_query(
        """
        INSERT INTO mitigation_actions
        (run_id, event_date, port, sku, risk_label, summary, recommendation,
         urgency, cost_delta, rag_citations_json, india_sourcing_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            event_date,
            port,
            sku,
            risk_label,
            summary,
            json.dumps(recommendations),
            urgency,
            cost_delta,
            json.dumps(rag_citations),
            json.dumps(india_sourcing_recommendations),
        ),
    )


def fetch_mitigation_action_output(run_id: str) -> Optional[Dict[str, Any]]:
    """Read the agent's persisted mitigation output for a pipeline run_id."""
    ensure_schema()
    rows = execute_query(
        "SELECT * FROM mitigation_actions WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if not rows:
        return None
    row = dict(rows[0])
    return {
        "run_id": run_id,
        "summary": row.get("summary") or None,
        "urgency": _normalize_mitigation_urgency(row.get("urgency")),
        "recommendations": json.loads(row.get("recommendation") or "[]"),
        "cost_delta": row.get("cost_delta") or None,
        "rag_citations": json.loads(row.get("rag_citations_json") or "[]"),
        "india_sourcing_recommendations": json.loads(row.get("india_sourcing_json") or "[]"),
    }


def insert_llm_call_log(**kwargs) -> None:
    """Persist one LLM call record. Accepts all llm_call_log columns as keyword args."""
    execute_non_query(
        """
        INSERT INTO llm_call_log (
            run_id, agent_name, model, prompt_preview, full_prompt, full_response,
            input_tokens, output_tokens, total_tokens, cost_usd, latency_ms,
            status, retry_count, error_message, langfuse_trace_id, langfuse_generation_id
        ) VALUES (
            :run_id, :agent_name, :model, :prompt_preview, :full_prompt, :full_response,
            :input_tokens, :output_tokens, :total_tokens, :cost_usd, :latency_ms,
            :status, :retry_count, :error_message, :langfuse_trace_id, :langfuse_generation_id
        )
        """,
        {
            "run_id": kwargs.get("run_id"),
            "agent_name": kwargs.get("agent_name"),
            "model": kwargs.get("model"),
            "prompt_preview": kwargs.get("prompt_preview"),
            "full_prompt": kwargs.get("full_prompt"),
            "full_response": kwargs.get("full_response"),
            "input_tokens": kwargs.get("input_tokens", 0),
            "output_tokens": kwargs.get("output_tokens", 0),
            "total_tokens": kwargs.get("total_tokens", 0),
            "cost_usd": kwargs.get("cost_usd", 0.0),
            "latency_ms": kwargs.get("latency_ms", 0.0),
            "status": kwargs.get("status", "success"),
            "retry_count": kwargs.get("retry_count", 0),
            "error_message": kwargs.get("error_message"),
            "langfuse_trace_id": kwargs.get("langfuse_trace_id"),
            "langfuse_generation_id": kwargs.get("langfuse_generation_id"),
        },
    )


def insert_agent_execution(**kwargs) -> None:
    """Insert a new agent execution row with status=Running."""
    execute_non_query(
        """
        INSERT INTO agent_execution_log (run_id, agent_name, status, started_at)
        VALUES (:run_id, :agent_name, :status, :started_at)
        """,
        {
            "run_id": kwargs.get("run_id"),
            "agent_name": kwargs.get("agent_name"),
            "status": kwargs.get("status", "Running"),
            "started_at": kwargs.get("started_at"),
        },
    )


def update_agent_execution(run_id: str, agent_name: str, **kwargs) -> None:
    """Update an existing agent execution row (status, timing, error, Langfuse ids)."""
    execute_non_query(
        """
        UPDATE agent_execution_log
        SET status = :status,
            completed_at = :completed_at,
            duration_ms = :duration_ms,
            error_message = :error_message,
            langfuse_trace_id = :langfuse_trace_id,
            langfuse_span_id = :langfuse_span_id,
            updated_at = CURRENT_TIMESTAMP
        WHERE run_id = :run_id AND agent_name = :agent_name
          AND id = (
              SELECT id FROM agent_execution_log
              WHERE run_id = :run_id AND agent_name = :agent_name
              ORDER BY id DESC LIMIT 1
          )
        """,
        {
            "run_id": run_id,
            "agent_name": agent_name,
            "status": kwargs.get("status"),
            "completed_at": kwargs.get("completed_at"),
            "duration_ms": kwargs.get("duration_ms"),
            "error_message": kwargs.get("error_message"),
            "langfuse_trace_id": kwargs.get("langfuse_trace_id"),
            "langfuse_span_id": kwargs.get("langfuse_span_id"),
        },
    )


def ensure_simulation_schema() -> None:
    """Create simulation_runs audit table."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT,
                port TEXT,
                sku TEXT,
                disruption_type TEXT,
                risk_label TEXT,
                trials_run INTEGER,
                stockout_p10 REAL,
                stockout_p50 REAL,
                stockout_p90 REAL,
                revenue_p10_usd REAL,
                revenue_p50_usd REAL,
                revenue_p90_usd REAL,
                days_to_stockout_p50 REAL,
                alternate_route TEXT,
                model_version TEXT,
                payload_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_simulation_run(
    event_date: str,
    port: str,
    sku: str,
    disruption_type: str,
    risk_label: str,
    result: Any,
    payload_json: str,
) -> None:
    ensure_simulation_schema()
    execute_non_query(
        """
        INSERT INTO simulation_runs (
            event_date, port, sku, disruption_type, risk_label, trials_run,
            stockout_p10, stockout_p50, stockout_p90,
            revenue_p10_usd, revenue_p50_usd, revenue_p90_usd,
            days_to_stockout_p50, alternate_route, model_version, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_date,
            port,
            sku,
            disruption_type,
            risk_label,
            result.trials_run,
            result.stockout_probability_p10,
            result.stockout_probability_pct,
            result.stockout_probability_p90,
            result.revenue_impact_usd_p10,
            result.revenue_impact_usd_p50,
            result.revenue_impact_usd_p90,
            result.days_to_stockout_p50,
            result.alternate_route,
            result.model_version,
            payload_json,
        ),
    )


def fetch_ops_kpi_priors(sku: str, region: str) -> Optional[Dict[str, float]]:
    """Return mean lead-time and demand variation priors from ops_kpi (parameterization only)."""
    try:
        rows = execute_query(
            """
            SELECT
                AVG(demand_actual) AS mean_demand,
                AVG(lead_time_baseline_days) AS mean_lead_time,
                AVG(lead_time_after_days - lead_time_baseline_days) AS mean_lead_time_inflation,
                AVG(ABS(demand_actual - forecast_baseline) / NULLIF(forecast_baseline, 0)) AS demand_cv
            FROM ops_kpi
            WHERE region = ?
              AND (sku_id = ? OR ? = '')
            """,
            (region, sku, sku),
        )
    except RuntimeError:
        return None
    if not rows or rows[0]["mean_demand"] is None:
        rows = execute_query(
            """
            SELECT
                AVG(demand_actual) AS mean_demand,
                AVG(lead_time_baseline_days) AS mean_lead_time,
                AVG(lead_time_after_days - lead_time_baseline_days) AS mean_lead_time_inflation,
                AVG(ABS(demand_actual - forecast_baseline) / NULLIF(forecast_baseline, 0)) AS demand_cv
            FROM ops_kpi
            WHERE region = ?
            """,
            (region,),
        )
    if not rows or rows[0]["mean_demand"] is None:
        return None
    row = dict(rows[0])
    return {
        "mean_demand": float(row["mean_demand"] or 0.0),
        "mean_lead_time": float(row["mean_lead_time"] or 0.0),
        "mean_lead_time_inflation": float(row["mean_lead_time_inflation"] or 0.0),
        "demand_cv": float(row["demand_cv"] or 0.15),
    }


def ensure_risk_classification_table() -> None:
    """Create risk_classifications table if it doesn't exist, and add
    full_result_json to pre-existing tables that predate it (ALTER TABLE
    is a no-op-safe upgrade path ΓÇö CREATE TABLE IF NOT EXISTS alone would
    not add a new column to an already-existing table on disk)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_classifications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id        INTEGER,
                mode            TEXT NOT NULL,
                composite_score REAL NOT NULL,
                geo_component   REAL,
                supply_component REAL,
                freight_component REAL,
                defect_component REAL,
                duration_days   REAL,
                base_label      TEXT,
                final_label     TEXT,
                escalated       INTEGER,
                rag_citations   TEXT,
                rationale       TEXT,
                full_result_json TEXT,
                run_ts          TEXT DEFAULT CURRENT_TIMESTAMP,
                sku_id          TEXT
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(risk_classifications)")}
        if "full_result_json" not in existing_cols:
            conn.execute("ALTER TABLE risk_classifications ADD COLUMN full_result_json TEXT")
        if "sku_id" not in existing_cols:
            conn.execute("ALTER TABLE risk_classifications ADD COLUMN sku_id TEXT")


def insert_risk_classification(
    order_id: Optional[int],
    mode: str,
    composite_score: float,
    geo_component: float,
    supply_component: float,
    freight_component: float,
    defect_component: float,
    duration_days: Optional[float],
    base_label: str,
    final_label: str,
    escalated: bool,
    rag_citations: List[str],
    rationale: str,
    full_result_json: Optional[str] = None,
    sku_id: Optional[str] = None,
) -> None:
    """Persist one L4 classification run. full_result_json, when provided,
    is a serialized RiskClassificationResult (rule/distilbert/llm/judge
    signals) so a later cache-hit read (fetch_risk_classification) can
    return the complete ensemble detail instead of rule-signal-only ΓÇö
    see risk.py's _response_from_cached_row. Rows inserted before this
    column existed simply have it NULL, and the reader falls back to the
    rule-only view for those. sku_id is None for rows whose source record
    predates the SKU_Product_Mapping crosswalk."""
    import json as _json
    ensure_risk_classification_table()
    execute_non_query(
        """
        INSERT INTO risk_classifications
          (order_id, mode, composite_score, geo_component, supply_component,
           freight_component, defect_component, duration_days, base_label,
           final_label, escalated, rag_citations, rationale, full_result_json,
           sku_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, mode, round(composite_score, 4),
            round(geo_component, 4), round(supply_component, 4),
            round(freight_component, 4), round(defect_component, 4),
            duration_days, base_label, final_label,
            int(escalated), _json.dumps(rag_citations), rationale,
            full_result_json, sku_id,
        ),
    )


def fetch_record_by_order_id(order_id: int) -> Optional[Dict[str, Any]]:
    """Resolve an order_id to its most recent daily_records row ΓÇö the
    (event_date, port, sku) shape risk_classifier_agent's record dict
    needs. Used by GET /api/risk-classification/{run_id} (Screen 2) to
    turn the run_id (== order_id) path param into a record before
    invoking the ensemble directly, without running the full LangGraph
    pipeline."""
    rows = execute_query(
        "SELECT * FROM daily_records WHERE order_id = ? ORDER BY record_id DESC LIMIT 1",
        (order_id,),
    )
    return dict(rows[0]) if rows else None


def fetch_risk_classification(order_id: int) -> Optional[Dict[str, Any]]:
    """Read back the most recently persisted ensemble result for an
    order_id from risk_classifications. Returns None if this order has
    never been classified ΓÇö the caller (Screen 2's API handler) computes
    it live in that case. Read counterpart to insert_risk_classification."""
    rows = execute_query(
        "SELECT * FROM risk_classifications WHERE order_id = ? ORDER BY id DESC LIMIT 1",
        (order_id,),
    )
    return dict(rows[0]) if rows else None


def fetch_latest_classified_order_id() -> Optional[int]:
    """Return the order_id of the most recently persisted classification,
    or the most recent daily_records order if none have been classified
    yet. Backs GET /api/risk-classification/latest (Screen 2's default
    view before a demo-scenario/record picker exists)."""
    rows = execute_query("SELECT order_id FROM risk_classifications ORDER BY id DESC LIMIT 1")
    if rows:
        return int(rows[0]["order_id"])
    rows = execute_query("SELECT order_id FROM daily_records ORDER BY record_id DESC LIMIT 1")
    return int(rows[0]["order_id"]) if rows else None


def fetch_scenario_options() -> List[Dict[str, Any]]:
    """Return valid region/product/date combinations with forecast history.

    Only genuine electronics categories are included. The DataCo source dataset
    labels sports/fashion products (golf balls, shoes) under 'Electronics';
    that category is excluded here. The clean categories are Cameras, Computers,
    Consumer Electronics, and Video Games.
    """
    rows = execute_query(
        """
        SELECT
            port,
            sku,
            MAX(event_date) AS event_date,
            COUNT(DISTINCT event_date) AS history_points
        FROM daily_records
        WHERE port IS NOT NULL
          AND sku IS NOT NULL
          AND category_name IN (
              'Cameras', 'Computers', 'Consumer Electronics', 'Video Games'
          )
        GROUP BY port, sku
        HAVING COUNT(DISTINCT event_date) >= 3
        ORDER BY history_points DESC, port, sku
        """
    )
    return [dict(row) for row in rows]


# ΓöÇΓöÇ Day 8 read helpers (Screen 1ΓÇô5) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

_HUB_CITIES_ORDER = [
    "Hsinchu", "Osaka", "Austin", "Shanghai", "Singapore", "Rotterdam",
]

_AGENT_LEVEL_MAP = {
    "L1_ingestion": "L1",
    "L2_news": "L2",
    "L3_weather": "L3",
    "L4_risk_classifier": "L4",
    "L5_forecast": "L5",
    "L6_simulation": "L6",
    "L7_mitigation": "L7",
}

_AGENT_TAB_MAP = {
    "L1_ingestion": 0,
    "L2_news": 1,
    "L3_weather": 1,
    "L4_risk_classifier": 1,
    "L5_forecast": 2,
    "L6_simulation": 2,
    "L7_mitigation": 3,
}

_AGENT_GANTT_COLORS = {
    "Complete": "#22C55E",
    "Running": "#818CF8",
    "Failed-Fallback": "#F59E0B",
    "Skipped-Optional": "#475569",
}

_VERDICT_TYPE_COLORS = {
    "unanimous": "#22C55E",
    "majority_rule": "#3B82F6",
    "override_distilbert": "#8B5CF6",
    "override_llm": "#8B5CF6",
    "defer_to_rules": "#F59E0B",
    "no_judge": "#475569",
}

_VERDICT_TYPE_LABELS = {
    "unanimous": "Unanimous",
    "majority_rule": "Majority Rule",
    "override_distilbert": "DistilBERT Override",
    "override_llm": "LLM Override",
    "defer_to_rules": "Defer to Rules",
    "no_judge": "No Judge",
}


def fetch_live_news(limit: int = 50) -> Dict[str, Any]:
    """Read the latest live_news_ingest batch for Screen 1 NewsPanel.

    Returns dict with run_id, count, fetched_at, items (list of row dicts).
    Table: live_news_ingest."""
    rows = execute_query(
        """
        SELECT hub_city, hub_country, supplier_country, headline,
               published_at, relevance_score, query_term, source_feed,
               fetched_at_utc, run_id
        FROM live_news_ingest
        WHERE run_id = (
            SELECT run_id FROM live_news_ingest ORDER BY fetched_at_utc DESC LIMIT 1
        )
        ORDER BY relevance_score DESC
        LIMIT ?
        """,
        (limit,),
    )
    if not rows:
        return {"run_id": None, "count": 0, "fetched_at": None, "items": []}
    items = [dict(row) for row in rows]
    return {
        "run_id": rows[0]["run_id"],
        "count": len(items),
        "fetched_at": rows[0]["fetched_at_utc"],
        "items": items,
    }


def fetch_live_weather() -> Dict[str, Any]:
    """Read live_weather_ingest for the 6 fab hub cities (Screen 1 WeatherPanel).

    Recomputes is_trigger_hub from raw_severity_score >= 7.0 ΓÇö never trusts the
    stored is_trigger_hub column. Table: live_weather_ingest."""
    rows = execute_query(
        """
        SELECT hub_city, wind_speed_kmh, precipitation_mm, weather_code,
               temperature_c, raw_severity_score, is_trigger_hub, fetched_at_utc, run_id
        FROM live_weather_ingest
        WHERE run_id = (
            SELECT run_id FROM live_weather_ingest ORDER BY fetched_at_utc DESC LIMIT 1
        )
        """
    )
    by_hub = {row["hub_city"]: dict(row) for row in rows}
    hubs: List[Dict[str, Any]] = []
    for hub_name in _HUB_CITIES_ORDER:
        row = by_hub.get(hub_name)
        if row is None:
            hubs.append({"hub_city": hub_name, "is_trigger_hub": False})
        else:
            severity = row.get("raw_severity_score")
            trigger = severity is not None and float(severity) >= 7.0
            hubs.append(
                {
                    "hub_city": row["hub_city"],
                    "wind_speed_kmh": row.get("wind_speed_kmh"),
                    "precipitation_mm": row.get("precipitation_mm"),
                    "weather_code": row.get("weather_code"),
                    "temperature_c": row.get("temperature_c"),
                    "raw_severity_score": severity,
                    "is_trigger_hub": trigger,
                    "fetched_at_utc": row.get("fetched_at_utc"),
                }
            )
    return {
        "run_id": rows[0]["run_id"] if rows else None,
        "fetched_at": rows[0]["fetched_at_utc"] if rows else None,
        "hubs": hubs,
    }


def fetch_run_logs(run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Read agent_execution_log for a pipeline run_id (Screen 1 AgentLogPanel).

    Returns clickable log lines with level, text, tab, source='real'.
    Table: agent_execution_log."""
    rows = execute_query(
        """
        SELECT agent_name, status, started_at, completed_at, duration_ms, error_message
        FROM agent_execution_log
        WHERE run_id = ?
        ORDER BY started_at ASC, id ASC
        LIMIT ?
        """,
        (run_id, limit),
    )
    lines: List[Dict[str, Any]] = []
    for row in rows:
        agent = row["agent_name"]
        level = _AGENT_LEVEL_MAP.get(agent, agent)
        tab = _AGENT_TAB_MAP.get(agent, 0)
        status = row["status"] or "Complete"
        dur_s = (row["duration_ms"] or 0) / 1000.0
        if row["error_message"]:
            text = f"{level}: {status} ΓÇö {row['error_message']}"
        else:
            text = f"{level}: {status} ({dur_s:.1f}s)"
        lines.append({"level": level, "text": text, "tab": tab, "source": "real"})
    return lines


def fetch_run_gantt(run_id: str) -> List[Dict[str, Any]]:
    """Build Gantt bars from agent_execution_log timing (Screen 1 GanttStrip).

    Converts started_at/completed_at per agent to {id, start, dur, color}.
    Table: agent_execution_log."""
    rows = execute_query(
        """
        SELECT agent_name, status, started_at, completed_at, duration_ms
        FROM agent_execution_log
        WHERE run_id = ?
        ORDER BY started_at ASC, id ASC
        """,
        (run_id,),
    )
    if not rows:
        return []

    def _parse_ts(ts: Optional[str]) -> float:
        if not ts:
            return 0.0
        try:
            normalized = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return 0.0

    base_ts = _parse_ts(rows[0]["started_at"])
    bars: List[Dict[str, Any]] = []
    for row in rows:
        agent = row["agent_name"]
        level = _AGENT_LEVEL_MAP.get(agent, agent)
        start_ts = _parse_ts(row["started_at"])
        start = round(max(0.0, start_ts - base_ts), 1)
        dur = round((row["duration_ms"] or 0) / 1000.0, 1)
        if dur <= 0 and row["completed_at"]:
            dur = round(max(0.1, _parse_ts(row["completed_at"]) - start_ts), 1)
        color = _AGENT_GANTT_COLORS.get(row["status"] or "Complete", "#22C55E")
        bars.append(
            {"id": level, "start": start, "dur": dur, "color": color, "source": "real"}
        )
    return bars


def fetch_latest_pipeline_run_id() -> Optional[str]:
    """Return the most recent pipeline run_id from agent_execution_log, or None."""
    rows = execute_query(
        "SELECT run_id FROM agent_execution_log ORDER BY id DESC LIMIT 1"
    )
    return rows[0]["run_id"] if rows else None


def insert_forecast_output(
    run_id: str,
    category: str,
    categories: List[str],
    series: List[Dict[str, Any]],
) -> None:
    """Persist one Prophet forecast snapshot for a pipeline run."""
    ensure_schema()
    execute_non_query(
        """
        INSERT INTO forecast_output (run_id, category, categories_json, series_json)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, category, json.dumps(categories), json.dumps(series)),
    )


def fetch_forecast(run_id: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read forecast_output for run_id (Screen 3 Forecast tab).

    Optional category filters the selected category label; series is always
    returned for the stored snapshot. Table: forecast_output."""
    rows = execute_query(
        "SELECT * FROM forecast_output WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if not rows:
        return None
    row = dict(rows[0])
    categories = json.loads(row["categories_json"])
    series = json.loads(row["series_json"])
    selected = category or row["category"] or (categories[0] if categories else "Laptops")
    return {
        "run_id": run_id,
        "category": selected,
        "categories": categories,
        "series": series,
    }


def insert_simulation_output(
    run_id: str,
    p10: float,
    p50: float,
    p90: float,
    revenue_at_risk_usd: float,
    alternate_route: str,
    histogram: List[Dict[str, Any]],
) -> None:
    """Persist one SimPy/Monte Carlo snapshot keyed by pipeline run_id."""
    ensure_schema()
    execute_non_query(
        """
        INSERT OR REPLACE INTO simulation_output (
            run_id, p10, p50, p90, revenue_at_risk_usd, alternate_route, histogram_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            p10,
            p50,
            p90,
            revenue_at_risk_usd,
            alternate_route,
            json.dumps(histogram),
        ),
    )


def fetch_simulation(run_id: str) -> Optional[Dict[str, Any]]:
    """Read simulation_output for run_id (Screen 3 Simulation tab).

    Table: simulation_output."""
    rows = execute_query(
        "SELECT * FROM simulation_output WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if not rows:
        return None
    row = dict(rows[0])
    return {
        "run_id": run_id,
        "p10": row["p10"],
        "p50": row["p50"],
        "p90": row["p90"],
        "revenue_at_risk_usd": row["revenue_at_risk_usd"],
        "alternate_route": row["alternate_route"],
        "histogram": json.loads(row["histogram_json"]),
    }


def insert_mitigation_output(
    run_id: str,
    urgency: str,
    ranked_actions: List[Dict[str, Any]],
    rag_query_trace: List[str],
    india_sourcing_recommendations: List[str],
    slack_preview: str,
    cost_delta_usd: float,
) -> None:
    """Persist full MitigationResponse payload for a pipeline run_id."""
    ensure_schema()
    execute_non_query(
        """
        INSERT OR REPLACE INTO mitigation_output (
            run_id, urgency, ranked_actions_json, rag_query_trace_json,
            india_sourcing_json, slack_preview, cost_delta_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            urgency,
            json.dumps(ranked_actions),
            json.dumps(rag_query_trace),
            json.dumps(india_sourcing_recommendations),
            slack_preview,
            cost_delta_usd,
        ),
    )


def fetch_mitigation(run_id: str) -> Optional[Dict[str, Any]]:
    """Read mitigation_output for run_id (Screen 4 Mitigation tab).

    Table: mitigation_output (+ RAG trace stored in rag_query_trace_json)."""
    ensure_schema()
    risk_row = fetch_risk_classification_output(run_id)
    action_row = fetch_mitigation_action_output(run_id)

    if action_row is None:
        rows = execute_query(
            "SELECT * FROM mitigation_output WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        )
        if not rows:
            return None
        row = dict(rows[0])
        ranked_actions = json.loads(row["ranked_actions_json"])
        return {
            "run_id": run_id,
            "risk_level": (risk_row or {}).get("final_label") if risk_row else None,
            "summary": None,
            "urgency": _normalize_mitigation_urgency(row["urgency"]),
            "ranked_actions": ranked_actions,
            "rag_citations": [],
            "rag_query_trace": [],
            "india_sourcing_recommendations": [],
            "slack_preview": None,
            "cost_delta": None,
            "cost_delta_usd": None,
        }

    ranked_actions = [
        {
            "rank": idx + 1,
            "text": text,
            "citations": action_row["rag_citations"],
        }
        for idx, text in enumerate(action_row["recommendations"])
    ]

    return {
        "run_id": run_id,
        "risk_level": (risk_row or {}).get("final_label") if risk_row else None,
        "summary": action_row["summary"],
        "urgency": _normalize_mitigation_urgency(action_row["urgency"]),
        "ranked_actions": ranked_actions,
        "rag_citations": action_row["rag_citations"],
        # Assumption: the pipeline does not persist a real per-query trace.
        # Keep this empty rather than reconstructing a fake trace from code.
        "rag_query_trace": [],
        "india_sourcing_recommendations": action_row["india_sourcing_recommendations"],
        "slack_preview": None,
        "cost_delta": action_row["cost_delta"],
        "cost_delta_usd": None,
    }


def insert_risk_classification_output(
    run_id: str,
    final_label: str,
    composite_score: Optional[float],
    critical_flag: bool,
    full_result_json: str,
) -> None:
    """Persist L4's RiskClassificationResult keyed by pipeline run_id.

    Separate from risk_classifications (order_id-keyed, written directly by
    risk_classifier_agent for Screen 2's historical-order lookup) ΓÇö this
    table is the run_id-keyed snapshot for a live/demo/replay pipeline run,
    following the same pattern as simulation_output/mitigation_output."""
    ensure_schema()
    execute_non_query(
        """
        INSERT OR REPLACE INTO risk_classification_output (
            run_id, final_label, composite_score, critical_flag, full_result_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, final_label, composite_score, int(critical_flag), full_result_json),
    )


def fetch_risk_classification_output(run_id: str) -> Optional[Dict[str, Any]]:
    """Read risk_classification_output for a pipeline run_id. Table:
    risk_classification_output (see insert_risk_classification_output)."""
    rows = execute_query(
        "SELECT * FROM risk_classification_output WHERE run_id = ? ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if not rows:
        return None
    row = dict(rows[0])
    return {
        "run_id": run_id,
        "final_label": row["final_label"],
        "composite_score": row["composite_score"],
        "critical_flag": bool(row["critical_flag"]),
        "full_result": json.loads(row["full_result_json"]),
    }


_PIPELINE_AGENTS = [
    ("L1", "L1_ingestion"),
    ("L2", "L2_news"),
    ("L3", "L3_weather"),
    ("L4", "L4_risk_classifier"),
    ("L5", "L5_forecast"),
    ("L6", "L6_simulation"),
    ("L7", "L7_mitigation"),
]


def build_idle_agents() -> List[Dict[str, Any]]:
    """The 7-entry all-Idle agent list, shared by fetch_pipeline_status()
    (a run_id with zero agent_execution_log rows yet) and the pipeline
    router's pre-L1 "fetching live data" placeholder status."""
    return [
        {
            "id": agent_id, "name": agent_name, "status": "Idle",
            "started_at": None, "completed_at": None, "error_message": None,
            "duration_ms": None,
        }
        for agent_id, agent_name in _PIPELINE_AGENTS
    ]


def fetch_recent_completed_run_ids(limit: int = 10) -> List[Dict[str, Any]]:
    """Recent run_ids whose L7_mitigation agent reached a terminal status ΓÇö
    the pool Replay mode's run picker offers (POST /run with mode="replay"
    only accepts an already-completed run_id)."""
    rows = execute_query(
        """
        SELECT run_id, MAX(started_at) AS last_started_at
        FROM agent_execution_log
        WHERE agent_name = 'L7_mitigation' AND status IN ('Complete', 'Failed-Fallback')
        GROUP BY run_id
        ORDER BY last_started_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [{"run_id": row["run_id"], "last_started_at": row["last_started_at"]} for row in rows]


def fetch_pipeline_status(run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Reshape agent_execution_log rows for one run_id into the 7-entry
    Idle/Running/Complete/Skipped-Optional/Failed-Fallback contract GET
    /api/pipeline/status returns. When run_id is None, resolves to the most
    recently active run (fetch_latest_pipeline_run_id()) so the status bar
    has something to show before any run has been explicitly selected.
    Returns None only when run_id was given explicitly and no rows exist
    for it at all."""
    resolved_run_id = run_id or fetch_latest_pipeline_run_id()
    if resolved_run_id is None:
        return None

    rows = execute_query(
        """
        SELECT agent_name, status, started_at, completed_at, duration_ms, error_message
        FROM agent_execution_log
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (resolved_run_id,),
    )
    if run_id is not None and not rows:
        return None

    latest_by_agent: Dict[str, sqlite3.Row] = {}
    for row in rows:
        latest_by_agent[row["agent_name"]] = row

    agents: List[Dict[str, Any]] = []
    for agent_id, agent_name in _PIPELINE_AGENTS:
        row = latest_by_agent.get(agent_name)
        if row is None:
            agents.append(
                {
                    "id": agent_id, "name": agent_name, "status": "Idle",
                    "started_at": None, "completed_at": None, "error_message": None,
                    "duration_ms": None,
                }
            )
            continue
        status = row["status"]
        # L5/L6 are optional: run_agent_sequence() catches their exception
        # outside agent_span(), after it already wrote "Failed-Fallback" ΓÇö
        # translate that into "Skipped-Optional" here rather than teaching
        # agent_span() (used by every agent, required and optional alike)
        # about per-agent optionality.
        if agent_id in ("L5", "L6") and status == "Failed-Fallback":
            status = "Skipped-Optional"
        agents.append(
            {
                "id": agent_id, "name": agent_name, "status": status,
                "started_at": row["started_at"], "completed_at": row["completed_at"],
                "error_message": row["error_message"], "duration_ms": row["duration_ms"],
            }
        )

    is_complete = latest_by_agent.get("L7_mitigation") is not None and latest_by_agent["L7_mitigation"]["status"] in (
        "Complete", "Failed-Fallback",
    )
    last_started = rows[0]["started_at"] if rows else None

    return {
        "run_id": resolved_run_id,
        "agents": agents,
        "last_ingested_at": last_started,
        "is_complete": is_complete,
    }


def fetch_cost_by_agent() -> List[Dict[str, Any]]:
    """SUM(cost_usd) GROUP BY agent_name from llm_call_log (Screen 5)."""
    rows = execute_query(
        """
        SELECT agent_name AS agent, ROUND(SUM(cost_usd), 6) AS cost
        FROM llm_call_log
        GROUP BY agent_name
        ORDER BY cost DESC
        """
    )
    return [{"agent": r["agent"], "cost": r["cost"] or 0.0} for r in rows]


def fetch_verdict_distribution() -> List[Dict[str, Any]]:
    """COUNT verdict_type from risk_classifications.full_result_json (Screen 5).

    Parses judge_verdict.verdict_type; legacy rows without JSON bucket as no_judge."""
    rows = execute_query(
        "SELECT full_result_json FROM risk_classifications WHERE full_result_json IS NOT NULL"
    )
    counts: Dict[str, int] = {}
    for row in rows:
        try:
            parsed = json.loads(row["full_result_json"])
            jv = parsed.get("judge_verdict") or {}
            vtype = jv.get("verdict_type") or "no_judge"
        except (json.JSONDecodeError, TypeError):
            vtype = "no_judge"
        counts[vtype] = counts.get(vtype, 0) + 1

    if not counts:
        legacy = execute_query(
            "SELECT final_label, COUNT(*) AS cnt FROM risk_classifications GROUP BY final_label"
        )
        for row in legacy:
            key = f"label_{row['final_label']}"
            counts[key] = row["cnt"]

    total = sum(counts.values()) or 1
    result = []
    for vtype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        label = _VERDICT_TYPE_LABELS.get(vtype, vtype.replace("_", " ").title())
        color = _VERDICT_TYPE_COLORS.get(vtype, "#64748B")
        result.append(
            {"name": label, "value": round(cnt / total * 100), "color": color}
        )
    return result


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def fetch_latency_percentiles() -> List[Dict[str, Any]]:
    """P50/P90 latency_ms per agent from llm_call_log (Screen 5).

    Percentiles computed in Python ΓÇö sqlite3 has no PERCENTILE_CONT."""
    rows = execute_query(
        """
        SELECT agent_name, latency_ms
        FROM llm_call_log
        WHERE latency_ms IS NOT NULL
        ORDER BY agent_name, latency_ms
        """
    )
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        buckets.setdefault(row["agent_name"], []).append(float(row["latency_ms"]) / 1000.0)

    result = []
    for agent, values in sorted(buckets.items()):
        result.append(
            {
                "agent": agent,
                "p50": round(_percentile(values, 0.50), 3),
                "p90": round(_percentile(values, 0.90), 3),
            }
        )
    return result


def fetch_prompt_log(limit: int = 100) -> List[Dict[str, Any]]:
    """Latest LLM call rows from llm_call_log (Screen 5 prompt inspector)."""
    rows = execute_query(
        """
        SELECT ts, agent_name, model, prompt_preview, full_prompt, full_response,
               total_tokens, cost_usd, latency_ms
        FROM llm_call_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [
        {
            "ts": r["ts"] or "",
            "agent": r["agent_name"] or "",
            "model": r["model"] or "",
            "prompt": r["prompt_preview"] or "",
            "full_prompt": r["full_prompt"] or r["prompt_preview"] or "",
            "resp": r["full_response"] or "",
            "tokens": r["total_tokens"] or 0,
            "cost": r["cost_usd"] or 0.0,
            "latency": round((r["latency_ms"] or 0.0) / 1000.0, 3),
        }
        for r in rows
    ]


def insert_guardrail_event(
    name: str,
    direction: str,
    agent: str,
    pass_count: int,
    fail_count: int,
    reason: str = "ΓÇö",
) -> None:
    """Insert or refresh one guardrail_events aggregate row."""
    ensure_schema()
    execute_non_query(
        """
        INSERT INTO guardrail_events (name, dir, agent, pass_count, fail_count, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, direction, agent, pass_count, fail_count, reason),
    )


def fetch_guardrail_events() -> List[Dict[str, Any]]:
    """Read guardrail_events rows (Screen 5 Guardrails tab)."""
    rows = execute_query(
        """
        SELECT name, dir, agent, pass_count, fail_count, reason
        FROM guardrail_events
        ORDER BY id ASC
        """
    )
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# L5 Demand Forecast schema + persistence helpers
# ---------------------------------------------------------------------------

def ensure_forecast_schema() -> None:
    """Create the demand_forecasts table used by DemandForecastingAgent v3."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS demand_forecasts (
                sku_id          TEXT NOT NULL,
                week_start      TEXT NOT NULL,
                demand_baseline REAL,
                demand_disrupted REAL,
                deviation_pct   REAL,
                stockout_prob   REAL,
                mape_prophet    REAL,
                generated_at_utc TEXT,
                PRIMARY KEY (sku_id, week_start)
            )
            """
        )


def fetch_forecast_for_sku(sku_id: str) -> List[Dict[str, Any]]:
    """Return all stored forecast weeks for one SKU, ordered chronologically."""
    ensure_forecast_schema()
    rows = execute_query(
        """
        SELECT sku_id, week_start, demand_baseline, demand_disrupted,
               deviation_pct, stockout_prob, mape_prophet, generated_at_utc
        FROM demand_forecasts
        WHERE sku_id = ?
        ORDER BY week_start ASC
        """,
        (sku_id,),
    )
    return [dict(row) for row in rows]


def build_stockout_histogram(stockout_pcts: List[float]) -> List[Dict[str, Any]]:
    """Bucket stockout probability samples into 10% histogram bins for simulation_output."""
    bins = [
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
    ]
    counts = [0] * len(bins)
    for pct in stockout_pcts:
        idx = min(int(max(pct, 0.0) / 10.0), len(bins) - 1)
        counts[idx] += 1
    return [{"range": label, "count": count} for label, count in zip(bins, counts)]


def list_forecast_skus() -> List[str]:
    """Return all SKU IDs that have at least one persisted forecast week."""
    ensure_forecast_schema()
    rows = execute_query(
        "SELECT DISTINCT sku_id FROM demand_forecasts ORDER BY sku_id"
    )
    return [row["sku_id"] for row in rows]
