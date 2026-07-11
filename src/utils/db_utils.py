import sqlite3
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
                event_date TEXT,
                port TEXT,
                sku TEXT,
                risk_label TEXT,
                recommendation TEXT,
                cost_delta TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
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
    ensure_simulation_schema()


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
    event_date: str,
    port: str,
    sku: str,
    risk_label: str,
    recommendation: str,
    cost_delta: str,
) -> None:
    execute_non_query(
        """
        INSERT INTO mitigation_actions
        (event_date, port, sku, risk_label, recommendation, cost_delta)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (event_date, port, sku, risk_label, recommendation, cost_delta),
    )


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
    is a no-op-safe upgrade path — CREATE TABLE IF NOT EXISTS alone would
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
                run_ts          TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(risk_classifications)")}
        if "full_result_json" not in existing_cols:
            conn.execute("ALTER TABLE risk_classifications ADD COLUMN full_result_json TEXT")


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
) -> None:
    """Persist one L4 classification run. full_result_json, when provided,
    is a serialized RiskClassificationResult (rule/distilbert/llm/judge
    signals) so a later cache-hit read (fetch_risk_classification) can
    return the complete ensemble detail instead of rule-signal-only —
    see risk.py's _response_from_cached_row. Rows inserted before this
    column existed simply have it NULL, and the reader falls back to the
    rule-only view for those."""
    import json as _json
    ensure_risk_classification_table()
    execute_non_query(
        """
        INSERT INTO risk_classifications
          (order_id, mode, composite_score, geo_component, supply_component,
           freight_component, defect_component, duration_days, base_label,
           final_label, escalated, rag_citations, rationale, full_result_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, mode, round(composite_score, 4),
            round(geo_component, 4), round(supply_component, 4),
            round(freight_component, 4), round(defect_component, 4),
            duration_days, base_label, final_label,
            int(escalated), _json.dumps(rag_citations), rationale,
            full_result_json,
        ),
    )


def fetch_record_by_order_id(order_id: int) -> Optional[Dict[str, Any]]:
    """Resolve an order_id to its most recent daily_records row — the
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
    never been classified — the caller (Screen 2's API handler) computes
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
