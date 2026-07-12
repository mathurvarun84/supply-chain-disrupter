/**
 * Admin page — SQLite database card. Shows current outputs/supply_chain.db
 * stats (get_sqlite_stats via GET /api/admin/status) and a "Load / Rebuild
 * Database" button that POSTs /api/admin/db/build (load_excel_into_sqlite +
 * ensure_schema + ensure_ingestion_schema, run as a BackgroundTask).
 */
import { Database, RefreshCw } from "lucide-react";
import { useAdminStatus, useBuildDatabase } from "../../hooks/useAdmin";
import { ElapsedSince } from "./ElapsedSince";

const JOB_LABEL: Record<string, string> = {
  idle: "Not yet built this session",
  running: "Building…",
  complete: "Build complete",
  failed: "Build failed",
};

const JOB_COLOR: Record<string, string> = {
  idle: "text-muted-foreground",
  running: "text-status-running",
  complete: "text-status-complete",
  failed: "text-risk-critical",
};

export function DatabaseStatusCard() {
  const { data, isLoading } = useAdminStatus();
  const { mutate, isPending } = useBuildDatabase();

  const db = data?.database;
  const job = data?.db_job;
  const isRunning = job?.status === "running" || isPending;

  return (
    <div className="rounded-lg p-4 bg-card border border-border">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database size={14} className="text-primary" />
          <span className="text-sm font-semibold text-foreground">SQLite Database</span>
        </div>
        <button
          onClick={() => mutate()}
          disabled={isRunning}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-white transition-opacity disabled:opacity-50 disabled:cursor-not-allowed bg-secondary border border-primary/20"
        >
          {isRunning ? <RefreshCw size={11} className="animate-spin" /> : <Database size={11} />}
          {isRunning ? "Building…" : db?.database_exists ? "Rebuild Database" : "Load Database"}
        </button>
      </div>

      {isLoading && <div className="text-xs text-muted-foreground">Loading status…</div>}

      {job && (
        <div className={`flex items-center gap-1.5 text-[10px] font-mono mb-2 ${JOB_COLOR[job.status]}`}>
          {job.status === "running" && <RefreshCw size={9} className="animate-spin shrink-0" />}
          {JOB_LABEL[job.status]}
          {job.status === "running" && job.started_at && (
            <>
              {" · "}
              <ElapsedSince isoTimestamp={job.started_at} />
            </>
          )}
          {job.status === "failed" && job.error ? ` — ${job.error}` : ""}
        </div>
      )}

      {db?.database_exists ? (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] font-mono text-muted-strong">
          <div>lite_master rows: {db.tables?.lite_master ?? "—"}</div>
          <div>ops_kpi rows: {db.tables?.ops_kpi ?? "—"}</div>
          <div>semiconductor_signals: {db.tables?.semiconductor_signals ?? "—"}</div>
          <div>daily_records: {db.tables?.daily_records ?? "—"}</div>
          <div>unique products: {db.unique_products ?? "—"}</div>
          <div>size: {db.size_mb != null ? `${db.size_mb} MB` : "—"}</div>
          <div className="col-span-2">date range: {db.date_range ?? "—"}</div>
          <div className="col-span-2">categories: {(db.categories ?? []).join(", ") || "—"}</div>
        </div>
      ) : (
        !isLoading && (
          <div className="text-xs text-muted-foreground">
            No database found at outputs/supply_chain.db — click "Load Database" to build it from
            data/raw/supply_chain_lite_master.xlsx.
          </div>
        )
      )}
    </div>
  );
}
