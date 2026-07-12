/**
 * Types for the Admin page — GET/POST /api/admin/*. Backed by
 * src/api/routers/admin.py and src/api/schemas.py (DatabaseStatus,
 * AdminJobStatus, AdminJobTriggerResponse, AdminStatusResponse).
 */
import type { CorpusHealth } from "./ragEval";

export interface DatabaseStatus {
  database_exists: boolean;
  tables: Record<string, number> | null;
  date_range: string | null;
  categories: string[] | null;
  unique_products: number | null;
  size_mb: number | null;
}

export type AdminJobState = "idle" | "running" | "complete" | "failed";

export interface AdminJobStatus {
  status: AdminJobState;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
}

export interface AdminJobTriggerResponse {
  status: "started" | "skipped_already_running";
  triggered_at: string;
}

export interface AdminStatusResponse {
  database: DatabaseStatus;
  db_job: AdminJobStatus;
  rag_job: AdminJobStatus;
  corpus: CorpusHealth[];
}
