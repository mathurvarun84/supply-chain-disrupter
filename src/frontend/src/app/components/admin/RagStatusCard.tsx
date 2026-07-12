/**
 * Admin page — RAG (ChromaDB) card. Shows live per-collection doc counts
 * (fetch_corpus_health via GET /api/admin/status — same data Screen 6's
 * corpus health cards read) and buttons to POST /api/admin/rag/build,
 * building both the monolithic electronics_supply_chain_knowledge
 * collection and the three named collections that feed Screen 6.
 */
import { RefreshCw, Database } from "lucide-react";
import { useAdminStatus, useBuildRag } from "../../hooks/useAdmin";
import { ElapsedSince } from "./ElapsedSince";

const JOB_LABEL: Record<string, string> = {
  idle: "Not yet built this session",
  running: "Building embeddings…",
  complete: "Build complete",
  failed: "Build failed",
};

const JOB_COLOR: Record<string, string> = {
  idle: "text-muted-foreground",
  running: "text-status-running",
  complete: "text-status-complete",
  failed: "text-risk-critical",
};

export function RagStatusCard() {
  const { data, isLoading } = useAdminStatus();
  const { mutate, isPending } = useBuildRag();

  const job = data?.rag_job;
  const corpus = data?.corpus ?? [];
  const isRunning = job?.status === "running" || isPending;
  const totalDocs = corpus.reduce((sum, c) => sum + c.docs, 0);

  return (
    <div className="rounded-lg p-4 bg-card border border-border">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database size={14} className="text-accent" />
          <span className="text-sm font-semibold text-foreground">RAG Database (ChromaDB)</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => mutate(false)}
            disabled={isRunning}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-white transition-opacity disabled:opacity-50 disabled:cursor-not-allowed bg-secondary border border-primary/20"
          >
            {isRunning ? <RefreshCw size={11} className="animate-spin" /> : <Database size={11} />}
            {isRunning ? "Building…" : totalDocs > 0 ? "Update Index" : "Create RAG Database"}
          </button>
          <button
            onClick={() => mutate(true)}
            disabled={isRunning}
            title="Wipe outputs/chromadb and rebuild every collection from scratch"
            className="px-3 py-1.5 rounded-btn text-xs font-semibold text-muted-strong transition-opacity disabled:opacity-50 disabled:cursor-not-allowed bg-background border border-border"
          >
            Rebuild from Scratch
          </button>
        </div>
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

      <div className="space-y-1">
        {corpus.map((c) => (
          <div
            key={c.name}
            className="flex items-center justify-between text-[11px] font-mono text-muted-strong px-2 py-1 rounded bg-background border border-border"
          >
            <span>{c.name}</span>
            <span>
              {c.docs} chunks
              <span className="text-muted-foreground"> ({c.real} real / {c.synth} synth)</span>
            </span>
          </div>
        ))}
        {!isLoading && corpus.length === 0 && (
          <div className="text-xs text-muted-foreground">
            No collections found — click "Create RAG Database" to embed the Excel workbook,
            mitigation playbooks, and data/raw/RAG_data/ reports into ChromaDB.
          </div>
        )}
      </div>
    </div>
  );
}
