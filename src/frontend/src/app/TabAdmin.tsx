/**
 * Admin tab — loads/rebuilds outputs/supply_chain.db from the source Excel
 * workbook and creates/updates the RAG (ChromaDB) index. Wires to
 * GET/POST /api/admin/* (src/api/routers/admin.py). Each card owns its own
 * data fetching via useAdminStatus(); this component only owns layout.
 */
import { ShieldAlert } from "lucide-react";
import { DatabaseStatusCard } from "./components/admin/DatabaseStatusCard";
import { RagStatusCard } from "./components/admin/RagStatusCard";

export function TabAdmin() {
  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <div className="flex items-start gap-2 text-[11px] text-muted-foreground bg-card border border-border rounded-lg p-3">
        <ShieldAlert size={13} className="shrink-0 mt-0.5 text-status-running" />
        <span>
          Rebuilding the database or RAG index reads from disk (Excel workbook, mitigation
          playbooks, data/raw/RAG_data/ reports) and can take a few minutes. Both builds run in
          the background — this page polls for progress every few seconds.
        </span>
      </div>
      <DatabaseStatusCard />
      <RagStatusCard />
    </div>
  );
}
