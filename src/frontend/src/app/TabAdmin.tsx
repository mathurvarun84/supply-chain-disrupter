/**
 * Admin tab — Data Management (loads/rebuilds outputs/supply_chain.db and
 * the RAG/ChromaDB index) and Data Explorer (read-only table browser) as
 * sub-tabs, same pattern as TabObservability's Observability/Guardrails
 * split. Data Management wires to GET/POST /api/admin/* (db/rag build);
 * Data Explorer wires to GET /api/admin/tables* — both in
 * src/api/routers/admin.py. Each card/sub-tab owns its own data fetching;
 * this component only owns layout and sub-tab state.
 */
import { useState } from "react";
import { ShieldAlert } from "lucide-react";
import { DatabaseStatusCard } from "./components/admin/DatabaseStatusCard";
import { RagStatusCard } from "./components/admin/RagStatusCard";
import { DataExplorer } from "./components/admin/DataExplorer";

const SUB_TABS = ["Data Management", "Data Explorer"] as const;

export function TabAdmin() {
  const [sub, setSub] = useState<(typeof SUB_TABS)[number]>("Data Management");

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex gap-4 px-3 pt-3 shrink-0 border-b border-border">
        {SUB_TABS.map((t) => (
          <button
            key={t}
            onClick={() => setSub(t)}
            className="text-sm pb-2.5 font-medium transition-colors border-b-2"
            style={{
              color: sub === t ? "var(--primary)" : "var(--muted-foreground)",
              borderBottomColor: sub === t ? "var(--primary)" : "transparent",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {sub === "Data Management" ? (
          <>
            <div className="flex items-start gap-2 text-[11px] text-muted-foreground bg-card border border-border rounded-lg p-3">
              <ShieldAlert size={13} className="shrink-0 mt-0.5 text-status-running" />
              <span>
                Rebuilding the database or RAG index reads from disk (Excel workbook, mitigation
                playbooks, data/raw/RAG_data/ reports) and can take a few minutes. Both builds run
                in the background — this page polls for progress every few seconds.
              </span>
            </div>
            <DatabaseStatusCard />
            <RagStatusCard />
          </>
        ) : (
          <DataExplorer />
        )}
      </div>
    </div>
  );
}
