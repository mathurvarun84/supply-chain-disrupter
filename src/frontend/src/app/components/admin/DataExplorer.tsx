/**
 * Admin page — Data Explorer sub-tab. Read-only browser over
 * outputs/supply_chain.db: table dropdown (useAdminTables) -> paginated
 * grid (useAdminTableRows). No sort/filter/search/export — see the Day 11
 * scope cuts in CLAUDE_CODE_PROMPT_AdminDataExplorer.md. Neutral/
 * informational styling (border-border, not destructive) so this reads as
 * distinct from the Recreate DB / Recreate RAG cards next to it.
 */
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Table2 } from "lucide-react";
import { useAdminTableRows, useAdminTables } from "../../hooks/useAdmin";

const DEFAULT_PAGE_SIZE = 50;

export function DataExplorer() {
  const { data: tablesData, isLoading: tablesLoading } = useAdminTables();
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const tables = tablesData?.tables ?? [];

  useEffect(() => {
    if (!selectedTable && tables.length > 0) {
      setSelectedTable(tables[0].name);
    }
  }, [tables, selectedTable]);

  const {
    data: rowsData,
    isLoading: rowsLoading,
    isError: rowsErrored,
  } = useAdminTableRows(selectedTable, page, DEFAULT_PAGE_SIZE);

  const handleSelectTable = (name: string) => {
    setSelectedTable(name);
    setPage(1);
  };

  if (tablesLoading) {
    return <div className="text-xs text-muted-foreground">Loading tables…</div>;
  }

  if (tables.length === 0) {
    return (
      <div className="rounded-lg p-4 bg-card border border-border text-xs text-muted-foreground">
        No tables found — load the database from Data Management first.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg p-4 bg-card border border-border">
        <div className="flex items-center gap-2 mb-3">
          <Table2 size={14} className="text-muted-foreground" />
          <select
            value={selectedTable ?? ""}
            onChange={(e) => handleSelectTable(e.target.value)}
            className="text-xs font-semibold bg-secondary border border-border rounded-btn px-2 py-1.5 text-foreground"
          >
            {tables.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name} ({t.row_count.toLocaleString()} rows)
              </option>
            ))}
          </select>
          {rowsData && (
            <span className="text-[11px] font-mono text-muted-strong ml-auto">
              {rowsData.total_rows.toLocaleString()} rows · {rowsData.columns.length} columns
            </span>
          )}
        </div>

        {rowsLoading && <div className="text-xs text-muted-foreground">Loading rows…</div>}

        {rowsErrored && (
          <div className="text-xs text-risk-critical">
            Failed to load "{selectedTable}" — it may no longer exist. Pick another table.
          </div>
        )}

        {rowsData && !rowsLoading && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-[10px] font-mono">
                <thead>
                  <tr className="text-muted-foreground" style={{ borderBottom: "1px solid var(--border)" }}>
                    {rowsData.columns.map((col) => (
                      <th key={col} className="text-left py-1.5 px-2 font-medium whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rowsData.rows.map((row, i) => (
                    <tr
                      key={i}
                      className="hover:bg-secondary/40 transition-colors"
                      style={{ borderBottom: "1px solid var(--border)" }}
                    >
                      {rowsData.columns.map((col) => (
                        <td key={col} className="py-1.5 px-2 text-card-foreground max-w-[240px] truncate">
                          {formatCell(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {rowsData.rows.length === 0 && (
                    <tr>
                      <td colSpan={rowsData.columns.length} className="py-6 text-center text-muted-foreground">
                        No rows in this table.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-center gap-3 mt-3">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="flex items-center gap-1 px-2 py-1 rounded-btn text-[11px] text-muted-foreground border border-border disabled:opacity-40 disabled:cursor-not-allowed hover:text-foreground transition-colors"
              >
                <ChevronLeft size={12} />
                Prev
              </button>
              <span className="text-[11px] font-mono text-muted-strong">
                Page {rowsData.page} of {rowsData.total_pages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(rowsData.total_pages, p + 1))}
                disabled={page >= rowsData.total_pages}
                className="flex items-center gap-1 px-2 py-1 rounded-btn text-[11px] text-muted-foreground border border-border disabled:opacity-40 disabled:cursor-not-allowed hover:text-foreground transition-colors"
              >
                Next
                <ChevronRight size={12} />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
