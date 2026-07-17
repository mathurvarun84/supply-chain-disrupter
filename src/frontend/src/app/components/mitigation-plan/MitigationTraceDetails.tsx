/**
 * Screen 4 mitigation RAG trace disclosure. Uses native <details>/<summary>
 * so the query trace can expand without introducing a new UI dependency.
 */
export function MitigationTraceDetails({ trace }: { trace: string[] }) {
  return (
    <details className="rounded-panel border border-border bg-card p-4">
      <summary className="cursor-pointer list-none text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        RAG Query Trace
      </summary>
      <div className="mt-3 space-y-2">
        {trace.length > 0 ? (
          trace.map((item, index) => (
            <div key={`${item}-${index}`} className="rounded-btn border border-border bg-background px-3 py-2 text-xs text-foreground">
              {item}
            </div>
          ))
        ) : (
          <div className="text-xs text-muted-foreground">
            No persisted query trace for this run.
          </div>
        )}
      </div>
    </details>
  );
}
