/**
 * Screen 4 mitigation RAG trace disclosure. Uses native <details>/<summary>
 * so the query trace can expand without introducing a new UI dependency.
 * Always renders all 3 fixed rows (historical/export-control/india), even
 * for a query with fired=false — showing *why* a query didn't fire is as
 * important as showing what came back when it did.
 */
import type { RagTraceQuery } from "../../types/mitigation";

const QUERY_LABEL: Record<RagTraceQuery["query_name"], string> = {
  historical_disruption_lookup: "Historical Disruption Lookup",
  export_control_check: "Export Control Check",
  india_sourcing_query: "India Sourcing Query",
};

function TraceRow({ query }: { query: RagTraceQuery }) {
  return (
    <details className="rounded-btn border border-border bg-background px-3 py-2">
      <summary className="cursor-pointer list-none flex items-center gap-2 text-xs text-foreground">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            query.fired ? "bg-risk-low" : "bg-muted-foreground"
          }`}
        />
        <span className="font-semibold">{QUERY_LABEL[query.query_name]}</span>
        <span className="text-[10px] text-muted-foreground">{query.fire_condition}</span>
      </summary>
      <div className="mt-2 space-y-1.5">
        {query.retrieved_chunks.length > 0 ? (
          query.retrieved_chunks.map((chunk, index) => (
            <div
              key={`${chunk.source_file}-${index}`}
              className="rounded border border-border/60 bg-card px-2 py-1.5 text-[11px] text-foreground"
            >
              <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
                <span>{chunk.collection}</span>
                <span>·</span>
                <span>{chunk.source_file}</span>
                {chunk.similarity_score != null && (
                  <>
                    <span>·</span>
                    <span>score {chunk.similarity_score.toFixed(3)}</span>
                  </>
                )}
              </div>
              {chunk.snippet && <div className="mt-1 leading-5">{chunk.snippet}</div>}
            </div>
          ))
        ) : (
          <div className="text-[11px] text-muted-foreground">
            {query.fired ? "No chunks retrieved for this query." : "Not fired for this run."}
          </div>
        )}
      </div>
    </details>
  );
}

export function MitigationTraceDetails({ trace }: { trace: RagTraceQuery[] }) {
  return (
    <details className="rounded-panel border border-border bg-card p-4">
      <summary className="cursor-pointer list-none text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        RAG Query Trace
      </summary>
      <div className="mt-3 space-y-2">
        {trace.map((query) => (
          <TraceRow key={query.query_name} query={query} />
        ))}
      </div>
    </details>
  );
}
