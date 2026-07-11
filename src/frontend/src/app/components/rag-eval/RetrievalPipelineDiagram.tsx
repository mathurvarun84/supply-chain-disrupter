/**
 * Screen 6 retrieval pipeline diagram — static 3-box flow (bi-encoder →
 * cross-encoder → LLM context). Architecture reference content, same
 * treatment as Screen 2's Escalation Guard panel: not wired to a live
 * retrieval call (Screen 4's Mitigation Plan already shows a live
 * per-query RAG trace). Plain divs, no diagramming library — ported from
 * _reference/App.mockup.tsx:990-1009.
 */
import { ArrowRight } from "lucide-react";

const STAGES = [
  { label: "bi-encoder", detail: "all-MiniLM-L6-v2", out: "top-10" },
  { label: "cross-encoder", detail: "ms-marco-MiniLM-L-6-v2", out: "top-3/4" },
  { label: "LLM context", detail: "passed via prompt", out: "" },
];

export function RetrievalPipelineDiagram() {
  return (
    <div className="rounded-lg p-4 bg-card border border-border">
      <div className="text-xs font-semibold text-muted-strong mb-3">Retrieval Pipeline</div>
      {/* No animation on this row, including the "LLM context" box — this
          is a static architecture-reference diagram (see header comment),
          not a live-state indicator. A prior pass added an ambient dot that
          traveled across it and read as an unwanted live signal on the LLM
          context stage in visual QA; removed rather than repositioned. */}
      <div className="flex items-center gap-2 text-[10px] font-mono overflow-x-auto">
        {STAGES.map((stage, i) => (
          <div key={stage.label} className="flex items-center gap-2">
            {i > 0 && <ArrowRight size={10} className="text-muted-foreground shrink-0" />}
            <div className="p-2 rounded shrink-0 bg-background border border-border">
              <div className="text-primary">{stage.label}</div>
              <div className="text-muted-foreground">{stage.detail}</div>
              {stage.out && <div className="text-risk-low">{stage.out}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
