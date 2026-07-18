/**
 * Screen 4 ranked mitigation action card. Reuses CitationChip for any real
 * provenance the backend exposes, and keeps the card chrome aligned with the
 * rest of the command-center panels.
 */
import { CitationChip } from "../risk-classification/CitationChip";
import type { MitigationRankedAction } from "../../types/mitigation";

const ACTION_TYPE_LABEL: Record<MitigationRankedAction["action_type"], string> = {
  INVENTORY: "Inventory",
  ROUTING: "Routing",
  SOURCING: "Sourcing",
  "INDIA-SOURCING": "India Sourcing",
  MONITOR: "Monitor",
  FINANCIAL: "Financial",
};

export function MitigationActionCard({ action }: { action: MitigationRankedAction }) {
  return (
    <article className="rounded-panel p-4 bg-card border border-border">
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary/25 bg-primary/10 font-mono text-sm font-bold text-primary">
          {action.rank}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            {ACTION_TYPE_LABEL[action.action_type]}
          </div>
          <div className="mt-1 text-sm leading-6 text-foreground">{action.text}</div>
          {action.citations.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {action.citations.map((citation) => (
                <CitationChip
                  key={`${citation.collection}:${citation.source_file}`}
                  source={citation.source_file}
                  collection={citation.collection}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
