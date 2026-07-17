/**
 * Screen 4 ranked mitigation action card. Reuses CitationChip for any real
 * provenance the backend exposes, and keeps the card chrome aligned with the
 * rest of the command-center panels.
 */
import { CitationChip } from "../risk-classification/CitationChip";
import type { MitigationRankedAction } from "../../types/mitigation";

export function MitigationActionCard({ action }: { action: MitigationRankedAction }) {
  return (
    <article className="rounded-panel p-4 bg-card border border-border">
      <div className="flex gap-3">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary/25 bg-primary/10 font-mono text-sm font-bold text-primary">
          {action.rank}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm leading-6 text-foreground">{action.text}</div>
          {action.citations.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {action.citations.map((citation) => (
                <CitationChip key={citation} source={citation} collection="mitigation" />
              ))}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
