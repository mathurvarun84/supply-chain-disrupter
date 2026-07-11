/**
 * Shared RAG citation chip for Screen 2. Extracted from the inline
 * CitationChip in _reference/App.mockup.tsx:225-236. The real ensemble
 * only returns a citation source string (LLMSignal.rag_citations), not a
 * separate collection name, so `collection` is optional here and the
 * tooltip falls back to the source alone when absent.
 */
import { ExternalLink } from "lucide-react";

export function CitationChip({ source, collection }: { source: string; collection?: string }) {
  return (
    <span
      title={collection ? `${collection}: ${source}` : source}
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full font-mono cursor-pointer hover:opacity-80 transition-opacity bg-primary/10 text-primary border border-primary/25"
    >
      <ExternalLink size={9} />
      {source}
    </span>
  );
}
