/**
 * Screen 6 corpus health cards — one per ChromaDB collection, direct
 * visualization of the [SOURCE: REAL]/[SOURCE: SYNTHESIZED] tagging
 * discipline. Owns its own query via useRagCorpus — GET /api/rag/corpus.
 * Card order is fixed (historical_precedents, export_control_corpus,
 * india_sourcing_corpus) rather than sorted, so it stays stable across
 * fixture edits. Layout ported from _reference/App.mockup.tsx:971-987.
 */
import { useRagCorpus } from "../../hooks/useRagEval";
import { useCountUp } from "../../utils/animation";
import type { CorpusHealth } from "../../types/ragEval";

const COLLECTION_ORDER = ["historical_precedents", "export_control_corpus", "india_sourcing_corpus"];

function CorpusCard({ c }: { c: CorpusHealth }) {
  // useCountUp now returns the raw float tween (doc counts elsewhere
  // format as decimals), so round it here for an integer doc count.
  const docs = Math.round(useCountUp(c.docs));
  return (
    <div className="p-3 rounded bg-background border border-border">
      <div className="text-[11px] font-mono text-primary mb-1.5">{c.name}</div>
      <div className="flex items-center gap-2 text-[10px] flex-wrap">
        <span className="font-mono text-muted-foreground">{docs} docs</span>
        <span className="px-1.5 py-0.5 rounded font-mono text-risk-low bg-risk-low/10">
          REAL: {c.real}
        </span>
        {c.synth > 0 && (
          <span className="px-1.5 py-0.5 rounded font-mono text-accent bg-accent/10">
            SYNTH: {c.synth}
          </span>
        )}
      </div>
      <div className="text-[9px] font-mono text-muted-foreground mt-1.5">
        Last re-ingested: {c.last_ingested_at}
      </div>
    </div>
  );
}

export function CorpusHealthCards() {
  const { data, isLoading, isError } = useRagCorpus();

  if (isLoading) {
    return <div className="text-xs text-muted-foreground p-2">Loading corpus health…</div>;
  }

  if (isError || !data) {
    return <div className="text-xs text-risk-critical p-2">Could not load corpus health.</div>;
  }

  const ordered = [...data].sort(
    (a, b) => COLLECTION_ORDER.indexOf(a.name) - COLLECTION_ORDER.indexOf(b.name)
  );

  return (
    <div className="rounded-lg p-4 bg-card border border-border">
      <div className="text-xs font-semibold text-muted-strong mb-3">
        Corpus Health — 3 ChromaDB Collections
      </div>
      <div className="space-y-2">
        {ordered.map((c) => (
          <CorpusCard key={c.name} c={c} />
        ))}
      </div>
    </div>
  );
}
