/**
 * Screen 6 (RAG / RAGAS Evaluation) tab body — single-scroll layout, no
 * sub-tabs. Renders the RAGAS scorecard, corpus health cards, retrieval
 * pipeline diagram, Faithfulness Gate panel, and gold dataset table top
 * to bottom, matching _reference/App.mockup.tsx's TabRAGEval section
 * order. Wires to /api/rag/scorecard, /api/rag/corpus, /api/rag/gold-dataset
 * (still fixture JSON — Day 8 replaces fixtures.py with real RAGAS +
 * live ChromaDB collection stats). Each panel owns its own data fetching
 * via its hook; this component only owns layout.
 */
import { RagasScorecard } from "./components/rag-eval/RagasScorecard";
import { CorpusHealthCards } from "./components/rag-eval/CorpusHealthCards";
import { RetrievalPipelineDiagram } from "./components/rag-eval/RetrievalPipelineDiagram";
import { FaithfulnessGatePanel } from "./components/rag-eval/FaithfulnessGatePanel";
import { GoldDatasetTable } from "./components/rag-eval/GoldDatasetTable";

export function TabRagEval() {
  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <RagasScorecard />

      <div className="grid gap-3 grid-cols-2">
        <CorpusHealthCards />

        <div className="space-y-3">
          <RetrievalPipelineDiagram />
          <FaithfulnessGatePanel />
        </div>
      </div>

      <GoldDatasetTable />
    </div>
  );
}
