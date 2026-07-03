# RAGAS Evaluation

Chunk-grounded RAG evaluation for the semiconductor/electronics supply
chain RAG layer, built across three phases:

- **Phase 1** (`generate_test_dataset.py`) — gold QA dataset generation.
- **Phase 2** (`rag_tracer.py`) — non-invasive trace interceptor.
- **Phase 3** (`run_evaluation.py`) — evaluation runner (this section).
- **L4 live extension** (`run_l4_live_evaluation.py`) — RAGAS against the
  real L4 Risk Classifier agent (see below).

## Phase 3 — Running the evaluation

```bash
python -m evaluation.ragas.run_evaluation --mode retrieval-only
python -m evaluation.ragas.run_evaluation --mode full [--yes] [--limit N] \
    [--collections export_control_corpus,india_sourcing_corpus]
```

Other flags: `--styles agent_pattern,natural_question`, `--top-k` (rerank
top-k, default 3), `--bi-encoder-top-n` (default 10).

- `retrieval-only` makes **zero LLM calls** — safe to re-run constantly,
  works with no `OPENAI_API_KEY` set at all. Scores Hit Rate@k, MRR, and
  embedding-based context relevance/recall proxies using the same
  bi-encoder (`all-MiniLM-L6-v2` / fine-tuned variant) that powers
  production retrieval. Writes `ragas_scores_retrieval_only.json`.
- `full` requires `OPENAI_API_KEY` and runs the real RAGAS metric suite
  (Faithfulness, Answer Relevancy, Context Precision, Context Recall) via
  the RAGAS judge (`gpt-4.1-mini` / `MODEL_FAST` — chosen over `gpt-4o` to
  keep repeated Day-26 error-analysis runs cheap; answer generation itself
  still uses `gpt-4o` / `MODEL_REASONING`). Prints a cost estimate and
  prompts for confirmation above 20 cases unless `--yes` is passed. Writes
  `ragas_scores_full.json`.

### Locked target metrics

| Metric              | Target  |
|----------------------|--------|
| Faithfulness          | > 0.85 |
| Answer Relevancy       | > 0.80 |
| Context Precision      | > 0.75 |
| Context Recall         | > 0.75 (project convention: same bar as Context Precision, not separately specified — applied consistently) |

Collections/metrics below target are reported in `full` mode's `flagged`
list, sorted worst-first by gap, for Day 26 error analysis to act on.

### Known scope limitation — full mode's "answer" is RAG-only

Gold test cases have no accompanying live SQLite order record (Phase 1
generates them from ChromaDB chunks alone). Full-pipeline mode therefore
cannot replay the EXACT production prompts of `build_risk_classifier_context`
→ `LLMSignal` or `build_mitigation_context` → `MitigationLLMOutput`, both of
which require a real `lite_master` row. Instead, full mode:

1. Calls the REAL production retrieval function (`retrieve_and_rerank`)
   with the gold question as the query and the gold `source_collection`
   as the target — this is the actual retrieval layer, unmodified.
2. Generates an answer with a minimal, RAG-only "answer strictly from
   context" prompt via `call_openai_structured` — NOT the full agent
   system prompt (which needs SQLite fields we don't have per-question).

This means full-pipeline Faithfulness/Answer Relevancy scores measure the
RAG layer's retrieval-and-grounding quality, not the complete agent
pipeline's behavior end to end. This is an explicit, reasoned scope
decision, not an oversight (same convention as documented scope cuts
elsewhere in the project, e.g. drift detection, SHAP).

### `context_recall_proxy` vs `context_recall`

`retrieval-only` mode's `context_recall_proxy` (cosine similarity between
the ground truth and the best-matching retrieved chunk, using the
production bi-encoder) is **not the same measurement** as `full` mode's
`context_recall` (RAGAS's LLM-graded metric, using a gpt-4.1-mini judge). They
use different methods and different embedding models — don't compare them
directly across modes.

**Not in scope for Phase 3:** plugging into `fine_tuning/evaluate_all.py`
as `evaluate_ragas()` (Phase 4).

## L4 live extension — evaluating the real Risk Classifier (Signal 3)

Phase 3's `full` mode deliberately does **not** replay the real
`build_risk_classifier_context` → `LLMSignal` prompt (no live SQLite row per
gold question — see the scope note above). `run_l4_live_evaluation.py`
closes that gap for L4 specifically: it runs the actual
`risk_classifier_agent()` against real historical orders from
`outputs/supply_chain.db`, and scores Signal 3's real `LLMSignal.rationale`
with RAGAS.

```bash
python -m evaluation.ragas.run_l4_live_evaluation [--n-per-bucket N] [--yes]
```

- Samples `N` orders (default 5) from **each** `delivery_status` bucket
  (`Shipping canceled`, `Late delivery`, `Shipping on time`,
  `Advance shipping`) so Signal 3 gets exercised across the full
  LOW..CRITICAL label range, not just one easy bucket.
- Requires `OPENAI_API_KEY` (fails fast, no stack trace, if unset) — there's
  no zero-cost mode here since the whole point is scoring a real GPT-4o
  call. Prints a cost estimate and prompts above 20 orders unless `--yes`.
- Writes `ragas_scores_l4_live.json` (`overall`, `by_delivery_status`,
  `flagged`, `per_case`) and a trace JSONL under `traces/`.

### Why only Faithfulness + Answer Relevancy (no Context Precision/Recall)

Live classifications have no gold reference rationale to compare against —
there is no `ground_truth` column. Context Precision/Recall are
reference-dependent RAGAS metrics and cannot be computed here. Faithfulness
(is the rationale grounded in what Signal 3 was actually shown) and Answer
Relevancy (does the rationale address the RAG query) are both reference-free
and are exactly the two metrics that answer "did the model hallucinate
beyond the retrieved evidence" for this specific call.

### Why only Signal 3 — not `_gather_rag_citations` or the Judge

`risk_classifier_agent()` makes **two separate LLM calls** and uses **two
separate retrieval paths**, not one:

1. `_gather_rag_citations()` (`agent.py`) — a deterministic, non-LLM
   citation string built from its own `query_chroma_rag` call. It cannot
   hallucinate (no LLM involved), so RAGAS doesn't apply to it.
2. `run_llm_signal()` → Signal 3 — the real GPT-4o call, grounded in
   `build_risk_classifier_context()`'s two-stage `retrieve_and_rerank`
   chunks. **This is what gets scored.**
3. `run_judge()` — a third, unrelated `call_openai_structured` call with no
   retrieval step of its own. Patched to a no-op here to save an unrelated
   GPT-4o call per order; this only changes which branch of the
   judge → llm_signal → rule fallback chain sets `final_label` internally
   and does not affect Signal 3's rationale, which is captured directly.

Because of this, the harness does **not** reuse the general-purpose
`RAGTraceCollector` (Phase 2) — that tracer patches `retrieve_and_rerank`
AND `query_chroma_rag` globally, which would blend citation-only chunks
(never shown to the LLM) into Signal 3's context and would also capture the
Judge's context-less call as a second, noisy trace record. Instead
`Signal3Capture` narrowly patches only
`src.rag.retriever.retrieve_and_rerank` and
`src.agents.risk_classifier_agent.llm_signal.call_openai_structured`,
restoring both after every order.

### Reading low scores: expected behavior, not a bug

If Signal 3's rationale leans on SQLite record fields (composite score,
delivery status, semiconductor signals) rather than the retrieved
historical-precedent text — which it does for routine, non-dramatic orders,
and will often say so explicitly ("the RAG context does not provide any
relevant historical precedents") — Faithfulness will score low. That is
correct: RAGAS only credits claims traceable to the `contexts` we supply
(the RAG chunks), and the rationale's other, non-RAG inputs are out of
scope by design (see "Why only Faithfulness + Answer Relevancy" above). A
low score here means "retrieval wasn't relevant enough to ground this
rationale," which is a legitimate Day 26 signal — not proof the model
invented facts.
