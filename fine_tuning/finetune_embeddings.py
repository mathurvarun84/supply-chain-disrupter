"""
finetune_embeddings.py — Fine-tune all-MiniLM-L6-v2 for supply-chain RAG.

Technique: MultipleNegativesRankingLoss (in-batch negatives)
Target: top-3 retrieval accuracy > 85%
Produces: fine_tuning/models/supply_chain_embeddings/
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_tuning.generate_training_data import save_all_qa_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "all-MiniLM-L6-v2"
MODEL_OUTPUT = Path("fine_tuning/models/supply_chain_embeddings")
DATA_PATH = Path("fine_tuning/data/qa_pairs.json")

BATCH_SIZE = 32
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LR = 2e-5


def build_eval_set(pairs: list, n_eval: int = 100) -> tuple:
    """Held-out eval set for InformationRetrievalEvaluator."""
    random.seed(42)
    eval_pairs = random.sample(pairs, min(n_eval, len(pairs)))
    queries, corpus, relevant_docs = {}, {}, {}
    for i, pair in enumerate(eval_pairs):
        qid, cid = f"q{i}", f"c{i}"
        queries[qid] = pair["query"]
        corpus[cid] = pair["positive"]
        relevant_docs[qid] = {cid}
    return queries, corpus, relevant_docs


def _accuracy_at_3(eval_result: dict | float) -> float:
    """Extract Accuracy@3 from sentence-transformers evaluator output."""
    if isinstance(eval_result, (int, float)):
        return float(eval_result)
    for key, value in eval_result.items():
        if "accuracy@3" in key.lower():
            return float(value)
    raise KeyError(f"No accuracy@3 metric in evaluator result: {eval_result!r}")


def run_embedding_finetuning() -> str:
    """Full embedding fine-tuning pipeline. Returns path to saved model."""
    if not DATA_PATH.exists():
        logger.info("QA pairs not found — generating now...")
        save_all_qa_pairs()

    with open(DATA_PATH) as f:
        pairs = json.load(f)
    logger.info("Loaded %d QA pairs.", len(pairs))

    queries, corpus, relevant_docs = build_eval_set(pairs, n_eval=min(100, len(pairs) // 5))
    eval_queries = set(queries.values())
    train_pairs = [p for p in pairs if p["query"] not in eval_queries]

    train_examples = [InputExample(texts=[p["query"], p["positive"]]) for p in train_pairs]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    model = SentenceTransformer(BASE_MODEL)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name="supply_chain_ir", show_progress_bar=False,
    )

    logger.info("Evaluating base model (baseline)...")
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_result = evaluator(model, output_path=str(MODEL_OUTPUT / "eval"))
    baseline_score = _accuracy_at_3(baseline_result)
    logger.info("Baseline retrieval Accuracy@3: %.4f", baseline_score)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        optimizer_params={"lr": LR},
        evaluator=evaluator,
        evaluation_steps=200,
        output_path=str(MODEL_OUTPUT),
        save_best_model=True,
        show_progress_bar=True,
    )

    finetuned_model = SentenceTransformer(str(MODEL_OUTPUT))
    final_result = evaluator(finetuned_model)
    final_score = _accuracy_at_3(final_result)
    improvement = final_score - baseline_score
    logger.info(
        "Baseline Accuracy@3: %.4f → Fine-tuned: %.4f  (+%.1f%%)",
        baseline_score, final_score, improvement * 100,
    )

    metrics = {
        "baseline_accuracy_at_3": baseline_score,
        "finetuned_accuracy_at_3": final_score,
        "improvement": improvement,
        "baseline_full": baseline_result if isinstance(baseline_result, dict) else {},
        "finetuned_full": final_result if isinstance(final_result, dict) else {},
    }
    with open(MODEL_OUTPUT / "retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return str(MODEL_OUTPUT)


if __name__ == "__main__":
    saved = run_embedding_finetuning()
    print(f"\nFine-tuned embedding model: {saved}")
    print("IMPORTANT: Rebuild ChromaDB: python scripts/build_rag_collections.py")
