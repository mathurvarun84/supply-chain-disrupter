"""
finetune_gpt4o_mini.py — Fine-tune GPT-4o-mini via OpenAI API for News Agent (L2).

OPTIONAL for the ensemble. After completion, set OPENAI_FT_NEWS_MODEL in .env.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from fine_tuning.generate_training_data import generate_gpt_finetune_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JSONL_PATH = Path("fine_tuning/data/gpt_finetune_train.jsonl")
RESULTS_PATH = Path("fine_tuning/data/gpt_ft_result.json")
BASE_MODEL = "gpt-4o-mini-2024-07-18"
SUFFIX = "supply-chain-news-agent"


def run_gpt_finetuning() -> str:
    """Full GPT-4o-mini fine-tuning pipeline. Returns fine-tuned model ID."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)

    if not JSONL_PATH.exists():
        generate_gpt_finetune_jsonl()

    with open(JSONL_PATH, "rb") as f:
        upload_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = upload_resp.id
    logger.info("Uploaded training file: %s", file_id)

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        suffix=SUFFIX,
        hyperparameters={"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto"},
    )
    job_id = job.id
    logger.info("Fine-tuning job created: %s", job_id)

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info("Status: %s", job.status)
        if job.status == "succeeded":
            model_id = job.fine_tuned_model
            logger.info("Fine-tuning complete: %s", model_id)
            break
        elif job.status in ("failed", "cancelled"):
            raise RuntimeError(f"Job {job_id} failed: {job.error}")
        time.sleep(30)

    result = {"file_id": file_id, "job_id": job_id, "model_id": model_id}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Add to .env: OPENAI_FT_NEWS_MODEL=%s", model_id)
    return model_id


if __name__ == "__main__":
    run_gpt_finetuning()
