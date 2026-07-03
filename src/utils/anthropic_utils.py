"""
anthropic_utils.py — Shared Anthropic (Claude) structured-output helper.

Mirrors src/utils/openai_utils.py::call_openai_structured() but calls
Claude via client.messages.parse(). Used by evaluation/ragas dataset
generation, which is intentionally Anthropic-backed rather than OpenAI-backed
(production RAG agents still call call_openai_structured — unchanged).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Type, TypeVar

logger = logging.getLogger(__name__)


def _load_project_env() -> None:
    """Load .env from project root so scripts pick up ANTHROPIC_API_KEY."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


_load_project_env()

from anthropic import Anthropic, RateLimitError
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Model constants.
MODEL_CLAUDE = "claude-opus-4-8"
MODEL_CLAUDE_HAIKU = "claude-haiku-4-5"
MAX_TOKENS_DEFAULT = 4096

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _get_client() -> Anthropic:
    """Return a process-lifetime Anthropic client. Raises if ANTHROPIC_API_KEY is unset."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add to .env.")
    return Anthropic(api_key=api_key)


def has_anthropic_api_key() -> bool:
    """Return True when ANTHROPIC_API_KEY is set in the environment."""
    _load_project_env()
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def call_anthropic_structured(
    system_prompt: str,
    user_message: str,
    response_model: Type[T],
    model: str = MODEL_CLAUDE,
    max_tokens: int = MAX_TOKENS_DEFAULT,
) -> T:
    """
    Call the Claude Messages API and return a validated Pydantic instance.

    Uses client.messages.parse() with output_format=response_model. Retries
    on RateLimitError only, matching call_openai_structured()'s policy.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    def _call() -> T:
        client = _get_client()
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            output_format=response_model,
        )

        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude refused the request (model={model})")

        usage = response.usage
        logger.info(
            "[LLM] tool=%s model=%s in=%d out=%d",
            response_model.__name__, model,
            getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0),
        )
        return response.parsed_output

    return _call()
