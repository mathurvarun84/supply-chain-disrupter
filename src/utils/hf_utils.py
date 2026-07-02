"""
Shared Hugging Face Hub / transformers loading helpers.

Corporate networks that MITM TLS (or misconfigured CA bundles) often break
`from_pretrained()` calls to huggingface.co. Set HF_INSECURE_SSL=1 (or reuse
INGEST_INSECURE_SSL=1) to disable certificate verification for Hub downloads.

Local model directories always load with local_files_only=True so inference
never hits the network when weights are already on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

_SSL_CONFIGURED = False


def insecure_ssl_enabled() -> bool:
    """True when the user explicitly opted out of TLS verification for Hub fetches."""
    for key in ("HF_INSECURE_SSL", "INGEST_INSECURE_SSL"):
        if os.getenv(key, "").lower() in {"1", "true", "yes"}:
            return True
    return False


def configure_hf_hub_ssl() -> None:
    """
    Patch requests so Hugging Face Hub downloads skip TLS verify when opted in.

    Idempotent — safe to call before every from_pretrained().
    """
    global _SSL_CONFIGURED
    if _SSL_CONFIGURED or not insecure_ssl_enabled():
        return
    _SSL_CONFIGURED = True

    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    try:
        import requests

        _orig_session_request = requests.Session.request

        def _session_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_session_request(self, method, url, **kwargs)

        requests.Session.request = _session_request  # type: ignore[method-assign]
    except Exception:
        pass


def is_local_model_path(model_id_or_path: str) -> bool:
    return Path(model_id_or_path).is_dir()


def from_pretrained_kwargs(model_id_or_path: str) -> Dict[str, Any]:
    """Kwargs for AutoTokenizer / AutoModel from_pretrained."""
    configure_hf_hub_ssl()
    kwargs: Dict[str, Any] = {}
    if is_local_model_path(model_id_or_path):
        kwargs["local_files_only"] = True
    return kwargs


def distilbert_model_inputs(encoded: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip tokenizer outputs that DistilBERT rejects.

    AutoTokenizer may return token_type_ids; transformers 4.57+ DistilBERT
    forward() no longer accepts that kwarg.
    """
    return {k: v for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
