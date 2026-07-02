"""
Test suite for ensemble signals (DistilBERT + LLM Judge + two-stage RAG).
Run: python -m pytest tests/test_ensemble_signals.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_distilbert_signal_no_model():
    """Signal 2 must never raise when model is absent."""
    from src.agents import distilbert_signal
    from src.agents.distilbert_signal import run_distilbert_inference

    with patch.object(distilbert_signal, "_model_available", return_value=False):
        result = run_distilbert_inference(
            {"delivery_status": "Late delivery", "supply_disruption_index": 8.5},
            duration_days=30.0,
        )
    assert result.model_source == "not-available-skipped"
    assert result.predicted_label == "N/A"
    assert result.confidence == 0.0


def test_build_inference_text_matches_training_format():
    """Train/serve text format must be identical."""
    from src.agents.distilbert_signal import build_distilbert_text, build_inference_text

    row = {
        "order_region": "Eastern Asia",
        "product_name": "Laptop",
        "known_disruption_event": "COVID-19 Pandemic",
        "disruption_news_count": 5,
        "supply_disruption_index": 7.5,
        "defect_rate_pct": 2.1,
        "export_control_level": 3.5,
        "risk_score_composite": 0.62,
        "lead_time_variance_days": 12.0,
    }
    assert build_inference_text(row) == build_distilbert_text(row)
    assert "Known disruption event: COVID-19 Pandemic." in build_distilbert_text(row)
    assert "Delivery:" not in build_distilbert_text(row)
    assert "Disruption duration: 0 days." in build_distilbert_text(row)
    assert "Disruption duration: 30 days." in build_distilbert_text(row, duration_days=30.0)


def test_judge_hard_rule_canceled():
    """critical_flag must come from final_label, not judge alone."""
    from src.agents.state import JudgeVerdict

    jv = JudgeVerdict(
        final_label="CRITICAL",
        verdict_type="unanimous",
        reasoning="all agree",
        signals_agreed=True,
        final_critical_flag=True,
    )
    assert (jv.final_label == "CRITICAL") is True


def test_rag_retriever_cross_encoder_fallback():
    """Two-stage RAG degrades gracefully when cross-encoder unavailable."""
    from src.rag.retriever import rerank_results

    mock_candidates = [
        {"text": "Candidate A", "distance": 0.3},
        {"text": "Candidate B", "distance": 0.1},
        {"text": "Candidate C", "distance": 0.5},
    ]
    with patch("src.rag.retriever._get_cross_encoder", return_value=None):
        result = rerank_results("test query", mock_candidates, top_k=2)
    assert len(result) == 2
    assert result[0]["distance"] <= result[1]["distance"]


def test_resolve_embedding_model_hf_repo_id(monkeypatch):
    """EMBEDDING_MODEL_PATH overrides the default Hugging Face repo."""
    import src.rag.utils as rag_utils

    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "user/supply-chain-embeddings")
    assert rag_utils.resolve_embedding_model_name() == "user/supply-chain-embeddings"


def test_resolve_embedding_model_default_hf_repo(monkeypatch):
    """Default embedding model is the project fine-tuned Hugging Face repo."""
    import src.rag.utils as rag_utils

    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    assert rag_utils.resolve_embedding_model_name() == rag_utils.DEFAULT_EMBEDDING_REPO


def test_query_collection_uses_chroma_singleton():
    """Named-collection queries must reuse get_chroma_client(), not a new client."""
    from src.rag.collections import query_collection

    mock_client = MagicMock()
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_client.get_collection.return_value = mock_col

    with patch("src.rag.utils.get_chroma_client", return_value=mock_client):
        with patch("src.rag.utils.get_embedding_model"):
            query_collection("historical_precedents", "test query", n_results=3)
    mock_client.get_collection.assert_called_once()


def test_distilbert_model_inputs_drop_token_type_ids():
    """DistilBERT forward must not receive token_type_ids (transformers 4.57+)."""
    from src.utils.hf_utils import distilbert_model_inputs

    encoded = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "token_type_ids": [[0, 0, 0]],
    }
    filtered = distilbert_model_inputs(encoded)
    assert "token_type_ids" not in filtered
    assert filtered.keys() == {"input_ids", "attention_mask"}


def test_distilbert_inference_strips_token_type_ids():
    """run_distilbert_inference passes only input_ids/attention_mask to the model."""
    import torch
    from src.agents import distilbert_signal
    from src.agents.distilbert_signal import run_distilbert_inference

    class _FakeModel:
        eval = lambda self: self

        def __call__(self, **kwargs):
            assert "token_type_ids" not in kwargs
            assert "input_ids" in kwargs
            logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
            return type("Out", (), {"logits": logits})()

    fake_tok = lambda text, **kw: {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "token_type_ids": torch.tensor([[0, 0]]),
    }

    with patch.object(distilbert_signal, "_model_available", return_value=True):
        with patch.object(distilbert_signal, "_load_model_and_tokenizer", return_value=(fake_tok, _FakeModel())):
            result = run_distilbert_inference(
                {
                    "order_region": "Eastern Asia",
                    "product_name": "Laptop",
                    "known_disruption_event": "COVID-19",
                    "disruption_news_count": 1,
                    "supply_disruption_index": 7.0,
                    "defect_rate_pct": 2.0,
                    "export_control_level": 3.0,
                    "risk_score_composite": 0.5,
                    "lead_time_variance_days": 5.0,
                },
                duration_days=3.0,
            )
    assert result.model_source == "fine-tuned"
    assert result.predicted_label == "CRITICAL"


def test_embedding_weights_requires_st_config(tmp_path):
    """Partial weight files without modules.json must not be treated as loadable."""
    from src.rag.utils import _embedding_weights_present

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"partial")
    assert _embedding_weights_present(incomplete) is False

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "model.safetensors").write_bytes(b"ok")
    (complete / "modules.json").write_text("[]")
    assert _embedding_weights_present(complete) is True


def test_get_embedding_model_fallback_to_base(monkeypatch):
    """get_embedding_model falls back to base MiniLM when Hugging Face load fails."""
    import src.rag.utils as rag_utils

    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    with patch("src.rag.utils.SentenceTransformerEmbeddingFunction") as mock_ef:
        mock_ef.side_effect = [RuntimeError("hub unavailable"), object()]
        ef = rag_utils.get_embedding_model()
    assert ef is not None
    assert mock_ef.call_count == 2
    assert mock_ef.call_args_list[1].kwargs["model_name"] == rag_utils.EMBEDDING_MODEL
