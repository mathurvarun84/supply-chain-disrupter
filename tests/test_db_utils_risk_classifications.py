import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils.db_utils as db_utils
from src.utils.db_utils import fetch_recent_composite_scores, insert_risk_classification


def test_fetch_recent_composite_scores_does_not_crash_on_a_fresh_database(tmp_path, monkeypatch):
    # Regression test for the bug flagged in PR review: the risk-drift
    # query was not safe on a freshly built database. Root cause:
    # fetch_recent_composite_scores() called ensure_schema(), which only
    # creates agent_execution_log/llm_call_log — the risk_classifications
    # table is created by a *different* function,
    # ensure_risk_classification_table(), only ever invoked lazily inside
    # insert_risk_classification(). On a DB where no classification has
    # ever been written, the table genuinely doesn't exist yet.
    monkeypatch.setattr(db_utils, "DB_PATH", tmp_path / "fresh_supply_chain.db")

    scores = fetch_recent_composite_scores(30)

    assert scores == []


def test_fetch_recent_composite_scores_returns_values_after_a_real_insert(tmp_path, monkeypatch):
    monkeypatch.setattr(db_utils, "DB_PATH", tmp_path / "fresh_supply_chain.db")

    insert_risk_classification(
        order_id=None, mode="live", composite_score=0.62, geo_component=0.3,
        supply_component=0.2, freight_component=0.1, defect_component=0.1,
        duration_days=None, base_label="HIGH", final_label="HIGH",
        escalated=False, rag_citations=[], rationale="r",
    )

    scores = fetch_recent_composite_scores(30)

    assert scores == [0.62]
