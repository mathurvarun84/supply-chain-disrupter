import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.trulens_integration.config import get_session, launch_dashboard


def test_get_session_returns_same_instance_on_repeated_calls():
    s1 = get_session()
    s2 = get_session()
    assert s1 is s2


def test_get_session_uses_data_trulens_sqlite_path():
    session = get_session()
    assert "data/trulens/trulens.db" in str(session.connector.db.engine.url)


def test_launch_dashboard_calls_run_dashboard_with_port_8502():
    with patch("src.evaluation.trulens_integration.config.run_dashboard") as mock_run:
        launch_dashboard()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("port") == 8502
