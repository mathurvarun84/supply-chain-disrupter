"""
config.py — TruLens session + dashboard initialization.

Uses the current (2.8.x) TruSession API — NOT the deprecated Tru() class
from trulens-eval (removed from maintenance 2025-12-01).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from trulens.core import TruSession
from trulens.dashboard import run_dashboard

DB_PATH = Path("data/trulens/trulens.db")


@lru_cache(maxsize=1)
def get_session() -> TruSession:
    """Process-lifetime TruSession backed by SQLite at data/trulens/trulens.db."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return TruSession(database_url=f"sqlite:///{DB_PATH}")


def launch_dashboard(port: int = 8502) -> None:
    """Launch the TruLens Streamlit dashboard as its own process on `port`."""
    run_dashboard(get_session(), port=port)
