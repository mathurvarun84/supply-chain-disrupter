"""
cli.py — python -m src.evaluation.trulens_integration.cli {run,dashboard,query}
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from src.evaluation.trulens_integration.config import launch_dashboard
from src.evaluation.trulens_integration.feedback_functions import risk_score_stability
from src.evaluation.trulens_integration.wrapper import run_with_trulens
from src.utils.db_utils import fetch_recent_composite_scores

_SUPPORTED_METRICS = {"risk_drift"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trulens_integration")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run one scenario with TruLens instrumentation")
    run_parser.add_argument("--port", required=True)
    run_parser.add_argument("--sku", required=True)
    run_parser.add_argument("--event-date", required=True)

    sub.add_parser("dashboard", help="Launch the TruLens dashboard on port 8502")

    query_parser = sub.add_parser("query", help="Query historical TruLens metrics")
    query_parser.add_argument("--metric", required=True)
    query_parser.add_argument("--days", type=int, default=30)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        payload = {
            "affected_port": args.port,
            "sku": args.sku,
            "event_date": args.event_date,
        }
        result = run_with_trulens(payload)
        print(f"risk_label={result.risk_label}")
        return 0

    if args.command == "dashboard":
        launch_dashboard(port=8502)
        return 0

    if args.command == "query":
        if args.metric not in _SUPPORTED_METRICS:
            print(f"Unknown metric '{args.metric}'. Supported: {sorted(_SUPPORTED_METRICS)}")
            return 1
        scores = fetch_recent_composite_scores(args.days)
        stability = risk_score_stability(scores)
        print(f"metric=risk_drift days={args.days} n_runs={len(scores)} stability_score={stability:.3f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
