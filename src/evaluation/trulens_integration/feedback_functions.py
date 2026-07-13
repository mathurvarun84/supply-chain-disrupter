"""
feedback_functions.py — Four domain-specific TruLens metrics for the L1-L7
pipeline. Pure functions over plain values (no TruLens/SQLite dependency)
so they're unit-testable in isolation; wrapper.py and cli.py are
responsible for sourcing the historical data these take as input and for
recording the resulting score into TruLens.

Targets (from docs/specs/2026-07-06-trulens-integration-design.md):
  risk_score_stability: coefficient of variation across last 30 runs, target CV < 0.30
  ensemble_agreement:   fraction of runs where >=2 of 3 signals agree, target > 0.66
  node_latency_check:   binary pass/fail per node vs. threshold, target: all pass
  forecast_accuracy:    1 - relative forecast error, target > 0.80
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

# node label -> latency threshold in ms (from the spec's "Custom Feedback Functions" table)
_LATENCY_THRESHOLDS_MS: dict[str, float] = {
    "L2": 2000.0,
    "L3": 2000.0,
    "L4": 5000.0,
    "total": 15000.0,
}


def risk_score_stability(composite_scores: list[float]) -> float:
    """1.0 - coefficient_of_variation, clipped to [0, 1]. CV < 0.30 => score > 0.70."""
    if len(composite_scores) < 2:
        return 1.0
    avg = mean(composite_scores)
    if avg == 0:
        return 1.0
    cv = pstdev(composite_scores) / avg
    return max(0.0, min(1.0, 1.0 - cv))


def ensemble_agreement(label_triples: list[tuple[str, str, str]]) -> float:
    """Fraction of (rule, distilbert, llm) triples where >=2 labels match."""
    if not label_triples:
        return 1.0
    agreements = 0
    for triple in label_triples:
        counts = Counter(triple)
        if counts.most_common(1)[0][1] >= 2:
            agreements += 1
    return agreements / len(label_triples)


def node_latency_check(latencies_ms: dict[str, float]) -> float:
    """Fraction of measured nodes whose latency is within threshold. 1.0 = all pass."""
    checked = {k: v for k, v in latencies_ms.items() if k in _LATENCY_THRESHOLDS_MS}
    if not checked:
        return 1.0
    passing = sum(1 for label, ms in checked.items() if ms <= _LATENCY_THRESHOLDS_MS[label])
    return passing / len(checked)


def forecast_accuracy(predicted_drop_pct: float, actual_drop_pct: float) -> float:
    """1 - abs(predicted - actual) / max(predicted, actual), clipped to [0, 1]."""
    denom = max(predicted_drop_pct, actual_drop_pct)
    if denom == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(predicted_drop_pct - actual_drop_pct) / denom))
