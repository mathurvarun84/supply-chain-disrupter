"""Regression tests for the two L6 Monte Carlo bug fixes:

1. Defect-rate sign inversion (engine.py: defects now erode usable inbound
   supply instead of demand, so higher Defect_Rate_Pct increases modeled
   stockout risk instead of paradoxically lowering it).
2. Non-deterministic hash()-based seed (agent.py: seed is now derived from
   a stable sha256 hash, reproducible across process restarts).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from src.agents.simulation_agent.engine import run_monte_carlo
from src.agents.simulation_agent.priors import SimulationParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_params(**overrides) -> SimulationParams:
    defaults = dict(
        initial_inventory=50.0,
        incoming_supply=400.0,
        baseline_lead_time=7.0,
        mean_daily_demand=50.0,
        unit_price_usd=10.0,
        horizon_days=30,
        severity=0.5,
        shock_duration_days=5,
        disruption_type="port closure",
        composite_score=0.5,
        supply_disruption_index=6.0,
        defect_rate_pct=5.0,
        alternate_route="Suez Canal",
        logistics_disruption=True,
        trials=800,
        seed=42,
    )
    defaults.update(overrides)
    return SimulationParams(**defaults)


def test_higher_defect_rate_increases_stockout_risk():
    """Two otherwise-identical params differing only in defect_rate_pct:
    the higher-defect run's stockout probability must be >= the lower-defect
    run's. Before the fix, defects reduced modeled demand and could lower
    the stockout probability as defect rate rose."""
    low = run_monte_carlo(_base_params(defect_rate_pct=1.0, seed=99))
    high = run_monte_carlo(_base_params(defect_rate_pct=25.0, seed=99))
    assert high.stockout_probability_pct >= low.stockout_probability_pct


def test_simulation_seed_is_reproducible_across_processes():
    """Same active_record run through simulation_agent() in two separate
    fresh Python processes must produce identical p10/p50/p90 — the
    regression guard for the hash()-seed bug, which only manifests across
    process boundaries (hash() salting is per-process)."""
    script = textwrap.dedent(
        """
        import json
        import sys
        sys.path.insert(0, r"%s")
        from src.agents.simulation_agent.agent import simulation_agent
        from src.agents.state import EventMetadata, GlobalState, RiskClassificationResult

        risk = RiskClassificationResult(
            mode="live", composite_score=0.55, geo_component=0.4,
            supply_component=0.5, freight_component=0.6, defect_component=0.4,
            duration_days=3.0, base_label="HIGH", final_label="HIGH",
            escalated=False, rationale="test", critical_flag=False,
        )
        state = GlobalState(
            event_metadata=EventMetadata(
                disruption_type="port closure", affected_port="Eastern Asia",
                affected_route="test", severity=0.6, shock_duration_days=5,
                recovery_window_days=30, synthetic_ratio=0.0, simulation_trials=300,
            ),
            config={
                "route_maps": {"JNPT": {"backup_route": "Cape of Good Hope"}},
                "region_route_maps": {"Eastern Asia": {"backup_route": "Suez Canal"}},
            },
            active_record={
                "event_date": "2024-01-01", "port": "Eastern Asia",
                "order_region": "Eastern Asia", "sku": "CHIP_AP",
                "inventory_level": 200.0, "incoming_supply": 100.0,
                "lead_time_days": 7.0, "demand": 50.0, "sales_usd": 500.0,
                "unit_price_usd": 10.0, "supply_disruption_index": 6.5,
                "defect_rate_pct": 5.0,
            },
            risk_classification=risk,
        )
        result = simulation_agent(state)["simulation_result"]
        print(json.dumps({
            "p10": result.stockout_probability_p10,
            "p50": result.stockout_probability_pct,
            "p90": result.stockout_probability_p90,
        }))
        """
        % str(PROJECT_ROOT).replace("\\", "\\\\")
    )

    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    assert outputs[0] == outputs[1]
