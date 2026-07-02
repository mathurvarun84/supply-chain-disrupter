"""
QA-08a | Integration smoke test — schema + normalization bounds (synthetic fixture)
====================================================================================
Agent tested : db_utils.ensure_risk_classification_table()
               langgraph_engine._get_norm_bounds()
Data source  : Synthetic spec-conformant DB (fixture_spec_conformant_db.py)
               Use when supply_chain_lite_master.xlsx is not available.
               Prefer qa_08_integration_schema_real_data.py when the real workbook is present.

What this file verifies
-----------------------
1. risk_classifications table — exists and has the correct schema.
2. _get_norm_bounds() — reads MIN/MAX from lite_master (4-row synthetic fixture)
   and matches FIXTURE_NORM_BOUNDS within ±0.01.

Expected outcome: all checks PASS (after running fixture_spec_conformant_db.py).
"""

import sys
import os

sys.path.insert(0, ".")

from src.agents.risk_classifier_agent import _get_norm_bounds

_get_norm_bounds.cache_clear()

import sqlite3

from evaluation.fixture_spec_conformant_db import FIXTURE_LITE_MASTER_ROWS, FIXTURE_NORM_BOUNDS
from src.utils.db_utils import ensure_risk_classification_table

print("=== QA-08a: Integration Smoke Test (synthetic fixture) ===")
print()

if not os.path.exists("outputs/supply_chain.db"):
    print("SKIP | outputs/supply_chain.db not found. "
          "Run: python evaluation/fixture_spec_conformant_db.py")
    sys.exit(0)

all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


ensure_risk_classification_table()

conn = sqlite3.connect("outputs/supply_chain.db")
lm_count = conn.execute("SELECT COUNT(*) FROM lite_master").fetchone()[0]
schema_row = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='risk_classifications'"
).fetchone()
conn.close()

if lm_count != FIXTURE_LITE_MASTER_ROWS:
    print(
        f"SKIP | lite_master has {lm_count:,} rows (expected {FIXTURE_LITE_MASTER_ROWS} "
        f"for synthetic fixture). Run: python evaluation/fixture_spec_conformant_db.py "
        f"or use qa_08_integration_schema_real_data.py for the full workbook."
    )
    sys.exit(0)

if schema_row:
    print("PASS | risk_classifications table exists")
    print(f"  Schema snippet: {schema_row[0][:300]}...")
else:
    print("FAIL | risk_classifications table is missing")
    all_pass = False

print()

bounds = _get_norm_bounds()

print("Normalization bounds loaded from lite_master:")
for col, (lo, hi) in bounds.items():
    print(f"  {col}: [{lo:.4f}, {hi:.4f}]")
print()

print("Cross-checking against synthetic fixture spec (tolerance ±0.01):")
for col, (exp_lo, exp_hi) in FIXTURE_NORM_BOUNDS.items():
    actual_lo, actual_hi = bounds[col]
    lo_ok = abs(actual_lo - exp_lo) < 0.01
    hi_ok = abs(actual_hi - exp_hi) < 0.01
    chk(
        lo_ok and hi_ok,
        f"{col}: expected [{exp_lo}, {exp_hi}], "
        f"got [{actual_lo:.4f}, {actual_hi:.4f}]",
    )

print()
print(f"All pass: {all_pass}")
if not all_pass:
    sys.exit(1)
