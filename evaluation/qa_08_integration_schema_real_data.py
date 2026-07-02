"""
QA-08b | Integration smoke test — schema + normalization bounds (real v3.1 dataset)
=====================================================================================
Agent tested : db_utils.ensure_risk_classification_table()
               langgraph_engine._get_norm_bounds()
               etl_loader (via the ingested lite_master table)
Data source  : outputs/supply_chain.db built from supply_chain_lite_master.xlsx
               (ETL via scripts/build_databases.py or src/utils/etl_loader.py)

What this file verifies
-----------------------
1. ETL completeness — lite_master and daily_records VIEW both contain
   EXPECTED_LITE_MASTER_ROWS (11,559 for v3.1 workbook).

2. risk_classifications table — re-created after ETL.

3. _get_norm_bounds() from REAL data matches SPEC_NORM_BOUNDS within ±0.01.

4. delivery_status distribution — no "MEDIUM" value (live-mode only).

Expected outcome: all checks PASS.
"""

import sys

sys.path.insert(0, ".")

from src.agents.risk_classifier_agent import _get_norm_bounds

_get_norm_bounds.cache_clear()

import sqlite3

from src.utils.db_utils import ensure_risk_classification_table
from src.utils.etl_loader import EXPECTED_LITE_MASTER_ROWS, SPEC_NORM_BOUNDS

print("=== QA-08b: Integration Smoke Test (real v3.1 dataset) ===")
print()

all_pass = True


def chk(condition: bool, msg: str) -> None:
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


ensure_risk_classification_table()

conn = sqlite3.connect("outputs/supply_chain.db")

lm_count = conn.execute("SELECT COUNT(*) FROM lite_master").fetchone()[0]
dr_count = conn.execute("SELECT COUNT(*) FROM daily_records").fetchone()[0]

chk(
    lm_count == EXPECTED_LITE_MASTER_ROWS,
    f"lite_master contains {EXPECTED_LITE_MASTER_ROWS:,} electronics rows (got {lm_count:,})",
)
chk(
    dr_count == EXPECTED_LITE_MASTER_ROWS,
    f"daily_records VIEW exposes all {EXPECTED_LITE_MASTER_ROWS:,} rows (got {dr_count:,})",
)

schema_row = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='risk_classifications'"
).fetchone()

if schema_row:
    print("PASS | risk_classifications table exists")
    print(f"  Schema snippet: {schema_row[0][:300]}...")
else:
    print("FAIL | risk_classifications table missing")
    all_pass = False

print()

bounds = _get_norm_bounds()

print("Normalization bounds loaded from REAL lite_master:")
for col, (lo, hi) in bounds.items():
    print(f"  {col}: [{lo:.4f}, {hi:.4f}]")
print()

print("Cross-checking against v3.1 spec ground truth (tolerance ±0.01):")
for col, (exp_lo, exp_hi) in SPEC_NORM_BOUNDS.items():
    actual_lo, actual_hi = bounds[col]
    lo_ok = abs(actual_lo - exp_lo) < 0.01
    hi_ok = abs(actual_hi - exp_hi) < 0.01
    chk(
        lo_ok and hi_ok,
        f"{col}: expected [{exp_lo}, {exp_hi}], "
        f"got [{actual_lo:.4f}, {actual_hi:.4f}]",
    )

dist = conn.execute(
    """
    SELECT delivery_status, COUNT(*) AS cnt
    FROM lite_master
    GROUP BY delivery_status
    ORDER BY cnt DESC
    """
).fetchall()
conn.close()

print()
print("Delivery status distribution in real data:")
for ds_value, cnt in dist:
    print(f"  {repr(ds_value):30s}  {cnt:5,} rows")

status_values = {row[0] for row in dist}
chk(
    "MEDIUM" not in status_values,
    "No 'MEDIUM' delivery_status in historical rows — MEDIUM is live-mode only",
)

print()
print(f"All pass: {all_pass}")
if not all_pass:
    sys.exit(1)
