"""
GET /api/admin/tables + GET /api/admin/tables/{table_name} — the read-only
Data Explorer sub-tab's backing endpoints. Runs against a temp SQLite file
seeded with known tables/rows; no dependency on outputs/supply_chain.db or
any live OpenAI/ChromaDB service.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.utils.db_utils.DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT, qty INTEGER)")
    conn.execute("CREATE TABLE lite_master (record_id INTEGER PRIMARY KEY, sku TEXT)")
    conn.executemany(
        "INSERT INTO widgets (name, qty) VALUES (?, ?)",
        [(f"widget-{i}", i) for i in range(120)],
    )
    conn.commit()
    conn.close()
    yield db_path


def test_list_tables_excludes_sqlite_internal_tables(seeded_db):
    resp = client.get("/api/admin/tables")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["tables"]}
    assert names == {"widgets", "lite_master"}
    assert not any(n.startswith("sqlite_") for n in names)


def test_list_tables_row_and_column_counts_are_accurate(seeded_db):
    resp = client.get("/api/admin/tables")
    tables = {t["name"]: t for t in resp.json()["tables"]}
    assert tables["widgets"]["row_count"] == 120
    assert tables["widgets"]["column_count"] == 3
    assert tables["lite_master"]["row_count"] == 0
    assert tables["lite_master"]["column_count"] == 2


def test_get_table_rows_paginates_correctly(seeded_db):
    page1 = client.get("/api/admin/tables/widgets", params={"page": 1, "page_size": 50}).json()
    assert len(page1["rows"]) == 50
    assert page1["total_pages"] == 3
    assert page1["total_rows"] == 120

    page3 = client.get("/api/admin/tables/widgets", params={"page": 3, "page_size": 50}).json()
    assert len(page3["rows"]) == 20


def test_get_table_rows_unknown_table_returns_404_not_500(seeded_db):
    resp = client.get("/api/admin/tables/does_not_exist")
    assert resp.status_code == 404


def test_get_table_rows_rejects_sql_injection_in_table_name(seeded_db):
    resp = client.get('/api/admin/tables/lite_master"; DROP TABLE lite_master; --')
    assert resp.status_code == 404

    still_there = client.get("/api/admin/tables")
    names = {t["name"] for t in still_there.json()["tables"]}
    assert "lite_master" in names


def test_connection_is_read_only(seeded_db):
    from src.api.routers.admin import _readonly_connection

    conn = _readonly_connection()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO widgets (name, qty) VALUES ('x', 1)")
    conn.close()
