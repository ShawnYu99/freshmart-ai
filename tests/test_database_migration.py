import sqlite3

import pytest

from scripts.generate_data import generate_data
from src.database import COMPLAINT_SCHEMA, SCHEMA_VERSION, connection, initialize_database
from src.policy_search import index_policies, retrieve_passages


@pytest.fixture
def legacy_db(tmp_path):
    path = tmp_path / "legacy.db"
    generate_data(path)
    with connection(path) as conn:
        legacy = COMPLAINT_SCHEMA.replace("'Open', ", "").replace(
            "IF NOT EXISTS complaints", "legacy_complaints"
        )
        conn.execute(legacy)
        conn.execute("INSERT INTO legacy_complaints SELECT * FROM complaints")
        conn.execute("DROP TABLE complaints")
        conn.execute("ALTER TABLE legacy_complaints RENAME TO complaints")
        conn.execute("PRAGMA user_version = 1")
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('keep')")
    index_policies(path)
    return path


def test_migration_preserves_all_rows_index_and_unrelated_data(legacy_db):
    with connection(legacy_db) as conn:
        original = {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("stores", "products", "sales", "quality_inspections", "complaints", "policy_passages_fts")
        }
        assert "'Open'" not in conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='complaints'"
        ).fetchone()[0]
    # Retrieval stays compatible even before the application migrates operational storage.
    assert retrieve_passages("refund", legacy_db)
    initialize_database(legacy_db)
    initialize_database(legacy_db)
    with connection(legacy_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        for table, rows in original.items():
            assert [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")] == rows
        assert conn.execute("SELECT value FROM unrelated").fetchone()[0] == "keep"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='complaint_filters'").fetchone()
        conn.execute("UPDATE complaints SET Status='Open' WHERE CaseID='FM-DEMO-0001'")
    assert retrieve_passages("refund", legacy_db)


def test_failed_migration_is_atomic(legacy_db):
    with connection(legacy_db) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE complaints SET IssueDescription='bad' WHERE CaseID='FM-DEMO-0001'")
    with pytest.raises(sqlite3.IntegrityError):
        initialize_database(legacy_db)
    with connection(legacy_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] == 25
        assert not conn.execute(
            "SELECT name FROM sqlite_master WHERE name='complaints_v2'"
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM policy_passages_fts").fetchone()[0] == 17
