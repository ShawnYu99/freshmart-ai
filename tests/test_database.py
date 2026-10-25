from datetime import date
import sqlite3
import subprocess
import sys

import pytest

from scripts.generate_data import generate_data
from src.database import APPLICATION_ID, ROOT, connection, database_path, initialize_database

ANCHOR = date(2026, 10, 24)
COUNTS = {
    "stores": 6, "products": 4, "sales": 2000,
    "quality_inspections": 150, "complaints": 25,
}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "freshmart.db"
    generate_data(path, end_date=ANCHOR)
    return path


def snapshot(path):
    with connection(path) as conn:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in COUNTS
        }


def test_counts_relationships_and_integrity(db):
    with connection(db) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        for table, expected in COUNTS.items():
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
        assert [tuple(row) for row in conn.execute(
            "SELECT State, COUNT(*) FROM stores GROUP BY State ORDER BY State"
        )] == [("NSW", 2), ("QLD", 2), ("VIC", 2)]
        assert {row[0] for row in conn.execute("SELECT ProductName FROM products")} == {
            "Strawberries", "Tomatoes", "Broccoli", "Lettuce"
        }
        assert tuple(conn.execute(
            "SELECT StoreName, State FROM stores WHERE StoreID = ?", ("NSW-01",)
        ).fetchone()) == ("Chatswood", "NSW")
        assert conn.execute(
            "SELECT ProductName FROM products WHERE ProductID = ?", ("PRD-01",)
        ).fetchone()[0] == "Strawberries"


def test_same_seed_and_date_reproduce_all_records(db, tmp_path):
    other = tmp_path / "other.db"
    generate_data(other, seed=42, end_date=ANCHOR)
    assert snapshot(db) == snapshot(other)
    generate_data(other, seed=43, end_date=ANCHOR, reset=True)
    assert snapshot(db)["sales"] != snapshot(other)["sales"]


def test_date_window(db):
    with connection(db) as conn:
        for table, column in (
            ("sales", "Date"), ("quality_inspections", "Date"),
            ("complaints", "PurchaseDate"),
        ):
            dates = [date.fromisoformat(row[0]) for row in conn.execute(
                f"SELECT {column} FROM {table}"
            )]
            assert all(0 <= (ANCHOR - value).days < 90 for value in dates)
        assert conn.execute("SELECT MIN(Date), MAX(Date) FROM sales").fetchone()[:] == (
            "2026-07-27", "2026-10-24"
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM complaints WHERE PurchaseDate > substr(CreatedAt, 1, 10)"
        ).fetchone()[0] == 0


def test_revenue_is_calculated_from_quantity_and_integer_cents(db):
    with connection(db) as conn:
        for row in conn.execute(
            "SELECT Quantity, UnitPriceCents, UnitPriceAUD, NetRevenueAUD FROM sales"
        ):
            assert row["UnitPriceAUD"] == row["UnitPriceCents"] / 100
            assert row["NetRevenueAUD"] == row["Quantity"] * row["UnitPriceCents"] / 100
        with pytest.raises(sqlite3.OperationalError, match="generated column"):
            conn.execute("UPDATE sales SET NetRevenueAUD = 0")


def test_quality_issue_uses_weighted_inspections_and_complaints(db):
    with connection(db) as conn:
        rows = conn.execute(
            """SELECT StoreID, SUM(PassedKg) / SUM(InspectedKg) AS pass_rate
               FROM quality_inspections WHERE ProductID = ?
               GROUP BY StoreID""", ("PRD-01",)
        ).fetchall()
        rates = {row["StoreID"]: row["pass_rate"] for row in rows}
        assert rates["NSW-01"] < 0.75
        assert all(rate > 0.90 for store, rate in rates.items() if store != "NSW-01")
        counts = dict(conn.execute(
            """SELECT StoreID, COUNT(*) FROM complaints
               WHERE ProductID = ? GROUP BY StoreID""", ("PRD-01",)
        ).fetchall())
        assert counts["NSW-01"] == 12
        assert all(count < 12 for store, count in counts.items() if store != "NSW-01")
        assert conn.execute(
            """SELECT COUNT(*) FROM (
                SELECT StoreID, ProductID FROM quality_inspections
                GROUP BY StoreID, ProductID HAVING COUNT(*) >= 6
            )"""
        ).fetchone()[0] == 24


@pytest.mark.parametrize(("statement", "parameters"), [
    ("INSERT INTO stores VALUES (?, ?, ?)", ("BAD", "Unknown", "WA")),
    ("UPDATE sales SET StoreID = ? WHERE TransactionID = ?", ("missing", "TXN-00001")),
    ("UPDATE complaints SET ProductID = ? WHERE CaseID = ?", ("missing", "FM-DEMO-0001")),
    ("UPDATE sales SET Quantity = 0", ()),
    ("UPDATE sales SET Quantity = 1.5", ()),
    ("UPDATE sales SET UnitPriceCents = -1", ()),
    ("UPDATE sales SET Date = ?", ("2026-02-30",)),
    ("UPDATE sales SET Date = ?", ("not-a-date",)),
    ("UPDATE quality_inspections SET InspectedKg = 0", ()),
    ("UPDATE quality_inspections SET InspectedKg = ?", ("not-a-weight",)),
    ("UPDATE quality_inspections SET PassedKg = InspectedKg + 1", ()),
    ("UPDATE quality_inspections SET PassedKg = -1", ()),
    ("UPDATE quality_inspections SET QualityScore = 101", ()),
    ("UPDATE complaints SET Status = ?", ("Refund approved",)),
    ("UPDATE complaints SET IssueDescription = ?", (" ",)),
    ("UPDATE complaints SET IssueDescription = ?", ("x" * 2001,)),
    ("UPDATE complaints SET ProofOfPurchase = 2", ()),
    ("UPDATE complaints SET CreatedAt = ?", ("not-a-timestamp",)),
    ("UPDATE complaints SET CreatedAt = ?", ("2026-02-30T02:00:00Z",)),
    ("UPDATE complaints SET CreatedAt = ?", ("2026-10-24T24:00:00Z",)),
    ("UPDATE complaints SET PurchaseDate = ?", ("2099-01-01",)),
    ("UPDATE complaints SET SubmissionToken = ?", (" ",)),
])
def test_invalid_data_is_rejected(db, statement, parameters):
    with pytest.raises(sqlite3.IntegrityError):
        with connection(db) as conn:
            conn.execute(statement, parameters)


def test_case_and_submission_token_uniqueness(db):
    with connection(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE complaints SET CaseID = ?", ("FM-DEMO-0001",))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE complaints SET SubmissionToken = ?", ("demo-submission-0001",)
            )
        conn.execute("UPDATE complaints SET SubmissionToken = NULL")
        conn.execute("UPDATE complaints SET ProofOfPurchase = NULL")


def test_legacy_sql_default_is_preserved_across_connections(db):
    """Direct legacy inserts retain Submitted; the complaint service writes Open."""
    with connection(db) as conn:
        conn.execute(
            """INSERT INTO complaints (
                CaseID, StoreID, ProductID, PurchaseDate, IssueDescription,
                CreatedAt, ProofOfPurchase, SubmissionToken
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("FM-TEST-NEW", "NSW-01", "PRD-01", "2026-10-23",
             "Synthetic report: strawberries were mouldy.",
             "2026-10-24T02:00:00Z", 1, "test-new-submission"),
        )
    with connection(db) as conn:
        row = conn.execute(
            "SELECT CaseID, Status FROM complaints WHERE SubmissionToken = ?",
            ("test-new-submission",),
        ).fetchone()
        assert tuple(row) == ("FM-TEST-NEW", "Submitted")


def test_connection_rolls_back_and_closes(db):
    with pytest.raises(RuntimeError, match="abort"):
        with connection(db) as conn:
            conn.execute("DELETE FROM complaints")
            raise RuntimeError("abort")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with connection(db) as reopened:
        assert reopened.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] == 25


def test_generation_refuses_to_overwrite_existing_data(db):
    original = snapshot(db)
    with pytest.raises(ValueError, match="already contains data"):
        generate_data(db, end_date=ANCHOR)
    assert snapshot(db) == original


def test_reset_reproduces_original_and_preserves_unrelated_content(db, tmp_path):
    original = snapshot(db)
    note = tmp_path / "keep.txt"
    note.write_text("Unrelated file.", encoding="utf-8")
    with connection(db) as conn:
        conn.execute("DELETE FROM complaints")
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES (?)", ("keep",))
    generate_data(db, end_date=ANCHOR, reset=True)
    assert snapshot(db) == original
    assert note.read_text(encoding="utf-8") == "Unrelated file."
    with connection(db) as conn:
        assert conn.execute("SELECT value FROM unrelated").fetchone()[0] == "keep"


def test_failed_reset_rolls_back_deleted_records(db):
    original = snapshot(db)
    with connection(db) as conn:
        conn.execute(
            """CREATE TRIGGER fail_demo_insert BEFORE INSERT ON sales
               BEGIN SELECT RAISE(ABORT, 'simulated failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        generate_data(db, end_date=ANCHOR, reset=True)
    assert snapshot(db) == original


def test_unrelated_database_is_not_adopted_or_reset(tmp_path):
    path = tmp_path / "unrelated.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE important (value TEXT)")
        conn.execute("INSERT INTO important VALUES ('preserve')")
    with pytest.raises(ValueError, match="not owned by FreshMart"):
        generate_data(path, end_date=ANCHOR, reset=True)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM important").fetchone()[0] == "preserve"
        assert conn.execute("PRAGMA application_id").fetchone()[0] == 0


def test_initialization_is_idempotent_and_rejects_newer_schema(db):
    original = snapshot(db)
    initialize_database(db)
    assert snapshot(db) == original
    with connection(db) as conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(ValueError, match="Unsupported database schema version"):
        initialize_database(db)


def test_database_path_configuration(monkeypatch):
    monkeypatch.setenv("FRESHMART_DB_PATH", "data/configured.db")
    assert database_path() == ROOT / "data/configured.db"
    assert database_path("data/explicit.db") == ROOT / "data/explicit.db"


def test_reset_cli_requires_confirmation(db):
    original = snapshot(db)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reset_demo", "--db", str(db)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "requires --yes" in result.stderr
    assert snapshot(db) == original


def test_generate_and_reset_cli(tmp_path):
    path = tmp_path / "cli.db"
    common = ["--db", str(path), "--seed", "42", "--end-date", ANCHOR.isoformat()]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.generate_data", *common],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    original = snapshot(path)
    refused = subprocess.run(
        [sys.executable, "-m", "scripts.generate_data", *common],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert refused.returncode == 1
    assert "already contains data" in refused.stderr
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reset_demo", "--yes", *common],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert snapshot(path) == original


def test_policy_documents_have_unique_source_metadata():
    paths = sorted((ROOT / "data/knowledge").glob("*.md"))
    assert len(paths) == 5
    identifiers = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# ")
        assert "Version: 1.0" in text
        assert "fictional" in text.lower()
        identifier = next(line for line in text.splitlines() if line.startswith("Source ID: "))
        identifiers.add(identifier)
    assert len(identifiers) == 5
