from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import logging
import sqlite3

import pytest

from scripts.generate_data import generate_data
from src.complaints import (
    ComplaintDraft,
    ComplaintError,
    ComplaintValidationError,
    complaint_options,
    create_complaint,
    validate_complaint,
)
from src.database import connection


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "freshmart.db"
    generate_data(path)
    return path


@pytest.fixture
def draft():
    return ComplaintDraft(
        "NSW-01", "PRD-01", datetime.now(timezone.utc).date() - timedelta(days=1),
        "Synthetic complaint: the strawberries were mouldy.", True,
    )


def count(db):
    with connection(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]


def test_options_come_from_database(db):
    with connection(db) as conn:
        conn.execute("UPDATE stores SET StoreName = ? WHERE StoreID = ?", ("Demo store", "NSW-01"))
    stores, products = complaint_options(db)
    assert len(stores) == 6 and len(products) == 4
    assert next(s for s in stores if s.store_id == "NSW-01").name == "Demo store"
    assert next(p for p in products if p.product_id == "PRD-01").name == "Strawberries"


@pytest.mark.parametrize("proof", [None, True, False])
def test_confirmed_case_is_committed_open_with_actual_receipt(db, draft, proof):
    draft = replace(draft, proof_of_purchase=proof)
    receipt = create_complaint(draft, "confirmed-request", confirmed=True, path=db)
    assert receipt.case_id.startswith("FM-") and len(receipt.case_id) == 35
    assert receipt.status == "Open"
    with connection(db) as conn:
        row = conn.execute(
            "SELECT * FROM complaints WHERE CaseID = ?", (receipt.case_id,)
        ).fetchone()
        assert row is not None
        assert (
            row["StoreID"], row["ProductID"], row["PurchaseDate"],
            row["IssueDescription"], row["ProofOfPurchase"], row["Status"],
        ) == (
            draft.store_id, draft.product_id, draft.purchase_date.isoformat(),
            draft.issue_description, proof, "Open",
        )
        assert row["CreatedAt"] == receipt.created_at
        assert row["SubmissionToken"] == "confirmed-request"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert count(db) == 26


@pytest.mark.parametrize("confirmed", [False, None, 1, "yes"])
def test_explicit_boolean_confirmation_is_required(db, draft, confirmed):
    with pytest.raises(ComplaintValidationError, match="explicitly confirm"):
        create_complaint(draft, "request", confirmed=confirmed, path=db)
    assert count(db) == 25


@pytest.mark.parametrize(("field", "value"), [
    ("store_id", ""), ("store_id", None), ("store_id", "unknown"),
    ("store_id", "NSW-01'; DROP TABLE stores; --"),
    ("product_id", ""), ("product_id", None), ("product_id", "unknown"),
    ("purchase_date", "2026-02-30"), ("purchase_date", None),
    ("purchase_date", datetime(2026, 1, 1)),
    ("purchase_date", datetime.now(timezone.utc).date() + timedelta(days=1)),
    ("issue_description", ""), ("issue_description", "   "),
    ("issue_description", "x" * 9), ("issue_description", "x" * 2001),
    ("issue_description", None),
    ("proof_of_purchase", 1), ("proof_of_purchase", "Yes"),
])
def test_invalid_input_never_writes(db, draft, field, value):
    invalid = replace(draft, **{field: value})
    with pytest.raises(ComplaintValidationError):
        validate_complaint(invalid, db)
    with pytest.raises(ComplaintValidationError):
        create_complaint(invalid, "request", confirmed=True, path=db)
    assert count(db) == 25


def test_review_validates_but_does_not_create_case(db, draft):
    normalized = validate_complaint(
        replace(draft, issue_description=f"  {draft.issue_description}  "), db
    )
    assert normalized == draft
    assert count(db) == 25


@pytest.mark.parametrize("description", ["x" * 10, "x" * 2000, "Mouldy berries'); DROP TABLE sales; --"])
def test_description_boundaries_and_sql_values_are_preserved(db, draft, description):
    receipt = create_complaint(
        replace(draft, issue_description=description), "request", confirmed=True, path=db
    )
    with connection(db) as conn:
        assert conn.execute(
            "SELECT IssueDescription FROM complaints WHERE CaseID = ?", (receipt.case_id,)
        ).fetchone()[0] == description
        assert conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 2000


@pytest.mark.parametrize("token", ["", "  ", None, 123, "x" * 201])
def test_invalid_submission_token(db, draft, token):
    with pytest.raises(ComplaintValidationError, match="identifier"):
        create_complaint(draft, token, confirmed=True, path=db)
    assert count(db) == 25


def test_retries_return_same_case_but_distinct_requests_get_distinct_cases(db, draft):
    first = create_complaint(draft, "stable-request", confirmed=True, path=db)
    assert create_complaint(draft, "stable-request", confirmed=True, path=db) == first
    second = create_complaint(draft, "new-request", confirmed=True, path=db)
    assert second.case_id != first.case_id
    assert count(db) == 27


def test_token_cannot_be_reused_with_different_details(db, draft):
    first = create_complaint(draft, "stable-request", confirmed=True, path=db)
    with pytest.raises(ComplaintValidationError, match="different details"):
        create_complaint(
            replace(draft, product_id="PRD-02"), "stable-request", confirmed=True, path=db
        )
    assert create_complaint(draft, "stable-request", confirmed=True, path=db) == first
    assert count(db) == 26


def test_concurrent_retries_create_exactly_one_case(db, draft):
    def submit(_):
        return create_complaint(draft, "concurrent-request", confirmed=True, path=db)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(submit, range(2)))
    assert receipts[0] == receipts[1]
    assert count(db) == 26


def test_database_failure_rolls_back_and_same_request_can_be_retried(db, draft, caplog):
    with connection(db) as conn:
        conn.execute("""
            CREATE TRIGGER reject_complaint AFTER INSERT ON complaints
            BEGIN SELECT RAISE(ABORT, 'PRIVATE STORAGE ERROR'); END
        """)
    with caplog.at_level(logging.INFO), pytest.raises(ComplaintError, match="retry"):
        create_complaint(draft, "retry-request", confirmed=True, path=db)
    assert "PRIVATE STORAGE ERROR" not in caplog.text
    assert draft.issue_description not in caplog.text
    assert count(db) == 25
    with connection(db) as conn:
        conn.execute("DROP TRIGGER reject_complaint")
    receipt = create_complaint(draft, "retry-request", confirmed=True, path=db)
    assert receipt.status == "Open" and count(db) == 26


def test_missing_database_is_not_created(tmp_path, draft):
    missing = tmp_path / "missing.db"
    with pytest.raises(ComplaintError, match="missing"):
        create_complaint(draft, "request", confirmed=True, path=missing)
    assert not missing.exists()


def test_unrelated_database_is_not_modified(tmp_path, draft):
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE important (value TEXT)")
        conn.execute("INSERT INTO important VALUES ('keep')")
    with pytest.raises(ComplaintError):
        create_complaint(draft, "request", confirmed=True, path=path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM important").fetchone()[0] == "keep"
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='complaints'").fetchone()[0] == 0


def test_empty_catalog_has_actionable_error(db):
    with connection(db) as conn:
        for table in ("complaints", "quality_inspections", "sales", "stores", "products"):
            conn.execute(f"DELETE FROM {table}")
    with pytest.raises(ComplaintError, match="stores or products are missing"):
        complaint_options(db)
