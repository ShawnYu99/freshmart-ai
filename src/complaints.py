"""Validated, explicitly confirmed complaint writes; independent of model inference."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

from src.database import APPLICATION_ID, SCHEMA_VERSION, connection, database_path

LOGGER = logging.getLogger(__name__)


class ComplaintError(RuntimeError):
    """A safe, customer-facing complaint storage error."""


class ComplaintValidationError(ComplaintError):
    """Complaint input or confirmation needs correction."""


@dataclass(frozen=True)
class Store:
    store_id: str
    name: str
    state: str


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str


@dataclass(frozen=True)
class ComplaintDraft:
    store_id: str
    product_id: str
    purchase_date: date
    issue_description: str
    proof_of_purchase: bool | None = None


@dataclass(frozen=True)
class ComplaintReceipt:
    case_id: str
    status: str
    created_at: str


@contextmanager
def complaint_connection(path: str | Path | None) -> Iterator[sqlite3.Connection]:
    try:
        target = database_path(path)
        if not target.is_file():
            raise ComplaintError("Demo database missing. Run python -m scripts.generate_data first.")
        with connection(target) as conn:
            if (
                conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
            ):
                raise ComplaintError("The demo database needs initialization. Restart the application.")
            yield conn
    except (sqlite3.Error, OSError) as exc:
        LOGGER.error("Complaint storage failed: type=%s", type(exc).__name__)
        raise ComplaintError(
            "Unable to access complaint storage. Check database access and retry; "
            "the same confirmed request will not create a duplicate."
        ) from None


def complaint_options(
    path: str | Path | None = None,
) -> tuple[list[Store], list[Product]]:
    with complaint_connection(path) as conn:
        stores = [
            Store(*row) for row in conn.execute(
                "SELECT StoreID, StoreName, State FROM stores ORDER BY State, StoreName"
            )
        ]
        products = [
            Product(*row) for row in conn.execute(
                "SELECT ProductID, ProductName FROM products ORDER BY ProductName"
            )
        ]
    if not stores or not products:
        raise ComplaintError("Demo stores or products are missing. Prepare the synthetic database first.")
    return stores, products


def validate_fields(draft: ComplaintDraft) -> ComplaintDraft:
    if not isinstance(draft.store_id, str) or not draft.store_id.strip():
        raise ComplaintValidationError("Select a store.")
    if not isinstance(draft.product_id, str) or not draft.product_id.strip():
        raise ComplaintValidationError("Select a product.")
    if type(draft.purchase_date) is not date:
        raise ComplaintValidationError("Enter a valid purchase date.")
    if draft.purchase_date > datetime.now(timezone.utc).date():
        raise ComplaintValidationError("Purchase date cannot be in the future.")
    if not isinstance(draft.issue_description, str):
        raise ComplaintValidationError("Enter an issue description.")
    description = draft.issue_description.strip()
    if not 10 <= len(description) <= 2000:
        raise ComplaintValidationError("Issue description must contain 10 to 2,000 characters.")
    if draft.proof_of_purchase is not None and type(draft.proof_of_purchase) is not bool:
        raise ComplaintValidationError("Proof of purchase must be Yes, No, or Not specified.")
    return replace(draft, issue_description=description)


def validate_references(conn: sqlite3.Connection, draft: ComplaintDraft) -> None:
    if not conn.execute(
        "SELECT 1 FROM stores WHERE StoreID = ?", (draft.store_id,)
    ).fetchone():
        raise ComplaintValidationError("Select an existing FreshMart store.")
    if not conn.execute(
        "SELECT 1 FROM products WHERE ProductID = ?", (draft.product_id,)
    ).fetchone():
        raise ComplaintValidationError("Select an existing FreshMart product.")


def validate_complaint(
    draft: ComplaintDraft, path: str | Path | None = None,
) -> ComplaintDraft:
    draft = validate_fields(draft)
    with complaint_connection(path) as conn:
        validate_references(conn, draft)
    return draft


def create_complaint(
    draft: ComplaintDraft,
    submission_token: str,
    *,
    confirmed: bool = False,
    path: str | Path | None = None,
) -> ComplaintReceipt:
    """Return the same real case for retries of the same confirmed submission."""
    if confirmed is not True:
        raise ComplaintValidationError("Review the details and explicitly confirm before submission.")
    draft = validate_fields(draft)
    if not isinstance(submission_token, str) or not 1 <= len(submission_token.strip()) <= 200:
        raise ComplaintValidationError("Submission identifier is invalid. Review the complaint again.")
    values = (
        draft.store_id, draft.product_id, draft.purchase_date.isoformat(),
        draft.issue_description, draft.proof_of_purchase,
    )
    with complaint_connection(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        validate_references(conn, draft)
        existing = conn.execute(
            """SELECT CaseID, Status, CreatedAt, StoreID, ProductID, PurchaseDate,
                      IssueDescription, ProofOfPurchase
               FROM complaints WHERE SubmissionToken = ?""",
            (submission_token,),
        ).fetchone()
        if existing:
            if tuple(existing)[3:] != values:
                raise ComplaintValidationError(
                    "This submission identifier belongs to different details. Review a new complaint."
                )
            receipt = ComplaintReceipt(*tuple(existing)[:3])
            LOGGER.info("Complaint retry resolved to existing case: case_id=%s", receipt.case_id)
        else:
            receipt = ComplaintReceipt(
                f"FM-{uuid4().hex.upper()}", "Open",
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            conn.execute(
                """INSERT INTO complaints (
                    CaseID, StoreID, ProductID, PurchaseDate, IssueDescription,
                    Status, CreatedAt, ProofOfPurchase, SubmissionToken
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    receipt.case_id, *values[:4], receipt.status, receipt.created_at,
                    values[4], submission_token,
                ),
            )
    LOGGER.info("Complaint submission completed: case_id=%s status=%s", receipt.case_id, receipt.status)
    return receipt
