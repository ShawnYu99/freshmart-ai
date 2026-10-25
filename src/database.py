"""SQLite storage for synthetic retail data; no model-driven database writes."""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ID = 0x464D4154
SCHEMA_VERSION = 2

COMPLAINT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS complaints (
        CaseID TEXT PRIMARY KEY NOT NULL,
        StoreID TEXT NOT NULL REFERENCES stores(StoreID),
        ProductID TEXT NOT NULL REFERENCES products(ProductID),
        PurchaseDate TEXT NOT NULL CHECK (
            length(PurchaseDate) = 10 AND date(PurchaseDate, '+0 days') IS NOT NULL
            AND date(PurchaseDate, '+0 days') = PurchaseDate
        ),
        IssueDescription TEXT NOT NULL CHECK (
            length(trim(IssueDescription)) BETWEEN 10 AND 2000
        ),
        Status TEXT NOT NULL DEFAULT 'Submitted' CHECK (
            Status IN ('Open', 'Submitted', 'Under review', 'Closed')
        ),
        CreatedAt TEXT NOT NULL CHECK (
            length(CreatedAt) = 20
            AND strftime('%Y-%m-%dT%H:%M:%SZ', CreatedAt, '+0 seconds') IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%SZ', CreatedAt, '+0 seconds') = CreatedAt
            AND PurchaseDate <= substr(CreatedAt, 1, 10)
        ),
        ProofOfPurchase INTEGER CHECK (ProofOfPurchase IN (0, 1)),
        SubmissionToken TEXT UNIQUE CHECK (
            SubmissionToken IS NULL OR length(trim(SubmissionToken)) > 0
        )
    )
"""

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS stores (
        StoreID TEXT PRIMARY KEY NOT NULL,
        StoreName TEXT NOT NULL UNIQUE,
        State TEXT NOT NULL CHECK (State IN ('NSW', 'VIC', 'QLD'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS products (
        ProductID TEXT PRIMARY KEY NOT NULL,
        ProductName TEXT NOT NULL UNIQUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sales (
        TransactionID TEXT PRIMARY KEY NOT NULL,
        Date TEXT NOT NULL CHECK (
            length(Date) = 10 AND date(Date, '+0 days') IS NOT NULL
            AND date(Date, '+0 days') = Date
        ),
        StoreID TEXT NOT NULL REFERENCES stores(StoreID),
        ProductID TEXT NOT NULL REFERENCES products(ProductID),
        Quantity INTEGER NOT NULL CHECK (
            typeof(Quantity) = 'integer' AND Quantity > 0
        ),
        UnitPriceCents INTEGER NOT NULL CHECK (
            typeof(UnitPriceCents) = 'integer' AND UnitPriceCents > 0
        ),
        UnitPriceAUD REAL GENERATED ALWAYS AS (UnitPriceCents / 100.0) VIRTUAL,
        NetRevenueAUD REAL GENERATED ALWAYS AS
            (Quantity * UnitPriceCents / 100.0) VIRTUAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quality_inspections (
        InspectionID TEXT PRIMARY KEY NOT NULL,
        Date TEXT NOT NULL CHECK (
            length(Date) = 10 AND date(Date, '+0 days') IS NOT NULL
            AND date(Date, '+0 days') = Date
        ),
        StoreID TEXT NOT NULL REFERENCES stores(StoreID),
        ProductID TEXT NOT NULL REFERENCES products(ProductID),
        InspectedKg REAL NOT NULL CHECK (
            typeof(InspectedKg) IN ('integer', 'real') AND InspectedKg > 0
        ),
        PassedKg REAL NOT NULL CHECK (
            typeof(PassedKg) IN ('integer', 'real')
            AND PassedKg >= 0 AND PassedKg <= InspectedKg
        ),
        QualityScore REAL NOT NULL CHECK (
            typeof(QualityScore) IN ('integer', 'real')
            AND QualityScore BETWEEN 0 AND 100
        )
    )
    """,
    COMPLAINT_SCHEMA,
    "CREATE INDEX IF NOT EXISTS sales_filters ON sales(StoreID, ProductID, Date)",
    """CREATE INDEX IF NOT EXISTS inspection_filters
       ON quality_inspections(StoreID, ProductID, Date)""",
    """CREATE INDEX IF NOT EXISTS complaint_filters
       ON complaints(StoreID, ProductID, CreatedAt)""",
)


def database_path(path: str | Path | None = None) -> Path:
    """Resolve paths independently of the caller's working directory."""
    value = path if path is not None else os.getenv(
        "FRESHMART_DB_PATH", "data/freshmart.db"
    )
    resolved = Path(value).expanduser()
    return resolved if resolved.is_absolute() else ROOT / resolved


@contextmanager
def connection(path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on failure, and always close."""
    target = database_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database(path: str | Path | None = None) -> None:
    """Initialize FreshMart storage, refusing unrelated or newer databases."""
    with connection(path) as conn:
        application_id = conn.execute("PRAGMA application_id").fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        objects = conn.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if application_id not in (0, APPLICATION_ID) or (
            application_id == 0 and objects
        ):
            raise ValueError("Refusing to modify a database not owned by FreshMart.")
        if version not in (0, 1, SCHEMA_VERSION):
            raise ValueError(f"Unsupported database schema version: {version}.")
        conn.execute("BEGIN IMMEDIATE")
        if version == 1:
            conn.execute(COMPLAINT_SCHEMA.replace(
                "IF NOT EXISTS complaints", "complaints_v2"
            ))
            conn.execute("""
                INSERT INTO complaints_v2 (
                    CaseID, StoreID, ProductID, PurchaseDate, IssueDescription,
                    Status, CreatedAt, ProofOfPurchase, SubmissionToken
                )
                SELECT CaseID, StoreID, ProductID, PurchaseDate, IssueDescription,
                       Status, CreatedAt, ProofOfPurchase, SubmissionToken
                FROM complaints
            """)
            conn.execute("DROP TABLE complaints")
            conn.execute("ALTER TABLE complaints_v2 RENAME TO complaints")
        for statement in SCHEMA:
            conn.execute(statement)
        conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
