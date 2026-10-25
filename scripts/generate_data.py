"""Generate fictional data with a fixed seed and an explicit date anchor."""

import argparse
from datetime import date, datetime, time, timedelta, timezone
import logging
from pathlib import Path
import random
import sqlite3

from src.database import connection, database_path, initialize_database

LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 42
STORES = (
    ("NSW-01", "Chatswood", "NSW"),
    ("NSW-02", "Parramatta", "NSW"),
    ("VIC-01", "Melbourne CBD", "VIC"),
    ("VIC-02", "Richmond", "VIC"),
    ("QLD-01", "Brisbane CBD", "QLD"),
    ("QLD-02", "South Brisbane", "QLD"),
)
PRODUCTS = (
    ("PRD-01", "Strawberries"),
    ("PRD-02", "Tomatoes"),
    ("PRD-03", "Broccoli"),
    ("PRD-04", "Lettuce"),
)
PRICES_CENTS = {"PRD-01": 650, "PRD-02": 480, "PRD-03": 390, "PRD-04": 320}
CONCERN_PAIR = ("NSW-01", "PRD-01")


def generate_data(
    path: str | Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    end_date: date | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """Populate all tables atomically; replacing demo records is explicit."""
    anchor = end_date if end_date is not None else date.today()
    rng = random.Random(seed)
    pairs = [(store[0], product[0]) for store in STORES for product in PRODUCTS]
    start = anchor - timedelta(days=89)
    sales = []
    for index in range(2000):
        store_id, product_id = pairs[index % len(pairs)] if index < 24 else rng.choice(pairs)
        sale_date = start + timedelta(days=rng.randrange(90))
        sales.append((
            f"TXN-{index + 1:05d}", sale_date.isoformat(), store_id, product_id,
            rng.randint(1, 6), PRICES_CENTS[product_id] + rng.choice((-20, 0, 20)),
        ))

    inspections = []
    inspection_pairs = pairs * 6 + [CONCERN_PAIR] * 6
    for index, (store_id, product_id) in enumerate(inspection_pairs):
        inspection_date = start + timedelta(days=rng.randrange(90))
        inspected = round(rng.uniform(10, 45), 2)
        pass_ratio = rng.uniform(0.55, 0.72) if (
            store_id, product_id
        ) == CONCERN_PAIR else rng.uniform(0.91, 0.99)
        inspections.append((
            f"INSP-{index + 1:04d}", inspection_date.isoformat(), store_id,
            product_id, inspected, round(inspected * pass_ratio, 2),
            round(pass_ratio * 100, 1),
        ))

    complaints = []
    other_pairs = [pair for pair in pairs if pair != CONCERN_PAIR]
    for index in range(25):
        store_id, product_id = CONCERN_PAIR if index < 12 else rng.choice(other_pairs)
        created_date = anchor - timedelta(days=rng.randrange(30 if index < 12 else 88))
        purchase_date = max(start, created_date - timedelta(days=rng.randint(0, 2)))
        created_at = datetime.combine(
            created_date, time(2, 0), tzinfo=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        issue = (
            "Synthetic complaint: strawberries were mouldy shortly after purchase."
            if (store_id, product_id) == CONCERN_PAIR
            else "Synthetic complaint: produce appeared bruised or wilted after purchase."
        )
        complaints.append((
            f"FM-DEMO-{index + 1:04d}", store_id, product_id,
            purchase_date.isoformat(), issue,
            "Under review" if index % 3 == 0 else "Submitted",
            created_at, index % 2, f"demo-submission-{index + 1:04d}",
        ))

    initialize_database(path)
    with connection(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        populated = any(
            conn.execute(f"SELECT EXISTS(SELECT 1 FROM {table})").fetchone()[0]
            for table in (
                "stores", "products", "sales", "quality_inspections", "complaints"
            )
        )
        if populated and not reset:
            raise ValueError(
                "Database already contains data. Use python -m scripts.reset_demo "
                "--yes to deliberately replace demo records."
            )
        if reset:
            for table in (
                "complaints", "quality_inspections", "sales", "products", "stores"
            ):
                conn.execute(f"DELETE FROM {table}")
        conn.executemany("INSERT INTO stores VALUES (?, ?, ?)", STORES)
        conn.executemany("INSERT INTO products VALUES (?, ?)", PRODUCTS)
        conn.executemany(
            """INSERT INTO sales
               (TransactionID, Date, StoreID, ProductID, Quantity, UnitPriceCents)
               VALUES (?, ?, ?, ?, ?, ?)""", sales,
        )
        conn.executemany(
            "INSERT INTO quality_inspections VALUES (?, ?, ?, ?, ?, ?, ?)",
            inspections,
        )
        conn.executemany(
            """INSERT INTO complaints (
                CaseID, StoreID, ProductID, PurchaseDate, IssueDescription,
                Status, CreatedAt, ProofOfPurchase, SubmissionToken
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", complaints,
        )
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError("Generated data failed foreign-key validation.")
    counts = {
        "stores": len(STORES), "products": len(PRODUCTS), "sales": len(sales),
        "quality_inspections": len(inspections), "complaints": len(complaints),
    }
    LOGGER.info(
        "Demo data committed: database=%s seed=%s end_date=%s counts=%s",
        database_path(path), seed, anchor, counts,
    )
    return counts


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--db", type=Path, help="SQLite file (default: data/freshmart.db)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--end-date", type=date.fromisoformat, default=date.today(),
        help="Last date in the 90-day dataset, YYYY-MM-DD (default: today)",
    )
    return parser


def main() -> None:
    parser = argument_parser(__doc__)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        generate_data(args.db, seed=args.seed, end_date=args.end_date)
    except (ValueError, sqlite3.Error, OSError) as exc:
        LOGGER.error("Data generation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
