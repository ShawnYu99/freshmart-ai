from datetime import date, datetime, timedelta, timezone
import logging
import sqlite3

import pandas as pd
import pytest

from src.analytics import (
    AnalyticsError,
    DashboardFilters,
    comparable_growth,
    load_dashboard,
)
from src.complaints import ComplaintDraft, create_complaint
from src.database import connection, initialize_database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "analytics.db"
    initialize_database(path)
    with connection(path) as conn:
        conn.executemany("INSERT INTO stores VALUES (?, ?, ?)", [
            ("NSW-01", "Chatswood", "NSW"), ("NSW-02", "Parramatta", "NSW"),
            ("VIC-01", "Melbourne CBD", "VIC"),
        ])
        conn.executemany("INSERT INTO products VALUES (?, ?)", [
            ("PRD-01", "Strawberries"), ("PRD-02", "Tomatoes"),
        ])
        sales = [
            ("2026-01-01", "NSW-01", "PRD-01", 1, 100),
            ("2026-02-01", "NSW-01", "PRD-01", 10, 100),
            ("2026-02-20", "NSW-02", "PRD-01", 20, 100),
            ("2026-02-25", "NSW-01", "PRD-02", 10, 200),
            ("2026-02-28", "VIC-01", "PRD-01", 5, 100),
            ("2026-03-01", "NSW-01", "PRD-01", 20, 100),
            ("2026-03-20", "NSW-02", "PRD-01", 10, 100),
            ("2026-03-25", "NSW-01", "PRD-02", 5, 200),
            ("2026-03-28", "VIC-01", "PRD-01", 10, 100),
            ("2026-03-31", "NSW-01", "PRD-01", 1, 100000),
        ]
        conn.executemany(
            """INSERT INTO sales (TransactionID, Date, StoreID, ProductID, Quantity, UnitPriceCents)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(f"TX-{i}", *values) for i, values in enumerate(sales)],
        )
        conn.executemany(
            """INSERT INTO quality_inspections
               (InspectionID, Date, StoreID, ProductID, InspectedKg, PassedKg, QualityScore)
               VALUES (?, '2026-03-01', ?, ?, ?, ?, 70)""",
            [
                ("Q-1", "NSW-01", "PRD-01", 1, 1),
                ("Q-2", "NSW-01", "PRD-01", 9, 3),
                ("Q-3", "NSW-02", "PRD-01", 20, 19),
                ("Q-4", "VIC-01", "PRD-01", 10, 9),
                ("Q-5", "NSW-01", "PRD-02", 5, 5),
            ],
        )
        conn.executemany(
            """INSERT INTO complaints
               (CaseID, StoreID, ProductID, PurchaseDate, IssueDescription, Status, CreatedAt)
               VALUES (?, ?, ?, '2026-03-01', 'Synthetic quality complaint.', ?,
                       '2026-03-31T12:00:00Z')""",
            [
                ("C-1", "NSW-01", "PRD-01", "Open"),
                ("C-2", "NSW-01", "PRD-01", "Submitted"),
                ("C-3", "NSW-01", "PRD-01", "Closed"),
                ("C-4", "NSW-02", "PRD-01", "Under review"),
                ("C-5", "NSW-01", "PRD-02", "Open"),
                ("C-6", "NSW-01", "PRD-02", "Submitted"),
            ],
        )
    return path


def test_all_kpis_are_calculated_without_fact_table_join_multiplication(db):
    data = load_dashboard(path=db)
    assert data.net_revenue_cents == 110600
    assert data.complaint_count == 6
    assert data.freshness_pass_rate == pytest.approx(37 / 45)
    assert data.revenue_growth.percent == pytest.approx((5000 - 5500) / 5500 * 100)
    assert data.revenue_growth.current_start == date(2026, 3, 1)
    assert data.revenue_growth.current_end == date(2026, 3, 28)
    assert data.revenue_growth.previous_start == date(2026, 2, 1)
    assert data.revenue_growth.previous_end == date(2026, 2, 28)
    assert int(data.revenue_trend["RevenueCents"].sum()) == 110600
    assert data.revenue_trend["RevenueAUD"].sum() == pytest.approx(1106)
    assert len(data.revenue_trend) == 90
    assert data.revenue_trend.loc[
        data.revenue_trend["Date"] == pd.Timestamp("2026-01-02"), "RevenueCents"
    ].iloc[0] == 0


@pytest.mark.parametrize(("filters", "cents", "complaints", "freshness"), [
    (DashboardFilters(state="NSW"), 109100, 6, 28 / 35),
    (DashboardFilters(state="VIC"), 1500, 0, 9 / 10),
    (DashboardFilters(product_id="PRD-02"), 3000, 2, 1),
    (DashboardFilters(store_id="NSW-02"), 3000, 1, 19 / 20),
    (DashboardFilters("NSW", "NSW-01", "PRD-01"), 103100, 3, 4 / 10),
    (DashboardFilters("NSW", "NSW-01", "PRD-02"), 3000, 2, 1),
])
def test_state_store_product_filters_update_kpis_trend_and_complaints(
    db, filters, cents, complaints, freshness,
):
    data = load_dashboard(filters, db)
    assert data.net_revenue_cents == cents
    assert data.complaint_count == complaints
    assert data.freshness_pass_rate == pytest.approx(freshness)
    assert int(data.revenue_trend["RevenueCents"].sum()) == cents
    assert len(data.recent_complaints) == complaints
    if filters.state:
        assert set(data.recent_complaints["State"]) <= {filters.state}
        assert set(data.freshness_by_store["State"]) == {filters.state}
    if filters.store_id:
        with connection(db) as conn:
            name = conn.execute(
                "SELECT StoreName FROM stores WHERE StoreID=?", (filters.store_id,)
            ).fetchone()[0]
        assert set(data.recent_complaints["StoreName"]) <= {name}
    if filters.product_id:
        with connection(db) as conn:
            name = conn.execute(
                "SELECT ProductName FROM products WHERE ProductID=?", (filters.product_id,)
            ).fetchone()[0]
        assert set(data.recent_complaints["ProductName"]) <= {name}


def test_weighted_freshness_and_observations_use_matching_peers(db):
    data = load_dashboard(DashboardFilters("NSW", "NSW-01", "PRD-01"), db)
    assert data.freshness_pass_rate == pytest.approx(0.4)
    assert data.revenue_growth.percent == pytest.approx(100)
    assert set(data.freshness_by_store["StoreID"]) == {"NSW-01", "NSW-02"}
    peer = data.freshness_by_store.set_index("StoreID").loc["NSW-02"]
    assert peer["FreshnessPassRate"] == pytest.approx(0.95)
    observations = " ".join(data.insights)
    assert "40.0%" in observations and "95.0%" in observations
    assert "3 quality complaints" in observations and "1.0 per store" in observations
    assert "do not establish causation" in observations
    with connection(db) as conn:
        conn.execute("UPDATE quality_inspections SET PassedKg=InspectedKg WHERE StoreID='NSW-01'")
        conn.execute("DELETE FROM complaints WHERE StoreID='NSW-01'")
    changed = load_dashboard(DashboardFilters("NSW", "NSW-01", "PRD-01"), db)
    assert changed.freshness_pass_rate == 1
    assert "No lower freshness or elevated" in " ".join(changed.insights)
    assert "40.0%" not in " ".join(changed.insights)


@pytest.mark.parametrize("filters", [
    DashboardFilters(state="WA"),
    DashboardFilters(state="NSW", store_id="VIC-01"),
    DashboardFilters(store_id="missing"),
    DashboardFilters(product_id="missing"),
    DashboardFilters(state="NSW'; DROP TABLE sales; --"),
    DashboardFilters(store_id="NSW-01' OR 1=1 --"),
])
def test_invalid_or_mismatched_filters_fail_explicitly(db, filters):
    with pytest.raises(AnalyticsError):
        load_dashboard(filters, db)
    assert load_dashboard(path=db).net_revenue_cents == 110600


def test_empty_sources_have_zero_counts_and_unavailable_rates(db):
    with connection(db) as conn:
        for table in ("sales", "quality_inspections", "complaints"):
            conn.execute(f"DELETE FROM {table}")
    data = load_dashboard(DashboardFilters("NSW", "NSW-01", "PRD-01"), db)
    assert data.net_revenue_cents == data.complaint_count == 0
    assert data.freshness_pass_rate is None
    assert data.revenue_growth.percent is None
    assert data.revenue_trend.empty and data.recent_complaints.empty
    assert data.freshness_by_store["FreshnessPassRate"].isna().all()


def test_no_matching_product_rows_does_not_become_unfiltered_data(db):
    data = load_dashboard(DashboardFilters("VIC", "VIC-01", "PRD-02"), db)
    assert data.net_revenue_cents == data.complaint_count == 0
    assert data.freshness_pass_rate is None and data.revenue_growth.percent is None
    assert data.revenue_trend.empty and data.recent_complaints.empty
    assert data.freshness_by_store["FreshnessPassRate"].isna().all()
    assert "No other stores" in data.insights[0]


def test_empty_database_without_catalog_is_safe(tmp_path):
    path = tmp_path / "empty.db"
    initialize_database(path)
    data = load_dashboard(path=path)
    assert data.net_revenue_cents == data.complaint_count == 0
    assert data.freshness_pass_rate is None
    assert data.freshness_by_store.empty


def test_newly_confirmed_case_is_visible_on_next_read(db):
    filters = DashboardFilters("NSW", "NSW-01", "PRD-01")
    before = load_dashboard(filters, db)
    receipt = create_complaint(
        ComplaintDraft(
            "NSW-01", "PRD-01", datetime.now(timezone.utc).date() - timedelta(days=1),
            "Synthetic new report of mouldy strawberries.",
        ),
        "dashboard-integration", confirmed=True, path=db,
    )
    after = load_dashboard(filters, db)
    assert after.complaint_count == before.complaint_count + 1
    assert receipt.case_id in after.recent_complaints["CaseID"].values
    assert after.net_revenue_cents == before.net_revenue_cents
    assert after.freshness_pass_rate == before.freshness_pass_rate
    assert receipt.case_id not in load_dashboard(
        DashboardFilters("NSW", "NSW-02", "PRD-01"), db,
    ).recent_complaints["CaseID"].values


def test_recent_table_limit_does_not_limit_total_complaint_kpi(db):
    with connection(db) as conn:
        conn.executemany(
            """INSERT INTO complaints
               (CaseID, StoreID, ProductID, PurchaseDate, IssueDescription, CreatedAt)
               VALUES (?, 'NSW-01', 'PRD-01', '2026-03-01',
                       'Synthetic extra complaint.', '2026-04-01T12:00:00Z')""",
            [(f"NEW-{i:03}",) for i in range(60)],
        )
    data = load_dashboard(path=db)
    assert data.complaint_count == 66
    assert len(data.recent_complaints) == 50
    assert data.recent_complaints.iloc[0]["CaseID"] == "NEW-059"


def sales_frame(rows):
    frame = pd.DataFrame(rows, columns=["Date", "RevenueCents"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


@pytest.mark.parametrize(("end", "previous_start", "previous_end", "current_end"), [
    ("2026-10-25", "2026-09-01", "2026-09-25", "2026-10-25"),
    ("2026-03-31", "2026-02-01", "2026-02-28", "2026-03-28"),
    ("2024-03-31", "2024-02-01", "2024-02-29", "2024-03-29"),
    ("2024-02-29", "2024-01-01", "2024-01-29", "2024-02-29"),
    ("2027-01-15", "2026-12-01", "2026-12-15", "2027-01-15"),
])
def test_comparable_periods_handle_partial_months_leap_years_and_year_boundaries(
    end, previous_start, previous_end, current_end,
):
    result = comparable_growth(sales_frame([]), date(2023, 1, 1), date.fromisoformat(end))
    assert result.current_end == date.fromisoformat(current_end)
    assert result.previous_start == date.fromisoformat(previous_start)
    assert result.previous_end == date.fromisoformat(previous_end)
    assert (result.current_end - result.current_start).days == (
        result.previous_end - result.previous_start
    ).days
    assert result.percent is None


def test_growth_with_insufficient_history_or_zero_prior_revenue_is_unavailable():
    sales = sales_frame([("2026-10-01", 1000)])
    result = comparable_growth(sales, date(2026, 9, 2), date(2026, 10, 25))
    assert result.percent is None and "history" in result.reason
    result = comparable_growth(sales, date(2026, 8, 1), date(2026, 10, 25))
    assert result.percent is None and "positive revenue" in result.reason


def test_zero_current_revenue_is_a_valid_decline():
    result = comparable_growth(
        sales_frame([("2026-09-01", 1000)]), date(2026, 8, 1), date(2026, 10, 25),
    )
    assert result.percent == -100


def test_dashboard_does_not_write_or_create_database(db, tmp_path):
    with connection(db) as conn:
        original = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    load_dashboard(path=db)
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0] == original
        assert conn.execute("PRAGMA user_version").fetchone()[0] == version
    missing = tmp_path / "missing.db"
    with pytest.raises(AnalyticsError, match="missing"):
        load_dashboard(path=missing)
    assert not missing.exists()


def test_database_errors_are_sanitized(db, caplog):
    with connection(db) as conn:
        conn.execute("DROP TABLE sales")
    with caplog.at_level(logging.ERROR), pytest.raises(AnalyticsError, match="refresh"):
        load_dashboard(path=db)
    assert "Dashboard query failed" in caplog.text
    assert "SELECT" not in caplog.text


def test_unrelated_database_is_rejected(tmp_path):
    path = tmp_path / "unrelated.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE important (value TEXT)")
        conn.execute("INSERT INTO important VALUES ('keep')")
    with pytest.raises(AnalyticsError, match="FreshMart"):
        load_dashboard(path=path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM important").fetchone()[0] == "keep"
