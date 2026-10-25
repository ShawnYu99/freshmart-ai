"""Read-only retail metrics from a consistent SQLite snapshot."""

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from pathlib import Path
import sqlite3

import pandas as pd

from src.database import APPLICATION_ID, SCHEMA_VERSION, connection, database_path

LOGGER = logging.getLogger(__name__)


class AnalyticsError(RuntimeError):
    """A safe dashboard validation or database error."""


@dataclass(frozen=True)
class DashboardFilters:
    state: str | None = None
    store_id: str | None = None
    product_id: str | None = None


@dataclass(frozen=True)
class RevenueGrowth:
    percent: float | None
    current_start: date | None = None
    current_end: date | None = None
    previous_start: date | None = None
    previous_end: date | None = None
    reason: str | None = None


@dataclass(frozen=True)
class DashboardData:
    net_revenue_cents: int
    revenue_growth: RevenueGrowth
    freshness_pass_rate: float | None
    complaint_count: int
    revenue_trend: pd.DataFrame
    freshness_by_store: pd.DataFrame
    recent_complaints: pd.DataFrame
    insights: tuple[str, ...]


def validate_filters(conn: sqlite3.Connection, filters: DashboardFilters) -> None:
    if filters.state is not None and not conn.execute(
        "SELECT 1 FROM stores WHERE State = ?", (filters.state,)
    ).fetchone():
        raise AnalyticsError("Select an existing store state.")
    if filters.store_id is not None:
        store = conn.execute(
            "SELECT State FROM stores WHERE StoreID = ?", (filters.store_id,)
        ).fetchone()
        if not store or (filters.state is not None and store["State"] != filters.state):
            raise AnalyticsError("Select a store belonging to the selected state.")
    if filters.product_id is not None and not conn.execute(
        "SELECT 1 FROM products WHERE ProductID = ?", (filters.product_id,)
    ).fetchone():
        raise AnalyticsError("Select an existing product.")


def filter_sql(
    filters: DashboardFilters, *, include_store: bool = True,
) -> tuple[str, list[str]]:
    clauses, parameters = [], []
    for column, value in (
        ("s.State", filters.state),
        ("s.StoreID", filters.store_id if include_store else None),
        ("f.ProductID", filters.product_id),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters


def comparable_growth(
    sales: pd.DataFrame, coverage_start: date | None, coverage_end: date | None,
) -> RevenueGrowth:
    """Compare equal calendar-day counts; zero current revenue is valid."""
    if coverage_start is None or coverage_end is None:
        return RevenueGrowth(None, reason="No recorded sales are available.")
    current_start = coverage_end.replace(day=1)
    previous_last = current_start - timedelta(days=1)
    previous_start = previous_last.replace(day=1)
    days = min(coverage_end.day, previous_last.day)
    current_end = current_start + timedelta(days=days - 1)
    previous_end = previous_start + timedelta(days=days - 1)
    periods = (current_start, current_end, previous_start, previous_end)
    if coverage_start > previous_start:
        return RevenueGrowth(
            None, *periods,
            reason="Recorded sales history does not cover the full comparison period.",
        )
    current = int(sales.loc[
        sales["Date"].between(pd.Timestamp(current_start), pd.Timestamp(current_end)),
        "RevenueCents",
    ].sum())
    previous = int(sales.loc[
        sales["Date"].between(pd.Timestamp(previous_start), pd.Timestamp(previous_end)),
        "RevenueCents",
    ].sum())
    if previous <= 0:
        return RevenueGrowth(
            None, *periods, reason="The matching prior period has no positive revenue.",
        )
    return RevenueGrowth((current - previous) / previous * 100, *periods)


def comparative_insights(
    comparison: pd.DataFrame, filters: DashboardFilters,
) -> tuple[str, ...]:
    if filters.store_id is None:
        return ("Select a store to compare freshness and complaint counts with matching peer stores.",)
    selected = comparison.loc[comparison["StoreID"] == filters.store_id].iloc[0]
    peers = comparison.loc[comparison["StoreID"] != filters.store_id]
    if peers.empty:
        return ("No other stores match the state/product scope for a peer comparison.",)
    messages = []
    inspected = float(peers["InspectedKg"].sum())
    rate = selected["FreshnessPassRate"]
    if pd.notna(rate) and inspected > 0:
        peer_rate = float(peers["PassedKg"].sum()) / inspected
        if round(float(rate) * 100, 1) < round(peer_rate * 100, 1):
            messages.append(
                f"{selected['StoreName']} has a {rate:.1%} freshness pass rate versus "
                f"{peer_rate:.1%} across other matching stores "
                f"({(peer_rate - rate) * 100:.1f} percentage points lower)."
            )
    average = float(peers["ComplaintCount"].mean())
    if int(selected["ComplaintCount"]) > average:
        messages.append(
            f"{selected['StoreName']} has {int(selected['ComplaintCount'])} quality complaints "
            f"versus a peer average of {average:.1f} per store. "
            "These counts are not adjusted for sales volume."
        )
    if messages:
        messages.append(
            "Consider further investigation. These comparisons do not establish causation "
            "or approve any refund."
        )
    else:
        messages.append(
            "No lower freshness or elevated complaint count is indicated by the available "
            "peer comparison. Missing inspections are not evidence of good freshness."
        )
    return tuple(messages)


def load_dashboard(
    filters: DashboardFilters = DashboardFilters(), path: str | Path | None = None,
) -> DashboardData:
    """Read all-time KPIs and the latest 50 complaints; do not cache mutable data."""
    target = database_path(path)
    if not target.is_file():
        raise AnalyticsError("Demo database missing. Run python -m scripts.generate_data first.")
    where, parameters = filter_sql(filters)
    peer_where, peer_parameters = filter_sql(filters, include_store=False)
    try:
        with connection(target) as conn:
            conn.execute("BEGIN")
            if (
                conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or conn.execute("PRAGMA user_version").fetchone()[0] not in (1, SCHEMA_VERSION)
            ):
                raise AnalyticsError("Dashboard requires an initialized FreshMart database.")
            validate_filters(conn, filters)
            bounds = conn.execute("SELECT MIN(Date), MAX(Date) FROM sales").fetchone()
            start, end = (date.fromisoformat(value) if value else None for value in bounds)
            sales = pd.read_sql_query(
                """SELECT f.Date, SUM(f.Quantity * f.UnitPriceCents) AS RevenueCents
                   FROM sales f JOIN stores s ON s.StoreID = f.StoreID"""
                + where + " GROUP BY f.Date ORDER BY f.Date",
                conn, params=parameters, parse_dates=["Date"],
            )
            stores = pd.read_sql_query(
                "SELECT StoreID, StoreName, State FROM stores"
                + (" WHERE State = ?" if filters.state is not None else "")
                + " ORDER BY State, StoreName",
                conn, params=[filters.state] if filters.state is not None else [],
            )
            inspections = pd.read_sql_query(
                """SELECT f.StoreID, SUM(f.InspectedKg) AS InspectedKg,
                          SUM(f.PassedKg) AS PassedKg
                   FROM quality_inspections f JOIN stores s ON s.StoreID = f.StoreID"""
                + peer_where + " GROUP BY f.StoreID",
                conn, params=peer_parameters,
            )
            complaints = pd.read_sql_query(
                """SELECT f.StoreID, COUNT(*) AS ComplaintCount
                   FROM complaints f JOIN stores s ON s.StoreID = f.StoreID"""
                + peer_where + " GROUP BY f.StoreID",
                conn, params=peer_parameters,
            )
            recent = pd.read_sql_query(
                """SELECT f.CaseID, s.State, s.StoreName, p.ProductName,
                          f.PurchaseDate, f.IssueDescription, f.Status, f.CreatedAt
                   FROM complaints f JOIN stores s ON s.StoreID = f.StoreID
                   JOIN products p ON p.ProductID = f.ProductID"""
                + where + " ORDER BY f.CreatedAt DESC, f.CaseID DESC LIMIT 50",
                conn, params=parameters,
            )
    except (sqlite3.Error, pd.errors.DatabaseError, OSError) as exc:
        LOGGER.error("Dashboard query failed: type=%s", type(exc).__name__)
        raise AnalyticsError("Unable to read dashboard data. Check database access and refresh.") from None
    comparison = stores.merge(inspections, on="StoreID", how="left").merge(
        complaints, on="StoreID", how="left",
    )
    for column in ("InspectedKg", "PassedKg", "ComplaintCount"):
        comparison[column] = pd.to_numeric(comparison[column]).fillna(0)
    comparison["ComplaintCount"] = comparison["ComplaintCount"].astype(int)
    comparison["FreshnessPassRate"] = comparison["PassedKg"].div(
        comparison["InspectedKg"].where(comparison["InspectedKg"] > 0)
    )
    selected = comparison if filters.store_id is None else comparison.loc[
        comparison["StoreID"] == filters.store_id
    ]
    inspected = float(selected["InspectedKg"].sum())
    freshness = float(selected["PassedKg"].sum()) / inspected if inspected > 0 else None
    growth = comparable_growth(sales, start, end)
    total_cents = int(sales["RevenueCents"].sum())
    if not sales.empty:
        sales = sales.set_index("Date").reindex(
            pd.date_range(start, end, name="Date"), fill_value=0,
        ).reset_index()
    sales["RevenueAUD"] = sales["RevenueCents"] / 100
    LOGGER.info(
        "Dashboard read completed: sales_days=%d complaints=%d",
        len(sales), int(selected["ComplaintCount"].sum()),
    )
    return DashboardData(
        total_cents, growth, freshness, int(selected["ComplaintCount"].sum()),
        sales, comparison, recent, comparative_insights(comparison, filters),
    )
