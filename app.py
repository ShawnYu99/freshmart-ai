"""FreshMart customer support and calculated store management insights."""

from datetime import datetime, timedelta, timezone
import logging
import sqlite3
from uuid import uuid4

from dotenv import load_dotenv
import streamlit as st
import plotly.express as px

from src.analytics import AnalyticsError, DashboardFilters, load_dashboard
from src.complaints import (
    ComplaintDraft,
    ComplaintError,
    complaint_options,
    create_complaint,
    validate_complaint,
)
from src.database import ROOT, database_path, initialize_database
from src.policy_search import PolicySearchError, index_policies
from src.rag_service import RagAnswer, RagError, answer_question

LOGGER = logging.getLogger(__name__)


def show_answer(result: RagAnswer) -> None:
    st.markdown(result.answer)
    st.caption(f"Policy evidence: {result.evidence_status}")
    if result.sources:
        with st.expander("Policy sources used"):
            documents = {}
            for source in result.sources:
                documents.setdefault(source.source_id, []).append(source)
            for sections in documents.values():
                document = sections[0]
                numbers = sorted(
                    {section.passage_id.split("#section-", 1)[1] for section in sections},
                    key=int,
                )
                label = "Section" if len(numbers) == 1 else "Sections"
                st.write(f"{document.title} - {label} {', '.join(numbers)}")
                st.caption(f"{document.source_id} | Version {document.version}")
                st.text(document.source_path)
                for section in sections:
                    st.text(f"{section.passage_id} | {section.section}")
                    quotes = dict.fromkeys(
                        citation.quote for citation in result.citations
                        if citation.passage.passage_id == section.passage_id
                    )
                    for quote in quotes:
                        st.text(quote)
    else:
        st.caption("No supporting policy source is claimed.")


def customer_chat() -> None:
    st.subheader("Ask about FreshMart policies")
    st.caption(
        "For example: I bought strawberries from Chatswood yesterday, "
        "but some were mouldy. Can I get a refund?"
    )
    history = st.session_state.setdefault("chat_history", [])
    for message in history:
        with st.chat_message(message["role"]):
            if "result" in message:
                show_answer(message["result"])
            elif "error" in message:
                st.error(message["error"])
            else:
                st.markdown(message["content"])
    question = st.chat_input("Ask a policy question", max_chars=2000)
    if question:
        question = question.strip()
        if not 1 <= len(question) <= 2000:
            st.warning("Enter a policy question containing 1 to 2,000 characters.")
            return
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking fictional policies with Microsoft Foundry..."):
                    result = answer_question(question)
            except (RagError, PolicySearchError) as exc:
                error = f"No AI answer is available. {exc}"
                st.error(error)
                history.append({"role": "assistant", "error": error})
            else:
                show_answer(result)
                history.append({"role": "assistant", "result": result})


def complaint_form() -> None:
    st.subheader("Report a product quality concern")
    st.caption(
        "Use synthetic details only. Do not include names, contact details, "
        "payment information, or other personal data."
    )
    if "complaint_receipt" in st.session_state:
        receipt = st.session_state["complaint_receipt"]
        st.success(f"Complaint saved. Case ID: {receipt.case_id}")
        st.write(f"Status: {receipt.status}")
        st.info("This is a request for review, not an approved refund. No payment has been initiated.")
        if st.button("Start another complaint"):
            del st.session_state["complaint_receipt"]
            st.session_state.pop("complaint_draft", None)
            st.session_state.pop("submission_token", None)
            st.rerun()
        return
    try:
        stores, products = complaint_options()
    except ComplaintError as exc:
        st.error(str(exc))
        return
    draft = st.session_state.get("complaint_draft")
    if draft is None:
        today = datetime.now(timezone.utc).date()
        with st.form("complaint_details"):
            store_id = st.selectbox(
                "Store", [store.store_id for store in stores],
                format_func=lambda value: next(
                    f"{store.name} ({store.state})" for store in stores if store.store_id == value
                ),
                key="complaint_store",
            )
            product_ids = [product.product_id for product in products]
            product_id = st.selectbox(
                "Product", product_ids,
                index=next((i for i, p in enumerate(products) if p.name == "Strawberries"), 0),
                format_func=lambda value: next(
                    product.name for product in products if product.product_id == value
                ),
                key="complaint_product",
            )
            purchase_date = st.date_input(
                "Purchase date", value=today - timedelta(days=1), max_value=today,
                key="complaint_date",
            )
            description = st.text_area(
                "Issue description", max_chars=2000, key="complaint_description",
                help="Describe the quality concern in 10 to 2,000 characters.",
            )
            proof = st.selectbox(
                "Proof of purchase (optional)", ("Not specified", "Yes", "No"),
                key="complaint_proof",
            )
            review = st.form_submit_button("Review complaint", type="primary")
        if review:
            try:
                draft = validate_complaint(ComplaintDraft(
                    store_id, product_id, purchase_date, description,
                    {"Not specified": None, "Yes": True, "No": False}[proof],
                ))
            except ComplaintError as exc:
                st.error(str(exc))
            else:
                st.session_state["complaint_draft"] = draft
                st.session_state["submission_token"] = uuid4().hex
                st.rerun()
        return
    st.write("Review your complaint")
    st.write("Store:", next(f"{s.name} ({s.state})" for s in stores if s.store_id == draft.store_id))
    st.write("Product:", next(p.name for p in products if p.product_id == draft.product_id))
    st.write("Purchase date:", draft.purchase_date.isoformat())
    st.text(draft.issue_description)
    st.write(
        "Proof of purchase:",
        "Not specified" if draft.proof_of_purchase is None else (
            "Yes" if draft.proof_of_purchase else "No"
        ),
    )
    st.info("Confirming creates an Open case for review. It does not approve a refund.")
    with st.form("complaint_confirmation"):
        confirmed = st.checkbox("I confirm these details and want to submit this complaint.")
        submit = st.form_submit_button("Submit confirmed complaint", type="primary")
    if submit:
        try:
            receipt = create_complaint(
                draft, st.session_state["submission_token"], confirmed=confirmed,
            )
        except ComplaintError as exc:
            st.error(str(exc))
        else:
            st.session_state["complaint_receipt"] = receipt
            st.rerun()
    if st.button("Edit complaint"):
        st.session_state["complaint_store"] = draft.store_id
        st.session_state["complaint_product"] = draft.product_id
        st.session_state["complaint_date"] = draft.purchase_date
        st.session_state["complaint_description"] = draft.issue_description
        st.session_state["complaint_proof"] = (
            "Not specified" if draft.proof_of_purchase is None else (
                "Yes" if draft.proof_of_purchase else "No"
            )
        )
        st.session_state.pop("complaint_draft", None)
        st.session_state.pop("submission_token", None)
        st.rerun()


def manager_dashboard() -> None:
    st.subheader("Store Manager Dashboard")
    st.caption("Internal demo view. All KPIs use recorded SQLite data, not model-generated figures.")
    try:
        stores, products = complaint_options()
    except ComplaintError as exc:
        st.error(str(exc))
        return
    state_column, store_column, product_column = st.columns(3)
    with state_column:
        state = st.selectbox(
            "State", [None, *sorted({store.state for store in stores})],
            format_func=lambda value: value or "All states", key="manager_state",
        )
    available_stores = {s.store_id: s for s in stores if state is None or s.state == state}
    if st.session_state.get("manager_store") not in (None, *available_stores):
        st.session_state["manager_store"] = None
    with store_column:
        store_id = st.selectbox(
            "Store", [None, *available_stores],
            format_func=lambda value: available_stores[value].name if value else "All stores",
            key="manager_store",
        )
    product_names = {p.product_id: p.name for p in products}
    with product_column:
        product_id = st.selectbox(
            "Product", [None, *product_names],
            format_func=lambda value: product_names[value] if value else "All products",
            key="manager_product",
        )
    st.button("Refresh dashboard", key="manager_refresh")
    st.caption("All recorded dates. Each rerun reads fresh data; complaint counts include every status.")
    try:
        data = load_dashboard(DashboardFilters(state, store_id, product_id))
    except AnalyticsError as exc:
        st.error(str(exc))
        return
    growth = data.revenue_growth
    revenue_card, growth_card, freshness_card, complaint_card = st.columns(4)
    revenue_card.metric("Net Revenue", f"A${data.net_revenue_cents / 100:,.2f}")
    growth_card.metric(
        "Month-over-Month Revenue Growth",
        f"{growth.percent:+.1f}%" if growth.percent is not None else "N/A",
    )
    freshness_card.metric(
        "Freshness Pass Rate",
        f"{data.freshness_pass_rate:.1%}" if data.freshness_pass_rate is not None else "N/A",
    )
    complaint_card.metric("Quality Complaint Count", str(data.complaint_count))
    if growth.current_start is not None:
        st.caption(
            f"MoM: {growth.current_start} to {growth.current_end} versus "
            f"{growth.previous_start} to {growth.previous_end}. Equal day counts, "
            "anchored to the latest recorded sales date across the database."
        )
    if growth.reason:
        st.caption(f"Revenue growth unavailable: {growth.reason}")
    st.caption("Freshness = total passed kilograms / total inspected kilograms, never inferred from sales.")
    if data.freshness_pass_rate is None:
        st.caption("No inspected weight matches the selected filters. Freshness is unavailable, not zero.")
    st.subheader("Revenue trend")
    if data.revenue_trend.empty:
        st.info("No sales match these filters.")
    else:
        trend = px.line(
            data.revenue_trend, x="Date", y="RevenueAUD",
            labels={"RevenueAUD": "Net revenue (AUD)", "Date": "Date"},
        )
        trend.update_traces(hovertemplate="%{x|%d %b %Y}<br>A$%{y:,.2f}<extra></extra>")
        st.plotly_chart(trend, width="stretch", key="manager_revenue_chart")
    st.subheader("Freshness pass-rate comparison")
    st.caption(
        "Comparison includes all stores in the selected state and product scope; "
        "the selected store is highlighted when inspected. Stores without inspections "
        "are omitted. KPIs and complaints use all three filters."
    )
    comparison = data.freshness_by_store.dropna(subset=["FreshnessPassRate"]).copy()
    if comparison.empty:
        st.info("No freshness inspections are available for this comparison.")
    else:
        comparison["Scope"] = comparison["StoreID"].map(
            lambda value: "Selected store" if value == store_id else "Peer store"
        )
        chart = px.bar(
            comparison, x="StoreName", y="FreshnessPassRate", color="Scope",
            color_discrete_map={"Selected store": "#0078D4", "Peer store": "#8AA6BA"},
            labels={"StoreName": "Store", "FreshnessPassRate": "Freshness pass rate"},
            hover_data={"InspectedKg": ":.1f", "PassedKg": ":.1f"},
        )
        chart.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(chart, width="stretch", key="manager_freshness_chart")
    st.subheader("Calculated observations")
    for insight in data.insights:
        st.write(insight)
    st.subheader("Recent customer complaints")
    st.caption("Most recent 50 matching cases, ordered by creation time (UTC). These are requests for review.")
    if data.recent_complaints.empty:
        st.info("No quality complaints match these filters.")
    else:
        st.dataframe(data.recent_complaints, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="FreshMart AI | Support & Retail Insights",
        page_icon=":material/storefront:", layout="centered",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    st.title("FreshMart AI")
    st.caption("Customer support and retail insights | Fictional Australian retail demonstration")
    st.info("This project uses synthetic retail data and fictional company policies for demonstration purposes.")
    path = database_path()
    if not path.is_file():
        st.error("Demo database missing. Run python -m scripts.generate_data, then reload.")
        st.stop()
    if st.session_state.get("initialized_database") != str(path):
        try:
            initialize_database(path)
            index_policies(path)
        except (ValueError, OSError, sqlite3.Error, PolicySearchError) as exc:
            LOGGER.error("Application initialization failed: type=%s", type(exc).__name__)
            st.error("Unable to initialize the demo database or policy index. Check local database access and setup.")
            st.stop()
        st.session_state["initialized_database"] = str(path)
    chat_tab, complaint_tab, manager_tab = st.tabs(
        ["Customer chat", "Quality complaint", "Store Manager Dashboard"]
    )
    with chat_tab:
        customer_chat()
    with complaint_tab:
        complaint_form()
    with manager_tab:
        manager_dashboard()


if __name__ == "__main__":
    main()
