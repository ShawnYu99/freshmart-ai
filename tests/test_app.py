"""Streamlit UI tests use local databases and mocked inference, never Azure."""

from datetime import datetime, timedelta, timezone

import pytest
from streamlit.testing.v1 import AppTest

from scripts.generate_data import generate_data
from src import analytics, complaints, rag_service
from src.analytics import DashboardFilters, load_dashboard
from src.database import ROOT, connection
from src.policy_search import retrieve_passages
from src.rag_service import PolicyCitation, RagAnswer, RagError, evidence_excerpts


@pytest.fixture
def app(tmp_path, monkeypatch):
    path = tmp_path / "ui.db"
    generate_data(path)
    monkeypatch.setenv("FRESHMART_DB_PATH", str(path))
    monkeypatch.setattr(
        rag_service, "answer_question",
        lambda *args, **kwargs: pytest.fail("An offline UI test must not call Foundry"),
    )
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception
    return at, path


def click(at, label):
    next(button for button in at.button if button.label == label).click().run()
    assert not at.exception


def count(path):
    with connection(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]


def review(at):
    at.selectbox(key="complaint_store").select("NSW-01")
    at.selectbox(key="complaint_product").select("PRD-01")
    at.date_input(key="complaint_date").set_value(
        datetime.now(timezone.utc).date() - timedelta(days=1)
    )
    at.text_area(key="complaint_description").set_value("Synthetic complaint: mouldy strawberries.")
    at.selectbox(key="complaint_proof").select("Yes")
    click(at, "Review complaint")


def test_complaint_requires_review_confirmation_and_survives_reruns(app):
    at, path = app
    review(at)
    assert count(path) == 25
    assert any("mouldy strawberries" in text.value for text in at.text)
    click(at, "Submit confirmed complaint")
    assert any("explicitly confirm" in error.value for error in at.error)
    assert count(path) == 25
    at.checkbox[0].check()
    click(at, "Submit confirmed complaint")
    receipt = at.session_state["complaint_receipt"]
    assert receipt.case_id in at.success[0].value
    with connection(path) as conn:
        row = conn.execute("SELECT * FROM complaints WHERE CaseID=?", (receipt.case_id,)).fetchone()
        assert row["Status"] == "Open"
        assert row["StoreID"] == "NSW-01" and row["ProductID"] == "PRD-01"
        assert row["ProofOfPurchase"] == 1
    at.run()
    assert not at.exception and count(path) == 26
    assert receipt.case_id in at.success[0].value
    click(at, "Start another complaint")
    assert at.text_area(key="complaint_description").value == ""
    assert count(path) == 26


def test_edit_preserves_values_and_invalid_form_does_not_write(app):
    at, path = app
    click(at, "Review complaint")
    assert any("10 to 2,000" in error.value for error in at.error)
    review(at)
    click(at, "Edit complaint")
    assert at.text_area(key="complaint_description").value == "Synthetic complaint: mouldy strawberries."
    assert at.selectbox(key="complaint_proof").value == "Yes"
    at.text_area(key="complaint_description").set_value("Updated synthetic report: spoiled berries.")
    click(at, "Review complaint")
    assert at.session_state["complaint_draft"].issue_description.startswith("Updated")
    assert count(path) == 25


def test_chat_displays_sources_and_preserves_history_without_repeating_inference(app, monkeypatch):
    at, path = app
    passage = retrieve_passages("mouldy strawberries refund", path)[0]
    result = RagAnswer(
        "Offline fixture: a review is possible; approval is not guaranteed.",
        (passage,), (passage,), True, True, evidence_status="partial",
        citations=(PolicyCitation(passage, passage.content, "Offline fixture claim."),),
    )
    questions = []

    def respond(question):
        questions.append(question)
        return result

    monkeypatch.setattr(rag_service, "answer_question", respond)
    at.chat_input[0].set_value("Can I get a refund for mouldy strawberries?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert any(item.value == result.answer for item in at.markdown)
    assert any(passage.source_id in item.value for item in at.caption)
    assert any(passage.source_path in item.value for item in at.text)
    at.run()
    assert not at.exception and len(at.chat_message) == 2
    assert len(questions) == 1
    assert count(path) == 25


def test_source_summaries_group_sections_and_preserve_only_cited_evidence(app, monkeypatch):
    at, path = app
    hits = retrieve_passages("mouldy strawberries refund guarantee", path, limit=12)
    selected = tuple(
        next(p for p in hits if p.passage_id == identifier)
        for identifier in (
            "FM-POL-002#section-3", "FM-POL-002#section-1", "FM-POL-001#section-1",
        )
    )
    citations = tuple(
        PolicyCitation(p, next(iter(evidence_excerpts((p,)).values()))[1], "Offline fixture claim.")
        for p in selected
    )
    result = RagAnswer(
        "Offline fixture: use the Quality complaint tab to request review, not approval.",
        selected, hits, True, True, evidence_status="partial",
        citations=(*citations, citations[0]),
    )
    monkeypatch.setattr(rag_service, "answer_question", lambda question: result)
    at.chat_input[0].set_value("Can I request a quality review?").run()
    assert not at.exception
    summaries = [item.value for item in at.markdown]
    assert summaries.count("Refund and Exchange Policy - Sections 1, 3") == 1
    assert summaries.count("Fresh Produce Quality Policy - Section 1") == 1
    details = [item.value for item in at.text]
    for passage in selected:
        assert f"{passage.passage_id} | {passage.section}" in details
        assert details.count(passage.source_path) == 1
    for citation in citations:
        assert details.count(citation.quote) == 1
    assert len(at.expander) == 1 and at.expander[0].label == "Policy sources used"
    assert all(
        not any(p.passage_id in text for text in details)
        for p in hits if p not in selected
    )
    assert count(path) == 25


@pytest.mark.parametrize("question", ["   ", "\t\n", "x" * 2001])
def test_invalid_chat_input_is_rejected_without_calling_model(app, question):
    at, path = app
    at.chat_input[0].set_value(question).run()
    assert not at.exception
    assert any("1 to 2,000 characters" in warning.value for warning in at.warning)
    assert not at.session_state["chat_history"]
    assert count(path) == 25


def test_foundry_error_is_explicit_and_does_not_block_complaints(app, monkeypatch):
    at, path = app

    def fail(question):
        raise RagError("Foundry rate limit or quota reached (429).")

    monkeypatch.setattr(rag_service, "answer_question", fail)
    at.chat_input[0].set_value("Can I request a refund review?").run()
    assert not at.exception
    assert any("No AI answer is available" in error.value and "429" in error.value for error in at.error)
    assert all("result" not in message for message in at.session_state["chat_history"])
    review(at)
    at.checkbox[0].check()
    click(at, "Submit confirmed complaint")
    assert count(path) == 26


def test_failed_submission_retains_same_token_for_retry(app, monkeypatch):
    at, path = app
    review(at)
    token = at.session_state["submission_token"]
    original = complaints.create_complaint

    def fail(*args, **kwargs):
        raise complaints.ComplaintError("Storage is temporarily unavailable.")

    monkeypatch.setattr(complaints, "create_complaint", fail)
    at.checkbox[0].check()
    click(at, "Submit confirmed complaint")
    assert count(path) == 25
    assert at.session_state["submission_token"] == token
    monkeypatch.setattr(complaints, "create_complaint", original)
    click(at, "Submit confirmed complaint")
    assert count(path) == 26
    with connection(path) as conn:
        assert conn.execute(
            "SELECT SubmissionToken FROM complaints WHERE CaseID=?",
            (at.session_state["complaint_receipt"].case_id,),
        ).fetchone()[0] == token


def test_missing_database_shows_setup_error_without_creating_it(tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setenv("FRESHMART_DB_PATH", str(path))
    at = AppTest.from_file(str(ROOT / "app.py")).run()
    assert not at.exception and not path.exists()
    assert any("scripts.generate_data" in error.value for error in at.error)


def manager_values(at):
    return {metric.label: metric.value for metric in at.metric}


def select_manager_scenario(at):
    at.selectbox(key="manager_state").select("NSW").run()
    at.selectbox(key="manager_store").select("NSW-01").run()
    at.selectbox(key="manager_product").select("PRD-01").run()
    assert not at.exception


def test_manager_tab_filters_kpis_charts_and_complaints(app):
    at, path = app
    assert "Store Manager Dashboard" in [tab.label for tab in at.tabs]
    select_manager_scenario(at)
    data = load_dashboard(DashboardFilters("NSW", "NSW-01", "PRD-01"), path)
    values = manager_values(at)
    assert values["Net Revenue"] == f"A${data.net_revenue_cents / 100:,.2f}"
    assert values["Freshness Pass Rate"] == f"{data.freshness_pass_rate:.1%}"
    assert values["Quality Complaint Count"] == str(data.complaint_count)
    assert len(at.get("plotly_chart")) == 2
    table = at.dataframe[0].value
    assert set(table["StoreName"]) == {"Chatswood"}
    assert set(table["ProductName"]) == {"Strawberries"}
    assert any("do not establish causation" in text.value for text in at.markdown)
    at.selectbox(key="manager_state").select("VIC").run()
    assert not at.exception
    assert at.selectbox(key="manager_store").value is None
    assert "Chatswood" not in at.selectbox(key="manager_store").options
    assert set(at.dataframe[0].value["State"]) <= {"VIC"}


def test_customer_submission_appears_in_filtered_dashboard(app):
    at, path = app
    select_manager_scenario(at)
    before = manager_values(at)
    review(at)
    at.checkbox[0].check()
    click(at, "Submit confirmed complaint")
    receipt = at.session_state["complaint_receipt"]
    after = manager_values(at)
    assert after["Quality Complaint Count"] == str(int(before["Quality Complaint Count"]) + 1)
    assert after["Net Revenue"] == before["Net Revenue"]
    assert after["Freshness Pass Rate"] == before["Freshness Pass Rate"]
    assert receipt.case_id in at.dataframe[0].value["CaseID"].values


def test_dashboard_refresh_reads_external_complaint_without_cache(app):
    at, path = app
    select_manager_scenario(at)
    before = int(manager_values(at)["Quality Complaint Count"])
    receipt = complaints.create_complaint(
        complaints.ComplaintDraft(
            "NSW-01", "PRD-01", datetime.now(timezone.utc).date() - timedelta(days=1),
            "Synthetic complaint from another session.",
        ), "external-session", confirmed=True, path=path,
    )
    assert int(manager_values(at)["Quality Complaint Count"]) == before
    click(at, "Refresh dashboard")
    assert int(manager_values(at)["Quality Complaint Count"]) == before + 1
    assert receipt.case_id in at.dataframe[0].value["CaseID"].values


def test_dashboard_empty_data_shows_na_and_explicit_empty_states(app):
    at, path = app
    with connection(path) as conn:
        for table in ("sales", "quality_inspections", "complaints"):
            conn.execute(f"DELETE FROM {table}")
    click(at, "Refresh dashboard")
    values = manager_values(at)
    assert values == {
        "Net Revenue": "A$0.00", "Month-over-Month Revenue Growth": "N/A",
        "Freshness Pass Rate": "N/A", "Quality Complaint Count": "0",
    }
    assert any("No sales" in info.value for info in at.info)
    assert any("No freshness inspections" in info.value for info in at.info)
    assert any("No quality complaints" in info.value for info in at.info)
    assert not at.dataframe


def test_dashboard_failure_does_not_block_complaint_review(app, monkeypatch):
    at, path = app

    def fail(*args, **kwargs):
        raise analytics.AnalyticsError("Unable to read dashboard data. Check database access and refresh.")

    monkeypatch.setattr(analytics, "load_dashboard", fail)
    at.run()
    assert not at.exception
    assert any("Unable to read dashboard data" in error.value for error in at.error)
    review(at)
    assert count(path) == 25
