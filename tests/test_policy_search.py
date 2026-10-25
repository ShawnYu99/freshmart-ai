"""FTS5 tests using real policy text and temporary operational databases."""

import sqlite3

import pytest

from scripts.generate_data import generate_data
from src.database import connection
from src.policy_search import (
    KNOWLEDGE_DIR,
    PolicySearchError,
    index_policies,
    keyword_query,
    read_policy_passages,
    retrieve_passages,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "freshmart.db"
    generate_data(path)
    return path


def operational_snapshot(path):
    with connection(path) as conn:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("stores", "products", "sales", "quality_inspections", "complaints")
        }


def test_logical_sections_preserve_full_original_text_and_metadata():
    passages = read_policy_passages()
    assert len(passages) == 17
    assert len({p.passage_id for p in passages}) == 17
    for path in KNOWLEDGE_DIR.glob("*.md"):
        matching = [p for p in passages if p.source_path.endswith(path.name)]
        assert matching
        assert matching[0].preamble + "".join(p.content for p in matching) == path.read_text()
        assert all(p.version == "1.0" and p.effective_date == "2026-01-01" for p in matching)
        assert all(f"Source ID: {p.source_id}" in p.preamble for p in matching)
        assert all(f"# {p.title}" in p.preamble for p in matching)


def test_index_is_repeatable_separate_and_preserves_records_and_files(db):
    original = operational_snapshot(db)
    original_files = {p: p.read_bytes() for p in KNOWLEDGE_DIR.glob("*.md")}
    assert index_policies(db) == index_policies(db) == 17
    assert operational_snapshot(db) == original
    assert {p: p.read_bytes() for p in original_files} == original_files
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM policy_passages_fts").fetchone()[0] == 17
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    generate_data(db, reset=True)
    assert len(retrieve_passages("refund", db)) > 0


@pytest.mark.parametrize(("question", "expected"), [
    ("I bought mouldy strawberries from Chatswood yesterday. Can I get a refund?", "FM-POL-002"),
    ("My online grocery order is missing an item. What should I do?", "FM-POL-003"),
    ("Can I reject a substituted product?", "FM-POL-004"),
    ("Can you guarantee that I will receive a refund?", "FM-POL-002"),
])
def test_requested_questions_rank_the_right_policy_first(db, question, expected):
    index_policies(db)
    hits = retrieve_passages(question, db)
    assert hits[0].source_id == expected
    assert [p.rank for p in hits] == sorted(p.rank for p in hits)
    assert len(hits) <= 6
    assert all(p.rank < 0 for p in hits)
    if "strawberries" in question:
        review = next(p for p in hits if p.section == "Fresh produce quality review")
        assert "seven calendar days" in review.content


@pytest.mark.parametrize("word", ["mouldy", "moldy", "spoiled", "rotten", "refund", "return"])
def test_required_synonyms_retrieve_actual_policy_passages(db, word):
    index_policies(db)
    hits = retrieve_passages(word, db)
    assert hits
    assert any(p.source_id in {"FM-POL-001", "FM-POL-002"} for p in hits)


def test_missing_evidence_and_safe_fts_syntax(db):
    index_policies(db)
    assert retrieve_passages(
        "What is FreshMart's policy on an issue not covered by the documents?", db
    ) == ()
    assert retrieve_passages("quantum teleportation", db) == ()
    assert retrieve_passages("!!!", db) == ()
    assert keyword_query('refund") OR *; DROP TABLE sales; --') == (
        '"drop" OR "exchange" OR "refund" OR "refunds" OR "return" OR "returns" '
        'OR "sales" OR "table"'
    )
    retrieve_passages('refund") OR *; DROP TABLE sales; --', db)
    assert len(operational_snapshot(db)["sales"]) == 2000


@pytest.mark.parametrize("question", ["", " ", "x" * 2001, None, 42])
def test_invalid_questions_are_explicit_errors(db, question):
    with pytest.raises(ValueError, match="Question"):
        retrieve_passages(question, db)


@pytest.mark.parametrize("limit", [0, 13, True, "six"])
def test_invalid_limits(db, limit):
    with pytest.raises(ValueError, match="limit"):
        retrieve_passages("refund", db, limit=limit)


def test_missing_database_and_index_are_explicit_errors(db, tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(PolicySearchError, match="database is missing"):
        index_policies(missing)
    assert not missing.exists()
    with pytest.raises(PolicySearchError, match="index is missing"):
        retrieve_passages("refund", db)


def test_refuses_unrelated_database_and_table_name_collision(db, tmp_path):
    unrelated = tmp_path / "other.db"
    with sqlite3.connect(unrelated) as conn:
        conn.execute("CREATE TABLE keep (value TEXT)")
    with pytest.raises(PolicySearchError, match="existing FreshMart"):
        index_policies(unrelated)
    with connection(db) as conn:
        conn.execute("CREATE TABLE policy_passages_fts (value TEXT)")
        conn.execute("INSERT INTO policy_passages_fts VALUES ('preserve')")
    with pytest.raises(PolicySearchError, match="already used"):
        index_policies(db)
    with connection(db) as conn:
        assert conn.execute("SELECT value FROM policy_passages_fts").fetchone()[0] == "preserve"


def test_invalid_documents_do_not_replace_existing_index(db, tmp_path):
    index_policies(db)
    original = retrieve_passages("refund", db)
    bad_docs = tmp_path / "bad-policies"
    bad_docs.mkdir()
    (bad_docs / "invalid.md").write_text("# No metadata\n\n## Invalid\nContent.")
    with pytest.raises(PolicySearchError, match="metadata"):
        index_policies(db, directory=bad_docs)
    assert retrieve_passages("refund", db) == original


def test_duplicate_document_identifiers_are_rejected(tmp_path):
    text = (KNOWLEDGE_DIR / "refund_and_exchange_policy.md").read_text()
    (tmp_path / "one.md").write_text(text)
    (tmp_path / "two.md").write_text(text)
    with pytest.raises(PolicySearchError, match="unique"):
        read_policy_passages(tmp_path)
