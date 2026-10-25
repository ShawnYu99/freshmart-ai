"""Local policy retrieval, separate from FreshMart's operational tables."""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sqlite3

from src.database import (
    APPLICATION_ID,
    ROOT,
    SCHEMA_VERSION,
    connection,
    database_path,
)

LOGGER = logging.getLogger(__name__)
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from get "
    "had has have how i if in is it its me my of on or our should some than "
    "that the their them there these they this to us was we were what when "
    "where which who will with would you your yesterday freshmart s policy "
    "policies document documents issue covered not".split()
)
SYNONYM_GROUPS = (
    frozenset("mould mouldy mold moldy spoiled spoilt spoilage rotten quality freshness".split()),
    frozenset("refund refunds return returns exchange".split()),
    frozenset("substitution substitutions substitute substituted replacement".split()),
    frozenset("missing absent undelivered".split()),
    frozenset("bought buy purchased purchase".split()),
    frozenset("complaint complaints report".split()),
    frozenset("guarantee guaranteed guarantees promise promises outcomes".split()),
)

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS policy_passages_fts USING fts5(
    passage_id UNINDEXED,
    source_id UNINDEXED,
    title,
    version UNINDEXED,
    effective_date UNINDEXED,
    source_path UNINDEXED,
    preamble UNINDEXED,
    section,
    content,
    tokenize = 'porter unicode61'
)
"""


class PolicySearchError(RuntimeError):
    """An explicit local indexing or retrieval failure."""


@dataclass(frozen=True)
class PolicyPassage:
    passage_id: str
    source_id: str
    title: str
    version: str
    effective_date: str
    source_path: str
    preamble: str
    section: str
    content: str
    rank: float = 0.0


def read_policy_passages(directory: Path = KNOWLEDGE_DIR) -> list[PolicyPassage]:
    """Keep original section text and the full introductory source metadata."""
    passages = []
    identifiers = set()
    try:
        paths = sorted(directory.glob("*.md"))
        if not paths:
            raise PolicySearchError("No Markdown policy documents were found.")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            sections = re.split(r"(?m)(?=^## )", text)
            preamble = sections[0]
            title_match = re.search(r"(?m)^# (.+)$", preamble)
            metadata = {}
            for label in ("Source ID", "Version", "Effective date"):
                match = re.search(rf"(?m)^{label}: (.+)$", preamble)
                if not match:
                    raise PolicySearchError(f"Policy metadata is missing in {path.name}.")
                metadata[label] = match.group(1).strip()
            if not title_match or not re.fullmatch(
                r"[A-Z0-9-]+", metadata["Source ID"]
            ):
                raise PolicySearchError(f"Invalid policy identity in {path.name}.")
            if metadata["Source ID"] in identifiers:
                raise PolicySearchError("Policy source identifiers must be unique.")
            identifiers.add(metadata["Source ID"])
            source_path = path.relative_to(ROOT).as_posix() if path.is_relative_to(
                ROOT
            ) else path.name
            for ordinal, content in enumerate(sections[1:] or [preamble], start=1):
                section = content.splitlines()[0].removeprefix("## ").strip()
                passages.append(PolicyPassage(
                    passage_id=f'{metadata["Source ID"]}#section-{ordinal}',
                    source_id=metadata["Source ID"],
                    title=title_match.group(1),
                    version=metadata["Version"],
                    effective_date=metadata["Effective date"],
                    source_path=source_path,
                    preamble=preamble,
                    section=section,
                    content=content,
                ))
    except (OSError, UnicodeError) as exc:
        LOGGER.error("Policy reading failed: type=%s", type(exc).__name__)
        raise PolicySearchError("Unable to read the local policy documents.") from exc
    return passages


def require_database(path: str | Path | None) -> Path:
    target = database_path(path)
    if not target.is_file():
        raise PolicySearchError(
            "The demo database is missing. Run python -m scripts.generate_data first."
        )
    return target


def check_database(conn: sqlite3.Connection) -> None:
    if (
        conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
        or conn.execute("PRAGMA user_version").fetchone()[0] not in (1, SCHEMA_VERSION)
    ):
        raise PolicySearchError("Policy search requires the existing FreshMart demo database.")
    existing = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", ("policy_passages_fts",)
    ).fetchone()
    if existing and "using fts5" not in (existing[0] or "").lower():
        raise PolicySearchError("The policy index name is already used by another table.")


def index_policies(
    path: str | Path | None = None,
    *,
    directory: Path = KNOWLEDGE_DIR,
) -> int:
    """Atomically refresh only the policy FTS table, never the retail records."""
    target = require_database(path)
    passages = read_policy_passages(directory)
    try:
        with connection(target) as conn:
            conn.execute("BEGIN IMMEDIATE")
            check_database(conn)
            conn.execute(FTS_SCHEMA)
            conn.execute("DELETE FROM policy_passages_fts")
            conn.executemany(
                """INSERT INTO policy_passages_fts (
                    passage_id, source_id, title, version, effective_date,
                    source_path, preamble, section, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        p.passage_id, p.source_id, p.title, p.version,
                        p.effective_date, p.source_path, p.preamble, p.section, p.content,
                    )
                    for p in passages
                ],
            )
    except sqlite3.Error as exc:
        LOGGER.error("Policy indexing failed: type=%s", type(exc).__name__)
        raise PolicySearchError(
            "Unable to update the policy index. Check database access and FTS5 support."
        ) from exc
    LOGGER.info("Policy index refreshed: passages=%d", len(passages))
    return len(passages)


def keyword_query(question: str) -> str:
    """Quote tokens instead of accepting user-supplied FTS operators."""
    tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - STOP_WORDS
    expanded = set(sorted(tokens)[:40])
    for group in SYNONYM_GROUPS:
        if expanded & group:
            expanded.update(group)
    return " OR ".join(f'"{token}"' for token in sorted(expanded))


def retrieve_passages(
    question: str,
    path: str | Path | None = None,
    *,
    limit: int = 6,
) -> tuple[PolicyPassage, ...]:
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        raise ValueError("Question must contain between 1 and 2,000 characters.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 12:
        raise ValueError("Retrieval limit must be an integer between 1 and 12.")
    target = require_database(path)
    query = keyword_query(question)
    try:
        with connection(target) as conn:
            check_database(conn)
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", ("policy_passages_fts",)
            ).fetchone()
            if not exists:
                raise PolicySearchError(
                    "The policy index is missing. Run index_policies() or "
                    "python scripts/test_rag.py to build it."
                )
            if not query:
                return ()
            rows = conn.execute(
                """SELECT *, bm25(
                       policy_passages_fts, 0, 0, 2, 0, 0, 0, 0, 3, 1
                   ) AS rank
                   FROM policy_passages_fts
                   WHERE policy_passages_fts MATCH ?
                   ORDER BY rank, passage_id LIMIT ?""",
                (query, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        LOGGER.error("Policy retrieval failed: type=%s", type(exc).__name__)
        raise PolicySearchError("Unable to search the policy index.") from exc
    LOGGER.info("Policy retrieval completed: passages=%d", len(rows))
    return tuple(PolicyPassage(**dict(row)) for row in rows)
