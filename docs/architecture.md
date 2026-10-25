# FreshMart AI architecture

This project uses synthetic retail data and fictional company policies for
demonstration purposes.

## Implemented: Phase 1

```text
scripts.generate_data / scripts.reset_demo
                  |
                  v
           src.database
                  |
                  v
        data/freshmart.db (ignored)

data/knowledge/*.md          tests/test_database.py
five fictional policies     SQLite + generator + reset validation
```

The runtime uses the Python standard library. Pytest is the only Phase 1
dependency. There are no Azure requests or credentials in this phase.

### Storage

| Table | Primary key | Relationships / validation |
|---|---|---|
| stores | StoreID | Unique names; NSW, VIC, QLD only |
| products | ProductID | Unique product names |
| sales | TransactionID | Store/product foreign keys; ISO date; positive integer quantity and price cents |
| quality_inspections | InspectionID | Store/product foreign keys; ISO date; positive inspected weight; valid passed weight and score |
| complaints | CaseID | Store/product foreign keys; ISO purchase date; UTC creation timestamp; description length; constrained status and proof indicator |

`UnitPriceAUD` and `NetRevenueAUD` are generated SQLite columns derived from
integer `UnitPriceCents`; callers cannot independently overwrite the totals.
Aggregate cents before conversion to AUD in later analytics to avoid accumulating
floating-point rounding errors.

`SubmissionToken` is nullable and unique when supplied. Phase 3 uses it for
idempotent confirmed complaint submission. Historical generator records retain
their original statuses. Schema version 2 adds `Open` to the allowed statuses,
with no new columns; application submissions explicitly use it. Initialization
transactionally copies the version 1 complaints table into the updated schema,
preserves every row, recreates its filter index, and leaves the FTS index and
other tables intact. Migration failure rolls back the entire change.
The legacy SQL default remains `Submitted` for direct inserts that omit status;
it is not the application's initial status. Historical seed records keep
`Submitted`/`Under review`. The current complaint procedure specifies `Open`,
matching the explicit application insert, without a further schema migration.

All connections enable foreign keys, commit on success, roll back on exceptions,
and close on exit. Filter indexes cover store, product, and date. An application
identifier and schema version prevent the generator/reset command from adopting
an existing unrelated SQLite database. This is an accidental-overwrite safeguard,
not an access-control boundary.

### Synthetic scenario

The generator uses a local seeded random-number generator and a 90-day inclusive
window. An explicit end date plus seed reproduces identical table contents.
The default end date is the machine's local date, so a demo can use "yesterday".
All complaint timestamps use UTC.

There are 2,000 sales, 150 inspections, and 25 historical complaints. Each
store/product pair has at least six inspections. Chatswood strawberries have
12 inspections with deliberately lower passed-weight ratios and 12 recent
complaints. Other pairs have higher inspection pass rates and fewer complaints.
Sales are generated independently of inspection quality: the data does not
establish that quality causes a revenue change.

Reset runs deletes and inserts inside one transaction, child tables first. It
does not delete the SQLite file, touch policy documents, or remove unrelated
tables/files. It replaces newly submitted demo cases as well as historical
records, and requires `--yes` at the command line.

## Implemented: Phase 2A/2B

```text
data/knowledge/*.md (fictional policy sources)
              |
              v
    src.policy_search.index_policies
              |
              v
    SQLite policy_passages_fts (separate from operational tables)
              ^
              | keyword synonyms + BM25
Customer question -> retrieve_passages -> up to six original sections
                                               |
                                               v
                            src.rag_service.generate_answer
                                               |
                              existing DeepSeek-V4-Flash in Foundry
                                               |
                              structured answer + passage identifiers
                                               |
                             Python resolves original excerpt identifiers
                                                |
                             same model: bounded grounding + status check
                                               |
                                answer + actual source references
```

`src/foundry.py` shares the existing environment configuration and output-tag
guard with the Phase 2A connection script. It does not provision anything.
Inference uses the officially documented OpenAI-compatible v1 Chat Completions
endpoint, the configured resource API key, and only the existing
`DeepSeek-V4-Flash` deployment. Cited answers use two calls to that same deployment:
generation, then bounded verification. Each request uses JSON output, a 60-second
timeout, and no automatic retries. Reasoning fields and raw service bodies are
not exposed. Available usage totals aggregate both calls.

The FTS5 index is a single virtual table plus SQLite-managed shadow tables.
It stores original section text and complete introductory metadata, with stable
section identifiers within each document version. A transactional refresh
replaces only index contents; the five retail tables and policy files are not
modified. `index_policies()` must be called after document changes; the live
script refreshes it on each run. No operational schema version change is needed.

Search uses a Porter tokenizer, quoted tokens, explicit synonym groups, and
BM25 ranking weighted toward titles and section headings. A maximum of six
passages is supplied to inference. Keyword relevance is not a semantic
answerability guarantee: loosely related evidence still needs cautious model
interpretation.

No lexical matches produce an explicit local unsupported result, with no model
call or claimed sources. With matches, inference classifies evidence as
`supported`, `partial`, or `unsupported`. An explicit policy denial can support
a complete negative answer; partial means a requested outcome/right is still
unresolved despite useful guidance. Review-request permission is separate from
the final refund decision. The existing `insufficient_evidence` field is derived
from status (`False` only for supported), preserving the public result fields.

Original statements inside each retrieved section receive transient excerpt IDs.
The model links customer-facing claims to those IDs, not to model-written
quotations. Python resolves IDs to original policy text and assembles the answer
from the claims and an explicit limitation, so a separate free-text answer cannot
append uncited material. It rejects unknown IDs, model-supplied quotations, and
uncited claims rather than supplying fallback citations.
Metadata comes from SQLite. Identical citations are deduplicated, and the
display groups documents into readable section summaries while keeping distinct
supporting sections, original IDs, paths, and quotations.

Both inference prompts share trusted application context identifying the existing
**Quality complaint** tab. They allow brief empathy and form navigation alongside
policy-supported guidance; these are not new policy evidence. All substantive
policy assertions still require retrieved excerpts and grounding verification.
The form is only for fresh produce quality concerns, not online-order actions.
The assistant receives no complaint-service receipt and cannot claim a write or
status change succeeded, invent a Case ID, or submit on the customer's behalf.

The same deployment then checks whether the selected original quotes support
every claim and classifies the evidence independently of the generator's draft
status. This distinguishes fully supported negative/process answers from an
unresolved personal outcome. The verifier cannot add claims, quotes, or source
IDs; failed verification or service errors never return an unverified draft.
An irrelevant-evidence verdict returns unsupported with no claimed sources.
It can omit an unasked outcome limitation from an otherwise supported answer.

This is **not** a proof of semantic entailment: the same model can make correlated
errors. Production still needs grounding evaluation and review.
User questions and retrieved text are untrusted
lower-priority data. An output guard rejects the known attack marker and common
affirmative refund approval/payment wording, common unconfirmed complaint-action
claims, and Case ID-shaped output. This is not a general-purpose
language safety verifier. Whole-request safety refusals must be explicitly
unsupported and uncited; live checks report them separately from policy answers.

Model errors, timeouts, invalid JSON, incomplete answers, unknown references,
and sensitive output produce explicit errors, not fabricated fallback answers.
Default logs contain counts and status, not questions, credentials, or raw model
output. Explicit local `--debug` output logs only the redacted final-answer field,
source/excerpt counts, draft and verified safety/status flags, and the validated
answer before assertions.
Hidden reasoning and raw SDK responses are never dumped. Neither retrieval nor
inference can write a complaint or approve a refund.

The explicit CLI `python scripts/test_rag.py` exercises policy questions and a
user-message injection scenario using real Foundry inference.
`--document-injection` adds a live attack in a copy of a retrieved passage,
without changing the corpus or database. Ordinary Pytest runs use only temporary
databases and labelled SDK transport fixtures, including malicious model-output
cases that must be rejected. Unit fixtures do not establish model resistance;
live results are reported separately.

## Implemented: Phase 3

```text
Streamlit customer chat -> local FTS5 retrieval -> Foundry -> answer + sources

Streamlit complaint form -> review + confirmation -> src.complaints -> SQLite
```

The app loads `.env` without overriding exported environment variables. A
database must already exist; startup initializes/migrates it and refreshes policy
passages once per session. Customer questions call the existing RAG service.
Only its validated answers and actual cited source metadata/excerpts are shown.
Errors are displayed explicitly without replacement answers. Chat history is
kept in Streamlit session state; prior messages are not sent to inference as
conversation context. Hidden reasoning is never displayed.

`src.complaints` has no dependency on inference. It reads store/product choices
from SQLite, validates identifiers, date, description, and the nullable proof
indicator, and requires explicit confirmation before writing. Dates are bounded
by the UTC creation date to match the existing database constraint. The UI holds
an immutable reviewed draft separately from editable widget values.

Each reviewed draft receives a session-held submission token. `BEGIN IMMEDIATE`
serializes the token lookup and insert; the database also enforces token
uniqueness. Replaying identical confirmed details returns the existing receipt;
reusing a token with changed details fails explicitly. A UUID-based Case ID and
`Open` status are committed before the actual receipt is shown. Failures roll
back and preserve the token for retry. Reading through another connection sees
the saved case immediately. Logs contain case IDs/status, never complaint text.

Editing requires a fresh review; starting another complaint creates a new
request. Tokens prevent rerun/retry duplicates, not deliberately repeated
reports or submissions from new sessions. Session history and review state are
not durable across server restarts or a new browser session.

Only deterministic Python application code creates complaints. No component
approves refunds or initiates payments. A complaint requests review only.
This local demo is not an authenticated public multi-user deployment. Proof of
purchase is a Yes/No/unspecified indicator; there is no document upload.

## Implemented: Phase 4

```text
Store Manager Dashboard tab -> src.analytics -> SQLite snapshot
                                            -> Pandas metrics -> Plotly charts
```

`src.analytics` issues parameterized read queries inside a single transaction so
the revenue, inspection, and complaint views come from one consistent snapshot.
Fact tables are aggregated separately to avoid multiplying revenue or inspected
weights through many-to-many joins. Integer revenue cents are summed before
conversion to AUD. No database schema changes are required for Phase 4.

State/store/product filters govern KPIs, revenue trend, and recent complaints.
The freshness comparison and peer observations intentionally retain other stores
in the selected state/product scope; the selected store is highlighted. A state
change clears an incompatible store selection. Missing inspections remain
unavailable rather than being counted as zero freshness.

Month-over-month revenue growth compares equal day counts in the latest
database-wide sales month and the previous month, bounded by the shorter month.
The first day of the previous month must lie inside observed sales history.
Missing or zero prior revenue yields `N/A`; positive prior revenue and zero
current revenue yield −100%. Exact comparison dates are displayed; all other
metrics cover all recorded dates. Observed sales dates are a POC coverage
assumption, not verification of production ingestion completeness.

Every rerun reads uncached data. **Refresh dashboard** triggers a rerun, making
new cases visible across sessions. The complaint KPI counts all matching cases
and statuses; the table is limited to the latest 50, sorted by UTC creation time.
Freshness and complaint comparisons use actual aggregate values, not hardcoded
store/product conclusions or model inference. Counts are not sales-adjusted,
and observations explicitly avoid causal or refund-approval claims.

Azure AI Search and Blob Storage are possible future enterprise replacements
for local indexing, not implemented services. Production authentication could
evolve to Entra ID rather than the local POC's API key; this dashboard has no
separate manager authorization boundary.

## Portfolio and enterprise documentation

The [enterprise proposal](microsoft_enterprise_solution.md) separates implemented
POC components from proposed Copilot Studio, Power Automate, Azure SQL, Power BI,
Entra ID, and Azure monitoring/hosting services. None of those enterprise
extensions has been implemented by writing the proposal.

The [interview script](interview_demo_script.md) follows the customer journey in
five minutes. The [validation checklist](validation.md) records actual checks,
live evidence, and limitations rather than treating mocked unit responses as
successful Azure integration.
