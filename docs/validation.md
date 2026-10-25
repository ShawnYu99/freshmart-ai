# Final validation checklist

Portfolio/browser validation was recorded **2026-10-25**, before the final
consistency improvements below. All retail information and policies are
synthetic. These results describe observed checks, not a
production certification or a guarantee about future model responses.

## Final consistency review: deterministic validation

The final review aligned the complaint policy with application status `Open`,
corrected both RAG prompts to acknowledge the existing complaint form, added
guards against common unconfirmed case-action claims, and simplified source
summaries without removing original evidence metadata.

| Check | Actual result |
|---|---|
| `python -m pytest -q` | **303 passed, 0 failed, 0 skipped** in 4.25 seconds |
| `python -m pip check` | No broken requirements found |
| `git diff --check` | Passed |
| Existing policy index refreshed | All **17 passages** match current source files; old initial-status wording removed |
| Policy provenance | Original document/section IDs, titles, versions, dates, and paths retained |
| Operational preservation | All six stores, four products, 2,000 sales lines, 150 inspections, and **27 existing complaints**, including Case IDs, unchanged |
| Database schema and integrity | Schema/version unchanged; integrity and foreign-key checks passed |
| Dashboard preservation | All four KPIs unchanged for all-store, NSW, and Chatswood/Strawberries scopes |
| Credential checks | `.env` ignored/untracked, example key a placeholder, configured key absent from publishable files |

Tests include supported/partial/unsupported RAG results, user/document injection,
invalid citations, model failures, grounded form guidance without writes,
unconfirmed-action rejection, grouped source rendering, stale-index replacement,
confirmed `Open` complaints, retry safety, filters, and dashboard calculations.
The legacy SQL default and historical `Submitted` records were deliberately
preserved; the complaint service explicitly inserts `Open`.

**No paid live Foundry calls were executed for these improvements.** Live
checks are separate manual commands below, not skipped Pytest cases. The
recorded browser evidence and screenshots do not validate the updated prompt's
exact wording or the latest source-summary presentation. No Azure resources,
additional models, application dependencies, or enterprise services were added.

## Recorded portfolio/browser acceptance checklist

- [x] **Genuine deployed RAG:** the running Streamlit application answered the
  Chatswood mouldy-strawberry question through the configured
  `DeepSeek-V4-Flash` deployment. It returned `partial` evidence with original
  Refund and Exchange Policy references, distinguishing review from approval.
  No mocked response was used for this browser run.
- [x] **Successful complaint persistence:** browser review alone inserted no
  record. Explicit confirmation created the actual case
  `FM-B655AE727FA8416F8014954CFA7A01C3`, status `Open`, for Chatswood/Strawberries.
  A separate SQLite connection verified it. The isolated database's complaint
  count increased from 25 to 26; refresh did not create a second copy.
- [x] **Correct dashboard KPIs:** under NSW / Chatswood / Strawberries,
  independent SQL queries reconciled the displayed net revenue `A$1,791.40`,
  freshness `61.1%`, and complaint count `13`. Comparable-period revenue was
  61,930 cents for October 1-25 versus 44,390 cents for September 1-25,
  reconciling to displayed growth `+39.5%`.
- [x] **New case visible to management:** the filtered browser dashboard showed
  the newly submitted case in its recent complaints table. Both Plotly charts
  rendered, and another refresh retained the correct count.
- [x] **Local setup verified:** a newly created virtual environment installed
  `requirements.txt`, generated an isolated seed-42 dataset, passed **276
  deterministic tests**, and launched Streamlit. HTTP health returned `ok`.
  Browser interactions used that clean installation.
- [x] **Credential hygiene checked:** `.env` is ignored and not tracked;
  `.env.example` uses a placeholder key. The configured API key was absent from
  Git-publishable files, including the generated screenshots. Browser text and
  inspected captures did not expose credentials or hidden reasoning.
- [x] **Dependency compatibility:** `python -m pip check` reported no broken
  requirements. All six declared dependencies have source/test imports; none
  was removed simply to make the list shorter.
- [x] **Data integrity:** the validation database passed SQLite integrity and
  foreign-key checks. The original application database and its existing cases
  were not reset or modified during the browser validation.

The screenshot Case ID belongs to the **isolated validation database**, not the
user's original demo database. Generated databases are not included in Git.
The screenshots are recorded evidence, not a substitute for live model inference.

## Original portfolio review scope and changes

Reviewed the Streamlit composition, model configuration and output handling,
FTS retrieval, schema/migration, generator/reset, complaint service, and analytics.
The implementation keeps model inference separate from deterministic writes
and calculations.

Record values are bound SQL parameters; dynamic query fragments and table names
are application constants. Complaint validation checks references, dates,
description bounds, proof indicator, and explicit confirmation. A database
transaction and unique submission token protect retries and concurrent submissions.
Analytics aggregates fact tables independently, sums money in integer cents,
weights freshness by kilograms, and explicitly handles missing comparison data.
Model failures are surfaced without fabricated fallback answers.

One input boundary bug was fixed: whitespace-only chat could raise an uncaught
retrieval validation error. The UI now validates stripped question length before
inference, with three regression cases for invalid input. SQLite journal/sidecar
ignore patterns were also extended to `.sqlite` and `.sqlite3` paths.
No application features, models, or enterprise services were added.

## Reproduce checks

From an activated environment after installing `requirements.txt`:

```bash
python -m pytest -q
python -m pip check
git check-ignore .env
git ls-files -- .env
```

The last command should produce no output. Inspect `.env.example` and staged
changes before publishing; do not print or copy your real key into the README,
an issue, screenshot, or terminal transcript.

Explicit live checks, requiring valid Azure configuration and incurring usage:

```bash
python scripts/test_foundry.py
python scripts/test_rag.py --document-injection
python -m streamlit run app.py
```

For the UI check, ask the genuine policy question, review and confirm a synthetic
complaint, record its actual Case ID, and select NSW / Chatswood / Strawberries
in the dashboard. Refresh and reconcile it with SQLite.

## Interpretation of the recorded portfolio run

- The 276 tests are offline tests, including mocked inference in unit/UI tests.
  They are distinct from the genuine browser inference described above.
- This final browser run exercised the core strawberry scenario. It did not
  rerun every optional CLI live scenario or establish universal injection
  resistance. The broader live command is provided explicitly above.
- Clean installation was validated on macOS with Python 3.14.5 and SQLite 3.50.4.
  Windows, all supported Python versions, remote cloning, GitHub publication,
  and cloud deployment were not independently exercised in this final pass.
- Browser automation was used only for validation and screenshot capture;
  Playwright is not an application runtime dependency.
- The credential check covers the configured key and current publishable files,
  not every possible secret format, external copies, or historical credential
  exposure. Keep secret scanning and credential rotation in the publishing process.
- No load test, production authentication, independent grounding benchmark,
  real-business impact measurement, or security certification is claimed.
- **Power BI, Copilot Studio, Power Automate, Azure AI Search, Azure SQL,
  Entra application access, and Azure Monitor remain proposals.**
