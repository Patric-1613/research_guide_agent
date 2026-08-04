# Backend backlog

**Status: capture/tracking document. No implementation happens here.**
Companion to `specs/legacy-cleanup-audit.md` (Phase 18A). Written against
tag `standardized-single-user-project`. Purpose: give future backend bug
fixes and feature work a single place to land, without re-litigating the
already-closed standardization arc (Phases 0–17) or the deferred
production-readiness arc (`specs/production-readiness-roadmap.md`,
Phase 18).

## 1. Known bugs

No currently-open bug is documented anywhere in this repo's docs, specs,
or test files as of this audit — not invented here, and not claimed
where none exists. (The mentor-review bugs from 2026-07-17 were all
found already fixed against the current codebase — see the Obsidian
vault's `Mentor-Feedback.md` for that full accounting; nothing from that
review carries forward as open.)

### Grouped report citation markers leaked into report body
- **Reported**: 2026-08-03, via a live report screenshot showing raw
  `[Paper 6, Paper 8]` text still in the rendered Methodology Landscape
  section.
- **Symptom**: when the model backed one claim with more than one
  source, it sometimes wrote a single bundled bracket (`[Paper 6, Paper
  8]`, or mixed kinds like `[Paper 3, Web 1]`) instead of one marker per
  citation. `research_agent/report.py`'s citation-marker regex only
  matched a bracket holding exactly one citation, so a bundled bracket
  didn't match at all and passed straight through to the rendered report
  as raw, unresolved text instead of a numbered citation.
- **Root cause**: `_SECTION_CITATION_MARKER_RE = re.compile(r"\[(Paper|
  Web) (\d+)\]")` structurally could not match more than one entry per
  bracket.
- **Fix**: generalized `_SECTION_CITATION_MARKER_RE` to match one-or-more
  comma-separated entries per bracket, extracted via a new
  `_SECTION_CITATION_MARKER_ENTRY_RE`; `_densify_section_markers` and
  `_build_references_and_renumber` both now resolve every entry in a
  bracket independently, emitting adjacent single-number brackets (e.g.
  `[6][8]`, not `[6, 8]`) since the frontend's marker renderer only
  recognizes single-number brackets. Invalid entries are dropped
  individually; an all-invalid group is stripped entirely. Same-source
  numbering still goes through the existing global `_ReferenceAssigner`
  registry, unchanged. Deterministic post-processing only — no prompt
  change. See `docs/architecture.md`'s "Report citation marker fix —
  grouped inline citations" section for the full design record.
- **Location**: `research_agent/report.py` (`_SECTION_CITATION_MARKER_RE`,
  `_densify_section_markers`, `_build_references_and_renumber`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-03). Commit `4e14024`.

### Revoked chat web source resurrected during report regeneration
- **Reported**: 2026-08-03. A web reference from a chat exchange was
  added to the report; the chat exchange was then deleted; a
  regeneration appeared to drop the reference correctly, but a SECOND
  regeneration brought it back into the report body and References list
  even though it was no longer a legitimate source.
- **Symptom**: `regenerate_report_with_new_sources` (the whole-pool
  path — both the Report tab's "Regenerate" button and chat's "update
  report with new sources" accept flow) always offered the model
  `session.web_articles_added` unfiltered. Phase B's delete/edit pruning
  only ever touched `session.report_approved_web_article_urls`, which
  this whole-pool path never consults — so a revoked source stayed
  structurally citable forever, and whether the model re-cited it on
  any given regeneration was pure chance.
- **Root cause**: no persistent record existed of "this URL lost its
  only live chat backing." An initial fix attempt that inferred
  revocation from whatever the *immediately prior* report happened to
  cite was insufficient — it self-defeated after one clean regeneration
  (the prior report stopped mentioning the excluded URL, so the next
  call's inference saw "nothing to revoke" and re-offered it).
- **Fix**: added a new persistent session field, `revoked_web_article_
  urls`. `curation_chat.py`'s `delete_chat_exchanges`/`edit_chat_
  exchange` snapshot live chat-backed URLs before/after each mutation
  and record any that lost backing (`live_cited_web_article_urls`,
  `_sync_revoked_web_article_urls`); `_accept_web_offer` un-revokes a
  URL the moment it's cited again. `report.py`'s shared regeneration
  body excludes every `revoked_web_article_urls` entry from the
  candidate web-article pool before building the schema/prompt, for
  both regeneration paths. See `docs/architecture.md`'s "Revoked web
  citation resurrection fix" section for the full design record,
  including the three-distinct-web-source-sets explanation.
- **Location**: `research_agent/report.py`
  (`_regenerate_report_sections_with_sources`), `research_agent/
  curation_chat.py` (`live_cited_web_article_urls`, `_sync_revoked_web_
  article_urls`, `delete_chat_exchanges`, `edit_chat_exchange`,
  `_accept_web_offer`), `research_agent/query_expansion.py` (new
  `revoked_web_article_urls` field), `research_agent/curation_
  session.py` (serialize/deserialize).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-03). Commit `f826ad6`.

### Foundational report leaked raw paper IDs as citations
- **Reported**: 2026-08-04, via live report screenshots showing raw
  identifiers like `[2308.06821v1]` and
  `[abd1c342495432171beb7ca8fd9551ef13cbd0ff]` in a Foundational-
  template report's rendered body instead of numbered citations.
- **Symptom**: the model sometimes ignored the instructed `[Paper N]`/
  `[Web N]` marker format entirely and cited a source using its own real
  identifier (arXiv id, Semantic Scholar-style hash id) instead —
  observed specifically in Foundational-template output. The existing
  citation-marker parser only recognized the exact `Paper N`/`Web N`
  shape, so a raw-identifier bracket was structurally invisible to it:
  never converted, never stripped, just leaked through as raw,
  unresolved text.
- **Root cause**: `_SECTION_CITATION_MARKER_RE` (and the whole
  downstream renumbering pipeline) had no way to recognize a bracket
  containing anything other than the literal `Paper N`/`Web N` shape.
- **Fix**: two-part. (1) Prompt reinforcement — the shared marker-
  instruction paragraph now explicitly forbids citing via paper_id/DOI/
  arXiv id/Semantic Scholar id/URL/title inside a bracket, and the
  Foundational template's own depth-guidance got an extra, template-
  specific reminder, since that's where the bug was observed. (2) A new
  deterministic backstop, `_resolve_raw_source_id_markers`, run before
  densify/the regular marker-resolve pass: a bracket with no internal
  whitespace, not already a valid `Paper N`/`Web N` marker, is resolved
  to the correct global reference number ONLY on an exact match against
  that section's own cited paper_ids/web urls, through the same shared
  `_ReferenceAssigner` the regular pipeline uses (so a raw-id citation
  and a correct-marker citation for the same source collapse to one
  number). An unrecognized raw id is stripped, not guessed at, same
  policy as an out-of-range `[Paper N]` marker. A bare digit string is
  never treated as a raw-id candidate, guarding against ever
  misinterpreting an already-final `[N]` marker. See
  `docs/architecture.md`'s "Raw source-id citation hardening fix"
  section for the full design record.
- **Location**: `research_agent/report.py`
  (`_build_report_system_prompt`, `_TEMPLATE_DEPTH_GUIDANCE`,
  `_RAW_SOURCE_ID_MARKER_RE`, `_resolve_raw_source_id_markers`,
  `_build_references_and_renumber`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-04). Commit `0189c2f`.

Placeholders below, ready for real entries:

### [Placeholder — bug 1]
- **Reported**:
- **Symptom**:
- **Location** (file:line if known):
- **Priority**:
- **Status**: Open

### [Placeholder — bug 2]
- **Reported**:
- **Symptom**:
- **Location**:
- **Priority**:
- **Status**: Open

*(Add more as needed — same shape.)*

## 2. Backend feature ideas

Two are already documented in this repo (`README.md`'s "Known
limitations" section) and listed here with their source reference, not
invented:

### Author-name parsing for multi-word surnames
- **Source**: `README.md:766` — "Author-name parsing for APA/BibTeX
  (`"First Last"` → `"Last, F."`) is a heuristic — it will mis-format
  multi-word surnames."
- **Where it lives**: `research_agent/citations.py`.
- **Priority**: Unset — needs your call.
- **Status**: Open, not started.

### PDF full-text ingestion
- **Source**: `README.md:761` — "Abstracts only — no PDF full-text
  ingestion (out of scope for v1)."
- **Scope note**: This is a real architectural expansion, not a small
  fix — would touch `ingestion.py` (or a new module), `embeddings.py`
  (what gets chunked/embedded), and likely `summarize.py`/`qa.py`'s
  grounding logic (currently grounded in abstracts specifically).
  Worth its own design pass before implementation, not a drop-in.
- **Priority**: Unset — needs your call.
- **Status**: Open, not started.

### Chat message actions + report inclusion
- **Phases 1–5 (persisted web-answer metadata, message menu/select
  mode, delete exchanges, add web-backed sources to the report, edit
  user question): Done — the whole planned arc is complete.** See
  `docs/architecture.md`'s "Chat feature: message actions and report
  inclusion (Phases 1–5) — complete" section for the full record — data
  model (`exchange_id`/`used_web_search`/`cited_web_articles`/
  `added_to_report`/`report_approved_web_article_urls`), the
  `qa.capped_history()` sanitization boundary, the selective-vs-
  whole-pool report regeneration split, and edit's truncate-and-
  regenerate behavior (new `exchange_id` per edit, `report_possibly_
  stale` reused as-is from Phase 3 — that open question from the
  previous note is now resolved). Final validation: backend 404 passed,
  frontend 154 passed, build clean.
- **Status**: Closed. Follow-up ideas below are optional, not scheduled.

### Chat feature follow-ups (optional, not scheduled)
- **Inline edit UI instead of `window.prompt`.** Phase 5 used the same
  native-dialog minimalism as Phase 3's `window.confirm` on purpose — a
  real inline-editable text field is a plausible future upgrade, not a
  correctness gap.
- **Stale-report remediation/regeneration UX.** `report_possibly_stale`
  (surfaced by both delete and edit) is only ever a warning today — no
  one-click "fix it now" action exists; the user has to know to
  manually regenerate.
- ~~Pruning `report_approved_web_article_urls` after delete/edit~~ —
  **Done (Phase B, 2026-08-03).** See `docs/architecture.md`'s "Phase B
  — approved report-source pruning after delete/edit" section and the
  "Phase B: report-source revocation after delete/edit" entry below.
- **A red-team/evaluation suite for chat + report behavior** —
  nothing in this arc has adversarial/eval-style coverage the way
  `report.py`'s citation-grounding does (`tests/test_report_
  grounding.py`) or the original pipeline's RAGAS harness does
  (`docs/evaluation.md`).
- **Priority**: Unset — needs your call, on any/all of the above.

### Phase B: report-source revocation after delete/edit
- **Goal**: define desired behavior: remove approved URLs only when no
  remaining added-to-report exchange cites them; keep existing report
  marked stale until regenerated.
- **Why it matters**: raised during chat-ux-polish Phase A (frontend-only
  dialog/badge/notice polish) — confirmed deleting a chat exchange did
  NOT remove anything from `web_articles_added` or `report_approved_
  web_article_urls`, so a subsequent regeneration (via either the
  selective or whole-pool path) would still very likely include the same
  reference even after the exchange that cited it was gone. Phase A
  deliberately left this semantic untouched; this was the tracking note
  for the follow-up.
- **Implementation**: `research_agent/curation_chat.py` gained
  `approved_web_article_urls_from_added_to_report_entries()` (pure) and
  `prune_report_approved_web_article_urls()` (mutator, recomputes the
  approved set from scratch as the union of `cited_web_articles` URLs
  over currently `added_to_report=True` assistant entries).
  `delete_chat_exchanges()`/`edit_chat_exchange()` call the latter only
  when `report_possibly_stale` is true. `web_articles_added` and
  `session.report` are untouched; no auto-regeneration; no endpoint,
  schema, or frontend changes. See `docs/architecture.md`'s "Phase B —
  approved report-source pruning after delete/edit" section for the full
  design record.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-03).

### Phase C: QA web-article relevance gate
- **Goal**: stop the accumulated `web_articles_added` pool from being
  reused blindly on later, unrelated curation-chat turns.
- **Why it matters**: reported as two symptoms of the same root cause —
  (1) nearly every later assistant answer got tagged `used_web_search=
  true` (blue web badge appeared almost everywhere) once one web search
  had ever been accepted in a session, and (2) a genuinely new/unrelated
  follow-up question stopped triggering a fresh web-search offer, because
  the model could always find *something* tangential in the stale pool
  to cite, propping up a false `answerable=true`.
- **Implementation**: `research_agent/qa.py`'s LangGraph gained a new
  node, `filter_web_relevance`, inserted between `retrieve` and
  `route_retrieved`. Cosine-similarity filter (`_filter_relevant_web_
  articles()`) reusing the existing `_embed_with_cache`/`_cosine_
  similarity` helpers — same embedding pathway `classify_message`'s
  non-substantive check already uses. Query = `standalone_query or
  question`; article text = `title + snippet`; threshold = `0.25`
  (documented starting point, not yet calibrated); fails open (keeps the
  unfiltered pool) on any embedding exception. `route_retrieved`'s own
  branching condition and `curation_chat.py`'s offer-and-decide flow are
  both unchanged. `used_web_search` needed no code changes — it derives
  correctly now as a side effect. See `docs/architecture.md`'s "Phase C —
  QA web-article relevance gate" section for the full design record.
- **Follow-ups not done in this phase** (see that same doc section for
  detail): the 0.25 threshold needs live calibration; `answerable` is
  still an LLM judgment call, not a hard rule; paper retrieval
  (`semantic_search`) still has no relevance threshold of its own,
  same as before this phase; `curation_chat.py`'s pending-web-offer flow
  was deliberately not restructured into LangGraph nodes.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-03).

### R2B.1: Analytical report prompt tuning and citation whitespace cleanup
- **Goal**: improve Analytical report body quality (R2B) before starting
  Foundational/Expert templates (R2C), so tuning happens once against a
  single template instead of being multiplied across several later.
- **Why it matters**: a real generated report showed uneven citation
  density (evidence-bearing claims without markers), awkward leftover
  spacing where an invalid marker was stripped (`"classification ,"`,
  `"training ."`), web sources cited for loosely-related adjacent
  context rather than direct evidence, "Contradictions & Open Debates"
  reading too narrow for paper sets with no direct contradictions (but
  real tensions/tradeoffs), Gap Analysis and Future Research Directions
  overlapping in content, and some sections reading generically.
- **Implementation**: prompt-only changes in `research_agent/report.py`
  — `_build_report_system_prompt` gained paragraphs/clauses for citation
  density, adjacent grouped-citation markers (`[Paper 2][Paper 5]`, not
  `[Paper 2, Paper 5]`), a stricter web-source-as-secondary-evidence
  constraint, and a generic-phrasing guard. Three `REPORT_SECTION_
  DEFINITIONS` descriptions (Contradictions & Open Debates, Gap
  Analysis, Future Research Directions, Thematic Findings) were
  decoupled from the legacy `FINDINGS_DESCRIPTION`/`LIMITATIONS_
  DESCRIPTION`/`FUTURE_SCOPE_DESCRIPTION` constants (which stay
  byte-identical for the untouched legacy path) and rewritten to be
  broader/more distinct/more synthesis-oriented. New deterministic
  `_cleanup_marker_stripped_whitespace()` collapses repeated spaces and
  removes a space before punctuation, applied only in `_build_
  references_and_renumber`'s fresh-report path, never in `derive_
  legacy_references`. No schema, section-key, section-title, endpoint,
  or frontend changes. See `docs/architecture.md`'s "R2B.1 — Analytical
  report prompt tuning and citation whitespace cleanup" section for the
  full design record.
- **Location**: `research_agent/report.py`
  (`_build_report_system_prompt`, `REPORT_SECTION_DEFINITIONS`,
  `_cleanup_marker_stripped_whitespace`, `_build_references_and_
  renumber`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-03). Commit `73e899b`.

### R2C: report templates / reader-depth modes
- **Goal**: let a user choose a reader-depth template (Foundational,
  Analytical, Expert) before generating or regenerating a report,
  without touching the section schema/layout.
- **Why it matters**: the tuned Analytical report (R2B/R2B.1) works well
  for a reader already comfortable with the topic, but is either too
  dense for a newcomer or not dense enough for an expert — a single
  fixed depth doesn't fit every reader.
- **Design decision**: all three templates share the exact same 8
  section keys/titles (Executive Summary, Introduction & Scope,
  Thematic Findings, Methodology Landscape, Contradictions & Open
  Debates, Gap Analysis, Future Research Directions, Conclusion) — only
  prompt instructions and word-budget guidance differ per template, kept
  deliberately low-risk over introducing per-template section layouts.
- **Implementation split across two chunks**:
  - **Chunk 1 (backend)**: `report_template` (`Literal["foundational",
    "analytical", "expert"]`) added to `research_agent/report.py`
    (`REPORT_TEMPLATES`, `_TEMPLATE_DEPTH_GUIDANCE`), stamped onto the
    report dict itself (the source of truth, not a session field), and
    exposed via `ReportOut.report_template` (defaults to `"analytical"`
    when absent — an old or otherwise template-less report). `POST
    /curation/{id}/report`'s optional `report_template` only matters on
    first generation (omitted → analytical); `POST /curation/{id}/report/
    regenerate`'s optional `report_template` omitted preserves the
    existing report's current template, an explicit value switches it.
    Chat's add-to-report regeneration has no `report_template` field on
    its request schema at all and always preserves the existing
    template. Also fixed a pre-existing `curation_session.py`
    serialization gap in this same chunk (`_serialize_report`/
    `_deserialize_report` only ever reconstructed the 3 legacy section
    keys on load, silently dropping the 5 non-legacy-mapped Analytical
    keys' `cited_papers`/`cited_web_articles`/`reference_numbers` on
    every save/load round trip) — fixed by iterating the full set of
    present section keys instead of a hardcoded 3-tuple.
  - **Chunk 2 (frontend)**: a compact segmented control
    (Foundational/Analytical/Expert) in `ReportModePanel`, initialized
    from the current report's `report_template` (or Analytical
    pre-generation), re-synced on report change; a small badge shows
    which template produced the current report; no confirmation dialog
    on switching. `curationApi.generateReport`/`regenerateReport` gained
    an optional `reportTemplate` parameter, threaded through
    `useCurationSession` and `CurationWorkspacePage` with no template
    state introduced at the page level.
  - See `docs/architecture.md`'s "R2C — report templates / reader-depth
    modes" section for the full design record.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-04). Commits `1020a02` (backend) and
  `68ea849` (frontend).

Placeholders below, ready for real entries:

### [Placeholder — feature idea 1]
- **Goal**:
- **Why it matters**:
- **Rough scope** (files likely touched):
- **Priority**:
- **Status**: Open

### [Placeholder — feature idea 2]
- **Goal**:
- **Why it matters**:
- **Rough scope**:
- **Priority**:
- **Status**: Open

*(Add more as needed — same shape.)*

## 3. Technical debt

Current single-user/backend debt only — scoped to what's actually true
today, cross-referenced to where each item is already tracked in detail:

- **Model-name constants not centralized.** 8 constants
  (`EMBEDDING_MODEL`, `SUMMARY_MODEL`, `REPORT_MODEL`, `AGENT_MODEL`,
  `TITLE_SUGGESTION_MODEL`, `CANONICALIZE_TOPIC_MODEL`, `CONDENSE_MODEL`,
  `ANSWER_MODEL`) across 6 files, several literal duplicates of the same
  string. Optional, not scheduled. See `docs/architecture.md`'s Phase 14
  section, `research_agent/config/settings.py`'s own module docstring.
- **Data/cache/Chroma path constants not centralized.** 4 constants
  (`DATA_DIR`, `DB_PATH`, `CHROMA_PERSIST_DIR`, `QA_CHECKPOINT_DB_PATH`),
  `DATA_DIR` independently duplicated in two files. Same status as
  above — optional, not scheduled, same source.
- **`eval_results/latency_history.csv` has no reproducing script.**
  Referenced directly by `README.md`'s "Search-call parallelization"
  section; no script in `scripts/` currently produces it. See
  `docs/evaluation.md`, `eval_results/archive/README.md`.
- **`api.py`'s compatibility re-exports retirement — deferred.**
  Every schema/helper/runtime object moved out of `api.py` across
  Phases 4–10 is still re-exported from it for `research_agent.api.
  <name>`/`patch.object(api, "<name>", ...)` callers. Retiring any of
  them requires updating every caller in lockstep — not attempted
  incrementally, not scheduled. See `docs/architecture.md`'s Phase 10
  section, `specs/legacy-cleanup-audit.md`'s `api.py` entry.
- **`research_agent/api_app/` → `api/` rename — deferred.** Blocked on
  the re-export retirement above (renaming now would either break
  `patch.object` mocking or require the whole re-export cleanup first).
  Not scheduled. See `docs/architecture.md`'s "Why `research_agent/
  api_app/`, not `research_agent/api/`" section.
- ~~`requirements-frozen-baseline.txt` is stale~~ — **Done (Phase 18B).**
  Archived to `docs/archive/requirements-frozen-baseline.txt` (content
  unchanged, `git mv`'d, not deleted); `docs/archive/README.md` added
  explaining the archive convention. See `specs/legacy-cleanup-audit.md`.
- ~~`research_agent_architecture.svg` is stale relative to current
  architecture~~ — **Done (Phase 18C).** Archived to `docs/archive/
  research_agent_architecture.svg` (content unchanged, `git mv`'d, not
  deleted); `README.md` now points to `docs/architecture.md`'s new
  Mermaid diagram as the current architecture instead. See
  `specs/legacy-cleanup-audit.md`.
- **No numeric pass/fail gate on the RAGAS eval** — still manually run
  and human-interpreted, not CI-enforced. See `docs/evaluation.md`, and
  the Obsidian vault's `Mentor-Feedback.md`/`TODO.md` (surfaced by the
  2026-07-17 mentor review, item 13).
- **Whole-pool `/report/regenerate` can silently discard a selectively-
  curated report.** If a session uses both the chat add-to-report path
  (Phase 4, selective — only approved web sources) and the pre-existing
  Report tab "Regenerate" button (whole-pool — every
  `web_articles_added` entry, approved or not), the whole-pool call will
  overwrite `session.report` with one reflecting the entire raw web
  pool, including sources never approved through chat. The two paths
  don't defer to each other. Documented in
  `research_agent/report.py`'s `regenerate_report_with_approved_web_
  sources` docstring and `docs/architecture.md`'s chat-feature section;
  not fixed, no scheduled owner.
- **No CI configuration exists anywhere in this repo** (confirmed: no
  `.github/workflows/`, no other CI config found). Validation throughout
  this entire project has been manual (`uv run pytest -q`, `npm test`,
  `npm run build`) — works, but isn't enforced automatically on push/PR.
  Noted as debt, not scheduled; `specs/production-readiness-roadmap.md`'s
  Phase 22 ("Config/deployment foundation") is the natural place this
  would eventually get picked up, if the production-readiness arc is
  ever started.

## 4. Explicitly deferred platform work

**Not next. Not implied by anything above.** Each of these already has
its own dedicated design document — this backlog does not duplicate or
re-scope them, only points at them:

- **OAuth/auth** — `specs/production-readiness-roadmap.md` §4.
- **PostgreSQL migration** — `specs/production-readiness-roadmap.md` §5.
- **Multi-user support** — `specs/production-readiness-roadmap.md` §6.
- **Tenant/workspace isolation** — `specs/production-readiness-roadmap.md`
  §2 (product questions), §6 (ownership model).

If any bug fix or feature idea above turns out to actually require one
of these (e.g. a feature idea that only makes sense per-user), that's a
signal to route it into the production-readiness roadmap instead of this
backlog — not to quietly start platform work inside a "backend backlog"
item.

## 5. Suggested backend work process

Recommended, not enforced by tooling (no CI exists yet — see §3):

1. **One bug/feature per branch or commit.** Matches this entire
   project's own established discipline since Phase 0 — small,
   reviewable, independently revertible changes, not bundled ones.
2. **Write a failing/targeted test first where practical.** Mirrors the
   pattern already used repeatedly in this project's own history (e.g.
   Phase 2's two coverage gaps closed *before* moving the routes they
   covered; `research_agent/config/settings.py`'s own
   `tests/test_config_settings.py` added alongside the module it tests).
3. **Run the targeted backend subset before the full suite** —
   `uv run pytest tests/test_<relevant_file>.py -q` first, then
   `uv run pytest -q` before considering the change done. Matches the
   validation cadence used throughout Phases 0–17.
4. **Update the Obsidian notes after meaningful changes** — tell
   Claude/Codex "update the Obsidian notes" once a bug fix or feature
   lands; the vault does not update itself (per its own `Home.md`/setup
   note).
5. **Preserve `standardized-single-user-project` as the rollback
   baseline.** Every future bug fix/feature branch should be able to
   diff cleanly against this tag; if a change ever needs to be reverted
   wholesale, this tag is the known-good point to return to — same role
   `pre-standardization-2026-07-29` and `standardized-single-user-backend`
   already play for their respective arcs.

## Related

`specs/legacy-cleanup-audit.md` (this phase's companion document) ·
`specs/production-readiness-roadmap.md` (where deferred platform work
actually lives) · `specs/remaining-standardization-plan.md` ·
`docs/architecture.md`
