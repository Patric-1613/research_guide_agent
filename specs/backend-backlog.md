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

### R4.1 refinement badge missing after reload
- **Reported**: 2026-08-06 — a report was regenerated with "Refine
  once" checked, the refined content looked improved, but the UI never
  showed the expected `"Refined once · score N"`/`"Evaluated · score
  N"` badge; the header only showed the template badge.
- **Symptom**: `report["refinement"]` (R4.1's minimal metadata dict)
  was present immediately after a refine-enabled generate/regenerate,
  but gone by the time the UI actually rendered — the frontend's own
  `loadState()` does a separate `GET /curation/{id}` right after the
  POST resolves, and that GET reads a session already round-tripped
  through the checkpointer.
- **Root cause**: `curation_session.py`'s `_serialize_report`/
  `_deserialize_report` build their output from an explicit, hardcoded
  key list that never accounted for R4.1's new top-level `refinement`
  key — silently dropped on the first save/load round trip. Same class
  of gap as the earlier "Foundational report leaked raw paper IDs"-
  adjacent section-key round-trip bug from R2C (see `docs/
  architecture.md`'s Phase R2C serialization notes).
- **Fix**: `refinement` added to both functions' existing, presence-
  checked opaque pass-through convention — the same one `references`/
  `sections`/`report_template` already use, no new logic introduced.
  Applies uniformly to `session.report` and every `report_versions[N].
  report`, since both go through the same two helpers. A report
  refinement was never requested for still round-trips with no
  `refinement` key at all, not a fabricated `{"enabled": false, ...}`
  placeholder. See `docs/architecture.md`'s R4.1 section's own
  "Bugfix" note for the full design record.
- **Location**: `research_agent/curation_session.py`
  (`_serialize_report`, `_deserialize_report`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-06). Commit `f6ae9f2`.

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

### R3: report history/versioning
- **Goal**: stop report generation/regeneration from silently
  overwriting the only stored report — let each generate/regenerate
  append a new, immutable version instead, with a clear active/current
  one and a way to switch back.
- **Why it matters**: R2C's templates made this a real, common need —
  users comparing Foundational vs. Analytical vs. Expert, or wanting to
  keep a good report before trying a regeneration, had no way to do
  either; a regeneration just destroyed whatever was there before. Also
  the right foundation to have in place before any future evaluator/
  refinement loop, which will want to attach scores/feedback to a
  specific report version, not "whatever the report currently is."
- **Implementation split across two chunks**:
  - **Chunk 1+2 (backend)**: `PaperPoolSession` gained `report_versions:
    list[dict]` (in-session, not a separate SQLite table — see
    `docs/architecture.md`'s own reasoning) and `active_report_
    version_id`. `report.py`'s `append_report_version`/`get_active_
    report_version`/`activate_report_version` are the one shared
    domain API every report-mutation call site goes through — all 4 of
    them (`get_or_create_report`, `regenerate_report`, chat's add-to-
    report, and chat's own auto-report-update-accept flow in
    `curation_chat.py`, the easiest of the four to miss). `session.
    report` stays the exact same active/current compatibility field it
    always was, kept in lockstep by construction. New `POST /curation/
    {session_id}/reports/{version_id}/activate` endpoint; `ReportOut`/
    `CurationStateResponse` gained additive version metadata fields.
    Old sessions with a `report` but no `report_versions` key derive
    one implicit version 1 at load time, never rewritten into storage
    until a real mutation happens.
  - **Chunk 3 (frontend)**: a compact version `<select>` dropdown in
    `ReportModePanel`, next to the template selector — hidden when
    there are no versions yet, labels read `Version N — Template —
    Reason`, no rename/delete/dashboard.
  - See `docs/architecture.md`'s "R3 — report history/versioning"
    section for the full design record.
- **Deferred follow-ups, not done in this phase**:
  - Report version rename/display-name support.
  - Report version deletion/archiving.
  - Version retention/capping — no limit today, unbounded growth per
    session, same explicitly-accepted tradeoff `turn_history` already
    set precedent for.
  - A real, table-backed `report_versions` model — deferred to the
    eventual Postgres/multi-user phase, once cross-session version
    queries are an actual need.
  - Evaluator/refinement scores/feedback attached to a specific report
    version — R3 exists partly to make this safe to build later, not to
    build it now.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commits `93e1a63` (backend) and
  `70875fa` (frontend).

### R3.1: approved web citation enforcement across regeneration
- **Superseded by R3.1b below** — this entry's force-include/restore
  mechanism was removed because it produced its own bug (an orphan
  References entry with no inline marker in the report body). Kept as
  historical record; see R3.1b for current behavior.
- **Goal**: a web source approved for the report via chat must actually
  survive report regeneration — not just papers, which already had this
  guarantee.
- **Why it matters**: reported live — a chat-approved web source added
  to the report disappeared after regenerating, even though it was
  never revoked. Papers already had `_restore_dropped_citations`
  keeping a cited paper in place across regeneration; web sources had
  no equivalent, so whether an approved web source survived any given
  regeneration was pure chance (whatever the model happened to
  re-cite).
- **Implementation**: `research_agent/report.py` gained
  `_restore_dropped_web_citations` (revocation-gated restoration of a
  web url a section cited in the prior report but this regeneration
  dropped — only restores if still currently allowed and not revoked,
  so this does not reopen the earlier revoked-source-resurrection fix)
  and `_force_include_allowed_web_articles` (deterministic
  force-inclusion of an approved web source the model has NEVER cited
  in any prior or current report, into whichever section already cites
  the most web sources this round, falling back to
  `thematic_findings`). Both are gated by a new `allowed_web_urls: set
  [str] | None = None` parameter threaded through
  `_regenerate_report_sections_with_sources` and both public regenerate
  functions — `regenerate_report_with_new_sources` passes `session.
  report_approved_web_article_urls`, `regenerate_report_with_approved_
  web_sources` passes `{a.url for a in approved_web_articles}`. Neither
  function ever edits section prose — only which `WebArticle` objects a
  section's citation/reference metadata carries. See
  `docs/architecture.md`'s "R3.1 — approved web citation enforcement
  across regeneration" section for the full design record.
- **Location**: `research_agent/report.py`
  (`_restore_dropped_web_citations`, `_force_include_allowed_web_
  articles`, `_regenerate_report_sections_with_sources`,
  `regenerate_report_with_new_sources`, `regenerate_report_with_
  approved_web_sources`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commit `bc4fc86`. Superseded by R3.1b.

### R3.1b: no orphan References entries for approved web sources
- **Goal**: a web reference should not appear in a report's References
  list unless at least one inline `[N]` citation marker in the body
  actually points to it — no orphan entries, ever.
- **Why it matters**: reported live via a report screenshot — R3.1's
  force-include mechanism guaranteed an approved web source's presence
  in References even when the model never cited it, by appending it to
  a section's `cited_web_articles` metadata without touching that
  section's prose. The result: a numbered, linked References entry
  with no `[N]` marker anywhere in the visible report body — reads as a
  broken or decorative reference to an actual reader.
- **Implementation**: `_force_include_allowed_web_articles`,
  `_restore_dropped_web_citations`, and `_resolve_prior_web_citations_
  for_regeneration` removed entirely from `research_agent/report.py` —
  not narrowed, since any append to `cited_web_articles` without a
  matching prose marker hits the same shared "structurally cited but
  unmarked" trailing pass that produced the orphan. The current round's
  own model output (`cited_web_urls`) is now the sole source of truth
  for which web sources a section cites; a previously-cited source the
  model drops this round simply disappears, rather than surviving as a
  metadata-only orphan. `_build_regeneration_system_prompt` gained an
  optional approved-sources paragraph (naming user-approved web sources
  by title, instructing the model to cite one inline with `[Web N]`
  only if directly relevant, omit otherwise) — a prompt nudge, not a
  guarantee; it's now the only mechanism giving an approved source any
  special treatment. Revoked-URL exclusion and paper citation
  preservation are both unchanged. Two options were weighed (a bounded
  one-round-orphan restoration vs. no restoration at all); Option B (no
  restoration at all) was chosen — preservation across regeneration now
  only holds as long as the model keeps citing a source on its own. See
  `docs/architecture.md`'s "R3.1b — no orphan References entries for
  approved web sources" section for the full design record.
- **Location**: `research_agent/report.py`
  (`_regenerate_report_sections_with_sources`, `_build_regeneration_
  system_prompt`, `regenerate_report_with_new_sources`, `regenerate_
  report_with_approved_web_sources`).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commit `58ab01e`.

### R3.2: chat-side references with independent numbering
- **Goal**: give curation chat its own resolved `[N]` citation numbering
  and References list, matching the report's own R1 treatment, with
  numbering kept structurally independent from the report's — the same
  source can be `[1]` in chat and `[5]` in the report, and deleting/
  editing a chat exchange must correctly update chat's own references
  without touching the report's.
- **Why it matters**: chat answers previously showed raw, unresolved
  `[Paper N]`/`[Web N]` marker text with no References list at all —
  the report had already solved this exact problem (R1) but chat never
  got the equivalent treatment.
- **Implementation split across three chunks**:
  - **Chunk 1 (backend)**: `ChatTurn.cited_papers` is now stamped
    alongside the existing `cited_web_articles` (same lightweight
    `{paper_id, title}` shape, never a full `Paper` object) —
    `qa.ask()`'s result already carried this; `curation_chat.py`'s
    `_attach_exchange_metadata` was simply discarding it before this
    chunk.
  - **Chunk 2 (backend)**: `research_agent/report.py`'s
    `_build_references_and_renumber` promoted to a public wrapper,
    `build_references_and_renumber` — a deliberate, one-off exception
    to this codebase's "reimplement small regexes rather than couple
    across module-private internals" precedent, justified because this
    is a large, multi-phase-hardened algorithm a third reimplementation
    would put at real maintenance risk. `curation_chat.py`'s new
    `derive_chat_references(session)` builds a fresh, `exchange_id`-
    keyed `sections_out` dict from `session.chat_history` and reuses
    that same function — since it builds a brand-new reference registry
    per call, chat-scoped and report-scoped numbering structurally
    cannot collide. Derived fresh every call, never persisted;
    `session.chat_history` itself is never mutated — only the response
    payload carries rewritten markers, same convention `_report_to_out`
    already established. This is also why delete/edit "just work" with
    zero extra bookkeeping. `services/curation_session_service.py`'s
    `get_state()` wires the result into the new `CurationStateResponse.
    chat_references: list[ReferenceEntry]` field (reusing the existing
    `ReferenceEntry` model, no new schema) and the rewritten `chat_
    history`.
  - **Chunk 3 (frontend)**: the report's marker renderer and
    References-list renderer were extracted into shared, reusable
    pieces (`frontend/src/lib/citationMarkers.tsx`, `frontend/src/
    components/shared/ReferencesList.tsx`) with report's own default
    parameters producing byte-identical output to before the
    extraction. `ChatMessage.tsx` renders assistant `[N]` markers as
    links (never a user's own typed text); `ChatModePanel.tsx` gained a
    compact "Chat references" panel above the input, hidden when empty,
    showing paper and web references (Globe icon for web), links
    opening in a new tab. Chat's own anchor/testid namespace (`chat-
    ref`/`chat-reference`) is kept distinct from report's as defense-
    in-depth, though the two panels are never simultaneously mounted
    today.
  - See `docs/architecture.md`'s "R3.2 — chat-side references with
    independent numbering" section for the full design record.
- **Location**: `research_agent/curation_chat.py`
  (`_attach_exchange_metadata`, `_resolve_cited_web_article`, `derive_
  chat_references`), `research_agent/report.py`
  (`build_references_and_renumber`), `research_agent/api_app/
  schemas.py` (`ChatTurn.cited_papers`, `CurationStateResponse.chat_
  references`), `research_agent/services/curation_session_service.py`,
  `frontend/src/lib/citationMarkers.tsx`, `frontend/src/components/
  shared/ReferencesList.tsx`, `frontend/src/components/TurnFeed/
  ChatMessage.tsx`, `frontend/src/components/ChatMode/ChatModePanel.
  tsx`, `frontend/src/types/index.ts`.
- **Deferred follow-ups, not done in this arc**:
  - Report version rename/delete/archive UI (carried over from R3,
    still not built).
  - Report version retention/capping — unbounded growth per session,
    same accepted tradeoff as R3 and `turn_history`.
  - A real, table-backed `report_versions` model for the eventual
    Postgres/multi-user phase.
  - Chat reference UX polish (e.g. scroll-to-reference animation,
    collapsing a long chat references list) — the current panel is
    deliberately minimal, not iterated on further.
  - Evaluator/refinement scores/feedback attached to a specific report
    version — R3 exists partly to make this safe to build later, not to
    build it now.
  - A chat/web-retrieval evaluation and red-team suite — nothing in the
    chat/report arc (R1–R3.2) has adversarial/eval-style coverage the
    way `report.py`'s citation-grounding does (`tests/test_report_
    grounding.py`) or the original pipeline's RAGAS harness does
    (`docs/evaluation.md`); worth its own scoped pass before R4.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commits `58e8c00` (Chunk 1),
  `6bb4c05` (Chunk 2), `e6941a4` (Chunk 3).

### R4.1: optional, bounded report refinement loop
- **Goal**: let a generated/regenerated report optionally go through a
  bounded draft → evaluate → revise (at most once) → finalize pass
  before being shown, without opening the door to an unbounded
  agentic loop.
- **Why it matters**: R1–R3.2 established grounding, citation
  correctness, versioning, and reference hygiene, but nothing ever
  judged a report's actual synthesis quality (paper-by-paper summary
  vs. real synthesis, Gap Analysis vs. Future Research Directions
  overlap, template appropriateness) or gave it a chance to improve
  before being finalized.
- **Implementation split across two commits**:
  - **Backend (`033d176`)**: implemented as plain, bounded functions in
    `research_agent/report.py` — deliberately NOT a LangGraph
    `StateGraph`, since this flow has exactly one conditional fork and
    no cycle (a revision, if it happens, is never re-evaluated), runs
    entirely inside one synchronous request, and has no human-interrupt
    point. `ReportEvaluation` (report.py-internal structured evaluator
    output), `_deterministic_report_checks` (hard gates: raw marker
    leaks, missing/empty sections, orphan references, malformed
    numbering; `skipped_papers` is warning-only, never a hard gate),
    `evaluate_report` (combines both layers), `revise_report` (reuses
    the exact same schema/citation/reference pipeline generation and
    regeneration already use — no new source discovery, same
    paper/web candidate set, same R3.1b web-citation product rule),
    `refine_report_if_requested` (the one orchestration point,
    `refinement_mode="off"` is a pure passthrough with zero extra LLM
    calls). Wired into `POST /curation/{id}/report` and `/report/
    regenerate` only — NOT chat's add-to-report or auto-report-update
    paths. Minimal `report.refinement` metadata
    (`enabled`/`rounds`/`initial_score`/`final_score`) persisted on the
    report body; full evaluator detail intentionally not persisted.
    `generation_reason` unchanged — no `"refined"` value added,
    refinement stays orthogonal metadata.
  - **Frontend (`8fc7cb4`)**: a compact "Refine once" checkbox in
    `ReportModePanel` (off by default, present in both the Generate and
    Regenerate views via shared lifted state), threaded through
    `useCurationSession`/`curationApi` as an optional `refinementMode`
    alongside the existing `reportTemplate` param — the API client only
    sends `refinement_mode` in the request body when it's actually
    `"single"`, keeping every existing caller's payload unchanged. A
    small score-only badge ("Refined once · score N" / "Evaluated ·
    score N") renders when a report carries refinement metadata — no
    evaluator-details UI.
  - See `docs/architecture.md`'s "R4.1 — optional, bounded report
    refinement loop" section for the full design record.
- **Location**: `research_agent/report.py`, `research_agent/api.py`,
  `research_agent/api_app/schemas.py`, `research_agent/api_app/
  serializers.py`, `research_agent/services/curation_report_service.py`,
  `research_agent/api_app/routers/curation_reports.py`,
  `frontend/src/types/index.ts`, `frontend/src/lib/api/client.ts`,
  `frontend/src/hooks/useCurationSession.ts`, `frontend/src/pages/
  CurationWorkspacePage.tsx`, `frontend/src/components/ReportMode/
  ReportModePanel.tsx`.
- **Deferred follow-ups, not done in this phase**:
  - ~~R4.2 — persist/display full evaluator detail (issues,
    revision_instructions, section_scores) if R4.1 proves useful in
    practice.~~ **Done** — see the R4.2 entry below.
  - R4.3 — configurable bounded refinement depth (off / 1 / 2
    revisions, never unlimited).
  - R4.4 — optional human-in-the-loop refinement using a LangGraph
    `interrupt()`, only if justified once a concrete need exists.
  - R4.5 — deeper, per-template evaluator rubric calibration (stricter
    density expectations for Expert, stronger completeness expectations
    for Foundational) — R4.1 already passes `report_template` into the
    evaluator prompt, so this is calibration, not a new capability.
  - A formal report evaluation harness/metrics — still needs its own
    scoped design pass (dataset, gold references, metrics), same
    caution this project already gave the original pipeline's RAGAS
    harness.
  - Chat/web retrieval evaluation and red-team guardrails — stays a
    separate, already-tracked backlog item (see R3.2's own deferred
    follow-ups above), unrelated to report refinement.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commits `033d176` (backend),
  `8fc7cb4` (frontend).

### R4.2: persist and display evaluator details for refined reports
- **Goal**: let a user see WHY a refined report was revised (or wasn't)
  — the evaluator's actual issues and per-section scores — without
  turning the report view into a noisy dashboard.
- **Why it matters**: R4.1 shipped with a compact score-only badge and
  deliberately discarded the evaluator's real findings
  (`issues`/`revision_instructions`/`section_scores`) even though they
  were already computed by the one evaluation call every refined report
  goes through — a real, if minor, loss of useful information for zero
  cost savings (the data existed, it just wasn't kept).
- **Implementation**: `refine_report_if_requested` (`research_agent/
  report.py`) now stamps the full `evaluation` dict's `issues`/
  `revision_instructions`/`section_scores` onto `report["refinement"]`
  alongside R4.1's original 4 fields — no new LLM call, the evaluation
  already ran. `ReportRefinementOut` (`api_app/schemas.py`) gained the
  same three fields with safe defaults, so an R4.1-only refinement dict
  (persisted before this phase) still constructs cleanly via Pydantic's
  own defaults — confirmed no changes were needed to `curation_
  session.py`'s serialize/deserialize (the opaque whole-dict pass-
  through from the R4.1 persistence bug fix already covers new keys)
  or to `api_app/serializers.py`'s `_report_to_out`, both proven by
  test rather than assumed. Frontend: the existing compact badge is
  unchanged; a new collapsed-by-default "Evaluation details"
  disclosure in `ReportModePanel` appears only when there's real
  content (non-empty issues or section_scores), explicitly states
  findings describe the draft before revision (not necessarily the
  current report — R4.1/R4.2 never re-evaluate after a revision), caps
  visible issues at 5 with a "+N more" line, tolerates a partial
  section_scores dict, and never renders raw `revision_instructions`.
  See `docs/architecture.md`'s "R4.2 — persist and display evaluator
  details for refined reports" section for the full design record.
- **Location**: `research_agent/report.py`
  (`refine_report_if_requested`), `research_agent/api_app/schemas.py`
  (`ReportRefinementOut`), `frontend/src/types/index.ts`, `frontend/
  src/components/ReportMode/ReportModePanel.tsx`.
- **Deferred follow-ups, not done in this phase**: R4.3 (configurable
  depth), R4.4 (human-in-the-loop), R4.5 (deeper template calibration),
  a formal eval harness, and chat/web retrieval red-team all remain
  exactly as scoped in the R4.1 entry above — none of them are
  prerequisites for or blocked by R4.2.
- **UI polish (2026-08-06)**: the disclosure's original single
  "Initial score: N · Final score: N" line read as a before/after
  comparison even when nothing was compared (the no-revision case).
  Replaced with three distinct score-summary shapes matched to what
  actually happened (`"Score N" / "No revision needed"` when
  `rounds===0`; `"Initial score N" / "Revised once" / "Final score not
  re-evaluated"` when revised but never re-scored; `"Score N → M" /
  "Revised once"` for an actual transition), section scores rendered as
  individual labeled rows with a compact progress bar each (reusing the
  existing shared `ProgressBar` component, no new dependency), and
  issues moved under an explicit "Draft issues" heading. Purely visual/
  copy — no change to `report["refinement"]`'s shape, the evaluator, or
  the API. See `docs/architecture.md`'s R4.2 section's own "UI polish"
  note for the full design record. Commit `88aac1f`.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-06). Commits `1918487` (persist/display),
  `88aac1f` (UI polish).

### R5A/R5B: export the active report version as Markdown
- **Goal**: let a user download the report they're currently looking at
  as a standalone Markdown file, without any re-processing or a second
  source of truth for what "the report" contains.
- **Why it matters**: report content only lived inside the app's own
  UI — there was no way to hand a report to someone else, paste it
  somewhere, or keep an offline copy of a specific version.
- **Implementation**: `GET /curation/{session_id}/report/export?
  format=markdown` (R5A, backend) exports `get_active_report_version
  (session)` — the ACTIVE version, never just the latest, matching
  every other report read's existing invariant, so no new
  version-resolution logic was needed. `render_report_markdown`
  (`research_agent/report.py`) is a deterministic renderer with no LLM
  call: it walks a version's own already-finalized `sections`/
  `references` exactly as stored, falling back to the same
  `derive_sections_from_legacy_report`/`derive_legacy_references`
  functions `_report_to_out` already uses for a legacy-shaped report.
  Deliberately excludes `report["refinement"]` (R4.1/R4.2 evaluator
  metadata — internal QA information, not report content) and chat
  history/chat references (a separate, unversioned scratchpad).
  Response is `text/markdown; charset=utf-8` with `Content-Disposition:
  attachment; filename="<slug>-v<N>.md"`, `<slug>` sanitized from
  `display_title`/`topic` and `<N>` the exported version's own
  `version_number`. R5B (frontend) adds a compact "Export Markdown"
  link in `ReportModePanel`, next to Regenerate — a real browser-native
  download link (`<a href download>`, no `fetch`/blob), hidden when
  there's no report, disabled/suppressed via `aria-disabled` +
  `preventDefault()` while a report action is in progress (`<a>` has no
  native `disabled` attribute). `curationApi.getReportExportUrl(...)`
  is computed in `CurationWorkspacePage` and passed down as a plain
  string prop — `ReportModePanel` never imports `curationApi` directly,
  matching its existing presentational convention. See `docs/
  architecture.md`'s "R5A — backend Markdown export for the active
  report version" and "R5B — frontend Export Markdown link" sections
  for the full design record.
- **Location**: `research_agent/report.py` (`render_report_markdown`,
  `report_export_filename`), `research_agent/services/
  curation_report_service.py` (`export_active_report`), `research_agent/
  api_app/routers/curation_reports.py`, `frontend/src/lib/api/
  client.ts` (`getReportExportUrl`), `frontend/src/types/index.ts`
  (`ReportExportFormat`), `frontend/src/components/ReportMode/
  ReportModePanel.tsx` (`ExportMarkdownLink`), `frontend/src/pages/
  CurationWorkspacePage.tsx`.
- **Deferred follow-ups, not done in this phase (R5C)**: PDF export,
  DOCX export, and a cleaner document template/layout for exported
  files (the current Markdown renderer is intentionally minimal — plain
  headings and a References list, no cover page or styling). The active
  report version should remain the export target for any of these —
  no reason to diverge from R5A's own scope decision.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-05). Commits `a6d8a8c` (R5A backend),
  `37221e9` (R5B frontend).

### R5C: PDF and DOCX report export via a shared document model
- **Goal**: let a user download the active report version as a clean,
  document-style PDF or DOCX — not an export of the dark app UI —
  alongside R5A/R5B's existing Markdown export.
- **Why it matters**: Markdown is convenient for pasting/version
  control but isn't what most people want to hand someone else or
  submit somewhere; PDF/DOCX were the two concrete document formats
  people actually share.
- **Implementation**: `build_report_export_document(session, version)`
  (`research_agent/report.py`) is a new shared `ReportExportDocument`
  model (`title`, `meta`, `sections`, `references`) — the one place the
  legacy-report fallback, title fallback, and paragraph-splitting are
  decided, consumed by all three renderers instead of each re-deriving
  it. `render_report_markdown` was refactored to consume it with
  byte-identical output to R5A (proven by every pre-existing R5A test
  passing unchanged). `render_report_docx` (python-docx) and
  `render_report_pdf` (ReportLab Platypus flowables, not raw canvas
  positioning) walk the same model into a clean, minimal layout: title,
  metadata lines, section headings/paragraphs, a page break, then
  numbered references with real hyperlinks where `link_url` exists.
  ReportLab was chosen over WeasyPrint specifically to avoid a
  system-level dependency (Pango/Cairo/GDK-Pixbuf) this single-user app
  has no CI/deployment story to absorb. PDF content is escaped via
  `xml.sax.saxutils.escape` (ReportLab parses `Paragraph` text as
  XML-like markup) and hyperlink URLs via `xml.sax.saxutils.quoteattr`
  (attribute-safe, not text-escaping) — covered by a dedicated
  adversarial test proving `<`, `>`, `&` in report content/reference
  text and URLs don't crash generation or corrupt the output.
  `GET /curation/{session_id}/report/export?format=docx|pdf` extends
  the existing R5A endpoint (format validation still runs before
  session lookup); DOCX/PDF use FastAPI's base `Response` (binary, no
  charset) rather than `PlainTextResponse`. Frontend: R5B's single
  direct "Export Markdown" link is replaced by a compact `ExportMenu`
  in `ReportModePanel` — an "Export ▾" trigger opening Markdown/PDF/
  DOCX options, each a real `<a href download>` link, still no
  `fetch`/blob. The trigger is a real `<button disabled>` (simpler than
  R5B's `<a>`-based manual workaround — disabling it fully prevents the
  menu from opening), and the menu closes on option click/outside
  click/Escape, reusing `ChatMessageRow`'s existing per-row action-menu
  pattern rather than a new one. See `docs/architecture.md`'s "R5C —
  PDF and DOCX report export via a shared document model" section for
  the full design record.
- **Location**: `research_agent/report.py` (`ExportSection`,
  `ExportReference`, `ReportExportDocument`,
  `build_report_export_document`, `render_report_docx`,
  `render_report_pdf`), `research_agent/services/
  curation_report_service.py`, `research_agent/api_app/routers/
  curation_reports.py`, `frontend/src/types/index.ts`
  (`ReportExportFormat`), `frontend/src/lib/api/client.ts`
  (`getReportExportUrl`), `frontend/src/components/ReportMode/
  ReportModePanel.tsx` (`ExportMenu`), `frontend/src/pages/
  CurationWorkspacePage.tsx`.
- **Implementation split**: R5C.1 (shared export document model +
  byte-identical Markdown refactor + DOCX backend), R5C.2 (PDF
  backend), R5C.3 (frontend Export menu for all three formats) — same
  commit-per-chunk granularity as every prior phase, each independently
  testable and revertable.
- **Dependencies added**: `python-docx==1.2.0`, `reportlab==5.0.0` —
  both pure Python, no system-level dependency, added via `uv add` and
  repinned to exact versions matching this project's existing
  convention.
- **Deferred follow-ups, not done in this phase**:
  - Manual visual QA polish of the rendered DOCX/PDF layout, if it
    turns out to need it — only structural/round-trip validity has
    been machine-checked so far, same as R5A's markdown got one manual
    look before being called done.
  - An optional cover/title page (explicitly skipped as a distinct
    branding decision outside this phase's actual goal).
  - Optional export of evaluator/refinement details — deliberately
    excluded from every export format so far (R4.2's "don't turn it
    into a dashboard" precedent applied to exports); would need to be
    an explicit, separate opt-in, not a default.
  - Optional chat transcript export — chat is a separate, unversioned
    scratchpad, explicitly out of scope for every R5 sub-phase.
  - Exporting an arbitrary (not-currently-active) report version
    directly by `version_id`, without first requiring a separate
    `/activate` call, if that workflow becomes a real need.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-06). Commits `34625a2` (R5C.1),
  `d610755` (R5C.2), `34a8b7b` (R5C.3).

### R5D: Literature Review export template polish
- **Goal**: make the DOCX/PDF exports look like a proper literature-
  review document by default — a real title block, serif body
  typography, sensible margins/spacing, a visible title-vs-heading
  hierarchy, and page numbers where clean to add — not plain dumped
  text or an export of the dark app UI.
- **Why it matters**: R5C shipped structurally correct DOCX/PDF export,
  but both used Word/ReportLab's own unmodified default styling. PDF's
  Title and Heading1 in particular rendered identically (both 18pt
  Helvetica-Bold) — no visual hierarchy at all between the document
  title and a section heading.
- **Implementation**: shared style constants
  (`_EXPORT_BODY_FONT_SIZE_PT`, `_EXPORT_LINE_SPACING`,
  `_EXPORT_MARGIN_INCHES`, `_EXPORT_REFERENCE_HANGING_INDENT_INCHES`,
  `_EXPORT_HEADING_COLOR_HEX`) in `research_agent/report.py` drive both
  renderers so DOCX and PDF share identical formatting decisions, not
  just similar ones. A "Literature Review" subtitle now sits under the
  title in both (Word's own built-in `Subtitle` style for DOCX, a
  dedicated `ParagraphStyle` for PDF). Times New Roman / Times-Roman
  family (PDF's is ReportLab's built-in Base-14 font, no embedding, no
  new dependency) at 12pt with 1.15 line spacing, 1in margins on all
  sides made explicit in both (DOCX's own default was actually 1.25in
  left/right, not 1in — verified, not assumed), a restrained dark-navy
  (`#1F3864`) heading color, and PDF's Title/Heading1 now genuinely
  differ in size (20pt vs 15pt). References keep their existing new-
  page-break and preserved hyperlinks, plus a new 0.5in hanging indent
  in both formats via each library's own first-class paragraph-format
  support. PDF page numbers are drawn via ReportLab's documented
  `onFirstPage`/`onLaterPages` canvas callback — the only first-class
  way Platypus exposes per-page footer content, scoped strictly to
  footer text. `pageCompression=0` on PDF's `SimpleDocTemplate` trades
  a slightly larger file for an uncompressed content stream, making
  rendered text (and the page-number footer's own operators) directly
  greppable in raw bytes — real content-level PDF tests for the first
  time, with no new dependency. `ReportExportDocument` and Markdown
  output are both untouched; no metadata inconsistency was found
  between formats to justify the one exception that would have allowed
  a Markdown change. See `docs/architecture.md`'s "R5D — Literature
  Review export template polish" section for the full design record.
- **Location**: `research_agent/report.py`
  (`_apply_docx_literature_review_style`, `_draw_pdf_page_number`, the
  shared `_EXPORT_*` constants, restyled `render_report_docx`/
  `render_report_pdf`). No API, frontend, or `ReportExportDocument`
  changes.
- **Deferred follow-ups, not done in this phase**:
  - DOCX page numbers — python-docx has no first-class API for them,
    only a hand-built field-code XML workaround of the same shape as
    the hyperlink helper already shipped; deliberately left out of an
    already-multi-part chunk rather than added on top.
  - An A4 page-size option — no locale signal anywhere in
    `session`/`report` to base the choice on; would want a real
    user-facing setting first (itself still not started).
  - A cover/title page, a table of contents, branding or logos,
    multiple export style modes/themes — none started.
  - Exporting evaluator/refinement details or chat/chat references —
    both remain permanently excluded from every export format (not
    merely deferred), matching every R5 sub-phase's own exclusion
    rationale.
  - Manual visual QA of the actual rendered layout — only structural/
    content-presence checks have been machine-verified so far.
  - Everything already listed as deferred under R5C above (evaluator/
    chat export overlaps with this list; arbitrary-version export by
    `version_id` without `/activate` remains untouched).
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-07). Commit `46abd89`.

### R7: chat/web retrieval relevance guardrails (R7A/R7B/R7C)
- **Goal**: stop the chat/web-search path from citing topically
  irrelevant web sources, and stop such a citation (if one ever slips
  through) from being promotable into the report.
- **Why it matters**: a real chat session about AI governance retrieved
  and cited an unrelated housing/zoning web source that merely shared
  governance-adjacent vocabulary. The existing per-turn relevance
  filter (`_WEB_ARTICLE_RELEVANCE_THRESHOLD`, from the earlier
  curation-chat-web-relevance work) only ever checked a candidate
  against the current turn's own (sometimes drifted) query — nothing
  anchored a check to what the session was actually about, and nothing
  gated whether a chat citation was ever safe to promote into the
  report.
- **Implementation**: three deliberately small, independently
  revertable chunks.
  - **R7A** (foundation, no live behavior change): `qa.ChatSession`
    gains an optional `topic: str = ""`, populated from
    `PaperPoolSession.topic`. `_filter_relevant_web_articles` gains a
    `topic` parameter — when given, an article must clear the existing
    threshold against BOTH the per-turn query AND the topic (AND, not
    OR); `topic=""` (the only value the live call site passed as of
    R7A) reproduces pre-R7A behavior exactly. Six red-team fixtures
    (housing-vs-AI-governance, a genuinely relevant source, query-
    relevant/topic-irrelevant, topic-relevant/query-irrelevant, empty-
    topic parity, the temporal-query-trap pattern, and a title-looks-
    relevant-but-snippet-isn't case) prove the mechanism in isolation.
  - **R7B** (live wiring, two failure postures): the answer-time gate
    (`_filter_web_relevance_node`) now passes `topic=` through and
    stays fail-open (re-filters an already-vetted pool every turn, so a
    transient failure degrading to "less scrutiny this once" is low-
    stakes). The insertion-time gate (`_accept_web_offer`) filters
    `search_web()`'s deduped candidates through the same function with
    a new `fail_open: bool = True` parameter set to `False` — the only
    gate deciding whether a brand-new article joins a pool that
    outlives the turn, so a failure there rejects rather than silently
    admits. `new_web_articles_found` now means "new, deduped, AND
    relevant." When real candidates exist but none pass, the assistant
    answer gets an honest suffix appended (never a replacement); no new
    offer loop; `used_web_search` stays correctly derived from actual
    citations.
  - **R7C** (report-promotion gate): `_filter_relevant_web_articles`
    gains an `outcome: dict` parameter recording whether a genuine
    check ran or fail-opened this turn — purely observational, surfaced
    as `web_relevance_verified` through `ask()`'s result.
    `_attach_exchange_metadata` stamps it on an assistant turn only
    when `cited_web_articles` is non-empty (`True`/`False`; absent when
    nothing was cited or on a pre-R7C turn).
    `select_eligible_exchanges_for_report` excludes only an *explicit*
    stored `False` — missing/legacy stays eligible. `ChatTurn` (backend
    schema + frontend type) gains the matching optional field; the
    frontend's `isEligibleForAddToReport` mirrors the backend gate
    exactly, so the client-side pre-check and server enforcement can't
    disagree.
  - See `docs/architecture.md`'s "R7 — chat/web retrieval relevance
    guardrails" section for the full design record, including why a
    single exchange's citations can never be "mixed" pass/fail under
    this mechanism.
- **Location**: `research_agent/qa.py` (`ChatSession.topic`,
  `_filter_relevant_web_articles`, `_filter_web_relevance_node`,
  `QAState.web_relevance_verified`), `research_agent/curation_chat.py`
  (`_build_chat_session`, `_accept_web_offer`,
  `_attach_exchange_metadata`, `select_eligible_exchanges_for_report`),
  `research_agent/api_app/schemas.py` (`ChatTurn`),
  `frontend/src/types/index.ts`,
  `frontend/src/components/TurnFeed/ChatMessageRow.tsx`.
- **Deferred follow-ups, not done in this arc**:
  - **R7D**: formalizing the red-team fixture set as a tracked eval
    suite, wiring the proposed metrics (query-topic preservation,
    source-relevance pass rate, citation-support correctness, false-
    positive web-offer rate, irrelevant-source-blocked count, answer-
    abstention correctness) into Langfuse trace metadata, and an
    `docs/evaluation.md` section documenting this arc the way the
    existing two harnesses are documented. Not started.
  - An LLM binary relevance judgment for borderline embedding scores
    (a "gray zone" secondary check) — not added; the guardrail is
    embedding-similarity-only throughout R7A–R7C.
  - A live threshold calibration pass — `_WEB_ARTICLE_RELEVANCE_
    THRESHOLD` is unchanged from its original, explicitly-uncalibrated
    0.25 value the whole way through this arc.
  - Gating `agent.py`'s one-shot `search_web_tool` path — still calls
    the same underlying `search_web()` with no relevance check; the
    reported bug was curation-chat-specific, extending the shared
    filter there later would be cheap but wasn't in scope.
  - No Neo4j or graph-database work of any kind — never proposed, not
    part of this arc.
- **Priority**: n/a — done (R7A–R7C); R7D not scheduled.
- **Status**: Closed (2026-08-08). Commits `f6f1f93` (R7A), `9511b33`
  (R7B), `f0d04bc` (R7C).

### E0: evaluation architecture decision checkpoint
- **Goal**: decide the shape of this project's next eval work (R7D,
  R6) before building any of it — audit a mentor repo's `backend/evals/`
  folder as a reference pattern, adopt what fits this project, and
  explicitly record what's deliberately not being copied, so the
  distinction doesn't get blurred once implementation starts.
- **Why it matters**: this project already has two working, documented
  eval harnesses (`scripts/eval_retrieval.py`, `scripts/ragas_eval.py`)
  with their own established artifact conventions (`docs/
  evaluation.md`). A new eval surface for chat/web relevance (R7D) and
  report quality (R6) needs a shape that extends those conventions
  rather than fragmenting them — and needs deciding once, deliberately,
  rather than improvised chunk-by-chunk the way an earlier attempt at
  this (`specs/migration-plan.md`'s own original Phase 6) was left
  half-finished.
- **Decision**: studied github.com/cwijayasundara/document_intelligence_
  adv_v2's `backend/evals/` folder in full (structure, eval-case format,
  runner/CLI pattern, evaluator layers, result persistence). Decided:
  1. A small future `research_agent/evals/` code package —
     `cli.py`, `runners/`, `evaluators/`, shared base-runner utilities.
     Not a new idea — matches `specs/migration-plan.md`'s original,
     never-executed Phase 6 shape almost exactly; the mentor repo is
     independent validation of that original plan, not a new direction.
  2. Fixtures stay in `eval_data/`, not moved into `research_agent/
     evals/` — one canonical location for eval input data, old and new
     alike.
  3. Results stay in `eval_results/`, one new CSV per new suite
     (`chat_relevance_history.csv`, `report_quality_history.csv`),
     never appended into the existing `retrieval_history.csv`/
     `history.csv`; per-run detail follows the existing `eval_results/
     runs/` gitignored convention.
  4. `scripts/eval_retrieval.py`/`scripts/ragas_eval.py` are unchanged
     — not wrapped or migrated into the new package in this phase.
  5. Planned CLI shape: `python -m research_agent.evals.cli
     list-suites`, `run --suite <name> --mode mock|live [--subset N]
     [--tags ...]`.
  6. `--mode` defaults to `mock` (offline) — `live` is always an
     explicit, opt-in flag.
  7. pytest and eval runners stay separate: pytest proves deterministic
     *code* behavior (mocked, zero API key, part of the dev loop); eval
     runners measure *product/agent* behavior over scenarios (separate,
     manual, cost-aware) — the same line `docs/evaluation.md` already
     draws for the two existing harnesses, carried forward.
  8. Borrowed from the mentor repo: the JSONL example format, the
     evaluator function shape (`(prediction, expected) -> {"key",
     "score", "comment"}`), `--subset`/`--tags`, a deterministic +
     LLM-as-judge evaluator split, the red-team-suite concept, and a
     consolidated result-summary table.
  9. Deliberately not copied yet: Postgres-backed eval persistence (no
     DB engine beyond SQLite exists here); LangSmith as a dataset/
     experiment system of record (present in `uv.lock` only as a
     transitive pin, nothing in this project's own code imports it); a
     FastAPI `/evals` dashboard (downstream of the Postgres call above);
     a synthetic LLM-generated dataset builder (this project has
     consistently preferred small, hand-verified fixture sets at its
     current scale); automated regression harvesting from production
     correction memory (no persistent memory store exists to harvest
     from — R7A already applied the same underlying principle by hand,
     building its entire red-team set from one real reported incident).
  10. Phase order: **R7D** (chat/web retrieval eval foundation) →
      **R6** (report quality eval foundation) → later (threshold
      calibration, an LLM gray-zone judge, Langfuse metrics for the new
      relevance signals, trend reports/a dashboard).
  See `docs/evaluation.md`'s "Planned evaluation architecture" section
  for the same record in workflow-doc form, and `docs/architecture.md`'s
  R7 section for the short cross-reference.
- **Why R7D before R6**: R7D targets a real failure already observed in
  the app (the housing-vs-AI-governance citation, see the R7 entry
  above) — a concrete, already-diagnosed problem to measure against.
  R6 follows because report quality should be measured once report
  generation/export/refinement are already stable, which they now are
  (R2C through R5D complete).
- **Location**: docs/spec only — `docs/evaluation.md`,
  `docs/architecture.md`. No code, dependency, or eval run.
- **Deferred**: everything under decision points 1–6 above is design
  only — no `research_agent/evals/` package, no new CLI, no new eval_data/
  eval_results files exist yet. R7D is the next phase that actually
  builds any of it.
- **Priority**: n/a — done (decision recorded).
- **Status**: Closed (2026-08-08).

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
