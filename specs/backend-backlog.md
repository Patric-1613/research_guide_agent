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
  - ~~**R7D**: formalizing the red-team fixture set as a tracked eval
    suite~~ — done, see the dedicated **R7D** entry below. Wiring the
    remaining proposed metrics (query-topic preservation, source-
    relevance pass rate, citation-support correctness, false-positive
    web-offer rate, irrelevant-source-blocked count, answer-abstention
    correctness) into Langfuse trace metadata is still not started —
    R7D built an eval *harness* for the relevance guardrail, not these
    Langfuse signals.
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
- **Priority**: n/a — done (R7A–R7C; R7D also done, see its own entry
  below).
- **Status**: Closed (2026-08-08). Commits `f6f1f93` (R7A), `9511b33`
  (R7B), `f0d04bc` (R7C).

### R7D: chat/web relevance eval foundation — mock (R7D.1) + opt-in live (R7D.2)
- **Goal**: turn R7A–R7C's hand-verified red-team scenarios into a
  tracked, re-runnable eval suite, per the architecture E0 decided —
  mock mode first (deterministic, no API key, safe to run anytime),
  live mode as a deliberate, opt-in follow-up once the harness itself
  was proven out.
- **Why it matters**: R7A's red-team fixtures were real but only ever
  exercised as one-off pytest cases inside `tests/test_qa.py` — no
  standing way to re-run them as a suite, subset/tag-filter them, or
  track pass/fail/score history over time the way `scripts/
  eval_retrieval.py`/`scripts/ragas_eval.py` already do for their own
  domains. R7D closes that gap for chat/web relevance specifically.
- **Decision/what shipped**:
  - **R7D.1 (mock mode, commit `69f07be`)**: the `research_agent/
    evals/` package E0 designed — `cli.py` (`list-suites`, `run
    --suite --mode --subset --tags`), `runners/_base.py` (JSONL
    loading with the mentor-repo-inspired metadata/`expected_`-prefix
    split, the shared predict → evaluate → aggregate loop, CSV
    append), `evaluators/relevance.py`
    (`chat_relevance_correctness`), and the `chat_relevance` suite
    against `eval_data/chat_web_relevance_redteam.jsonl` (9
    hand-curated cases: topic drift, query-only/topic-only mismatches,
    a genuinely relevant positive case, a temporal "latest" trap,
    stale web-pool reuse, an empty candidate pool, and fail-open/
    fail-closed embedding-failure behavior). Mock mode patches
    `research_agent.qa._embed_with_cache` with small, fixed vectors
    keyed off each case's own `mock_relevance` label, then calls the
    real, unmodified `_filter_relevant_web_articles` — a genuine
    regression test of the production relevance logic, not a parallel
    reimplementation.
  - **R7D.2 (opt-in live mode, commit `5c95bec`)**: `--mode live`
    constructs a real `OpenAI()` client (same construction `qa.ask()`
    uses) and lets the real embedding API decide relevance through the
    same unmodified `_filter_relevant_web_articles` call. Never runs by
    default (`--mode` still defaults to `mock`); fails cleanly with no
    traceback if credentials are missing (`LiveModeSetupError`, caught
    by the CLI, non-zero exit); prints a cost warning before running.
    The two embedding-failure fixture cases are marked `mock_only:
    true` (a new, backward-compatible optional fixture field) and are
    skipped in live mode with a clear reason — a real API call can't
    be forced to fail the way the mock does, so R7D.2 skips rather than
    fakes it. `runners/_base.py::run_suite` gained a generic `skip_if`
    predicate so mock and live share one predict → evaluate →
    aggregate loop; no scoring logic is duplicated between modes.
  - Every run appends one row to `eval_results/
    chat_relevance_history.csv`, kept to the same 11-column header
    R7D.1 established (`run_id, date, git_commit, suite, mode, total,
    passed, failed, average_score, tags, note`) — a live run's
    skipped-case count and mean latency are folded into the free-text
    `note` column instead of adding columns, so every row (mock or
    live) reads against one stable header, matching the append-only
    convention `retrieval_history.csv`/`history.csv` already set.
  - `scripts/eval_retrieval.py`/`scripts/ragas_eval.py` and the
    existing `retrieval_history.csv`/`history.csv` are untouched by
    either R7D.1 or R7D.2, exactly as E0 decided.
  - See `docs/evaluation.md`'s "Planned evaluation architecture"
    section and `docs/architecture.md`'s R7 section for the same record
    in doc/workflow form.
- **Location**: `research_agent/evals/` (`cli.py`, `runners/__init__.py`,
  `runners/_base.py`, `runners/run_chat_relevance.py`,
  `evaluators/__init__.py`, `evaluators/relevance.py`),
  `eval_data/chat_web_relevance_redteam.jsonl`, `eval_results/
  chat_relevance_history.csv`, `tests/test_evals_chat_relevance.py`.
- **Deferred follow-ups, not done here**: wiring the R7-planning-era
  Langfuse metrics (query-topic preservation, source-relevance pass
  rate, etc.) into trace metadata; a `report_quality` suite (R6, a
  separate phase); threshold calibration; an LLM gray-zone judge for
  borderline relevance scores; trend reports/a dashboard over
  `chat_relevance_history.csv`.
- **Priority**: n/a — done.
- **Status**: Closed (2026-08-08). Commits `69f07be` (R7D.1), `5c95bec`
  (R7D.2).

### R7E: chat relevance evaluation arc — live red-teaming, guardrail fixes, judge hardening (R7E.1-R7E.5b)
- **Goal**: actually run the R7D harness live against the real pipeline
  as a red-team tool, and fix whatever it found — not just confirm the
  harness works.
- **Why it matters**: R7D built a re-runnable suite but had only run it
  live once, on a small fixture set, without acting on the results. R7E
  is where that live loop closed: each live run's findings drove the
  next fix, and the fix was re-validated live before moving on.
- **Decision/what shipped** (six sub-steps, each its own commit):
  - **R7E.1 (commit `4956c1c`)**: per-example live detail — a `debug`
    param on `_filter_relevant_web_articles`, persisted per run to
    gitignored `eval_results/runs/chat_relevance_run_<run_id>.json`,
    correlated to the tracked CSV by `run_id`.
  - **R7E.2 (commit `5b01103`, live evidence commit `bebf5f1`)**: web
    article provenance metadata (which query originally surfaced a pool
    member), recorded at insertion time. First live redteam baseline:
    run_id 7, 2/5 passed, score 0.5.
  - **R7E.3 (commit `acee821`, evidence commit `e6a78ab`)**:
    provenance-aware stale-pool guard — a stricter re-check
    (`_STALE_POOL_QUERY_THRESHOLD = 0.50`) for a pool member whose
    provenance shows a different source query than the current turn's,
    fixing a real leak the run_id 7 baseline found (a stale executive-
    order summary re-surfacing against an unrelated later question).
    Post-fix: run_id 8, 3/5 passed, score 0.7.
  - **R7E.4 (commit `f4aa5f5`, evidence commit `e62de34`)**: temporal
    freshness guard — a 5-tier query-intent parse into an optional
    recency cutoff checked against `published_date`
    (`_DEFAULT_RECENCY_WINDOW_DAYS = 180` fallback); missing/malformed
    dates always pass. Post-fix: run_id 9, 4/5 passed, score 0.8.
  - **R7E.5 (commit `3544d13`, evidence commit `9200bf9`)**: selective
    direct-relevance judge — candidates in a similarity "gray zone" sent
    to a new batched LLM judge for a `relevant`/`not_relevant`/
    `uncertain` verdict. Fixture set expanded 5→11 cases for adversarial
    coverage; the live run this expansion enabled (run_id 10, 8/11
    passed, score 0.7273 — a **temporary, expected** drop from 0.8, not
    a regression) found three real bugs: an unsafe high-similarity
    bypass leak (an Atari RL source at `query_similarity=0.6287`), a
    live prompt-injection success against the real judge model
    (`verdict="relevant"`, `confidence=1.0`), and an invalid live
    fixture expectation (forced `"uncertain"`, mock-only behavior).
  - **R7E.5b (commit `94eb621`, evidence commit `ac4b9b0`; skip-reason
    fix commit `ac1f325`)**: removed the gray-zone bypass entirely —
    every deterministic survivor is now judged; added a deterministic
    `_detect_retrieved_prompt_injection` guard strictly before the
    judge, rejecting immediately and never through `fail_open` at either
    call site; bumped `_DIRECT_RELEVANCE_PROMPT_VERSION` to invalidate
    stale cache entries; fixed mock-only skip messaging to be
    fixture-specific (`mock_only_reason`) rather than one hardcoded
    string. **Final live validation**: run_id 11, 10/10 evaluated cases
    passed, 0 failed, 3 correctly skipped as mock-only, score 1.0, mean
    latency ≈1083 ms — 100% on the current 10-case synthetic live
    chat-relevance red-team set (not a claim of universal accuracy).
  - New persistent `direct_relevance_cache` SQLite table (same physical
    file as the embedding cache) keyed on model/prompt-version/topic/
    query/url/content-hash; only definite `relevant`/`not_relevant`
    verdicts cached; both production call sites
    (`qa.py::_filter_web_relevance_node`, `curation_chat.py::
    _accept_web_offer`) now pass `enable_direct_relevance_judge=True`
    unconditionally.
  - See `docs/architecture.md`'s R7 section (R7E.1-R7E.5b subsections)
    and `docs/evaluation.md`'s "R7E — chat relevance evaluation arc"
    section for the full design record, live-run evidence table, and
    known-limitations list.
- **Location**: `research_agent/qa.py`
  (`_filter_relevant_web_articles`, `_detect_retrieved_prompt_injection`,
  `_judge_direct_web_relevance`, `_init_direct_relevance_cache_db`, the
  `_STALE_POOL_QUERY_THRESHOLD`/`_DEFAULT_RECENCY_WINDOW_DAYS`/
  `_DIRECT_RELEVANCE_JUDGE_MAX_BATCH_SIZE`/`_DIRECT_RELEVANCE_PROMPT_
  VERSION` constants), `research_agent/curation_chat.py`
  (`_accept_web_offer` provenance recording),
  `research_agent/evals/runners/run_chat_relevance.py` (debug
  persistence), `eval_data/chat_web_relevance_redteam.jsonl` (expanded
  to 11 cases), `eval_results/chat_relevance_history.csv`,
  `eval_results/runs/` (new per-run JSON detail, gitignored).
- **Explicitly NOT part of this arc**: `report_quality`/R6 (report
  generation quality evaluation) is a wholly separate, still-not-started
  phase — R7E only closes the chat/web relevance guardrail arc, nothing
  about report evaluation shipped here.
- **Deferred follow-ups, still open** (not closed by this arc):
  - A real-world, labelled chat relevance dataset — the current suite
    is entirely synthetic/hand-curated red-team fixtures, not sampled
    real user sessions.
  - Threshold/recency calibration — `_WEB_ARTICLE_RELEVANCE_THRESHOLD`
    (0.25), `_STALE_POOL_QUERY_THRESHOLD` (0.50), and
    `_DEFAULT_RECENCY_WINDOW_DAYS` (180) all remain explicitly
    provisional, none calibrated against a real dataset.
  - A broader prompt-injection corpus — the current deterministic guard
    covers the one pattern family the R7E.5 live run surfaced; encoded,
    multilingual, indirect, and quoted-attack variants are unevaluated.
  - Latency/cost measurement under realistic pool sizes — the ≈1083 ms
    mean latency is evidence from this 10-case fixture run, not a
    production SLO measured under realistic candidate-pool sizes or
    load.
  - Langfuse/production observability for the relevance signals
    (query-topic preservation, source-relevance pass rate, etc.) —
    still not wired into trace metadata; R7E extended the eval
    *harness*, not production tracing.
  - Gating `agent.py`'s one-shot `search_web_tool` path with this same
    cascade, if still applicable — still calls `search_web()` with no
    relevance check at all; unchanged from the R7 entry's original
    deferred-follow-up note above.
- **Priority**: n/a — done (deferred items above tracked separately).
- **Status**: Closed (2026-08-09). Commits `4956c1c` (R7E.1), `5b01103`
  (R7E.2), `bebf5f1` (R7E.2 evidence), `acee821` (R7E.3), `e6a78ab`
  (R7E.3 evidence), `f4aa5f5` (R7E.4), `e62de34` (R7E.4 evidence),
  `3544d13` (R7E.5), `9200bf9` (R7E.5 evidence), `94eb621` (R7E.5b),
  `ac4b9b0` (R7E.5b evidence), `ac1f325` (skip-reason fix). Full backend
  suite → 780 passed.

### R6A: report quality evaluation — rubric/schema frozen, first fixture set
- **Goal**: freeze the R6 (report quality evaluation) result schema,
  hard-failure identifiers, informational-signal list, fixture
  architecture, and future R6B/R6C/R6D scoring semantics, and build the
  first reviewable, fully synthetic fixture set — design/fixtures only,
  no evaluator or runner code yet.
- **Why it matters**: R4's own in-generation evaluator
  (`research_agent/report.py::evaluate_report`) shares a model with the
  report it grades, blends 8 qualitative dimensions into one
  `overall_score`, and never re-evaluates after its one revision round
  (`refine_report_if_requested` sets `final_score=None` post-revision
  by design). R6 is a separate, standing measurement system that must
  not treat R4's score as ground truth — R6A is where that
  independence is designed and locked in before any scoring code is
  written.
- **Decision/what shipped**:
  - **Result schema** (`schema_version: "r6a-v1"`) separating
    `structural_integrity` (deterministic, pass/fail),
    `informational_signals` (deterministic, never a gate), and
    `judge_dimensions` (categorical `pass`/`fail`/`not_applicable`/
    `unknown` now; a future 0-1 `score` is informational only until
    R6E calibrates it — no overall score, no invented weights).
  - **6 frozen hard-failure identifiers**: `missing_required_section`,
    `empty_required_section`, `unresolved_citation_marker`,
    `non_sequential_reference_numbering`, `orphan_reference` (split
    from R4's existing combined checks) and `reference_source_
    unavailable` (new — R4's live generation path can never produce
    this by construction; only a stored/regressed report dict can).
  - **7 judge dimensions frozen**: `citation_correctness`,
    `groundedness`, `synthesis_quality`, `analytical_quality`,
    `template_fit`, `coherence`, `source_balance` — split into a
    future bounded claim/source judge (citation + groundedness) and a
    separate holistic judge (the other five), decided now so R6B's
    fixtures don't need reworking when R6C lands.
  - **8 fixtures** under `eval_data/report_quality/` (manifest +
    individual JSON files, not JSONL — a report-quality fixture is too
    large for one line): `good_foundational`, `good_analytical`,
    `good_expert` (identical evidence across all three templates),
    `citation_and_grounding_failure`, `verbose_low_synthesis`,
    `source_prompt_injection`, `evaluator_injection_in_report`,
    `structural_and_metadata_corruption`. Every fixture's expected
    dimension label carries a rationale a human reviewer can verify
    against the fixture's own evidence — explicitly synthetic,
    never described as human calibration data.
  - **R6D (pairwise refinement evaluation) is documented, not built** —
    blinded A/B labels, swapped order, per-dimension A/B/tie,
    positional disagreement, at least one human-labelled longer-but-
    not-better pair reserved for R6E. No pairwise fixtures exist yet.
  - **A real, currently undefended gap was surfaced, not fixed**:
    `research_agent/qa.py`'s prompt-injection guard (R7E.5b) is wired
    only into the chat/web-relevance path — never applied to paper
    abstracts, and `research_agent/report.py`'s own prompt construction
    has no independent injection defense at all. The
    `source_prompt_injection` fixture proves this with a report that
    shows the injection succeeding. **Not addressed in R6A** — closing
    it is separate production work (see "Deferred follow-ups" below).
  - See `specs/report-quality-evaluation-plan.md` for the full frozen
    design and `docs/evaluation.md`'s "R6A" section for the workflow-doc
    cross-reference.
- **Location**: `specs/report-quality-evaluation-plan.md` (new),
  `eval_data/report_quality/README.md` (new), `eval_data/
  report_quality/manifest.jsonl` (new), `eval_data/report_quality/
  fixtures/*.json` (new, 8 files), `eval_data/README.md` (updated),
  `docs/evaluation.md` (updated). **No changes to `research_agent/`,
  `tests/`, `frontend/`, `eval_results/`, `pyproject.toml`, or
  `uv.lock`.**
- **Explicitly NOT part of this checkpoint**: R6B (the deterministic/
  mock evaluator and runner code itself) has not started. R4/R4.1/R4.2
  are untouched. Report generation, `research_agent/report.py`, and the
  prompt-injection gap the `source_prompt_injection` fixture surfaces
  are all unmodified.
- **Deferred follow-ups, still open** (not addressed by R6A):
  - R6B — the actual deterministic evaluator/runner code, CLI
    registration, and `tests/test_evals_report_quality.py`.
  - R6C — the live claim/source and holistic judges (model choice
    deliberately deferred to R6C itself, not decided in R6A).
  - R6D — the pairwise refinement fixtures and harness (schema
    documented, nothing built).
  - R6E — human-labelled calibration.
  - Closing the report-generation prompt-injection gap the
    `source_prompt_injection` fixture surfaces — tracked as its own
    security-debt entry below ("No independent prompt-injection defense
    in report generation / paper abstracts"), not folded into R6's own
    scope.
- **Priority**: n/a — done (checkpoint closed).
- **Status**: Closed (2026-08-10). Superseded by R6B below (also
  closed) — see that entry for what actually shipped.

### R6B: deterministic report quality evaluation suite
- **Goal**: build the real `report_quality` eval suite against R6A's
  frozen schema — deterministic structural/citation checks only, no
  LLM judge, no network call, fully independent of R4.
- **Why it matters**: R6A froze the design; R6B is where "R6 must not
  treat R4's score as ground truth" becomes an actual, running,
  independently-implemented checker rather than a design commitment.
- **Decision/what shipped**:
  - `research_agent/evals/runners/run_report_quality.py` — a thin
    manifest+fixture loader (tags/subset filtering, path-traversal/
    duplicate-id/schema-version/template-identity validation) and
    `predict()`, implementing all 6 frozen hard-failure checks
    (`missing_required_section`, `empty_required_section`,
    `unresolved_citation_marker`, `non_sequential_reference_
    numbering`, `orphan_reference`, `reference_source_unavailable`)
    plus informational signals (section word counts, citation density,
    citation frequency by reference, selected-source coverage,
    skipped-paper rate, dominant-source share) and two warnings
    (missing abstract, empty snippet) — none of which gate pass/fail.
    Every check is a fresh, independent implementation — never calls
    `research_agent.report`'s `generate_report`/`evaluate_report`/
    `revise_report`, and never reuses R4's own
    `_deterministic_report_checks`.
  - `research_agent/evals/evaluators/report_quality.py` — one
    evaluator, `report_quality_hard_failure_agreement`, a pure set
    comparison between a prediction's detected `hard_failures` and a
    fixture's `expected_hard_failures`. Never reads
    `expected_dimension_labels` — those are reserved for R6C.
  - `research_agent/evals/cli.py` — `report_quality` registered
    alongside `chat_relevance`. `--mode live` raises a clean
    `LiveModeSetupError` (exit 2, no traceback, no CSV/detail side
    effects) — not implemented yet, R6C's job.
  - `research_agent/evals/runners/_base.py` — one small, additive
    change: `run_suite` gained an optional `examples: list[Example] |
    None = None` parameter, letting a suite whose fixtures don't fit
    flat JSONL (this one) reuse the exact predict → evaluate →
    aggregate loop without forking it. Zero behavior change for
    `chat_relevance`, which never passes it.
  - Crucial documented distinction: `structural_integrity.status`
    describes the REPORT; the evaluator's `score` describes whether
    the HARNESS correctly detected that state. The deliberately broken
    `structural_and_metadata_corruption` fixture has
    `structural_integrity.status="fail"` and all 6 hard failures
    present, and still scores `1.0` — correctly detected, not silently
    passed. "8/8 passed" in the mock baseline means the checker matched
    every fixture's expectation, not that all 8 reports are good.
  - Mock baseline: `total=8, passed=8, failed=0, average_score=1.0`.
    Full backend suite: 831 passed (780 + 51 new).
  - See `docs/evaluation.md`'s "R6B" section and `docs/architecture.md`'s
    "R6A/R6B" section for the full record and the manifest → fixture →
    prediction → evaluator → CSV/detail diagram.
- **Location**: `research_agent/evals/runners/run_report_quality.py`
  (new), `research_agent/evals/evaluators/report_quality.py` (new),
  `tests/test_evals_report_quality.py` (new, 51 tests),
  `research_agent/evals/cli.py` (updated), `research_agent/evals/
  runners/_base.py` (updated, additive only), `eval_results/
  report_quality_history.csv` (new, tracked), `eval_data/README.md` /
  `eval_data/report_quality/README.md` / `eval_results/README.md`
  (updated). **No changes to `research_agent/report.py`,
  `research_agent/qa.py`, `research_agent/curation_session.py`,
  `research_agent/api_app/`, `frontend/`, any fixture file, `pyproject.
  toml`, or `uv.lock`.**
- **Explicitly NOT part of this checkpoint**: no LLM judge of any kind
  — `judge_dimensions`/`judge_metadata` are always `null` in every
  R6B prediction. `expected.dimension_labels` is loaded but never
  scored. R4/R4.1/R4.2 and report generation are untouched.
- **Deferred follow-ups, open at this checkpoint** (not addressed by
  R6B; superseded by the R6C entry below, now closed):
  - R6C — the live claim/source and holistic judges (see that entry's
    plan; not yet implemented as of this R6B checkpoint).
  - R6D — the pairwise refinement fixtures and harness (schema
    documented in `specs/report-quality-evaluation-plan.md` section 8,
    nothing built).
  - R6E — human-labelled calibration.
  - The report-generation prompt-injection gap — tracked as its own
    security-debt entry below, not R6's to fix.
- **Priority**: n/a — done (R6C is next).
- **Status**: Closed (2026-08-10). Commits `9430013` (R6A), `e783ec6`
  (R6B).

### R6C: live judges, calibration, and freeze
- **Goal**: build the two live judges R6A's §10 design specified
  (bounded claim/source judge + separate holistic judge), then
  calibrate the whole benchmark — fixtures, expected labels, and the
  aggregation rule mapping judge verdicts to categorical labels —
  against real live evidence, and freeze the result.
- **Why it matters**: R6B proved the deterministic gate works; R6C is
  where R6 gets an actual qualitative signal, and where that signal
  gets checked against real model behavior rather than assumed
  correct from design alone.
- **Decision/what shipped**:
  - **R6C.1** (`research_agent/evals/report_quality_inputs.py`):
    bounded claim extraction (`extract_claim_units`, sentence-level,
    over raw report content, marker-merged), section-round-robin
    sampling (`sample_claim_units`, capped and recorded in `judge_
    metadata.sampling_coverage`), a deduplicated evidence registry
    with injection blocking (`build_evidence_registry` — a flagged
    source's text is blanked to `""` before any prompt is built), and
    report-prose sanitization for the holistic judge only (`build_
    sanitized_report_and_findings`, `[BLOCKED_UNTRUSTED_INSTRUCTION]`
    placeholder).
  - **R6C.2** (`research_agent/evals/judges/claim_source.py`,
    `research_agent/evals/judges/holistic.py`): two independent live
    judges wired into `predict_live`, both against `REPORT_QUALITY_
    JUDGE_MODEL` (default `gpt-5.6-terra`, independently configurable
    from `research_agent.report.REPORT_MODEL`), structured outputs,
    no silent fallback (a malformed response degrades to a recorded
    `error`).
  - **R6C.2a**: added `not_a_verifiable_claim` as a fifth collective
    verdict for framing/organizational prose; rejected as malformed
    for a cited claim; excluded from groundedness's judged set.
  - **R6C.2b/R6C.2c**: evidence-based correction of real citation/
    grounding defects found in the fixtures' own report prose by live
    smoke runs (not evaluator tuning), then a recalibration of the
    citation/groundedness aggregation rule (`r6c2-citation-
    aggregation-v2`) after the original rule was found to mechanically
    fail well-formed, correctly-cited comparative claims. Claim/source
    prompt bumped to `r6c2-claim-source-v3` (bounded-negative-claim and
    prospective-recommendation guidance).
  - **R6C.3**: first full 8-fixture live benchmark (run_id 6) — 14
    judge calls, zero errors, hard-failure agreement 8/8, `average_
    score=0.5` because every fixture had at least one of 7 categorical
    dimension mismatches (36/56 individual dimensions agreed — the
    all-or-nothing per-fixture score is a stricter, different
    measurement from dimension-level agreement, not evidence "the
    judges failed"). A calibration audit classified every mismatch as
    a fixture defect, a stale expected label, a judge-prompt/schema
    issue, an aggregation issue, an intentional skip-semantics
    mismatch, or model variability, followed by one bounded offline
    pass (R6C.3a: 4 fixtures' expected labels corrected, 6 fixtures'
    prose corrected, every edit independently verified against
    evidence — never rewritten merely because a judge called it
    partial) and 3 targeted live reruns (baseline, single-fixture
    stability, security) confirming the corrections held.
  - **Security validation** (run_id 9): both the source-side and
    report-prose injection fixtures remained fully blocked/rejected
    after calibration — poisoned evidence produces `insufficient_
    evidence`, never fabricated support; the report-prose injection is
    redacted before the holistic judge and, when seen by the claim/
    source judge as ordinary claim text, is fact-checked and rejected
    (`unsupported`), never obeyed. No injection bypass found.
  - **Accepted, documented residual policy debt** (not fixed): the
    strict "any partial claim fails the whole report" groundedness
    rule doesn't cleanly separate all 8 synthetic fixtures;
    Analytical/Expert-template synthesis often reads as
    `partially_supported` to the strict per-claim verifier even when
    it's defensible, evidence-adjacent inference; no materiality/
    severity threshold was invented from 8 synthetic fixtures;
    `good_foundational.template_fit` showed borderline stability
    across 3 repeated live calls (pass/fail/pass); synthetic fixtures
    remain synthetic, not a substitute for R6E's real human-labelled
    dataset; judge cost/latency/reliability need production-scale
    measurement; evidence scope is abstracts/snippets, never full
    papers; `REPORT_QUALITY_JUDGE_MODEL` availability/pricing is
    environment/account dependent; `.env` loading is not uniform
    across suite import paths (`uv run --env-file .env ...` is now the
    documented live-run command form). Full list: `specs/
    report-quality-evaluation-plan.md` §14.
  - See `docs/evaluation.md`'s "R6C.1" through "R6C.3" sections and
    `docs/architecture.md`'s "R6C" section for the full narrative, and
    `specs/report-quality-evaluation-plan.md` §12-14 for the frozen
    aggregation semantics and residual-debt record.
- **Location**: `research_agent/evals/report_quality_inputs.py` (new),
  `research_agent/evals/judges/claim_source.py` (new), `research_
  agent/evals/judges/holistic.py` (new), `research_agent/evals/
  runners/run_report_quality.py` (updated — `predict_live`,
  `_aggregate_claim_source_dimensions`, `CITATION_AGGREGATION_POLICY_
  VERSION`), `tests/test_evals_report_quality.py` (updated, many new
  tests across the R6C.1-R6C.3 phases), `eval_data/report_quality/`
  (4 fixtures' expected labels corrected, 6 fixtures' prose corrected,
  manifest notes updated), `eval_results/report_quality_history.csv`
  (run_ids 2-9), `docs/evaluation.md` / `docs/architecture.md` /
  `specs/report-quality-evaluation-plan.md` / `eval_data/report_
  quality/README.md` / `eval_results/README.md` (updated). **No
  changes to `research_agent/report.py`, `research_agent/qa.py`,
  `research_agent/curation_session.py`, `research_agent/api_app/`,
  `frontend/`, `pyproject.toml`, or `uv.lock`.**
- **Explicitly NOT part of this checkpoint**: R6C is not wired into
  `research_agent/report.py`'s runtime generation path anywhere and
  displays no per-report pass/fail to end users. The report-generation
  prompt-injection gap (see the security-debt entry below) is not
  closed by R6C — R6C's own injection fixtures test the *evaluation
  harness's* defenses, not `report.py`'s.
- **Deferred follow-ups, still open** (not addressed by R6C):
  - R6D — paired refinement-effectiveness evaluation (see next entry).
  - R6E — human-labelled calibration, materiality/severity
    classification, threshold calibration, judge stability/repeated-
    run study, production observability.
  - The report-generation prompt-injection gap (separate entry below).
- **Priority**: n/a — done (R6D is next).
- **Status**: Closed (2026-08-11). Commits `cf60191`, `bf8541d`,
  `d0c4982`, `ff67113` (R6C.2/2a/2b/2c); `2544e4e`, `f0eea0a`,
  `3a14d6f`, `193b27c` (R6C.3/R6C.3a).

### R6D.1: pairwise refinement-effectiveness fixture schema and fixtures
- **Goal**: freeze the pair schema and build the first synthetic pair
  fixtures R6D.2+ will run judges against — schema/fixtures/loader
  only, no judge, no aggregation, no runtime call.
- **Why it matters**: R6D's eventual purpose is measuring whether
  `research_agent/report.py`'s refinement/revision step
  (`revise_report`, `refine_report_if_requested`) actually improves
  report quality — not assumed, measured. Before that measurement can
  happen, the *shape* of a pair comparison and a reviewable set of
  synthetic pairs need to exist, the same "design + fixtures before
  code" sequencing R6A used for R6B/R6C.
- **Decision/what shipped**:
  - Pair schema (`schema_version: "r6d1-v1"`): `draft_report`/
    `refined_report` reuse the real stored report-dict shape R6A/R6C
    already validated; `selected_papers`/`approved_web_articles`
    shared once at the pair level (loader rejects a report that embeds
    its own copy); `expected.hard_failure_direction` plus a per-
    dimension `dimension_directions` block (`{direction, rationale}`
    for each of R6C's 7 frozen dimensions) — direction is one of
    `improved`/`unchanged`/`regressed`/`unknown`. **No
    `overall_direction`, `overall_score`, `accept_refinement`, or
    `winner` field anywhere** — the loader actively rejects a fixture
    that adds one; those are calibration decisions for a later phase.
  - `research_agent/evals/report_refinement_inputs.py` — manifest+
    fixture loader, `ReportRefinementFixtureError`, and 14 enforced
    pair invariants: unique/matching id, exact schema version, strict
    path containment, template agreement (pair + both reports),
    shared-evidence-only (no per-report duplication), canonical
    8-section order (checked directly, not assumed), structural
    validity linked to `hard_failure_direction` (an independent copy
    of R6A/R6B's 6 hard-failure checks — never imports R6C's own
    `run_report_quality.py`), every reference resolving in the shared
    evidence pool, complete non-empty per-dimension rationale,
    `revision_applied` matching real report equality/inequality in
    both directions, and no fixture rationale text leaking verbatim
    into report content. Also exports `reports_are_equal`/`diff_
    report_sections` (deterministic helpers) and the canonical
    `REQUIRED_DIMENSION_NAMES`/`VALID_DIRECTIONS`/`VALID_TEMPLATES`
    constants for R6D.2 to reuse. Never mutates a loaded fixture dict.
  - 7 fixtures under `eval_data/report_refinement/`, a fresh two-paper
    synthetic evidence pool (SpanCite, DriftGuard — distinct from
    R6A/R6C's ChunkRank/LongMem/CiteGuard set): `clear_grounding_
    improvement`, `holistic_synthesis_improvement`, `justified_no_
    revision`, `cosmetic_rewrite_tie`, `citation_regression`,
    `mixed_tradeoff`, `structural_regression` — covering all 3
    templates, paper+web evidence, a grouped `[1][2]` citation, and
    every direction value at least once. See `eval_data/report_
    refinement/README.md` for the per-fixture intended-direction table.
  - See `docs/evaluation.md`'s "R6D.1" section for the full narrative.
- **Location**: `research_agent/evals/report_refinement_inputs.py`
  (new), `eval_data/report_refinement/` (new: `README.md`,
  `manifest.jsonl`, `fixtures/*.json` ×7), `tests/test_evals_report_
  refinement.py` (new, 58 tests), `eval_data/README.md` / `docs/
  evaluation.md` / `specs/report-quality-evaluation-plan.md` (updated).
  **No changes to any CLI suite registration, `research_agent/report.py`,
  `research_agent/evals/runners/run_report_quality.py`, `research_
  agent/evals/report_quality_inputs.py`, `research_agent/evals/
  judges/`, `eval_results/`, `pyproject.toml`, or `uv.lock`.**
- **Explicitly NOT part of this checkpoint**: no live judge of any
  kind, no mock/live CLI suite, no pairwise aggregation/scoring, no
  blinded A/B ordering or swap-order logic, no citation-integrity-
  preservation check against a real revision, no result CSV. **No
  claim that refinement is effective** — that is what R6D.2+ has to
  measure, not assume.
- **Priority**: n/a — done (R6D.2 is next).
- **Status**: Closed (2026-08-11).

### R6D.2: deterministic/mock pair-evaluation runner
- **Goal**: register a real `report_refinement` CLI suite and actually
  run something against R6D.1's 7 pair fixtures — deterministic/mock
  only, mirroring R6B's own "deterministic first, live later"
  sequencing before R6C existed.
- **Why it matters**: R6D.1 froze the pair schema but ran nothing;
  R6D.2 is where a report pair's *structural* direction (R6B's own 6
  hard-failure identifiers) becomes an actual, checkable measurement,
  with the 7 semantic R6C dimensions explicitly left unmeasured rather
  than silently assumed.
- **Decision/what shipped**:
  - `research_agent/evals/runners/run_report_refinement.py` —
    `predict()` evaluates `draft_report` and `refined_report`
    independently by calling `run_report_quality.predict()` directly
    (wrapped in a throwaway `Example`) for each side — the exact same
    function R6B's own suite uses, never a second interpretation of
    the 6 hard-failure identifiers. `_hard_failure_direction` derives
    `improved`/`unchanged`/`regressed`/`mixed` from the two resulting
    failure sets, set-subset-based rather than count-based (a report
    that fixes 3 defects while introducing 1 new one is `mixed`, not
    `improved`, even though its raw failure count went down) —
    `mixed` is checked first and never collapsed into `unchanged`.
    `dimension_directions` is always `None` and `semantic_evaluation_
    status` is always `"not_evaluated_in_mock_mode"` — never inferred
    from informational signals, never copied from a fixture's own
    `expected.dimension_directions`.
  - `research_agent/evals/evaluators/report_refinement.py` — two
    evaluators mirroring `report_quality.py`'s own pair exactly:
    `report_refinement_hard_failure_direction_agreement` (1.0/0.0,
    the only thing mock mode scores) and `report_refinement_semantic_
    dimensions_not_evaluated` (always `score=None`, present to make
    "not measured yet" explicit in every run's own detail JSON).
  - `research_agent/evals/cli.py` — `report_refinement` registered
    alongside `chat_relevance`/`report_quality`. `--mode live` raised
    `LiveModeSetupError` (exit 2, no traceback, no CSV/detail side
    effects, truthful "not implemented until R6D.3" message) at the
    time this checkpoint closed — implemented in R6D.3, below.
  - Mock baseline: `total=7, passed=7, failed=0, average_score=1.000`
    — **this score is structural hard-failure-direction agreement
    only, never a report-quality or refinement-effectiveness score.**
  - See `docs/evaluation.md`'s "R6D.2" section for the full record.
- **Location**: `research_agent/evals/runners/run_report_refinement.py`
  (new), `research_agent/evals/evaluators/report_refinement.py` (new),
  `research_agent/evals/cli.py` (updated, additive), `tests/test_
  evals_report_refinement.py` (updated, +41 tests), `eval_results/
  report_refinement_history.csv` (new, tracked), `docs/evaluation.md`
  / `eval_data/report_refinement/README.md` / `eval_results/README.md`
  (updated). **No changes to `research_agent/report.py`,
  `research_agent/evals/runners/run_report_quality.py`,
  `research_agent/evals/report_quality_inputs.py`, `research_agent/
  evals/judges/`, `research_agent/evals/report_refinement_inputs.py`,
  any existing fixture, `eval_results/report_quality_history.csv`,
  `eval_results/chat_relevance_history.csv`, `pyproject.toml`, or
  `uv.lock`.**
- **Explicitly NOT part of this checkpoint**: no R6C live judge is
  called anywhere. No blinded A/B ordering, no swap-order logic, no
  citation-integrity-preservation check against a real revision, no
  human-labelled pair. **No claim that refinement improves report
  quality** — R6D.2 answers a narrower, purely structural question.
- **Priority**: n/a — done (R6D.3 is next).
- **Status**: Closed (2026-08-11). Commit `e97b910` (R6D.1);
  R6D.2's own commit follows in this same backlog update.

### R6D.3: live paired semantic judging (independent per-side judging, not a blinded pairwise judge)
- **Goal**: run R6C's claim/source and holistic judges against both
  halves of a pair and compute real *semantic* directions for the 7
  R6C dimensions — the question R6D.2 explicitly left unmeasured.
- **Design actually built (deviates from `specs/report-quality-
  evaluation-plan.md` §8's original blinded-A/B sketch)**: each side
  of a pair is judged **completely independently** through R6C's
  existing single-report live path (`run_report_quality.predict_
  live`, reused directly — no second judge implementation, no third
  pairwise LLM judge, no blinded `Report A`/`Report B` labels, no
  swap-order calls). Direction is derived afterward in plain Python by
  comparing the two already-independent results
  (`run_report_refinement._dimension_direction`, rules A-G — see
  `docs/evaluation.md`'s "R6D.3" section for the full rule set). This
  was the explicit scope given for this checkpoint; the §8 blinded-A/B
  design, `positional_disagreement`, and the citation-integrity-
  preservation check remain **not built** and are not re-scoped here.
- **Cost bound**: at most 4 judge calls per pair (1 claim/source + 1
  holistic, per side) — never a 5th. Identical-input optimization: a
  `revision_applied=false` pair with byte-identical `draft_report`/
  `refined_report` (exact equality, `report_refinement_inputs.
  reports_are_equal`) is judged once and deep-copied, not twice.
- **Direction thresholds**: `citation_correctness`/`groundedness`
  direction is categorical-only (never score-derived); the 5 holistic
  dimensions use `HOLISTIC_DIRECTION_MIN_DELTA = 0.10` —
  **explicitly provisional and uncalibrated**, not derived from any
  calibration study.
- **Live evaluator**: `report_refinement_semantic_direction_agreement`
  (`score = matched / 7` against a fixture's own
  `expected.dimension_directions`, `unknown` never a wildcard) —
  labeled expectation agreement, never an invented overall quality
  score or winner.
- **Mock mode unchanged/byte-compatible**: `predict()`, the hard-
  failure-direction rules, and the mock evaluator pair are exactly as
  R6D.2 left them.
- **Tests**: `tests/test_evals_report_refinement.py` → 144 passed (was
  99), all against a mocked OpenAI/judge boundary — no real paid call
  made anywhere in this checkpoint's implementation or validation.
  `report_quality` + `report_refinement` together → 336 passed. Full
  backend suite → 1116 passed.
  Mock non-regression re-run: `total=7, passed=7, failed=0,
  average_score=1.000` (unchanged from R6D.2's own baseline).
- **Location**: `research_agent/evals/runners/run_report_refinement.py`
  (rewritten to add `predict_live` + helpers, mock `predict()`
  unchanged), `research_agent/evals/evaluators/report_refinement.py`
  (new `report_refinement_semantic_direction_agreement`, existing 2
  evaluators unchanged), `research_agent/evals/cli.py` (`live_warning`
  text updated, additive), `tests/test_evals_report_refinement.py`
  (updated, +45 tests), `docs/evaluation.md` / `eval_data/report_
  refinement/README.md` / `eval_results/README.md` (updated). **No
  changes to `research_agent/report.py`, `research_agent/evals/
  runners/run_report_quality.py`, `research_agent/evals/judges/`,
  `research_agent/evals/report_quality_inputs.py`, `research_agent/
  evals/report_refinement_inputs.py`, any existing fixture, R6C's
  prompts or aggregation rules, or the production R4 refinement loop.**
- **Priority**: n/a — done (R6D.3a is next).
- **Status**: Closed (2026-08-11). Superseded by R6D.3a immediately
  below — R6D.3's own first paid live pair (run_id 3) surfaced real
  calibration problems in this checkpoint's design.

### R6D.3a: calibrate refinement evaluation around changed claims and paired holistic judgment
- **Goal**: fix two calibration problems R6D.3's own first paid live
  pair (run_id 3, `clear_grounding_improvement`, commit `4aae124`, 4
  judge calls, 35.6s, only 1/7 semantic-direction agreement) exposed:
  independent whole-report groundedness aggregation can hide a real,
  isolated fix behind unrelated judge-call sampling noise, and two
  independent standalone holistic calls are two independently sampled
  judgments that can (and did) disagree over content that never
  changed at all.
- **Fix, both required together**: (1) derive `citation_correctness`/
  `groundedness` direction from ONLY the claim units that actually
  changed between draft and refined (matched by `claim_id`, exact
  field equality on `claim_text`/`claim_kind`/`reference_numbers`/
  `evidence_ids` — never fuzzy similarity, never an LLM); (2) replace
  two independent standalone holistic calls with one pairwise call
  (`research_agent/evals/judges/refinement_holistic.py`, new prompt
  version `r6d3a-pairwise-holistic-v1`, independent of `judges/
  holistic.py`'s own `HOLISTIC_JUDGE_PROMPT_VERSION`) that sees both
  reports together and returns direction only (`improved`/`unchanged`/
  `regressed`/`unknown` + confidence + bounded reason) — never an
  absolute score, never a winner, never an accept/reject decision.
- **Cost bound dropped from 4 to 3** for a normal pair (1 claim/source
  call per side + 1 pairwise holistic call); identical-pair
  optimization tightened to 1 call (pairwise holistic is skipped
  entirely for byte-identical reports, not just deep-copied).
  `HOLISTIC_DIRECTION_MIN_DELTA`'s 0.10 score-subtraction rule is fully
  removed from the live pair path — superseded by the pairwise judge's
  own direct direction output, not merely deprecated in place.
- **Extraction**: `run_report_quality.prepare_and_judge_claims_only`
  factors `predict_live`'s claim/source step out (no standalone
  holistic call), reused directly by `report_refinement`'s own live
  path; `predict_live` itself calls this same function internally, and
  `report_quality`'s own full, UNMODIFIED test suite (192 tests)
  continues to pass byte-for-byte, proving equivalence.
- **Fixture correction**: `clear_grounding_improvement`'s `expected.
  dimension_directions.citation_correctness.direction` changed
  `unchanged` → `improved` (the draft's attached source genuinely
  `does_not_support`s the "eliminates all" overclaim; the refined
  claim's identical source genuinely `supports`s the accurate
  restatement — a real fail→pass transition on the one claim that
  changed, under the new changed-claim-only derivation). No other
  fixture, expected direction, report prose, or evidence touched.
- **Tests**: `tests/test_evals_report_refinement.py` → 175 passed (was
  144). `report_quality` + `report_refinement` together → 367 passed.
  Full backend suite → 1147 passed. No real paid live call made
  anywhere in implementation or validation. Mock non-regression re-run:
  `total=7 passed=7 failed=0 average_score=1.000` (run_id 4, commit
  `84b75d8`, note "R6D.3a mock non-regression").
- **Location**: `research_agent/evals/runners/run_report_quality.py`
  (extraction, `predict_live` refactored but behavior-equivalent),
  `research_agent/evals/runners/run_report_refinement.py` (rewritten
  live path), `research_agent/evals/judges/refinement_holistic.py`
  (new), `research_agent/evals/cli.py` (`live_warning` text updated to
  the 3-call bound), `eval_data/report_refinement/fixtures/clear_
  grounding_improvement.json` (citation_correctness expectation
  corrected), `tests/test_evals_report_refinement.py` (+31 tests net),
  `docs/evaluation.md` / `eval_data/report_refinement/README.md`
  (updated). **No changes to `research_agent/report.py`, `research_
  agent/evals/judges/claim_source.py`, `research_agent/evals/judges/
  holistic.py`'s own prompt/version, `research_agent/evals/report_
  quality_inputs.py`, `research_agent/evals/report_refinement_
  inputs.py`, any OTHER fixture, `CLAIM_SOURCE_JUDGE_PROMPT_VERSION`,
  `HOLISTIC_JUDGE_PROMPT_VERSION`, `CITATION_AGGREGATION_POLICY_
  VERSION`, `report_quality`'s own CLI output/behavior, or the
  production R4 refinement loop.**
- **Explicitly NOT part of this checkpoint**: no extra refinement
  round, no change to R4's production report generation/refinement, no
  new rubric dimension, no threshold calibration beyond removing the
  old provisional 0.10 delta, no paid live evaluation run, no overall
  winner/acceptance decision.
- **Priority**: n/a — done. Next: a deliberately small rerun of ONLY
  `clear_grounding_improvement` live (same single-pair scope as run_id
  3) to confirm the recalibrated pipeline now agrees with its own
  corrected expectation, then R6D.4.
- **Status**: Closed (2026-08-11). Followed by R6D.3b (fixture
  adjudication of `clear_grounding_improvement`'s analytical_quality/
  template_fit/coherence, run_id 5) and R6D.3c (coherence-boundary
  clarification after run_id 6's own stability check disagreed with
  run_id 5 on exactly `coherence`; prompt version bumped to
  `r6d3c-pairwise-holistic-v2`, `coherence` reverted to `unchanged`)
  — see `docs/evaluation.md`'s "R6D.3b"/"R6D.3c" sections for the full
  record. **R6D.3c is the final calibration allowed for this fixture**;
  one more paid stability run may still confirm it, after which any
  remaining disagreement is documented as a residual judge-stability
  limitation rather than tuned further. Both closed (2026-08-11).

  **R6D closure (2026-08-11)**: run_id 7 (commit `398e5a1`, "R6D.3c
  final clear-grounding stability run") confirmed the coherence
  clarification and finished at 6/7 semantic-direction agreement —
  `template_fit` was the one remaining disagreement (`improved` on
  runs 5/6, `unchanged` at confidence 0.62 on run 7). **`clear_
  grounding_improvement`'s calibration is now CLOSED — `template_fit`
  is NOT tuned again.** This is accepted as residual judge/rubric
  ambiguity, documented rather than repeatedly chased; exact 7/7
  agreement on one hand-authored fixture was never the bar for this
  machinery being useful. See `docs/evaluation.md`'s "R6D closure"
  section for the full record, including the explicit non-claim that
  this synthetic-fixture calibration says nothing about whether
  production R4 refinement improves real reports (R6D.4 remains
  necessary for that). **This does NOT close R6D as a whole** — only
  this one fixture's calibration. Next: calibrate `holistic_synthesis_
  improvement` (cross-source synthesis, analytical distinction,
  coherence improvement) under the same stopping rule — read-only
  adjudication first, code changed only for a reproducible
  implementation defect, never to force perfect agreement.

  **`holistic_synthesis_improvement` fixture correction (2026-08-11)**:
  run_id 8 (commit `71cffbe`, "R6D holistic-synthesis live validation")
  found `groundedness`/`coherence` regressions and a `template_fit`
  disagreement (5/7). Read-only adjudication against the fixture's own
  evidence (applying the stopping rule established above) found the
  `groundedness`/`coherence` regressions were a REAL fixture-evidence
  defect, not a judge false positive: the refined prose incorrectly
  described SpanCite as checking sentences "after generation", directly
  contradicting both SpanCite's own abstract (decoding-time, before-
  emission) and this fixture's own unchanged Methodology section.
  Corrected against the frozen abstract; `template_fit` separately
  adjudicated `unchanged → improved` against the frozen Analytical-
  template definition. **No judge or runner code changed** — fixture-
  only, same pattern as `clear_grounding_improvement`'s own R6D.3b. See
  `docs/evaluation.md`'s "R6D — `holistic_synthesis_improvement` live
  validation, run 8" section for the full record. One corrected live
  rerun remains; **R6D as a whole is still not closed.**

  **`holistic_synthesis_improvement` closed (2026-08-11)**: run_id 9
  (commit `70d70d5`, "R6D corrected holistic-synthesis validation")
  confirmed `synthesis_quality`/`analytical_quality`/`template_fit` are
  stable (6/7). One further evidence-language defect was found and
  corrected (a broad "grounding failures" umbrella statement narrowed
  to name SpanCite's/DriftGuard's two distinct, evidence-established
  intervention points); `groundedness` separately adjudicated
  `unchanged → improved`. `coherence` disagreed with the live run
  (predicted `regressed`, expected `improved`) and is recorded as a
  documented residual rubric-ambiguity limitation, NOT corrected again
  — the coherence rationale itself was deliberately left untouched.
  **No judge or runner code changed. `holistic_synthesis_improvement`
  is now CLOSED — no further live run will be performed for this
  fixture.** See `docs/evaluation.md`'s "R6D closure —
  `holistic_synthesis_improvement` is closed" section for the full
  record. Next: the two no-change controls, `justified_no_revision`
  and `cosmetic_rewrite_tie`. **R6D as a whole is still not closed.**

  **R6D no-change controls, run 10 (2026-08-11)**: run_id 10 (commit
  `5cfadb1`, tags `tie`, "R6D no-change controls validation") ran both
  no-change control fixtures: 2/2 passed, 1.0 average expectation
  agreement. `justified_no_revision` (byte-identical draft/refined)
  confirmed the identical-input cost optimization (1 call, pairwise
  holistic skipped) and all 7 dimensions correctly `unchanged` — a
  strong draft receiving no revision is never falsely labelled
  improved. `cosmetic_rewrite_tie` (reworded, no substantive change)
  confirmed the full 3-call changed-pair path still correctly returns
  all 7 dimensions `unchanged` — cosmetic rewriting alone is never
  rewarded as semantic improvement. **No fixture, judge, prompt, or
  runner correction was needed.** These results support two candidate
  bounded-refinement stopping principles (don't revise a report that
  already passes; don't accept a revision merely because it rewrites
  the prose) as evaluation EVIDENCE only — **not wired into production
  R4**, pending R6D.4's real output pairs. See `docs/evaluation.md`'s
  "R6D no-change controls, run 10" section. Next: `citation_regression`
  (semantic/citation damage from an apparently smoother revision) and
  `structural_regression` (misleading semantic approval of a
  structurally invalid refinement). **R6D remains open.**

  **R6D regression controls, run 11 (2026-08-11)**: run_id 11 (commit
  `921838b`, tags `regression`, "R6D regression controls validation")
  ran both regression control fixtures: 1/2 passed. `structural_
  regression` matched exactly as frozen (`hard_failure_direction=
  regressed`, all 7 semantic dimensions `unknown`, semantic judging and
  the pairwise holistic call both correctly skipped) — **structural
  hard-failure gating works exactly as designed**: a structurally
  invalid refined report never receives a misleading semantic verdict.
  `citation_regression` matched `citation_correctness`/`groundedness`
  but all 5 holistic dimensions came back `regressed` where the
  fixture originally expected `unchanged` (5/7). Read-only adjudication
  against the frozen rubric found the judge was right: the same
  misattribution (DriftGuard's finding wrongly reattributed to
  SpanCite) necessarily also damages `synthesis_quality` (lost
  cross-source contrast), `analytical_quality` (a false attribution),
  `template_fit` (the Analytical template's cross-source framing
  collapses), `coherence` (the refined Thematic Findings now
  contradicts the unchanged Methodology Landscape section), and
  `source_balance` (Thematic Findings' representation shifts to
  SpanCite-centred). **All 7 dimensions now expected `regressed`** —
  report prose/evidence/references/citation markers deliberately left
  untouched, only expectations corrected. **No judge or runner code
  changed.** Both regression-control fixtures are now CLOSED. See
  `docs/evaluation.md`'s "R6D regression controls, run 11" section.
  Next and final synthetic fixture: `mixed_tradeoff`. **R6D and R6
  remain open — production refinement is not proven effective by any
  synthetic-fixture checkpoint; that remains R6D.4's job.**

  **R6D synthetic-fixture stage CLOSED, run 12 (2026-08-11)**: run_id
  12 (commit `e2b1b38`, tags `tradeoff`, "R6D mixed-tradeoff
  validation") ran the last of the 7 synthetic fixtures, `mixed_
  tradeoff`: 3 judge calls, no errors, 73.3s, 5/7 matched. Read-only
  adjudication confirmed a genuine multidimensional trade-off: the
  refined report's unsupported "industry-standard best practice"
  overclaim simultaneously damages `citation_correctness` (neither
  source establishes it), `groundedness` (unsupported claim, already
  correctly expected), `analytical_quality` (evidence overreach), and
  `coherence` (contradicts the report's own stated narrow scope) —
  while the SAME refinement genuinely improves `synthesis_quality`
  (real controlled-vs-production comparison, already correctly
  expected) and `template_fit` (that comparison better fits the Expert
  template). `source_balance` unchanged. **No overall winner, accept/
  reject field, or composite score anywhere in the fixture.** **No
  judge, runner, or production code changed.**

  **This closes R6D's synthetic-fixture calibration stage — all 7
  hand-authored pairs (clear_grounding_improvement, holistic_
  synthesis_improvement, justified_no_revision, cosmetic_rewrite_tie,
  citation_regression, structural_regression, mixed_tradeoff) have now
  been exercised live and adjudicated at least once.** Accepted
  limitations carried forward: some holistic dimensions have genuinely
  overlapping boundaries (multiple checkpoints found one edit
  legitimately touching several frozen dimensions at once); `template_
  fit`/`coherence` showed real instability across repeated runs on the
  same fixture and were accepted as residual judge/rubric ambiguity
  rather than tuned indefinitely; live latency ranged materially
  (16.4s–73.3s) with no SLO established. **None of this proves
  production R4 refinement helps.** See `docs/evaluation.md`'s "R6D
  synthetic-fixture stage: closed" section for the full record.
  **R6D remains open. R6 remains open. No production refinement
  decision has been made.** Next: **R6D.4** — evaluate a deliberately
  small set of real R4-generated draft/refined pairs.

### R6D.4: evaluate real R4-generated draft/refined report pairs
- **Goal**: run R6D's live paired evaluation against *real* R4 output
  (actual `generate_report_for_session`/`refine_report_if_requested`
  draft/refined pairs from real or realistic topics), not R6D.1's 7
  synthetic fixtures — the first point at which this project can make
  an actual, evidence-backed claim about whether refinement improves
  report quality.
- **Why it matters**: R6D's full synthetic-fixture stage (R6D.1-R6D.3c,
  closed 2026-08-11 — see that section above) validates that the
  *measurement* machinery works correctly against hand-authored
  fixtures with known, human-adjudicated answers. None of that is
  evidence about real R4 behavior at scale.
- **Status**: **Closed (2026-08-11)**. All four sub-phases (R6D.4a-d,
  below) complete — 3 real R4 pairs captured, adjudicated, and live-
  evaluated once (bounded, not rerun); see R6D.4d and the top-level "R6
  — closed" entry below for the final result and product decision.

#### R6D.4a: developer-only capture helper — complete (2026-08-11)
- **Goal**: an in-memory, eval-only helper that captures a genuine R4
  draft/refined pair using the exact real production generation path,
  without touching production persistence.
- **What shipped**: `research_agent/evals/r6d4_capture.py` —
  `capture_real_refinement_pair(session, client, *, report_template,
  pair_id, source_session_ref, ...)` calls `report.generate_report_
  for_session` then `report.refine_report_if_requested` (`refinement_
  mode="single"`, `web_articles=[]` — the same literal empty list
  `get_or_create_report`'s own initial-generation call hardcodes, never
  `session.web_articles_added`) and returns a complete, **unlabelled**
  `r6d4-capture-v1` artifact dict. Deep-copies (and JSON-native-strips
  — see below) the draft BEFORE it is ever handed to refinement, since
  `refine_report_if_requested` mutates its own `draft` argument in
  place on the no-revision branch. A separate `validate_r6d4_capture`
  pure validator checks schema version, template validity, revision-
  applied/body-equality consistency, canonical section shape, evidence-
  reference resolution, absence of any `refinement`/`expected`/
  `winner`/`accept_refinement` key, and a recursive forbidden-key scan
  (`session_id`/`chat_history`/`turn_history`/`pending_web_offer`/
  `pending_report_update`/`api_key`/`openai_api_key`) — kept completely
  separate from `report_refinement_inputs.py`'s own frozen `r6d1-v1`
  synthetic loader (proved via AST inspection that neither module
  imports the other).
- **No production/API/session change**: no schema, route, or service
  function touched; `append_report_version`/`save_curation_session` are
  never imported (not just "never called this time" — structurally
  impossible); `session.report`/`report_versions`/`active_report_
  version_id` are never assigned to.
- **Unlabelled by construction, on purpose**: no `expected` block, no
  overall-winner/composite field, matching the same "no overall score,
  weighted composite, or winner field" requirement R6D.1's own schema
  already enforces. **If human labels are ever collected for a
  captured pair, they must be frozen BEFORE that pair's own R6D live
  evaluation runs** — never assigned or corrected after seeing live
  judge output, which would be post-hoc answer-key bias.
- **Tests**: `tests/test_evals_r6d4_capture.py`, 54 passed, all against
  a mocked client (no real network call, no real report captured, no
  file written). `test_report.py` + this file → 190 passed
  (non-regression). Full backend suite → 1280 passed.
- **Location**: `research_agent/evals/r6d4_capture.py` (new),
  `tests/test_evals_r6d4_capture.py` (new), `docs/evaluation.md` /
  `specs/backend-backlog.md` (updated). No other file touched.
- **Priority**: next, **R6D.4b** — CLI capture command + atomic
  artifact-file writing/validation (still no paid call required to
  build); then **R6D.4c** — capture a maximum of 3 real pairs (the
  first phase that spends real R4 calls).
- **Status**: Closed (2026-08-11). **Does not close R6D.4 or R6D or R6
  as a whole** — no real report has been captured, and no live
  evaluation of a real pair has happened under this schema yet.

#### R6D.4b: guarded CLI for capture/validate — complete (2026-08-11)
- **Goal**: an explicit, guarded CLI over R6D.4a's capture helper — a
  session-loading `capture-refinement` command and a read-only
  `validate-refinement-capture` command.
- **What shipped**: two new commands in `research_agent/evals/cli.py`.
  `capture-refinement` requires an explicit `--allow-paid-calls` flag
  (zero side effects without it — no session load, no `OpenAI()`, no
  file); loads the requested session read-only through the exact same
  `research_agent.qa.sqlite_checkpointer`/`research_agent.curation_
  session.load_curation_session` pair production's own FastAPI
  dependency is built from (no second SQLite interpretation); enforces
  the same 3 preconditions `get_or_create_report` itself does (session
  exists, `stage=="synthesize"`, non-empty `selected_papers`), never
  echoing the raw `--session-id` in any output; refuses to overwrite an
  existing destination file (checked before `OpenAI()` construction, no
  `--force` in this chunk); writes the artifact atomically (serialize
  → temp file in the same directory → flush+fsync → `os.replace`, temp
  file removed on any failure). `validate-refinement-capture` makes
  zero OpenAI calls, loads no session, writes nothing, and rejects a
  synthetic `r6d1-v1` fixture as the wrong schema rather than silently
  treating it as a real capture.
- **Captures are gitignored**: `eval_results/captures/` added to
  `.gitignore` — a temporary, local working area, separate from
  `eval_data/report_refinement/`'s tracked synthetic fixtures and
  `eval_results/runs/`'s judge-detail files.
- **Tests**: `tests/test_evals_r6d4_cli.py`, 43 passed, all mocked (no
  real network call, no real session, no real report captured).
  Combined with R6D.4a's own tests → 97 passed. Full backend suite →
  1323 passed.
- **Location**: `research_agent/evals/cli.py` (updated), `tests/
  test_evals_r6d4_cli.py` (new), `.gitignore` / `docs/evaluation.md` /
  `specs/backend-backlog.md` / `eval_results/README.md` (updated).
- **Priority**: next, **R6D.4c** — read-only candidate-session
  inventory, then precommit exactly 3 template slots (Foundational/
  Analytical/Expert) BEFORE any paid capture call.
- **Status**: Closed (2026-08-11). **Does not close R6D.4** — no real
  report has been captured under this CLI yet.

#### R6D.4c: precommit and capture 3 real pairs — complete (2026-08-11)
- **Goal**: spend the real, paid R4 calls — capture exactly 3 real
  draft/refined pairs (Foundational, Analytical, Expert), template
  slots precommitted before any capture call so no pair can be dropped
  or replaced after seeing its own content.
- **What shipped**: `capture-refinement --allow-paid-calls` run once
  per precommitted slot against R6D.4b's own CLI. Written to
  `eval_results/captures/real-{foundational,analytical,expert}-01.json`
  (gitignored, never committed), each independently `validate_r6d4_
  capture`-clean. Foundational and Expert came back byte-identical
  (`revision_applied=false` — R4 chose not to revise either draft);
  Analytical is the one pair R4 actually revised (`revision_applied=
  true`) — the only one of the three that can measure a real
  refinement effect.
- **Status**: Closed (2026-08-11). Does not close R6D.4 — no
  adjudication or live evaluation exists yet.

#### R6D.4d: adjudication, real-capture evaluation adapter, one bounded live run, and structural correction — complete, R6D closed (2026-08-11)
- **Goal**: adjudicate the 3 real captures (blind, frozen before any
  live result), build the minimal adapter to run R6D's existing live
  machinery against them, run it exactly once, and correct any
  objectively-verifiable label defect the run surfaces — without ever
  rewriting the run's own recorded result.
- **What shipped**: blind human (AI-assisted, human-confirmed) review
  of Analytical against R6C's 7 dimensions, frozen and committed
  BEFORE the draft/refined mapping was read, then mechanically
  translated into R6D's own vocabulary (`eval_data/report_refinement/
  real_reviews/*.json`, schemas `r6d4-review-v1`/`r6d4-adjudication-
  v1`); Foundational/Expert needed no blind review — byte-identical
  bodies make all 7 directions `unchanged` by construction. A new
  `report_refinement_real` suite (`research_agent/evals/report_
  refinement_real_inputs.py` + `research_agent/evals/runners/run_
  report_refinement_real.py`) reuses `run_report_refinement.py`'s own
  `predict`/`predict_live` and evaluators completely unchanged, hash-
  binds each capture to its adjudication before constructing any
  OpenAI client, and logs to its own separate history/detail files. One
  live run (run_id 2, commit `003ad81`, 3 real judge calls, never
  rerun): Foundational/Expert matched the frozen labels exactly (7/7
  dimensions each); Analytical's refined report tripped R6B's own
  deterministic structural gate (a newly-introduced `orphan_reference`
  the draft didn't have), so its 7 semantic dimensions were correctly
  forced `unknown` rather than judged. Afterward, `real-analytical-01`'s
  `hard_failure_direction` label (`unchanged`, an unverified assumption
  carried from the blind review) was independently re-checked against
  R6B's deterministic checker and corrected to `regressed` — objective
  and mechanically reproducible (2 orphan references, #9/#10, present
  in `refined_report` only), never a re-judgment; all 7 semantic
  `dimension_directions` stayed byte-identical. Run_id 2's own CSV row
  and detail JSON (`average_score=0.6667`) were never rewritten or
  rerun — the correction changes only the label a future run would be
  compared against.
- **Result**: directional evidence, not statistical proof. Full
  narrative, per-pair/per-dimension breakdown, and the evaluator-
  validity-vs-refinement-effectiveness distinction: `docs/
  evaluation.md`'s "R6D.4d" section.
- **Final product decision**: retain "Refine Once" as optional, **off
  by default**; no autonomous multi-round refinement; a later phase
  should target specific sections for revision instead of regenerating
  all eight; require human comparison/approval before a revision
  replaces the draft; preserve both versions for rollback. See the "R6
  follow-on" backlog entry below for the concrete future items this
  motivates, kept explicitly outside R6's own scope.
- **Tests**: `tests/test_evals_report_refinement_real.py`, 34 passed
  (30 adapter/loader + 4 structural-correction regression, none calling
  a real API). Full backend suite → 1357 passed.
- **Location**: `eval_data/report_refinement/real_reviews/*.json`,
  `research_agent/evals/report_refinement_real_inputs.py` (new),
  `research_agent/evals/runners/run_report_refinement_real.py` (new),
  `research_agent/evals/cli.py` (suite registration), `tests/
  test_evals_report_refinement_real.py` (new), `eval_results/
  report_refinement_real_history.csv` (new, tracked).
- **Status**: Closed (2026-08-11). **Closes R6D and R6 as a whole** —
  no further R6 calibration or live reruns are planned.

### R6 — closed (2026-08-11)
- **R6A**: froze the 7-dimension rubric, the 6 hard-failure identifiers,
  and the first synthetic fixture set.
- **R6B**: built the deterministic structural evaluator and
  informational signals (no judge model, no network call).
- **R6C**: built the two bounded live judges (claim/source, holistic),
  completed synthetic calibration; known limitations (see `docs/
  evaluation.md`'s "R6C.3" section) were documented and retained, not
  resolved by fiat.
- **R6D**: 7 synthetic refinement pairs completed (R6D.1-R6D.3c); 3 real
  R4 captures precommitted and captured (R6D.4c) — Foundational and
  Expert byte-identical no-revision controls, Analytical the only real
  revision; blind AI-assisted/human-confirmed adjudication found
  citation correctness regressed, synthesis quality and template fit
  improved, coherence regressed, groundedness/analytical quality/
  source balance unchanged; the one bounded live run (run_id 2) found
  the refined Analytical report had introduced an orphan reference, and
  structural gating correctly withheld semantic judgment rather than
  producing an unsupported one; **that live run's `average_score=
  0.6667` is permanent and was never rewritten**; after the run, the
  `hard_failure_direction` label was objectively corrected to
  `regressed` — but the run itself was not rerun.
- **Final product decision**: keep "Refine Once" optional and **off by
  default**; do not build an autonomous multi-round refinement loop;
  preserve both draft and revised versions; require human comparison/
  approval before a revision becomes active; future refinement work
  should target the affected section(s) instead of regenerating all
  eight. R6 provides directional evidence, not statistical proof of
  universal benefit or harm — three real pairs, one of them
  semantically evaluable, can never be that.
- **R6 is complete. No further R6 calibration or live reruns are
  planned.** Full narrative: `docs/evaluation.md`'s "R6A" through
  "R6D.4d" sections. Frozen result schema: `specs/report-quality-
  evaluation-plan.md`.

### R6 follow-on: targeted-section refinement, human approval, broader dataset, observability (not part of R6)
- **Why these are listed separately**: R6's own final product decision
  (above) motivates each of these, but none of them is an R6
  measurement task — R6 is closed; these are ordinary product/backend
  work items, gated on real product need, same as everything else in
  this backlog.
- **Targeted-section refinement**: R4's "Refine Once" currently
  regenerates all 8 sections even when the evaluator's own `issues`/
  `revision_instructions` point at 1-2 specific sections (see R4.2's
  own entry above for what's already persisted). A scoped design pass
  for revising only the affected section(s) — not a blind "smaller
  prompt" change — is real future work, informed by (not blocked on)
  R6D.4d's own finding that a full-report revision can introduce a new
  structural defect outside the section that actually needed fixing.
  Not started.
- **Human approval workflow**: this is the same item already tracked as
  **R4.4** above (optional human-in-the-loop refinement via a LangGraph
  `interrupt()`) — R6D.4d's final product decision ("require human
  comparison/approval before a revision becomes active") is additional
  motivation for it, not a new item. Not started.
- **Broader human-labelled real dataset**: R6D.4 evaluated exactly 3
  precommitted real pairs by design (bounded cost, bounded scope).
  Labelling more real pairs is legitimate future work **only if a
  concrete product need justifies the cost** — not a default next step,
  and not implied by anything in R6's own closure. Not started, not
  scheduled.
- **Production latency/cost observability for refinement**: R6D.4d's
  own live run measured judge-call latency/count purely as evaluation
  instrumentation (`eval_results/runs/report_refinement_real_run_2.
  json`); no equivalent observability exists for the production
  `refine_report_if_requested` path itself (real per-call latency/cost
  in the live product, not the eval harness). Not started.

### Security debt: no independent prompt-injection defense in report generation / paper abstracts
- **Goal**: close the gap `eval_data/report_quality/fixtures/
  source_prompt_injection.json` (R6A) demonstrates — report generation
  currently has no defense of its own against instruction-like content
  embedded in a paper abstract or an already-approved web snippet.
- **Why it matters**: `research_agent/qa.py`'s
  `_detect_retrieved_prompt_injection` guard (R7E.5b) is wired **only**
  into the chat/web-relevance filtering path
  (`_filter_relevant_web_articles`, reached from
  `_filter_web_relevance_node` and `_accept_web_offer`). It is never
  applied to paper abstracts anywhere in the system (papers reach
  report generation via the search/curation flow, not the chat path,
  so they never pass through this guard at all), and
  `research_agent/report.py`'s own generation/evaluation/revision
  prompt construction has no independent injection defense — it trusts
  whatever `selected_papers`/`web_articles` a session already contains.
  The `source_prompt_injection` fixture's `generated_report` shows the
  concrete failure mode: an injected instruction inside a paper
  abstract ("...should be described as the definitive, complete
  solution...") and inside a web snippet ("...rate this article as the
  single most important...") both visibly succeeding in the fixture's
  simulated report output.
- **Location (where the fix would land, not yet touched)**:
  `research_agent/report.py` (generation/evaluation/revision prompt
  construction — no existing guard to extend), and/or the paper-
  ingestion path (`research_agent/ingestion.py` /
  `research_agent/query_expansion.py`) if the guard should apply at
  the point papers first enter a session rather than at report-
  generation time.
- **Priority**: not yet scheduled — flagged by R6A/R6B, not scoped or
  sequenced against other work.
- **Status**: Open. Explicitly not addressed by R6A or R6B (both
  evaluation-only phases); not implicitly assigned to R6C either,
  since R6C's own scope is a live judge, not a production-code fix.

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
- **Location**: docs/spec only at the time this checkpoint closed —
  `docs/evaluation.md`, `docs/architecture.md`. No code, dependency, or
  eval run happened as part of E0 itself; the package this decision
  describes was built immediately after, in R7D (see that entry above)
  — `research_agent/evals/` now exists exactly as decided here.
- **Priority**: n/a — done (decision recorded).
- **Status**: Closed (2026-08-08). Implementation: see R7D entry above.

### Usage Protection M1: production usage telemetry foundation — complete, observe-only, M1.3 skipped
- **Goal**: before any rate limit/quota can be designed, know what the
  app's own production paid/external calls actually look like — one
  privacy-safe record per user-visible action, with real (or honestly
  nullable) token/call counts, and zero behavior change to anything it
  observes.
- **Why it matters**: every prior usage-cost discipline in this project
  (embedding caches, the direct-relevance judge cache, the R6C/R6D
  bounded-call-count design) was reasoned about by hand, per call site,
  with no aggregate, cross-request record of what a real session
  actually costs. A later limits phase (M2) needs that record — or an
  honest "not enough data yet" — rather than inventing numbers.
- **M1.1 — request/action telemetry foundation** (`research_agent/
  telemetry.py`, new; `research_agent/api_app/app.py` updated): a pure
  ASGI `RequestTelemetryMiddleware` (never `BaseHTTPMiddleware` — see
  `docs/architecture.md`'s own "Usage Protection M1" section for why
  that matters for a future streaming endpoint) records one
  `http_requests` row per request (route TEMPLATE, method, status,
  outcome, latency, `X-Request-ID` correlation) in a new, dedicated
  `data/usage_telemetry.sqlite`. A `contextvars.ContextVar`-based
  `paid_action(...)`/`record_child_call(...)` pair — the same mechanism
  `ingestion.py`'s own `_rate_limit_tracker` already proved works across
  this project's sync call chains — lets nested work attach to one
  active action ("first active action wins") rather than duplicate top-
  level rows. Fail-open throughout: a telemetry write/serialization
  failure is logged and swallowed, never allowed to change a real
  request's own result. Tests: 32 passed. Full backend suite: 1389
  passed.
- **M1.2 — instrumentation of production call sites** (15 domain/
  service files): wired `paid_action(...)` into the 8 frozen action
  types (`search`, `summarize`, `search_chat`, `curation_start`,
  `curation_refill`, `curation_chat`, `report_generate`, `report_
  regenerate`) at their exact service-layer boundaries, and `timed_
  child_call(...)` into every production `client.chat.completions.*`/
  `client.embeddings.create`/Tavily/arXiv/Semantic Scholar/OpenAlex/
  Unpaywall/CrossRef call site outside `research_agent/evals/**` and
  `agent.py`'s own internal LangChain tool-loop (one opaque, null-
  tokened `agent_loop_unmetered` record marks that the loop ran at all
  — exact per-turn agent tokens remain Langfuse's job, not reconstructed
  here). Closed four confirmed usage-capture gaps that predated M1
  entirely (report evaluation, offer classification, direct-relevance
  judging, non-substantive-classification embeddings — none recorded
  usage anywhere before this phase, not even in Langfuse). A cache hit
  (embeddings, direct-relevance judge) never produces a child-call
  record at all, rather than a `cache_hit=True` placeholder that could
  read as a real call. `discard_if_empty=True` (added to `paid_action`)
  lets `curation_refill` wrap every `/picks`/`/reopen` call
  unconditionally while still producing no row on the common no-refill
  path, without the service layer duplicating `curation_loop.py`'s own
  refill-routing decision. An AST-based coverage guard test fails if a
  new production external/model call site is ever added without an
  instrumentation decision. Tests: 90 focused (`test_telemetry.py` +
  `test_telemetry_instrumentation.py`). Full backend suite: **1446
  passed**. No test touches the real `data/usage_telemetry.sqlite`,
  confirmed directly every run.
- **M1.3 (admin/read API over the telemetry data) — deliberately
  skipped.** Nothing in the current product needs an HTTP-exposed read
  path over `usage_telemetry.sqlite` yet; a later M2 enforcement phase
  can query the SQLite file directly. Revisit only if M2's own design
  turns out to need one.
- **Explicitly not built**: rate limiting, 429s, quotas, an admin
  dashboard, authentication, billing, context summarization, streaming
  — M1 is observe-only in full; no production decision or limit depends
  on this data yet.
- **Location**: `research_agent/telemetry.py` (new), `research_agent/
  api_app/app.py`, 15 instrumented domain/service files (`agent.py`,
  `curation_chat.py`, `embeddings.py`, `enrichment.py`, `ingestion.py`,
  `qa.py`, `query_expansion.py`, `report.py`, `summarize.py`, `web_
  search.py`, and 6 `services/*.py` files), `tests/test_telemetry.py`
  (new), `tests/test_telemetry_instrumentation.py` (new). See `docs/
  architecture.md`'s "Usage Protection M1" section for the full
  narrative.
- **Priority**: next, **M2** — usage limits/admission control, designed
  against this data (or an honest "not enough data yet" if the real
  local `usage_telemetry.sqlite` doesn't have enough genuine
  application traffic in it).
- **Status**: Closed (2026-08-11). Commits: `649c5e5` (M1.1), `01b948a`
  (M1.2). **M2 below is also now closed** — the "not enough data yet"
  branch above did apply: M2's thresholds were set from topology
  inspection, not real-traffic calibration (see M2's own "Priority"
  line below).

### Usage Protection M2: agent execution limits, admission/leases, static/session/provider limits, frontend UX, red-team coverage — complete
- **Goal**: turn M1's observe-only telemetry into real enforcement —
  bound the standalone agent's own run length, admit/reject paid work
  against per-session and global budgets, prevent concurrent paid work
  on the same session, cap request/text/ID/session size at the static
  layer, and make every resulting rejection understandable and
  recoverable in the frontend — all provisional and configurable, not
  calibrated against real traffic yet.
- **Why it matters**: M1 closed with an explicit "no rate limiting, no
  429s, no quotas" scope line — this is the phase that actually adds
  those, so a single runaway agent loop, a scripted burst of paid
  requests, or two concurrent expensive actions on one session can no
  longer run unbounded or corrupt shared session state.
- **M2.1 — agent execution limits + admission/lease foundation**
  (`research_agent/agent.py`, `research_agent/admission.py` (new),
  `research_agent/leases.py` (new), `research_agent/config/limits.py`
  (new), `research_agent/telemetry.py`): `ModelCallLimitMiddleware`
  (10/run, `exit_behavior="end"`) and `ToolCallLimitMiddleware` (10/run,
  `exit_behavior="continue"` — see `docs/architecture.md`'s M2 section
  for why `"continue"`, not `"end"`, is correct here) wrap the
  standalone agent; explicit `recursion_limit=15` passed to
  `agent.stream()`. `admission.py`/`leases.py` land as read-but-not-yet-
  applied-by-any-route foundations (hourly/daily/global budget checks;
  one atomic SQLite lease per session via `INSERT ... ON CONFLICT ...
  DO UPDATE ... WHERE action_leases.expires_at < excluded.acquired_at`).
  Tests: extensive (`test_admission.py`, `test_agent_limits.py`,
  `test_config_limits.py`, `test_leases.py`, new).
- **M2.1b — telemetry-test DB isolation fix**: a real gap found while
  building M2.1 — some usage-guard/agent-limit tests could still touch
  the real `data/usage_telemetry.sqlite` path under certain monkeypatch
  orderings. Fixed by adding `tests/_usage_db_fingerprint.py` (shared
  before/after fingerprint helper) and tightening the isolation fixtures
  in `test_admission.py`/`test_agent_limits.py`/`test_leases.py`/
  `test_telemetry.py`/`test_telemetry_instrumentation.py`.
- **M2.2A — paid-action budgets and session concurrency enforcement**
  (`research_agent/usage_guard.py`, new): composes M2.1's admission
  checks and leases with M1's `paid_action(...)` into one reusable
  `guard_paid_action(...)` context manager. Fail-closed (unlike M1's
  fail-open telemetry) — a storage error during an admission/lease
  check is treated as a rejection, never a silent pass-through. One
  centralized FastAPI exception handler maps the resulting
  `UsageGuardRejection` to 429/409/503 with `Retry-After` where
  meaningful. Tests: `test_usage_guard.py` (new, 305 lines), plus
  `test_api.py`/`test_curation_api.py` additions.
- **M2.2B — conditional paid-workflow guard placement**
  (`curation_loop.py`, `services/chat_service.py`,
  `services/curation_core_service.py`, `services/curation_helpers.py`,
  `services/curation_history_service.py`, `services/summary_service.py`):
  moved the refill guard *inside* `_refill_node` (only a turn that
  actually refills opens the guard); confirmed/hardened the cache-hit-
  bypasses-guard, cache-re-check-after-lease-acquisition pattern for
  report/summary generation; confirmed `search_chat` is guarded for
  budget but deliberately lease-free (stateless per turn, no shared
  session state to conflict over). Tests: `test_curation_loop.py`,
  `test_summary_service.py` (new), `test_api.py`/`test_curation_api.py`
  additions.
- **M2.2C — static request/text/ID/session/provider limits**
  (`research_agent/request_limits.py` (new), `research_agent/
  session_limits.py` (new), `api_app/constrained_types.py`,
  `api_app/schemas.py`, `provider_clients.py`): 64 KiB request body (a
  pure ASGI middleware, draining `receive()` itself rather than
  trusting `Content-Length`), 2,000-char text fields, 30-ID mutation
  cap, 60-selected-paper/100-chat-turn session capacity (`409`, no
  `Retry-After` — a capacity conflict, not a rate one), 60s OpenAI
  timeout (arXiv/Tavily don't expose a compatible timeout parameter yet
  — documented gap, not silently assumed covered), 20-call provider
  fan-out ceiling (every fan-out path actually exercised today stays
  ≤5). Tests: `test_request_limits.py`, `test_schema_limits.py`,
  `test_session_limits.py`, `test_provider_clients.py` (all new).
- **M2.3 — frontend usage-protection UX and red-team regression
  coverage** (`frontend/src/types/index.ts`, `frontend/src/lib/api/
  errorMessages.ts` (new), `frontend/src/hooks/useCurationSession.ts`,
  `frontend/src/pages/CurationWorkspacePage.tsx`, chat/review/edit-
  question components; `tests/test_red_team_bypass.py`, new): one
  extended, structured `ApiError`; a centralized reason-code → user-
  message mapping (no raw reason codes ever shown); one shared,
  accessible (`role="alert"`) error banner reused by every curation
  action, with existing report/chat/selection state always preserved
  after a rejection and no automatic retry; client-side preventative
  caps (2,000-char `maxLength`, 30-ID/60-paper selection caps) layered
  on top of, never instead of, backend enforcement. 29 new backend red-
  team tests covering body/schema/session-capacity/admission-
  concurrency bypass attempts and explicit validation → capacity →
  admission → lease → provider ordering assertions, plus 34 new
  frontend tests. Full backend suite: **1638 passed** (1609 before this
  chunk). Frontend suite: **303 passed**. Frontend build: clean.
- **Explicitly not built**: telemetry admin/dashboard endpoint (M1.3
  remains skipped, M2 doesn't add one), authentication/IP-based
  identity, distributed rate limiting (SQLite admission/leases suit
  only the current single-instance phase), cross-request session-
  mutation versioning (the pre-existing non-paid-mutation race noted in
  `docs/architecture.md`'s M2 section is untouched), long-chat
  summarization, response streaming.
- **Location**: see each sub-entry above; full narrative in `docs/
  architecture.md`'s "Usage Protection M2" section.
- **Priority**: thresholds throughout M2 are provisional (topology-
  inspection-derived, not real-traffic-calibrated) — recalibrating them
  against `usage_telemetry.sqlite` once it holds enough genuine traffic
  is real, useful future work, tracked as its own open item below. Next
  substantive phase is **M3** (long-chat summarization); **M4**
  (streaming) follows M3.
- **Status**: Closed (2026-08-12). Commits: `67035be` (M2.1), `1ef7a1c`
  (M2.1b), `49dbd2c` (M2.2A), `633abd1` (M2.2B), `e5418bf` (M2.2C),
  `f4679d0` (M2.3).

### Usage Protection follow-on: threshold calibration, distributed rate limiting, admin dashboard, session-mutation versioning, broader provider timeouts (not part of M1/M2)
- **Goal**: the real, useful next-layer work M1/M2 deliberately left
  open, tracked so it isn't lost, not because any of it is scheduled.
- **Items**:
  - **Threshold calibration using real telemetry** — once
    `usage_telemetry.sqlite` holds enough genuine production traffic,
    revisit every M2.1/M2.2 provisional number (agent call limits,
    session/global budgets, lease TTL, static caps) against real usage
    instead of topology inspection.
  - **Telemetry admin/dashboard endpoint** — M1.3 was skipped, M2 didn't
    revisit it; still no HTTP-exposed read path over
    `usage_telemetry.sqlite`. Worth building only once something in the
    product actually needs it (an ops view, a self-serve usage page);
    query the SQLite file directly until then.
  - **Distributed rate limiting/storage for multi-instance deployment**
    — the current SQLite-backed admission counters and leases are
    correct for one process; a real multi-instance deployment needs a
    shared store (e.g. Redis) and true in-flight reservation counting,
    not just completion-based counting.
  - **Cross-request session-mutation versioning** — selected-paper
    capacity (and other non-paid session mutations) is not fully atomic
    across simultaneous requests for the same session; needs real
    transactional/versioned session persistence, not attempted here.
  - **Broader provider timeout support** — `provider_timeout_seconds`
    only reaches the OpenAI client today; arXiv's `arxiv` package and
    the current Tavily tool-client abstraction don't expose a compatible
    timeout parameter.
- **Priority**: open, none scheduled — each item above is real and
  worth doing, but none blocks M3/M4.
- **Status**: Open.

### Usage Protection M3: bounded curation-chat summarization — complete
- **Goal**: close the gap M1/M2 explicitly left open — `qa.py::
  capped_history` has always silently dropped anything before the last
  8 exchanges from the MODEL's own view of a curation-chat conversation,
  even though the full conversation stays visible to the user the whole
  time. M3 bounds model-bound context via a validated, persisted
  structured summary instead of silent truncation, without ever
  deleting, truncating, or rewriting the real, stored, user-visible
  `chat_history`.
- **Scope decision**: persisted curation chat only. Search chat is
  stateless/client-echoed (no server-persisted growth problem to
  solve); the standalone `create_agent` agent has no persisted
  multi-turn conversation at all (one topic, one shot, no `thread_id`).
  LangChain's `SummarizationMiddleware` (installed `langchain==1.3.11`,
  inspected directly at the source level) was deliberately NOT used:
  it hooks `AgentMiddleware.before_model` against a `create_agent`-
  built `AgentState`'s `list[AnyMessage]` channel, and curation chat's
  own graph (`qa.py::build_qa_graph`) is a hand-built `StateGraph
  (QAState)` with no middleware pipeline and a plain-dict `list[dict]`
  history — there is no compatible hook, and forcing the QA graph into
  `create_agent` merely to gain one would mean rebuilding the whole
  condense → retrieve → filter-web-relevance → generate pipeline as
  agent turns. `ContextEditingMiddleware` remains deferred too — it
  clears stale TOOL-CALL results inside one agent run, a different
  problem from persistent cross-request conversation-history loss, and
  no stale-tool-result problem has been observed to justify it.
- **M3.1 — deterministic summary state, policy, and context builder**
  (`research_agent/chat_summarization.py`, new; `research_agent/
  config/limits.py`, `research_agent/query_expansion.py`, `research_
  agent/curation_session.py` updated): `ChatHistorySummary` (Pydantic,
  `extra="forbid"`); four provisional policy fields
  (`chat_summary_trigger_tokens=6000`, `chat_summary_keep_recent_
  turns=8`, `chat_summary_max_output_tokens=800`, `chat_summary_min_
  new_turns=4`), independent of M2.2C's own 100-turn storage ceiling;
  three new `PaperPoolSession` fields (`chat_summary`, `chat_summary_
  covers_history_count` — a RAW entry count, never a turn count —
  `chat_summary_updated_at`), all `.get(key, default)` backward-
  compatible; exchange-boundary/selection/trigger/rendering/
  replacement-validation/invalidation helpers, all pure — no `OpenAI()`
  construction, no network call, nothing wired into the live request
  path yet (by design, matching the same "lay the foundation, activate
  later" precedent M2.1's admission/leases and `qa.py`'s own
  `sqlite_checkpointer()` already set). Tests: `tests/test_chat_
  summarization.py` (new, 76), `tests/test_curation_session.py` (+6),
  `tests/test_config_limits.py` (+19, exact-equality test updated for
  the 4 new fields, existing M1/M2 defaults spot-checked unchanged).
  Full backend suite: **1739 passed** (1638 before this chunk).
- **M3.2 — live curation-chat integration, telemetry, invalidation,
  and regression coverage** (`research_agent/chat_summarization.py`
  extended with the one real `client.chat.completions.parse(...)` call
  this phase makes; `research_agent/qa.py` — `ChatSession` gains the
  three summary fields, `ask(enable_chat_summarization=False)` defaults
  to today's exact `capped_history` behavior for every existing caller
  (search chat, RAGAS scripts) and only `curation_chat.py::
  ask_in_session` opts in; `research_agent/curation_chat.py` —
  `_build_chat_session`/`ask_in_session` copy summary state in and back
  out, `delete_chat_exchanges`/`edit_chat_exchange` wire in M3.1's
  invalidation rule before their own truncation/rerun): summary
  generation attaches as a `summarize_chat_history` child call under
  the already-open top-level `curation_chat` `paid_action` — no second
  admission check, no second lease, no duplicate `paid_actions` row,
  confirmed by a same-session-concurrency regression test (still
  exactly one 200/one 409) and a different-sessions-independence test.
  Failure policy: no retry, no fallback model; first-time failure falls
  back to bounded `capped_history`-equivalent behavior, incremental
  failure retains and reuses the previous valid summary; coverage/
  timestamp never advance on failure; nothing is exposed as a user-
  facing error. `derive_chat_references`/`select_eligible_exchanges_
  for_report` confirmed identical with/without a persisted summary
  present — neither reads `chat_summary`. No frontend/API response
  schema changes; no live long-chat evaluation suite added (deferred
  until real usage shows a genuine coherence/retention problem
  deterministic tests can't answer). Tests: `tests/test_chat_
  summarization.py` (+23), `tests/test_qa.py` (+27), `tests/test_
  curation_chat.py` (+15), `tests/test_curation_api.py` (+7),
  `tests/test_telemetry_instrumentation.py` (AST coverage-guard module
  list extended to include the new module's own instrumented call
  site). Full backend suite: **1811 passed** (1739 before this chunk).
- **No M3.3 required.** The originally-sketched optional third chunk
  (standalone-agent `SummarizationMiddleware`) was evaluated and
  explicitly not built — see the scope decision above; M3.1+M3.2 alone
  already satisfy M3's own stopping rule (model context always bounded,
  full stored history intact, summary correctly reused/invalidated,
  citation/report behavior unchanged, no further complexity without a
  measured failure).
- **Explicitly not built**: search-chat summarization, standalone-agent
  persisted conversation + `SummarizationMiddleware` wiring,
  `ContextEditingMiddleware`, any frontend summary display/status, a
  live long-chat evaluation suite, exact token-threshold calibration
  against real telemetry (thresholds remain topology-derived,
  provisional), any change to the pre-existing lease/final-save
  ordering limitation (the lease still releases before
  `save_curation_session()` — unchanged from M2, not newly claimed
  fixed).
- **Location**: see each sub-entry above; full narrative in `docs/
  architecture.md`'s "Usage Protection M3" section.
- **Priority**: **M4** (chat/report streaming) is the immediate next
  phase. Live long-chat evaluation and threshold calibration are real,
  useful follow-on work, tracked as open items below — neither is
  scheduled or blocks M4.
- **Status**: Closed (2026-08-13). Commits: `95fa448` (M3.1), `35572b7`
  (M3.2).

### M3 follow-on: live long-chat evaluation, threshold calibration (not part of M3)
- **Goal**: real, useful next-layer work M3 deliberately left open,
  tracked so it isn't lost, not because it's scheduled.
- **Items**:
  - **Live long-chat evaluation** — a small, opt-in-only eval suite
    (continuity/factual-retention checks across long simulated
    conversations, explicit cost/latency reporting) was deliberately
    NOT built in M3 — deterministic testing found no real gap
    justifying it yet. Build only if real usage surfaces a genuine
    coherence/retention problem deterministic tests can't answer.
  - **Token-threshold calibration using real telemetry** — the four
    M3.1 policy defaults (6000/8/800/4) are topology-derived, not
    calibrated against real curation-chat token data. Revisit once
    `usage_telemetry.sqlite` holds enough genuine long-chat traffic,
    same posture as every other M2 threshold's own open calibration
    item above.
  - **Standalone-agent summarization** — only worth reconsidering if a
    future phase gives `agent.py`'s `create_agent` path real persisted,
    multi-turn conversation state (a product change, not an M3-scope
    decision); no such state exists today.
  - **Session-persistence atomicity** — unchanged from M2's own open
    item above; M3's summary+exchange save inherits the same read-
    check-write/lease-releases-before-save limitation, not newly
    introduced.
- **Priority**: open, none scheduled — real work, doesn't block M4.
- **Status**: Open.

### Usage Protection M4: chat and report response streaming — complete
- **Goal**: give curation chat and report generation/regeneration real-
  time progress over Server-Sent Events, phase-first (never token-by-
  token — see M4.1's own SDK finding below), while every existing
  synchronous endpoint stays completely unchanged and remains the
  default for any client that hasn't adopted streaming.
- **M4.1 — protocol foundation and structured-stream feasibility**
  (`research_agent/sse.py`, `research_agent/chat_streaming.py`, new;
  `frontend/src/lib/api/sseDecoder.ts`, new, unwired): pure foundation,
  no live route yet. Tested directly against the installed
  `openai==2.44.0` SDK (not assumed): structured-output streaming's own
  `ContentDeltaEvent.parsed` omits an in-progress string field entirely
  (via `jiter`'s `partial_mode=True`, hardcoded inside the SDK's own
  accumulator with no public override) rather than including a
  truncated prefix — the `answer` field stays `None` almost the entire
  stream, then appears complete in one jump. This is why M4's own
  product contract everywhere is "phase-first, answer may arrive as one
  large delta," never a token-by-token promise. `LangGraph interrupt_
  before` was tested and rejected as a pause/resume mechanism for the
  QA graph — confirmed directly that resuming a paused, checkpointer-
  less graph reruns every preceding node from `START` a second time,
  not from the interrupt point. `stream_chat_answer` (one real
  `client.chat.completions.stream(...)` call, a monotonic-suffix delta
  algorithm, citations read only from the terminal completion) proves
  the adapter is real and safe even though token-level streaming isn't
  possible for this schema shape. Tests: `tests/test_sse.py`, `tests/
  test_chat_streaming.py` (new), `frontend/src/lib/api/sseDecoder.
  test.ts` (new). Status: complete. Commit: `27e7439`.
- **M4.2A — curation-chat streaming backend** (`research_agent/
  curation_chat_streaming.py`, new; `research_agent/usage_guard.py`
  extended with `open_admission_and_lease_for_streaming`/`Streaming
  LeaseHandle`; `research_agent/services/curation_chat_service.py`,
  `research_agent/api_app/routers/curation_chat.py` extended): `POST /
  curation/{session_id}/chat/stream`, alongside the unchanged `POST /
  curation/{session_id}/chat`. A real production bug was found
  empirically during this chunk, not by inspection alone — driving
  `guard_paid_action`'s combined admission+lease+telemetry context by
  hand across the route-handler/generator boundary raised a real
  `contextvars.Token` cross-Context `ValueError` under a genuine
  `TestClient`-driven test (Starlette's own ASGI-fallback path spawns
  the streaming generator in a separate Task with a copied `Context`).
  Fixed by splitting the guard: admission+lease (no `contextvars`
  involved) acquired synchronously before any `StreamingResponse`
  exists; `telemetry.paid_action` opened entirely inside the async
  generator's own body, in one uninterrupted execution frame — proven
  safe across a genuinely spawned Task boundary by a dedicated
  regression test. This split-guard pattern is the template both M4.2B
  and M4.3A reuse identically. Tests: `tests/test_usage_guard_
  streaming.py`, `tests/test_qa_prepare_turn.py`, `tests/test_curation_
  chat_streaming.py`, `tests/test_curation_chat_stream_api.py`, `tests/
  test_curation_chat_stream_service.py` (all new). Full backend suite:
  **1893 passed**. Status: complete. Commit: `2491322`.
- **M4.2A lifecycle-hardening fix** (same day): a handled failure
  previously self-converted to `error`+`done` and returned normally
  from inside the `telemetry.paid_action` block, leaving the top-level
  row `outcome="success"` for a turn that actually failed — fixed via a
  new typed `HandledStreamFailure` that propagates out of that block
  before being caught and converted, one layer up. A shielded
  persistence task could previously let the lease release before the
  save genuinely settled on outer cancellation — fixed by retaining the
  save as an explicit `asyncio.Task` and `asyncio.wait`-ing it (never a
  bare re-`await`) before re-raising. Both proven with dedicated tests,
  including a real `threading.Event`-gated mid-save cancellation.
  Status: complete. Commit: `9592e4d`.
- **M4.2B — chat-streaming frontend** (`frontend/src/lib/api/
  chatStream.ts`, new; `frontend/src/hooks/useCurationSession.ts`,
  `frontend/src/components/ChatMode/ChatModePanel.tsx` extended):
  fetch+`AbortController` adapter reusing M4.1's `sseDecoder.ts`
  unchanged; the hook owns the full chat-stream lifecycle and an
  explicit event-ordering state machine (any violation becomes the same
  safe transport-error bucket a malformed/truncated stream gets, never
  a crash); the panel renders one stable temporary-answer area with
  Send/Stop sharing a single slot. Streaming is the normal submission
  path; the non-streaming hook method and endpoint are untouched.
  Tests: `frontend/src/lib/api/chatStream.test.ts` (new), plus
  extensions to `useCurationSession.test.ts`/`ChatModePanel.test.tsx`.
  Frontend suite: **358 passed**. Status: complete. Commit: `912f079`.
- **M4.3A — report-generation/regeneration progress streaming backend**
  (`research_agent/report_streaming.py`, `research_agent/curation_
  report_streaming.py`, new; `research_agent/report.py` — two
  precondition checks extracted verbatim for preflight reuse, one
  optional backward-compatible `progress_callback` added to `refine_
  report_if_requested`; `research_agent/services/curation_report_
  service.py`, `research_agent/api_app/routers/curation_reports.py`
  extended): `POST /curation/{session_id}/report/stream` and `POST /
  curation/{session_id}/report/regenerate/stream`, alongside both
  unchanged synchronous endpoints. Reuses the identical split-guard
  pattern M4.2A established. Frozen phases (`generating`/`evaluating`/
  `revising`/`saving`) map to real execution boundaries only — no
  `finalizing` phase, since reference/section finalization happens
  synchronously inside the content-producing calls themselves, not as a
  separately orchestrable step. Initial-generation cache hit performs
  zero admission/lease/telemetry/provider work and emits `started ->
  completed -> done` with no phase events. Cancellation during ANY of
  the four phases (not just persistence, unlike M4.2's own narrower
  original scope) retains that phase's own thread task and waits for
  genuine settlement before releasing the lease — provider-call
  concurrency, not just mutation safety, is the reason, even though
  generate/evaluate/revise are confirmed pure relative to `session`.
  Commit-point mutations run against a `copy.deepcopy` of `session`,
  exposed only after `save_curation_session` succeeds on the copy.
  Tests: `tests/test_report_streaming.py`, `tests/test_curation_report_
  stream_service.py`, `tests/test_curation_report_stream_api.py` (all
  new), `tests/test_report.py` extended. Full backend suite: **1931
  passed**. Status: complete. Commit: `c04688d`.
- **M4.3B — report-streaming frontend** (`frontend/src/lib/api/
  reportStream.ts`, new; `frontend/src/hooks/useCurationSession.ts`,
  `frontend/src/components/ReportMode/ReportModePanel.tsx` extended):
  one shared internal generator, two thin wrappers (`streamGenerate
  Report`/`streamRegenerateReport`); the one payload check beyond
  M4.2's own framing-only validation — a `completed` event's four
  always-required `ReportOut` fields are structurally checked before
  being trusted. `reportStreamError` is its own dedicated hook field
  (not the shared error banner M4.2 reuses), so a handled report
  failure surfaces inline in the report panel. Chat and report streams
  are mutually exclusive at the hook level, not just via page-level
  `disabled` composition. The empty (Generate) view shows one stable
  progress area with no partial report shell; the regenerate view keeps
  the existing report fully visible and undimmed throughout, with
  Regenerate/Stop sharing one action slot. Cancellation shows a stable
  "Stopping" state held through the post-cancellation canonical reload
  — never presented as instantaneous, since the backend may genuinely
  still be waiting out an in-flight synchronous provider call. Tests:
  `frontend/src/lib/api/reportStream.test.ts` (new), plus extensions to
  `useCurationSession.test.ts`/`ReportModePanel.test.tsx`/`App.test.
  tsx`. Frontend suite: **411 passed**. Status: complete. Commit:
  `266102d`.
- **M4.2C/M4.3C — live browser validation**: a real successful chat
  turn, a real deliberate chat cancellation, a zero-cost report cache-
  hit check made directly against the real backend, one real Regenerate
  (refinement off), and one real cancelled Regenerate attempt — all
  driven against the actual running dev servers (headless-Chromium
  Playwright for the UI-driven scenarios) on pre-existing disposable
  sessions only, never an important user session. Confirmed live in
  both domains: the streaming endpoint (never the sync one) is what the
  UI actually calls; existing content stays visible during regeneration/
  ongoing chat; a genuine phase label and a stable Stop/Stopping
  sequence; canonical reload replacing any temporary preview with the
  real persisted result and exactly one new exchange/version, with no
  duplicate after a page refresh; no raw SSE/JSON, no application-error
  toast for deliberate cancellation, zero new console errors. One
  unrelated, pre-existing DOM-nesting console warning was found in
  `ReviewsList/ReviewCard.tsx` (last touched in `6b16baf`, well before
  any M4 work) — investigated and confirmed not an M4 defect, not
  fixed as part of this phase. Full narrative and screenshots-equivalent
  detail in `docs/architecture.md`'s M4 section.
- **Explicitly not built**: token-by-token structured-answer streaming
  (not possible against the current SDK/schema shape); report prose/
  section streaming (no `delta` event exists for reports; no proven
  non-prose use); heartbeat events; percentage/numeric progress
  indicators; production reverse-proxy/load testing of long-lived SSE
  connections; multi-instance streaming coordination (SQLite leases/
  admission remain single-instance, unchanged from M2); HITL report
  approval; targeted-section refinement; any change to R4's own bounded
  refinement loop; auth/deployment hardening.
- **Location**: see each sub-entry above; full narrative in `docs/
  architecture.md`'s "Usage Protection M4" section.
- **Priority**: M4 is now complete. No further milestone is scheduled
  or invented here — see "M4 follow-on" below for real, unscheduled
  next-layer work.
- **Status**: Closed (2026-08-13). Commits: `27e7439` (M4.1), `2491322`
  (M4.2A), `9592e4d` (M4.2A hardening), `912f079` (M4.2B), `c04688d`
  (M4.3A), `266102d` (M4.3B). Tagged `m4-chat-report-streaming`.

### Post-M4 UX hardening: user-journey verification and closure — complete
- **Goal**: close concrete desktop journey gaps discovered after M4 without
  reopening M4's protocol/backend design or starting another feature phase.
- **UXH.1 — selection/session consistency** (`1cf6d4c`): one deduplicated
  persisted+staged selected-paper derivation feeds every visible count;
  staged picks reset between reviews; stale session successes and failures
  cannot publish after the user switches away.
- **UXH.1b — chat progress/references** (`f7d45c8`): optimistic input,
  phases, streamed answer, and canonical replacement remain visible without
  duplication; auto-scroll respects deliberate scroll-up. Chat References is
  collapsed by default, count-labelled, accessible, height-bounded, reset on
  session change, and stable across same-session refreshes.
- **UXH.2 — action-specific progress** (`81ca35d`): truthful status for
  starting, continuing, searching-more, and finishing review mutations, with
  synchronous duplicate/conflicting-submission exclusion; report regeneration
  retains the existing report and promotes its truthful phase beside Stop.
- **UXH.3 — focus/live regions/safe errors** (`634b8ac`): chat and report
  commands restore focus after completion/cancellation when appropriate;
  progress surfaces are polite live regions; unexpected technical failures use
  only `Something went wrong. Please try again.`, while structured `ApiError`
  messages remain unchanged.
- **Verification**: committed-range inspection found no objective defect and
  produced no additional code/test commit. Focused frontend verification: **365
  passed** (8 files). Full frontend verification: **476 passed** (18 files).
  Production build clean. Lint exit 0 with the same 3 pre-existing warnings.
  The pre-HEAD Vite process was restarted and localhost module output confirmed
  current hardened code was served. Deterministic mocked/delayed requests and
  SSE sequences cover the required journeys; the configured browser controller
  had no available browser, so this run claims no new interactive screenshot or
  browser-control evidence. No backend compatibility run was warranted by the
  frontend-only diff. No paid provider call, production-session mutation,
  telemetry change, or evaluation-artifact change occurred.
- **Deferred/remaining**: responsive/mobile redesign is explicitly deferred;
  the existing desktop/fixed-sidebar layout is unchanged. No new real-provider
  timing evidence was gathered beyond M4's already-recorded live-browser pass;
  the existing thread-pool cancellation-settlement limit remains. Repeat the
  interactive smoke when a browser surface is available if fresh visual
  evidence is required.
- **Priority**: closed; verification checkpoint only, no follow-on feature
  phase invented.
- **Status**: Closed (2026-08-13). Tagged
  `uxh-post-m4-user-journey-hardening` at the documentation closure commit.

### Post-UXH follow-up: report progress observability — complete
- **Goal**: fix a real-browser-confirmed UX gap in the M4.3B/UXH.2 report-
  progress panel without reopening M4 or UXH's own closed scope.
- **Finding**: a real Regenerate + Refine Once run captured raw SSE timing
  (`generating` 13:58:57.753, `evaluating` 13:59:46.149, `saving`
  13:59:48.590, `completed`/`done` 13:59:48.708), proving the backend
  streamed every phase correctly and unbuffered. The defect was frontend-
  only: the panel showed just the single latest phase and discarded it
  the instant the next one arrived, so `evaluating` (~2.44s) and `saving`
  (~118ms) were functionally invisible even though both streamed in real
  time.
- **Fix** (`d9c018e`, test-coverage correction `f1bf192`):
  `useCurationSession.ts` now retains `reportStreamPhaseHistory`, the
  ordered, deduplicated phases genuinely received (a functional `setState`
  update, immune to same-batch overwrites); `ReportModePanel.tsx` renders
  one row per observed phase (check = done, spinner = current), never a
  future/predicted phase, and distinguishes Generate ("Generating
  report" / "Report generated") from Regenerate ("Regenerating report" /
  "Report regenerated"). A `reportStreamCompletionNotice` (e.g. "Report
  regenerated · Evaluated · Saved") replaces the trail once
  `completed → done` and the canonical reload both genuinely succeed,
  then auto-clears after `REPORT_STREAM_SUCCESS_NOTICE_MS` (5s) — never
  shown after cancellation, a handled error, a malformed stream, or a
  reload failure.
- **Verification**: independent review of `origin/main..HEAD` found one
  test-coverage gap (several tests asserted only the completion notice,
  never `reportStreamPhaseHistory` itself, despite their names) and no
  implementation defect; closed with one additional test (`f1bf192`).
  Focused and full frontend suite: **503 passed** (18 files). Production
  build clean. Lint exit 0 with the same 3 pre-existing warnings, zero
  new. Already manually confirmed working in a real browser before this
  checkpoint.
- **Scope**: frontend-only. No backend route, SSE vocabulary, API
  contract, provider behavior, persistence, telemetry, or evaluation
  artifact changed. No artificial delay, fake percentage, or invented
  phase added. No paid provider call made.
- **Priority**: closed; visibility fix only, no new feature phase
  invented.
- **Status**: Closed (2026-08-14). **M4 and UXH remain closed** — this is
  a follow-up fix within their existing scope, not a reopening of either.

### Paper Keywords and Filtering — complete
- **Goal**: up to 6 useful, deterministic keywords per paper (when a real
  abstract exists), displayed on each paper card, with a client-side
  filter over the currently visible candidate batch. Cheap and offline
  from the start — explicitly no LLM/embedding call, no network call.
- **K1 — extraction, persistence, API** (`1de6488`):
  `research_agent/keywords.py::extract_keywords` (YAKE, `n=2, top=12,
  dedupLim=0.85`), capped at 6, case-insensitively deduplicated, noise-
  filtered (URLs/DOIs/citation markers/pure numbers/single characters),
  deterministic. Computed exactly once per deduplicated paper inside
  `build_candidate_pool()`, immediately after `deduplicate()` — never
  per-source (would be silently dropped by `dedup.py`'s own `Paper(...)`-
  rebuilding merge) and never at read time. `Paper.keywords` defaults to
  `[]`, so every existing persisted Paper dict reconstructs unchanged
  (`curation_session.py` needed zero code changes, proven by a real-
  SQLite round-trip test). `PaperOut`/`_paper_to_out` are a pure pass-
  through.
- **K2 — display and filtering** (`b0b40d9`): `PaperCard.tsx` renders
  static keyword chips below the abstract, only when `showAbstract` and
  `keywords.length > 0` — no placeholder for an abstract-less/legacy
  paper, and `PoolSummaryPanel`'s compact list is untouched.
  `ReviewModePanel.tsx` adds a collapsed-by-default, presentation-only
  filter over `pending_batch`: multi-select, OR semantics, case-
  insensitive matching with a stable display label, options sorted by
  count then label, removable-chip + "Showing X of Y" + clear-all UI, a
  concise empty state for zero matches. Resets on a real session or
  pending-batch change (not on the array's own changing reference or an
  unrelated re-render); never touches selection state, counters, or
  submission payloads.
- **K3 — validation, docs, publication** (this entry): independent review
  of the K1/K2 range against a 12-point checklist (post-dedup-only
  computation, title+abstract input, determinism/offline/cap, backward
  compatibility, serializer pass-through-only, ranking/dedup/selection/
  report non-interference, chip/panel absence when empty, presentation-
  only OR filtering, reset-on-real-change-only, no submission-payload
  effect, no permanently-visible large panel, bounded/overflow-safe
  layout) found **no objective defect** — no corrective commit was
  needed. A bounded, no-network, no-paid-call audit of the real
  extractor against 12 representative local abstracts (realistic
  technical prose, noisy-citation, numeric-heavy, case-duplicate-heavy,
  short/empty/`None`, Unicode-heavy, all-uppercase, long-concatenated,
  Markdown/LaTeX-noisy) found zero contract violations. Full backend
  suite **1948 passed**; full frontend suite **524 passed**; build
  clean; lint unchanged (3 pre-existing warnings, 0 new); `uv.lock`
  confirmed synchronized with `pyproject.toml`. No browser-control
  surface was available in this environment and no paid search/refill
  was made solely to obtain a smoke-test session — component/
  integration tests are the recorded evidence instead, stated honestly
  rather than substituted silently.
- **K3-follow-up — real-session keyword-loss regression, fixed** (`cfa4fdd`):
  a user's brand-new (post-K1) curation session showed abstracts but no
  keywords and no filter control. Investigation (read-only session load,
  a direct no-cost `extract_keywords()` control run against the actual
  affected paper, and the dev server's own reload log) ruled out both a
  stale backend and a broken extractor before touching any code. Root
  cause: `research_agent/embeddings.py::_serialize_metadata`/
  `_paper_from_metadata` — the Chroma metadata round trip inside
  `rank_full_pool()`'s own `embed_and_index_papers()` → `semantic_search()`
  sequence, which every real `/curation/start` call runs immediately
  after `build_candidate_pool()` — never learned about `Paper.keywords`.
  `semantic_search()` never returns the original in-memory `Paper`
  objects, only ones reconstructed from Chroma metadata, so every paper
  leaving `rank_full_pool()` lost its just-computed keywords one call
  after `build_candidate_pool()` set them. Fixed by adding
  `keywords_json` to both metadata functions (same JSON-list convention
  as the existing `authors_json`/`source_urls_json`), with a safe `"[]"`
  default for metadata indexed before this fix. Regression test
  (`tests/test_query_expansion.py::
  test_real_initial_curation_path_keeps_keywords_through_rank_full_pool`)
  runs the real `build_candidate_pool()` → `rank_full_pool()` sequence
  against a real, unmocked ephemeral Chroma collection — confirmed to
  fail without the fix and pass with it. No ranking score, embedding
  text, dedup, paper-id, or selection behavior changed; no backfill —
  sessions/Chroma entries indexed before this fix keep `keywords: []`
  until genuinely re-fetched. Full backend suite **1949 passed** (1948 +
  this one new test); focused embeddings/query-expansion re-run **41
  passed** at this checkpoint's own review.
- **Explicitly deferred, not part of this feature's own scope**:
  - Historical-session backfill — an old session's papers stay
    keyword-less until naturally re-fetched (also true of sessions
    created between K1 and the K3-follow-up fix above, for the same
    reason).
  - Filtering the full selected-paper collection across batches/turns —
    today's filter only ever covers the current `pending_batch`.
  - Keyword search across sessions.
  - Manual keyword editing.
  - LLM/embedding-based extraction — YAKE only, by design.
  - Real-world extractor-quality calibration (stop words, thresholds,
    title-vs-abstract weighting) against genuine usage — the K3 audit's
    own 12 examples are contract validation, not a quality-calibration
    dataset; explicitly not attempted from that small a sample.
- **Priority**: closed; no further checkpoint scheduled.
- **Status**: Closed (2026-08-14). Commits: `1de6488` (K1), `b0b40d9`
  (K2), `cfa4fdd` (K3-follow-up corrective fix), plus this checkpoint's
  own validation/docs/publication commits. Tagged
  `paper-keywords-filtering` (unmoved — the tag still marks the K1–K3
  checkpoint on `dcc832b`; this corrective fix lands after it).

### Paper Keyword Quality and Visual Polish (K4) — complete
- **Goal**: fix the extraction-quality noise the K1–K3 feature's own
  contract validation had explicitly ruled out of scope (title-position
  domination, incomplete overlapping fragments, malformed punctuation,
  hyphen/case surface-variant duplicates, generic single-word noise), an
  explicit offline re-extraction path for already-persisted sessions, and
  a visual/filter redesign to make the resulting keywords legible instead
  of a flat wall of checkboxes.
- **K4.1 — extractor quality + maintenance tooling** (`f67d876`):
  `research_agent/keywords.py` bumped `KEYWORD_EXTRACTOR_VERSION` from
  `yake-v1` to `yake-v2`. Abstract and title are now extracted
  *separately* (`n=3`, was `n=2`), with the title admitting at most one
  candidate into the final six, appended last — evidence-backed over
  simple `abstract + title` reordering, which was directly confirmed
  insufficient for a self-referential abstract. Added: NFKC
  normalization; rejection of candidates with an embedded comma/semicolon
  (reproduces a real production defect, `"Agentic AI,this"`); a canonical
  comparison key (casefold + Unicode dash-variant-to-space +
  whitespace-collapse) for both exact-duplicate dedup and a new
  *bidirectional* redundancy pass that drops a candidate whenever its
  tokens are a contiguous subsequence of any other candidate's, with a
  standalone-uppercase-acronym (2–6 chars) exemption. `scripts/
  re_extract_keywords.py` — a new, explicit, dry-run-by-default,
  `--apply`-gated maintenance command — recomputes one session's keywords
  through the exact production session-load/save path, propagating each
  unique `paper_id`'s recomputed value identically to every occurrence in
  `reserve`/`selected_papers`/`turn_history`, saving at most once and
  only if something changed. 30 keyword tests (was 11), 9 new maintenance-
  command tests. Full backend suite **1977 passed**.
- **K4.2 — PaperCard hierarchy + Popular/Browse-all filter** (`e907753`):
  card content reordered to title/badges → keyword chips → metadata →
  abstract → action (was: metadata → abstract → keywords). Chips restyled
  to `text-xs font-medium text-accent bg-accent-soft border
  border-accent/30 rounded-md`, wrapping (`whitespace-normal break-words
  max-w-full`) instead of truncating. New `frontend/src/lib/keywords.ts`
  is the canonical frontend aggregation/comparison module (mirrors the
  backend's own casefold + dash-variant + whitespace rules), with 13
  direct unit tests. `ReviewModePanel.tsx`'s filter replaced the flat
  checkbox list with **Popular** (count ≥ 2 across distinct papers,
  sorted by count desc then label, capped at 12, never backfilled with
  count-one keywords) and **Browse all** (every keyword, labeled search
  input matching via the same canonical key, bounded/scrolling results,
  `"No keywords match your search."` empty state) — mutually exclusive
  inside one unframed panel, so no option is ever exposed by two visible
  controls at once. Active filters, OR semantics, source order, and
  submission payloads are all unaffected by construction. Full frontend
  suite **553 passed**; build clean; lint unchanged (3 pre-existing
  warnings, 0 new).
- **K4.3 — bounded review, read-only audit, approved session refresh,
  publication** (`09ec80c` + this checkpoint's own docs/publication
  commits): a range-scoped review against the K4.1/K4.2 contract
  checklists found one objective defect — `scripts/
  re_extract_keywords.py` was missing the project-root `sys.path`
  bootstrap every other `scripts/` file has, so running it exactly as its
  own usage docstring says failed with `ModuleNotFoundError` (never
  caught by K4.1's own tests, which all call `main()` in-process). Fixed
  in `09ec80c` with a subprocess-based regression test; no other
  checklist item was found violated, and no subjective keyword-quality
  tuning was performed. A read-only audit of session
  `8fa9857f21fb4a2dbd103ca771e54e7b`'s 10 originally-served papers
  confirmed, directly: the `"Agentic AI,this"` artifact is gone; single-
  word contained fragments no longer survive; complete three-word
  compounds now do where the source text supports them; title-only
  domination is structurally bounded (9/10 papers admit zero title-
  sourced candidates, 1/10 admits exactly one, never more); no URL/DOI/
  citation/numeric leakage; no canonical duplicates; deterministic;
  capped at six. With explicit user approval, `--apply` was run once
  against that session (96 unique papers; 95 changed) — a before/after
  fingerprint of every non-keyword field confirmed deep equality (the
  only serialization difference was `seen_paper_ids`/`seen_titles`'
  pre-existing, unrelated `set`-to-`list` iteration-order non-determinism,
  confirmed by content, not order); exactly one save; no provider call.
  The session's currently-pending interrupt batch (10 papers, mid-review)
  still shows `yake-v1` keywords — `pending_batch` lives in
  `curation_loop.py`'s own separate interrupt-state thread, outside this
  script's documented, production-path-only scope — and will pick up
  `yake-v2` keywords once the user advances to a new batch from the
  refreshed `reserve`.
- **Explicitly deferred, not part of this feature's own scope**:
  - Automatic historical backfill — an old or not-yet-refreshed session's
    papers keep their stored keyword values (`yake-v1` or otherwise)
    until the explicit maintenance command is run against them, or until
    genuinely re-fetched through a new search/refill.
  - Automatic extractor-version migration — `KEYWORD_EXTRACTOR_VERSION`
    remains documentation-only; nothing compares a paper's own version
    against the current one or triggers re-extraction automatically.
  - Filtering the full selected-paper collection across batches/turns —
    the filter (Popular/Browse-all alike) only ever covers the current
    `pending_batch`.
  - Semantic acronym/full-form merging — `RAG` and `Retrieval-Augmented
    Generation` are deliberately kept as two distinct keywords, on both
    backend and frontend; only literal hyphen/space/case surface variants
    of the *same* phrase merge.
  - Real-world, human-labelled keyword-quality evaluation — the K4.3
    audit is real-session contract/regression verification (has the
    known defect class gone, is the extractor deterministic/bounded/
    clean), not a scored quality benchmark against human judgments;
    explicitly not attempted from one session's worth of examples.
- **Priority**: closed; no further checkpoint scheduled.
- **Status**: Closed (2026-08-14). Commits: `f67d876` (K4.1), `e907753`
  (K4.2), `09ec80c` (K4.3 corrective fix), plus this checkpoint's own
  docs/publication commit. The existing `paper-keywords-filtering` tag is
  unmoved (still marks the K1–K3 checkpoint); a new annotated tag
  `paper-keywords-filtering-v2` marks this K4 checkpoint's own
  documentation commit.

### Curation checkpoint safety incident and hardening — complete
- **Goal**: an urgent safety checkpoint, discovered immediately after K4
  closed: the K4.3 `--apply` run against session
  `8fa9857f21fb4a2dbd103ca771e54e7b` (recorded above as fully successful
  and non-destructive) was itself the actual cause of that session's own
  pending curation interrupt being silently destroyed. Fix the mechanism,
  fix the resulting frontend dead end, and correct the record.
- **Incident**: `save_curation_session()` always writes a fresh
  checkpoint via `curation_session.py`'s smaller `{"session": dict}`-only
  graph, via a plain `graph.invoke()` -- this unconditionally becomes a
  thread's new "latest" checkpoint, silently discarding whatever pending
  task/interrupt `curation_loop.py`'s own graph held for that thread
  (both graphs share the same thread_id/checkpoint row by LangGraph's own
  design), with `session.stage` left untouched at `"curate"` and no error
  raised. K4.1's original safety model asserted `pending_batch` was
  simply unreachable by `scripts/re_extract_keywords.py` and therefore
  safe to leave alone -- wrong: the WRITE itself, not the script's own
  read path, is what destroys it, and the K4.3 `--apply` run against that
  session was genuinely mid-interrupt at the time.
- **Fix 1, `4c230b1` -- zero-selection dead end and frontend honesty**:
  found and fixed two independent, generalizable gaps while investigating
  the stuck session. (1) The backend never enforced "no zero-selection
  finish" -- only the frontend's disabled "I'm done" button did; new
  `session_limits.check_finish_requires_selection()` (409,
  `reason_code="zero_selection_finish"`, the existing
  `SessionCapacityError` convention) closes this. (2) `curationAction`'s
  "Starting new review…" busy state is one hook-wide flag with no session
  scoping -- new `startingReviewVisible` scopes it to the session open
  when the action began. `ReviewModePanel.tsx`'s completion view is now
  gated on `state.stage === "synthesize"`, never merely `!pendingBatch`
  -- an anomalous `curate`-stage/no-batch session renders an honest,
  non-mutating status instead of a fabricated "Curation complete."
  Full backend suite **1985 passed**, frontend **557 passed**.
- **Fix 2, `66d9e3d` -- the maintenance command itself**: new
  `curation_loop.has_unresolved_curation_work()` inspects
  `curation_loop.py`'s own graph snapshot (pending tasks, queued
  next-nodes, interrupts, task errors). `scripts/re_extract_keywords.py`
  now refuses `--apply` whenever `session.stage == "curate"` (catches a
  session whose interrupt was already lost some other way) OR that
  snapshot check reports unresolved work -- re-checked immediately before
  the one `save_curation_session()` call, not only at initial load. A
  refusal exits non-zero, writes nothing, never prints paper content.
  `--apply` remains available for a session with no unresolved work,
  positively proven (not assumed) via a real completed session's
  genuinely empty graph snapshot -- the same mechanism
  `curation_history_service.py`'s own `reopen_curation()` already uses in
  production. Regression tests build a REAL interrupted
  `curation_loop.py` graph (`start_curation_turn`, not just a saved
  session) and confirm: refusal fires; the graph snapshot is byte-
  identical before/after; the pending batch stays normally resumable; no
  save occurred; an already-corrupted (`stage="curate"`, interrupt
  already gone) session also refuses; dry-run stays non-mutating; a
  provably-empty completed session still applies successfully. Full
  backend suite **1992 passed**.
- **Deferred, found but not fixed**: LangGraph 1.2.9 replays a rejected
  node's ORIGINAL resume payload on any later retry against the same
  still-pending task, rather than the caller's corrected one -- confirmed
  by direct instrumented tracing during Fix 2's own investigation, not
  assumed. Affects both `zero_selection_finish` and the pre-existing
  `selected_paper_limit_reached` rejection paths; the new 409 guard closes
  the "can a review finish with zero papers" gap but does NOT by itself
  give a retryable user journey for either. Resolving it needs a
  LangGraph-version-specific fix or a checkpoint/interrupt-payload
  redesign -- out of scope for a checkpoint-safety patch scoped to
  refusing an unsafe write. See this file's own Technical Debt section
  for the full entry.
- **Residual, explicit**: session `8fa9857f21fb4a2dbd103ca771e54e7b`
  remains unrecovered -- its pending interrupt cannot be reconstructed by
  this or any current tool, and no automatic or manual repair was
  attempted (repairing a named session requires a separate, explicitly
  approved decision, not a side effect of a safety patch). Its
  `reserve`/`selected_papers`/`turn_history` keywords are `yake-v2` (from
  the K4.3 run); its actual pending batch (10 papers) was never refreshed
  and is not recoverable by any tool that exists today.
- **Priority**: closed; no further checkpoint scheduled for this
  incident specifically -- the LangGraph retry-replay item above remains
  open technical debt.
- **Status**: Closed (2026-08-14). Commits: `4c230b1` (zero-selection +
  frontend fix), `66d9e3d` (maintenance-command safety patch), plus this
  checkpoint's own docs/publication commit. No new milestone tag --
  neither `paper-keywords-filtering` nor `paper-keywords-filtering-v2`
  moved.

### K4.1b: exclude organization/affiliation entities from paper keywords — complete
- **Goal**: a narrowly bounded keyword-quality correction — a paper's own
  institutional affiliation was found ranking highly enough to become
  one of its top keywords, confirmed real, not hypothetical.
- **Evidence**: a read-only local-sample review (session
  `8fa9857f21fb4a2dbd103ca771e54e7b`, 96 papers) found `"Hai Phong
  University"` (the real institution named in both a paper's title and
  abstract) and `"SLAC National Accelerator Laboratory"` (a different,
  non-RAG paper) both surviving as genuine top-ranked candidates —
  K4.1's own completeness fixes (n=3, bidirectional redundancy
  resolution) correctly assemble the FULL institution name, which then
  ranks well precisely because it's complete; being complete does not
  make it a topic. No recurring author-name or venue-leakage pattern was
  found in the same sample (one isolated venue-title case from an
  atypical "workshop report" document is not a pattern, zero genuine
  author-name cases) — per instruction, no general rule was added for
  either.
- **Fix** (`7d8d304`): `_is_organization_candidate()`, wired into
  `_filter_candidates()` (the same per-candidate stage the existing
  noise/clause-join checks already use — never a whole-paper or
  whole-sentence exclusion), rejects any candidate whose canonical
  tokens include an organization/affiliation designator as a **complete
  token**: `university`, `college`, `department`, `faculty`, `school`,
  `institute`, `laboratory`, `lab`, `corporation`, `corp`, `company`,
  `consortium` — small, generic, topic-agnostic, no institution names
  hardcoded. Whole-token matching (via the same `_canonical_tokens()`
  normalization redundancy resolution already uses) is load-bearing,
  confirmed against real local data: a substring check would wrongly
  reject `"annotated scientific corpora"`/`"large textual corpora"`
  (`corp` only inside `corpora`) and `"conversation remains
  labor-intensive"` (`lab` only inside `labor`). Verified across the
  full 96-paper local sample: only the 2 genuinely affected papers
  changed output; no designator token survives anywhere afterward;
  `"Student Support"`, `"Question Answering"`, `"Question Answering
  Model"`, and existing acronyms/system names (`RAG`, `BERT`, `"Agentic
  RAG Chatbot"`) are untouched, none blacklisted by exact string. No NER
  model, LLM, embeddings, KeyBERT, author/venue database, or semantic
  classifier — a plain token-set membership check, same dependency
  footprint as before.
- **Bounded review confirmed unchanged**: K4.1/K4.2 extraction
  parameters, title-quota logic, redundancy resolution, canonical
  normalization, and every K4.1a maintenance/checkpoint-safety
  constraint (`4c230b1`/`66d9e3d`) — re-ran the active-session-refusal
  regression tests directly, all still passing. Full backend suite
  **1999 passed** (1992 pre-K4.1b, +7 new organization-exclusion tests).
- **Known, accepted residual gap**: a designator-less institutional
  fragment (e.g. `"SLAC National Accelerator"`, or `"Hai Phong"` — the
  city name, not `"University"`) can still survive, since it never
  contains a listed designator token. Not fixed — closing it would need
  a gazetteer/NER-style approach, explicitly out of scope for this
  narrowly bounded, topic-agnostic rule.
- **Explicitly deferred, not part of this checkpoint's own scope**:
  - Semantic keyword typing (classifying a surviving keyword as
    method/technology/task/dataset/domain) — this checkpoint only
    excludes affiliation entities, never classifies what remains.
  - Author/venue classification beyond the demonstrated structural
    rules here — no recurring pattern was found to justify one; not
    speculatively built.
  - Automatic historical refresh — an old or not-yet-refreshed session's
    papers keep their stored keyword values until the explicit
    maintenance command is run against them, or genuinely re-fetched.
  - Recovery of the damaged curation session
    (`8fa9857f21fb4a2dbd103ca771e54e7b`) — remains unrecovered; not
    attempted here, requires a separate, explicitly approved decision.
  - Human-labelled, real-world keyword-quality evaluation — this
    checkpoint's own local-sample review is contract/regression
    verification, not a scored benchmark against human judgments.
- **Priority**: closed; no further checkpoint scheduled.
- **Status**: Closed (2026-08-14). Commit: `7d8d304`, plus this
  checkpoint's own docs/publication commit. No new milestone tag --
  neither `paper-keywords-filtering` nor `paper-keywords-filtering-v2`
  moved.

### K5: keyword-quality evaluation and guarded Policy C production pilot — complete
- **Goal**: settle, with real evidence rather than assumption, whether an
  LLM-based keyword filter improves on the K4-era deterministic YAKE-v2
  extractor enough to justify production use, and if so, exactly how
  narrowly scoped it must be.
- **K5A/K5B -- baseline confirmed, YAKE-v2 unchanged**: YAKE-v2 vs the
  YAKE-v1 reference implementation, AI-assisted human-approved annotation
  over 8 headline product-local papers. YAKE-v2 wins descriptively
  (35.4% vs 29.2% resolved precision; 38.1% vs 32.5% macro concept
  coverage) -- recommendation "keep production YAKE-v2 unchanged."
  Nothing in K5 touched `research_agent/keywords.py`; it remains the
  default, deterministic, offline extractor throughout.
- **K5C -- broad LLM filtering, rejected**: one `gpt-4.1-mini` call per
  paper's full candidate set (decisions keep/remove/uncertain) over the
  same 8 headline papers. Precision improved (+24.6pp) but FAILED the
  frozen provisional gate on two of five conditions -- accepted-keyword
  retention 70.6% (< 90% threshold) and macro concept-coverage retention
  86.9% (< 90% threshold). Recommendation: "do not integrate."
- **K5C.1 -- post-hoc narrowing, Policy C identified**: a zero-cost,
  zero-provider-call re-analysis of the SAME K5C responses under four
  fixed candidate-removal policies (A: `malformed_fragment` only; B:
  `sentence_fragment` only; C: both; D: both plus `redundant_variant`).
  Policy C (and B) pass the frozen gate; Policy C's own precision
  improvement +17.9pp. Explicitly labelled post-hoc exploratory --
  suggestive, not independent confirmation.
- **K5D.1 -- independent 6-paper held-out validation**: 6 NEW
  product-local papers, selected by a deterministic, documented seed,
  disjoint from all 10 prior K5B/K5C papers, frozen BEFORE any candidate
  was examined. Human-approved annotation frozen BEFORE the one live
  `gpt-4.1-mini` call per paper (6 calls total) that ran the SAME frozen
  prompt/schema/Policy C definition validated in K5C.1. Result: 36 -> 19
  candidates; resolved precision 30.56% -> 52.63% (+22.08pp);
  accepted-keyword retention 90.91%; rejected-keyword removal 64.00%;
  false-removal rate 9.09%; macro concept coverage 37.78% -> 34.44%
  (coverage retention 91.18%); 2 `uncertain` decisions, both retained; 0
  provider failures -- ALL 5 frozen gate conditions pass, independently
  of K5C/K5C.1. Conclusion: "Policy C may proceed to a guarded,
  off-by-default production pilot."
- **K5D.2 -- production implementation, off by default**:
  `research_agent/keyword_filter.py` (new, production-owned; copies the
  validated prompt/schema/policy semantics verbatim; never imports
  `scripts/`, proven by a dedicated AST-level regression test) wired into
  `curation_loop.py`'s `_serve_batch_node` (after `serve_next_batch()`,
  before the serialized batch is returned/appended to `turn_history`),
  gated on new `KEYWORD_FILTER_POLICY_C_ENABLED`
  (`research_agent/config/settings.py`, default `False`, strict boolean
  parsing, never read at all when the feature is off). One bounded call
  per newly displayed paper (<=10/turn) -- never one call for a whole
  batch, never the reserve, never a deterministic "suspicious candidate"
  pre-filter (none of those were validated). Content-hashed SQLite cache
  (`data/cache/keyword_filter_cache.sqlite`, WAL + busy_timeout, same
  convention as `telemetry.py`/`embeddings.py`) keyed on the EXACT
  ordered candidate list (never sorted) plus model/prompt-version/
  policy-version. Bounded concurrency (default 3, clamped to `[1,
  provider_fan_out_limit]` only once the feature is confirmed on) via
  `asyncio.Semaphore`/`asyncio.to_thread`, with `asyncio.run()` only at
  the existing synchronous node boundary (mirrors
  `query_expansion.py`'s own bounded-concurrency pattern) -- never turns
  the graph or public API async. Reuses `guard_paid_action`/
  `timed_child_call` unchanged: a cache-only batch opens zero admission/
  lease/telemetry; an uncached batch opens exactly one
  `curation_keyword_filter` paid action for the whole turn (new action
  type added to `telemetry.ACTION_TYPES`), one child-call telemetry
  record per real provider call, content-free by construction (call
  type/provider/model/tokens/cache-hit/latency/outcome/error-type only
  -- no phrase, title, abstract, paper ID, session ID, or prompt text has
  a field to go into). Complete fail-open: any provider error, timeout,
  malformed response, missing/duplicate/invented candidate ID, or
  malformed CACHE row retains the paper's complete original YAKE-v2 list
  untouched -- **K5D.2a** (Codex-reviewed correction) added strict
  cache-row validation (exact candidate-ID set, every decision a string
  in the frozen four-value set) after finding a malformed cache row could
  otherwise crash curation or silently under-filter; also moved
  concurrency-limit parsing out of the always-called `get_settings()`
  into a separate function only reached once the flag is confirmed on,
  and changed an over-limit concurrency request from a hard reject to a
  clamp. Never mutates `session.reserve`'s live `Paper` objects or Chroma
  metadata -- only the serialized batch dict handed to
  `current_batch`/`turn_history` is ever rewritten, confirmed directly
  against a real session (see K5D.3 below).
- **K5D.2c/2d -- pre-existing test-isolation gaps found and fixed**:
  `tests/test_curation_api.py` and `tests/test_api.py` each had a real
  (not simulated) `TestClient(api.app)`-triggered FastAPI lifespan that
  opened the real, gitignored Chroma database without ever inserting a
  document -- discovered while validating K5D.2a, not introduced by it.
  Fixed by patching `api.get_chroma_collection` in every such fixture
  (`_client()` and `_client_with_usage_db()` in each file), with a hard
  `chromadb.PersistentClient` tripwire proving no real client is ever
  constructed. **Residual, explicit, tracked separately (see this file's
  own Technical Debt section)**: the complete K5-focused test group
  still shows Chroma-fingerprint drift when the ~12 pre-existing test
  files that use `TestClient` run together, even though every one of
  them passes cleanly and repeatedly in isolation -- a distinct,
  pre-existing multi-file interaction, deliberately not chased down
  inside K5.
- **K5D.3 -- one bounded, explicitly approved production pilot**: one
  disposable, zero-selection curation session; one already-ranked
  10-paper batch (no refill/search/ranking/embeddings triggered); 10
  approved / 10 actual `gpt-4.1-mini` calls, no retries, no model
  substitution, concurrency 3. Wall-clock 6.90s; summed per-call provider
  latency 19.28s; 4,359 input / 681 output / 5,040 total tokens. 60
  candidates -> 29 retained (31 `sentence_fragment` removed, 0
  `malformed_fragment`, 0 `uncertain`, 0 failures). Verified directly
  against the real session: paper IDs and ranking order unchanged; only
  the displayed batch was filtered (the reserve `Paper` objects for the
  served papers, and the unserved tail, both still held their original,
  unfiltered keywords); `current_batch` and the persisted `turn_history`
  entry held identical filtered lists; selected-paper count unaffected
  (0 before and after); telemetry and cache contained no phrase/title/
  abstract/paper-ID/session-ID content; a cache replay of the exact same
  10 candidate lists against a provider tripwire returned 10/10 hits with
  zero further provider calls and zero new paid-action rows. The flag was
  restored to off immediately afterward via a graceful backend restart,
  confirmed via the worker process's own environment. **The pilot's
  60 -> 29 result is operational-behavior evidence only -- the live
  pilot's own papers were never human-labelled, so it is never read as a
  quality-improvement measurement**; that claim rests entirely on K5D.1's
  independent held-out result above.
- **Final product decision**: Policy C is implemented and validated but
  remains **off by default**; eligible for explicit opt-in use; this
  evidence alone does not make it the default. YAKE-v2 remains the
  default extractor and the universal fail-open/rollback behavior for
  every failure class this feature can hit.
- **Explicit limitations**: small (8-10-paper) product-local samples at
  every stage; AI-assisted, human-approved labels, not independent
  third-party annotation; no external benchmark or statistical-
  significance claim anywhere in K5; one model (`gpt-4.1-mini`) and one
  frozen prompt version evaluated throughout; the K5D.3 live pilot batch
  itself is unlabelled (see above); no manual browser/UI verification was
  performed during the pilot (no browser-automation tool was available in
  that session; relies on existing frontend contract tests); disabling
  the flag stops filtering FUTURE batches immediately but does not
  retroactively rewrite keywords already persisted into a session's
  `turn_history`/`selected_papers` from while it was on; the K5D.2c/2d
  multi-file Chroma test interaction remains open, tracked in this file's
  Technical Debt section.
- **Priority**: closed; no further keyword-quality checkpoint scheduled.
  The next roadmap item, whenever picked up, is not another keyword-
  quality experiment.
- **Status**: Closed (2026-08-19). Commits: `53244d7`..`070b9c0` (K5A-C.1
  evaluation harness and evidence), `7d907f6`..`40a9ac6` (K5D.1 held-out
  validation), `4d89ba3` (K5D.2 implementation), `d77a6de` (K5D.2a
  correction), `0c1074f`/`695f1f1` (K5D.2c/2d test isolation), plus this
  checkpoint's own docs/publication commit. New annotated milestone tag
  `k5-keyword-quality-evaluation-pilot` marks this checkpoint's
  documentation commit; `paper-keywords-filtering` and
  `paper-keywords-filtering-v2` are both unmoved.

### M4 follow-on: token-level streaming revisit, report prose streaming, production streaming hardening (not part of M4)
- **Goal**: real, useful next-layer streaming work M4 deliberately left
  open, tracked so it isn't lost, not because it's scheduled.
- **Items**:
  - **Token-level structured-answer streaming** — only worth
    revisiting if a future OpenAI SDK release exposes `jiter`'s own
    `partial_mode="trailing-strings"` mode publicly, or if the answer
    schema itself is redesigned to avoid the single-large-string-field
    shape that makes `partial_mode=True` unhelpful today. Not a bug in
    this codebase — a documented SDK/schema-shape limitation.
  - **Report prose/section streaming** — no proven use case yet;
    revisit only if a real product need for partial report text
    emerges, not speculatively.
  - **Production reverse-proxy/load testing of long-lived SSE
    connections** — this phase validated only against local dev
    servers (`uvicorn --reload`, Vite dev server); a real deployment
    (nginx/similar buffering, connection limits, timeout tuning) was
    never exercised.
  - **Multi-instance streaming coordination** — unchanged from M2's own
    already-documented limitation; SQLite-backed admission/leases are
    not a distributed mechanism.
  - **HITL report approval / targeted-section refinement** — both
    remain explicitly out of scope; no redesign of R4's own bounded
    draft→evaluate→revise loop was made or is planned here.
  - **Auth and deployment hardening** — unchanged, out of scope for the
    current single-user architecture.
- **Priority**: open, none scheduled — real work, doesn't block anything.
- **Status**: Open.

**M1/M2/M3/M4 are complete for the current single-user SQLite
architecture.** None of them claims authentication, distributed rate
limiting, or production-scale streaming infrastructure — that's
explicitly out of scope here, tracked in "M4 follow-on" above as real,
unscheduled work. No M5 (or any further milestone) exists anywhere in
this project's prior roadmap docs as of this checkpoint — none is
invented here.

### Research Lanes: optional multi-query curation (RL1–RL6) — complete, off by default
- **Goal**: let a curation review optionally start from up to four
  complementary search "lanes" (label + research question + one query
  each) instead of one topic string, with lane-of-origin provenance on
  every candidate paper. Single search unchanged and still the default.
- **Design**: RL0 was a no-code design review (Model C — hybrid
  LLM-suggested, user-editable lanes — recommended and chosen). Full
  architecture, API contracts, provenance/dedup model, test list, and
  the one approved live-journey provider-call record are in
  `docs/architecture.md`'s "Research Lanes" section; user-facing summary
  in `README.md`.
- **Feature flag**: `RESEARCH_LANES_ENABLED` (`config/settings.py`,
  strict boolean, default `False`). When off: the capability endpoint
  returns `false`, the frontend shows no lane affordance, and
  `POST /curation/lanes/suggest` / a `lanes` field on
  `POST /curation/start` return `403` before any admission / telemetry /
  provider work. An existing lane session keeps refilling after the flag
  is turned back off — `_refill_node` dispatches on `session.lanes`, not
  the flag.
- **New modules**: `research_lanes.py` (domain model + strict-construction
  vs. lenient-deserialization validation), `lane_suggestion.py` (one
  protected `gpt-4.1-mini` structured call, no retry / no fallback
  model), `research_lane_retrieval.py` (`retrieve_across_lanes` +
  `refill_lane_session`), `services/lane_suggestion_service.py`,
  `api_app/routers/curation_lanes.py`. `dedup.deduplicate_with_clusters`
  added as the sole cross-lane identity authority.
- **API**: `GET /curation/capabilities` → `{"research_lanes_enabled":
  bool}` (that key only); `POST /curation/lanes/suggest`; optional
  `lanes` on `POST /curation/start`; `lanes` / `paper_lane_ids` /
  `lane_result_counts` (all defaulted empty) added to state / turn /
  turn-history responses — pre-RL4 checkpoints and single-query clients
  deserialize unchanged.
- **Tests**: 230 in the dedicated backend group + 27 lane-specific
  across `test_curation_api`/`test_curation_session`/`test_curation_loop`;
  614 frontend (Vitest); one fully mocked Playwright browser journey
  (`frontend/e2e/research-lanes-mocked.spec.ts`, desktop + 375 px
  mobile). Full backend suite at the milestone: 2618 passed, 1 failed
  (the known pre-existing multi-file Chroma-fingerprint flake — see
  Technical Debt below; unrelated).
- **Live validation (RL6)**: one approved journey — 1 Suggest + 1 Start,
  two enabled lanes, disposable topic. Approved bound ≤ 8 OpenAI calls;
  actual 7 (4 `gpt-4.1-mini` + 3 `text-embedding-3-small`, ~4,636
  tokens, ≈ $0.0013), no retry / no model substitution / no OpenAlex.
  Session deleted, flag restored (nothing in `.env` changed).
- **Residual limitations**: no mid-curation lane editing (frozen at
  Start), no per-lane refill, no lane-aware chat/report synthesis, no
  coverage guarantee, no statistical/quality evaluation of multi-lane
  vs. single-query retrieval, single-user only. Duplicate lane *labels*
  are permitted at Start (only lane IDs are enforced unique).
- **Status**: Closed (2026-08-28). Commits: `909e300`..`29d334d`
  (RL1–RL5b) plus this checkpoint's docs/publication commit. Annotated
  milestone tag `research-lanes-v1` marks the documentation commit.

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
- **`eval_results/latency_history.csv` has no reproducing script.** It is
  a one-time historical measurement (search-call parallelization,
  before/after); no script in `scripts/` produces it. Either commit a
  small `scripts/eval_latency.py` or leave it labelled as historical. See
  `docs/evaluation.md`'s "Search-call parallelization" note,
  `eval_results/archive/README.md`.
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
- **LangGraph 1.2.9 replays a rejected node's ORIGINAL resume payload on
  retry, not the caller's new one — confirmed directly, not fixed.**
  Found during the curation-checkpoint-safety patch (`66d9e3d`): when
  `curation_loop.py`'s `_present_and_apply_node` raises
  `SessionCapacityError` (either existing reason code,
  `selected_paper_limit_reached`, or the new `zero_selection_finish`
  added in `4c230b1`) after `interrupt()` has already returned a resume
  payload, the pending task is correctly left resumable (`snap.tasks[0].
  interrupts` still present, confirmed by direct inspection) — but a
  SUBSEQUENT `resume_curation_turn()`/`Command(resume=...)` call against
  that SAME still-pending task, even with a genuinely different,
  corrected `picked_paper_ids`/`stop` payload, silently replays the
  FIRST (rejected) attempt's payload instead of the new one — reproduced
  directly with instrumented node-level tracing, using fresh
  `sqlite_checkpointer` connections per call to rule out same-process
  caching as the cause. This means: **a client that receives a 409 from
  either capacity-style guard cannot recover by simply resubmitting a
  corrected `/curation/{id}/picks` request against the same pending
  batch** — the retry would replay the original, still-rejected payload
  and 409 again, identically, no matter what the client actually sends.
  Not currently known to be reachable through the real UI in the
  `zero_selection_finish` case (the frontend's own `review-stop` button
  stays disabled at `totalSelected===0`, so a normal user session cannot
  construct the rejected request in the first place) — but genuinely
  reachable for `selected_paper_limit_reached` (a client legitimately
  picking more papers than the 60-paper cap allows in one request, then
  retrying with fewer). **The new backend 409 guard added in `4c230b1`
  closes the "can a review finish with zero papers" gap, but does NOT by
  itself give a retryable user journey for either rejection path** — that
  would require either a LangGraph-version-specific fix, a checkpoint/
  interrupt-payload redesign, or routing corrected resumes through a
  fresh turn (`start_curation_turn`) instead of `Command(resume=...)`
  against the same task — all explicitly out of scope for the
  checkpoint-safety patch that found this, which was scoped to
  refusing an unsafe write, not redesigning the resume/retry workflow.
  Not scheduled; no owner assigned. See `research_agent/session_limits.
  py`'s `check_selected_paper_capacity`/`check_finish_requires_selection`
  docstrings, `docs/architecture.md`'s matching entry, and `tests/
  test_curation_loop.py`'s zero-selection tests (which assert only that
  the interrupt remains present/resumable in principle, never that a
  differently-shaped retry against it is honored).
- **Multi-file Chroma-fingerprint test interaction (found during K5D.2c/
  2d).** `tests/test_curation_api.py` and `tests/test_api.py` each had a
  real `TestClient(api.app)` lifespan that opened the real, gitignored
  `data/chroma_db/` without patching `api.get_chroma_collection` -- both
  fixed (see `docs/architecture.md`'s K5 section). Every file that uses
  `TestClient` (those two, plus `tests/test_curation_chat.py`'s own
  extensive `get_chroma_collection` patching) is now confirmed clean,
  repeatedly, in isolation: zero Chroma-fingerprint drift. Running the
  ~12-file subset of the K5-focused group that includes them TOGETHER
  still shows drift, even though no single file in that subset drifts
  alone -- a genuine cross-file interaction (most likely shared,
  module-level mutable state in `research_agent.api._state`, a plain
  dict, persisting across test files within one pytest process; not
  confirmed further). Bisection was deliberately stopped once this shape
  was established, per explicit instruction not to open another repair
  chain during K5's own closure. Not fixed, not scheduled, no owner
  assigned. The dedicated K5D.2 production test files themselves (`tests/
  test_keyword_filter.py`, `tests/test_config_settings.py`, `tests/
  test_curation_loop_keyword_filter.py`) are unaffected and pass cleanly
  on their own. **Confirmed still present at the Research Lanes (RL6)
  milestone**: a full `uv run pytest -q` showed `2618 passed, 1 failed`,
  the single failure being `test_curation_loop_keyword_filter.py::
  test_real_chroma_db_untouched` from exactly this interaction — it fails
  identically on the clean `4ffe63c` baseline and is unrelated to
  Research Lanes.
- **`ReviewCard.tsx` nests its delete `<button>` inside the card
  `<button>` (frontend, pre-existing, not Research Lanes).** The card is
  a `<button onClick={onSelect}>` and the delete affordance is a second
  `<button>` inside it (wrapped in a `display: contents` div that fixes
  layout but not the nesting). React 19 logs a DOM-nesting warning
  whenever a review card renders. Surfaced during the RL6 live browser
  journey (the mocked journey used an empty reviews list so it never
  appeared). Last touched by `6b16baf` — well before the Research Lanes
  range (`909e300`..) — so it is explicitly **not** a Research Lanes
  defect and was left for a separate fix (the smallest correct fix is to
  make the card a non-`<button>` clickable region, or move the delete
  button out as a sibling). Not scheduled, no owner assigned.
  `frontend/src/components/ReviewsList/ReviewCard.tsx`.

## 4. Explicitly deferred platform work

**Not next. Not implied by anything above.** Each of these already has
its own dedicated design document — this backlog does not duplicate or
re-scope them, only points at them:

- **OAuth/multi-user auth** — `specs/production-readiness-roadmap.md` §4.
  Still fully deferred. Separately, a smaller, already-completed
  single-user HTTP Basic Auth gate exists (`docs/deployment.md`) — this
  is NOT that roadmap item and does not substitute for it; multi-user
  support still does not exist.
- **Automated/off-site backups, retention policy, encryption at rest** —
  `docs/deployment.md` §5. A local, manually-invoked create/verify/
  restore tool (`scripts/data_backup.py`) exists and was validated
  against real data — scheduling, remote/cloud storage, pruning, and
  encryption are still all open, tied to whichever hosting platform is
  eventually chosen.
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
