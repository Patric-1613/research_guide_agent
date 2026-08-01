# Legacy cleanup audit

**Status: audit only. Nothing in this document is deleted, moved, or
modified.** Phase 18A of the project's standardization effort — a
read-only pass over the codebase looking for files/directories that look
old, transitional, obsolete, duplicated, or confusing now that the
backend/frontend/config/eval standardization (Phases 0–17) is done.
Written against tag `standardized-single-user-project`.

## Method

For every file in `research_agent/` (18 domain modules + `api.py` +
`api_app/` + `services/` + `config/`), checked reference counts via
`rg`-style search across `research_agent/`, `tests/`, and `scripts/` —
**every single domain module has real, non-zero references**; nothing at
the module level is orphaned. That result, plus Phase 11's already-
completed dead-code audit of every `research_agent.api` reference
(`specs/remaining-standardization-plan.md` §5 — "zero dead code or
orphaned old-location imports found"), means this audit's genuine
findings are concentrated in **repo-root files and documentation**, not
in `research_agent/` itself. That's reported honestly below rather than
manufacturing findings to fill out the domain-module categories.

## Candidates

### `requirements-frozen-baseline.txt`

- **Path**: `/requirements-frozen-baseline.txt` (repo root, 130 lines)
- **Why it looks legacy**: Predates the pip → uv dependency-management
  migration (`54c613b`, 2026-07-16). `pyproject.toml`'s own comment
  describes it as the historical snapshot `uv.lock`'s
  `constraint-dependencies` was originally pinned from — a one-time
  provenance record, not something read live by any tooling.
- **Confirmed stale, not just old**: it still lists `streamlit==1.59.0`
  and `altair==6.2.2` — both removed from the live app by the
  `streamlit-removal` branch (`b9150fa`, 2026-07-28), **12 days after**
  the uv migration and 2 days after this file's own last content update
  (`03de039`, 2026-07-26). `pyproject.toml`'s actual
  `constraint-dependencies` list is already clean of both — confirmed by
  direct grep, zero matches. So this file isn't just historical, it's
  actively **wrong** about the dependency set today.
- **References found**: exactly one — `pyproject.toml`'s own comment
  naming it. No code imports it, no script reads it, no test references
  it.
- **Tests referencing it**: none.
- **Runtime risk if removed**: none — nothing reads this file at
  runtime, build, or `uv lock` time.
- **Recommendation**: **ARCHIVE CANDIDATE.** Real historical value (proof
  of what was captured at the pip→uv cutover) but actively misleading
  left in the repo root as-is, since it now describes a dependency set
  that hasn't been true for weeks. Either delete it (its provenance
  value is already fully captured in `pyproject.toml`'s comment and
  `uv.lock`) or move it under a clearly historical location (e.g.
  `docs/` with a note) if the provenance record itself is worth keeping.
- **Removable before future backend feature work?** Yes — zero coupling
  to anything live.

### `research_agent_architecture.svg`

- **Path**: `/research_agent_architecture.svg` (repo root, 16.6 KB)
- **Why it looks legacy**: Still embedded in `README.md`'s "Architecture"
  section (`![Architecture diagram](research_agent_architecture.svg)`)
  as *the* architecture diagram — but `docs/architecture.md`'s own
  opening paragraph already says this SVG "describe[s] the original
  single-pipeline research agent... predates the curation/report/chat
  system and the React UI added afterward." It also predates the entire
  `api.py` → `api_app/`/`services/` split (Phases 2–10) and the
  `research_agent/config/` module (Phase 14) — none of that structure
  appears in it.
- **References found**: `README.md` (embeds it), `docs/architecture.md`
  (names it in the "predates..." caveat above).
- **Tests referencing it**: none (it's a docs asset, not code).
- **Runtime risk if removed**: none — purely a documentation image.
- **Recommendation**: **UNKNOWN — needs a manual decision.** Two real
  options, both reasonable, neither mine to pick: (a) regenerate it to
  reflect current architecture (a real effort — it's a detailed,
  hand-maintained file-by-file diagram, not a quick edit), or (b) keep
  it as explicitly-labeled historical material for the original
  one-shot pipeline specifically (which is still accurate for *that*
  part of the system) and rely on `docs/architecture.md`'s Mermaid
  diagram (added in this project's Obsidian-notes work, not in this
  repo) or a new one for current architecture. Not classified as a
  simple ARCHIVE/REMOVE candidate because it's still linked from the
  README a reader sees first — removing the link without a replacement
  would be a real documentation regression, not a cleanup.
- **Removable before future backend feature work?** Not blocking either
  way — this doesn't interact with backend code at all. Purely a
  documentation-quality decision, on its own timeline.

### `docs/`/`specs/` document proliferation (not a single file — a pattern)

- **Paths**: `docs/architecture.md`, `docs/project-history.md`, `docs/
  evaluation.md`, `specs/migration-plan.md`, `specs/remaining-
  standardization-plan.md`, `specs/production-readiness-roadmap.md`,
  `specs/test-plan.md`, plus this audit and `specs/backend-backlog.md`
  (this phase's own output) — **9 markdown documents** tracking overlapping
  facets of the same standardization history.
- **Why it looks confusing, not necessarily legacy**: `docs/
  architecture.md` and `specs/migration-plan.md` in particular narrate
  almost the same Phase 0–17 history in parallel, from two different
  organizing principles (one "current state, with phase notes woven in,"
  the other "a phase log"). Both are internally accurate and
  cross-reference each other correctly — this isn't drift or
  contradiction (the kind of issue Phase 12 already found and fixed
  once, in `ranking.py`'s README description) — it's **volume**: a
  newcomer has no single obvious entry point for "what actually
  happened," only "start at `docs/architecture.md` or `specs/
  migration-plan.md`, either works, they agree."
- **References found**: extensively cross-linked to each other and to
  `README.md` throughout; all actively used as the basis for this
  project's Obsidian vault notes (external to this repo).
- **Tests referencing it**: none (docs, not code).
- **Runtime risk if removed/consolidated**: none directly, but real risk
  of losing cross-references or historical detail if merged carelessly
  — this is exactly the kind of change that should get its own explicit
  go-ahead, not be bundled into a "cleanup."
- **Recommendation**: **KEEP — historical docs/specs**, all of them, as
  they stand today. Flagged here only because "9 overlapping documents"
  is a legitimate thing to notice post-standardization, not because any
  individual one is wrong, dead, or unsafe. A future consolidation (e.g.
  one canonical "how we got here" doc, with the phase-by-phase detail
  demoted to an appendix) is a real, reasonable idea — but it's a
  documentation-architecture decision on its own, not a byproduct of
  this audit, and not attempted here.
- **Removable before future backend feature work?** N/A — not a removal
  candidate, a possible future reorganization.

### `research_agent/api.py`

- **Path**: `research_agent/api.py` (142 lines)
- **Classification**: **KEEP — public compatibility surface.** Per this
  phase's own instruction to treat it as KEEP unless proven otherwise —
  and nothing in this audit found reason to question that. It's the
  live `uvicorn research_agent.api:app` entrypoint, re-exports every
  schema/helper/runtime object still reached via `research_agent.api.
  <name>` or `patch.object(api, "<name>", ...)` (confirmed: 22
  referencing files across `research_agent/`, `tests/`, `scripts/`), and
  removing or shrinking any of its re-exports would require updating
  every one of those 22 files in lockstep — explicitly out of this
  audit's scope (no code changes).

### `research_agent/api_app/` (naming, not content)

- **Path**: `research_agent/api_app/`
- **Why it might look confusing to a newcomer**: the name reads as
  transitional/interim (and is — `docs/architecture.md`'s own "Why
  `research_agent/api_app/`, not `research_agent/api/`" section explains
  exactly why it can't yet be renamed `api/` without breaking
  `patch.object(api, "<name>", ...)` mocking semantics).
- **Classification**: **KEEP — public compatibility surface** (same
  reasoning as `api.py` — this package and `api.py` are a matched pair).
  Not a cleanup candidate; the "interim" naming is a deliberate,
  already-documented decision, not an oversight. Renaming it is already
  tracked as deferred work in `specs/remaining-standardization-plan.md`
  and `TODO.md` (the Obsidian vault), gated on retiring `api.py`'s
  compatibility re-exports first — not scheduled, not part of this
  audit's recommendations.

### `research_agent/ranking.py`

- **Path**: `research_agent/ranking.py`
- **Why it might look confusing**: README's own "Project structure"
  section used to contradict its later "Retrieval ranking experiments"
  section about whether this module is used by the live app — **already
  found and fixed in Phase 12** (`specs/remaining-standardization-plan.md`
  §4 audit, confirmed via `rg` that `agent.py` directly imports and calls
  `get_partition_n`/`partition_by_citation`/`merge_with_guaranteed_slots`
  from this module in its live rerank tool).
- **Classification**: **KEEP — live domain module.** Genuinely dual-use
  (BM25/hybrid functions are eval-only via `scripts/eval_retrieval.py`;
  citation-partitioned reranking functions are live via `agent.py`), and
  that dual-use nature is now correctly documented, not confusing. Not
  a legacy candidate — re-confirmed here, not re-litigated.

### Everything else in `research_agent/`

Every remaining domain module (`agent.py`, `citations.py`,
`curation_chat.py`, `curation_loop.py`, `curation_session.py`,
`dedup.py`, `embeddings.py`, `enrichment.py`, `ingestion.py`, `qa.py`,
`query_expansion.py`, `report.py`, `schema.py`, `storage.py`,
`summarize.py`, `tracing.py`, `web_search.py`), all 13 files under
`services/`, all 11 files under `api_app/routers/`, and `api_app/
{schemas,serializers,errors,runtime,app}.py` — **KEEP — live domain
module / live service / live router, still used by services/routes/tests.**
Reference counts confirmed non-zero for every one (see Method above);
no further detail warranted per-file since none showed any legacy
signal.

`research_agent/__init__.py` — empty package marker, trivial, not
flagged.

### `tests/`, `scripts/`, `eval_data/`, `eval_results/`

- **`tests/`** (21 files): every file maps to a real, currently-live
  module (confirmed via `specs/test-plan.md`'s own per-file coverage
  table, still accurate for every file present at the time it was
  written; `test_config_settings.py` was added later, Phase 14, and is
  correctly newer than that snapshot). **KEEP**, no candidates.
- **`scripts/`** (21 files): 14 of 21 are named directly in `README.md`/
  `specs/test-plan.md`; the other 7 (`test_citation_styles.py`,
  `test_curation_chat_offer.py`, `test_filters.py`,
  `test_refinement_live.py`, `test_report_regeneration.py`,
  `test_report_update_offer_live.py`, `test_top_k.py`) are **not**
  individually named anywhere, but `specs/test-plan.md`'s "Eval scripts"
  section explicitly covers "every `scripts/test_*.py` file" as a
  category, by design (manual, live, cost-real, run on demand — never
  meant to be an exhaustive individually-documented list). **KEEP —
  live domain-adjacent tooling**, not a cleanup candidate; this was a
  documentation-completeness question, not a legacy-code one, and the
  category-level documentation already answers it.
- **`eval_data/`**: `reference_topics.json`, `reference_topics.xlsx`
  (gitignored, correctly — Phase 11 finding), `stage1_ragas_questions.json`,
  `README.md` (Phase 15). **KEEP**, already standardized, no candidates.
- **`eval_results/`**: current running logs, `archive/` (Phase 13),
  `runs/` (gitignored, Phase 15), `README.md`. **KEEP**, already
  standardized, no candidates. (`latency_history.csv`'s missing
  reproducing script is a known, already-documented gap — tracked as
  technical debt in `specs/backend-backlog.md`, not a legacy-file
  finding, since the file itself isn't stale, just under-tooled.)

### Frontend/backend boundary (frontend read only enough to confirm this)

`frontend/src/lib/api/client.ts` is the entire boundary — one typed
fetch wrapper, every request going through `research_agent/api_app/
routers/`'s HTTP surface. No server-side code lives under `frontend/`,
no client-side code lives under `research_agent/`. Frontend structure
was already standardized in Phase 16; no legacy signal found there, and
per this audit's own scope ("frontend/ only enough to identify
frontend-vs-backend boundaries"), no deeper frontend audit was
performed here.

## Summary

| Classification | Count | Items |
|---|---|---|
| ARCHIVE CANDIDATE | 1 | `requirements-frozen-baseline.txt` |
| UNKNOWN (needs manual decision) | 1 | `research_agent_architecture.svg` |
| KEEP, flagged as a pattern worth knowing about (not a defect) | 1 | docs/specs proliferation (9 documents) |
| KEEP — public compatibility surface | 2 | `research_agent/api.py`, `research_agent/api_app/` (naming) |
| KEEP — live domain module/service/router | everything else in `research_agent/`, `services/`, `api_app/` | — |
| KEEP — live tooling | `tests/`, `scripts/`, `eval_data/`, `eval_results/` | — |
| REMOVE CANDIDATE | 0 | none found |

**No dead code was found.** This is consistent with — not a contradiction
of — Phase 11's own prior finding that the backend standardization left
zero orphaned imports. The two genuine findings here (a stale root-level
provenance file, and a stale-but-still-linked architecture diagram) are
both repo-root artifacts outside `research_agent/` entirely, exactly
where a decade-old-feeling "legacy cleanup audit" instinct would expect
to find something and a targeted, evidence-based one instead finds very
little — which is itself the expected outcome of a project that has
already been through 17 phases of standardization.

## Related

`specs/backend-backlog.md` (this phase's companion document) ·
`specs/remaining-standardization-plan.md` (Phase 11's own prior
dead-code audit) · `docs/architecture.md`
