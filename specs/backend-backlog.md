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
- **`requirements-frozen-baseline.txt` is stale** — still lists
  `streamlit==1.59.0`/`altair==6.2.2`, both removed from the live
  dependency set weeks ago. See `specs/legacy-cleanup-audit.md`. Low
  risk, zero runtime coupling — safe to archive/remove whenever
  convenient, not urgent.
- **`research_agent_architecture.svg` is stale relative to current
  architecture** (predates the curation system, the React frontend, and
  the entire `api.py` → `api_app/`/`services/` split) but is still the
  only diagram linked from `README.md`. See `specs/legacy-cleanup-audit.md`
  — flagged UNKNOWN there, needs a manual decision (regenerate vs.
  relabel as historical), not urgent.
- **No numeric pass/fail gate on the RAGAS eval** — still manually run
  and human-interpreted, not CI-enforced. See `docs/evaluation.md`, and
  the Obsidian vault's `Mentor-Feedback.md`/`TODO.md` (surfaced by the
  2026-07-17 mentor review, item 13).
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
