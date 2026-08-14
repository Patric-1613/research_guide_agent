#!/usr/bin/env python3
"""Paper Keywords and Filtering, K4.1: explicit, offline, dry-run-by-default
maintenance command that recomputes `Paper.keywords` for every paper in ONE
local curation session, using the CURRENT `research_agent.keywords.extract_keywords`
(see that module's own `KEYWORD_EXTRACTOR_VERSION`).

**No LLM call, no embeddings API, no OpenAI/Tavily/arXiv/Semantic Scholar
call, no network call -- ever.** Only local SQLite I/O (the session
checkpointer) and pure, deterministic keyword extraction.

**Safety model**:
- Dry-run by default. Nothing is written to the database unless `--apply`
  is passed explicitly.
- Even with `--apply`, a save only happens if at least one paper's
  keywords actually changed relative to what's currently stored. A
  no-op input (recomputed keywords already match stored keywords for
  every paper) is reported and never saved, `--apply` or not.
- Loads and saves through the exact same production path the app itself
  uses -- `research_agent.curation_session.load_curation_session`/
  `save_curation_session`, via `research_agent.qa.sqlite_checkpointer`
  over `QA_CHECKPOINT_DB_PATH` -- never a hand-rolled SQL query, so this
  script can never produce a session shape the app itself wouldn't.
- Every OTHER field on the session is preserved deep-equal. Only the
  `keywords` list on each Paper-bearing entry changes, computed fresh
  from that same paper's OWN (title, abstract) via the current
  extractor -- ranking, scores, paper_id/order, Chroma embeddings,
  reports, selection state, and turn_history structure are never
  touched.

**Scope note on Paper-bearing session locations** (confirmed by direct
inspection of `curation_session.py`/`query_expansion.py`, not assumed):
- `session.reserve` (list of `(Paper, score)`) and `session.selected_papers`
  (list of `Paper`) are real `Paper` objects -- updated directly.
- `session.turn_history` entries hold each served batch as
  `[[paper_dict, score], ...]`, where `paper_dict` is already a plain
  dict (not a `Paper` object) -- see `PaperPoolSession`'s own docstring.
  Every occurrence of a given `paper_id` across every turn_history entry
  is updated to the same recomputed keyword list.
- `session.report`/`session.report_versions` also embed serialized Paper
  dicts (`cited_papers`, `skipped_papers`) -- these are DELIBERATELY left
  untouched. A report is a generated, historical artifact; this script's
  own contract explicitly forbids modifying reports, so a report's
  snapshot of a paper's keywords at generation time is preserved exactly
  as generated, even if it now differs from the paper's freshly
  recomputed keywords.
- `pending_batch` (the batch currently awaiting a curate-stage pick) is
  NOT part of `PaperPoolSession` and is NOT reachable through
  `load_curation_session`/`save_curation_session` at all -- it lives
  inside `curation_loop.py`'s own, separate LangGraph interrupt state
  (`snap.tasks[0].interrupts[0].value["batch"]`), a different checkpoint
  thread this script deliberately does not touch, since the task this
  script exists for is scoped to the production checkpointer/session
  loader (`load_curation_session`/`save_curation_session`).

**Curation-checkpoint safety patch (post-incident, see docs/architecture.md
and specs/backend-backlog.md for the full record).** A real session was
corrupted by an EARLIER version of this script: it called
`save_curation_session()` against a session that was genuinely
mid-interrupt (`stage == "curate"`, a batch presented but not yet
picked). `save_curation_session()` always writes a FRESH checkpoint via
`curation_session.py`'s own smaller `{"session": dict}`-only graph, via a
plain `graph.invoke()` (never `Command(resume=...)`) -- this
unconditionally became the thread's new "latest" checkpoint, silently
discarding `curation_loop.py`'s own pending task/interrupt, with
`session.stage` left at `"curate"` (untouched, since the smaller graph
never touches it) and no error raised anywhere. The frontend then
misread the resulting `pending_batch == None` as "curation complete,"
though it genuinely was not -- see `fix: prevent zero-selection curation
dead ends` for the frontend-side correction. That real, still-unrecovered
session is `8fa9857f21fb4a2dbd103ca771e54e7b`; this script is NOT used to
repair it.

This script now REFUSES `--apply` for any session with unresolved
curation work, checked via `research_agent.curation_loop.
has_unresolved_curation_work()` (this graph's OWN checkpoint snapshot --
pending tasks, queued next-nodes, interrupts, or task errors) PLUS a
plain `session.stage == "curate"` check (catching a session whose
interrupt was ALREADY destroyed by some other cause, where the graph
snapshot alone would show nothing pending) -- checked immediately before
the write, not only at initial load, since this script's own read-diff-
save flow itself takes real (if brief) wall-clock time. Dry-run remains
fully available and read-only for ANY session regardless of this check
-- only `--apply`'s actual write is gated. A refused apply writes
NOTHING, exits non-zero, and never attempts to repair or resume graph
state itself -- that remains a product decision, not something this
maintenance command decides unilaterally.

**Provably safe for a genuinely completed session.** Confirmed directly
(not assumed): a session that reached `stage == "synthesize"` via a real
`stop=True` resume has an EMPTY graph snapshot for `curation_loop.py`'s
own thread (`snap.tasks == ()`, `snap.next == ()`) -- there is nothing
left for `save_curation_session()`'s overwrite to destroy, and this is
the exact same "plain `graph.invoke()`, no pending task needed" mechanism
`curation_history_service.py`'s own `reopen_curation()` already performs
routinely in production against a completed thread. `--apply` remains
available for a session with no unresolved curation work, on that basis.

Usage:
    python scripts/re_extract_keywords.py SESSION_ID              # dry-run (default)
    python scripts/re_extract_keywords.py SESSION_ID --apply       # write changes, if any (refused for active/interrupted curation)
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.curation_loop import has_unresolved_curation_work
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.keywords import KEYWORD_EXTRACTOR_VERSION, extract_keywords
from research_agent.qa import QA_CHECKPOINT_DB_PATH, sqlite_checkpointer
from research_agent.schema import Paper


def _is_active_curation(session_stage: str, session_id: str, checkpointer) -> bool:
    """True if this session must never be written to by save_curation_session()
    right now -- either it's still in the curate stage at all (whether or
    not a live interrupt currently exists for it -- a stage=="curate"
    session whose interrupt was already lost some other way is exactly as
    unsafe to overwrite as one that still has a live one, since the ONLY
    path back to a resumable state is a real product action, never this
    script), or curation_loop.py's own graph snapshot shows unresolved
    work (a pending task, a queued next-node, an interrupt, or a task
    error) regardless of what `stage` claims."""
    if session_stage == "curate":
        return True
    return has_unresolved_curation_work(session_id, checkpointer)


def _compute_keyword_map(papers_by_id: dict[str, Paper]) -> dict[str, list[str]]:
    """Computes each paper's keywords exactly once, keyed by paper_id."""
    return {
        paper_id: extract_keywords(paper.title, paper.abstract)
        for paper_id, paper in papers_by_id.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute Paper.keywords for one local curation session using the current extractor.",
    )
    parser.add_argument("session_id", help="Local curation session id to re-extract keywords for.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write recomputed keywords back to the session. Without this flag, the script only reports what would change.",
    )
    args = parser.parse_args(argv)

    with sqlite_checkpointer(QA_CHECKPOINT_DB_PATH) as checkpointer:
        session = load_curation_session(args.session_id, checkpointer)
        if session is None:
            print(f"error: no session found for session_id={args.session_id!r}", file=sys.stderr)
            return 1

        # Deep-copied up front so every non-keyword field can be verified
        # deep-equal to the original after mutation, and so a dry run
        # never touches the loaded object the caller might reuse.
        original_session = deepcopy(session)

        try:
            papers_by_id: dict[str, Paper] = {}
            for paper, _score in session.reserve:
                papers_by_id.setdefault(paper.paper_id, paper)
            for paper in session.selected_papers:
                papers_by_id.setdefault(paper.paper_id, paper)
            for entry in session.turn_history:
                for paper_dict, _score in entry["batch"]:
                    papers_by_id.setdefault(paper_dict["paper_id"], Paper(**paper_dict))
        except (KeyError, TypeError) as exc:
            print(f"error: session {args.session_id!r} has an unexpected/malformed shape: {exc!r}", file=sys.stderr)
            return 1

        keyword_map = _compute_keyword_map(papers_by_id)

        changed_paper_ids: set[str] = set()

        new_reserve = []
        for paper, score in session.reserve:
            new_keywords = keyword_map[paper.paper_id]
            if new_keywords != paper.keywords:
                changed_paper_ids.add(paper.paper_id)
                paper = replace(paper, keywords=new_keywords)
            new_reserve.append((paper, score))
        session.reserve = new_reserve

        new_selected_papers = []
        for paper in session.selected_papers:
            new_keywords = keyword_map[paper.paper_id]
            if new_keywords != paper.keywords:
                changed_paper_ids.add(paper.paper_id)
                paper = replace(paper, keywords=new_keywords)
            new_selected_papers.append(paper)
        session.selected_papers = new_selected_papers

        new_turn_history = []
        for entry in session.turn_history:
            new_batch = []
            for paper_dict, score in entry["batch"]:
                new_keywords = keyword_map[paper_dict["paper_id"]]
                if new_keywords != paper_dict.get("keywords", []):
                    changed_paper_ids.add(paper_dict["paper_id"])
                    paper_dict = {**paper_dict, "keywords": new_keywords}
                new_batch.append([paper_dict, score])
            new_turn_history.append({**entry, "batch": new_batch})
        session.turn_history = new_turn_history

        print(f"session_id: {args.session_id}")
        print(f"extractor version: {KEYWORD_EXTRACTOR_VERSION}")
        print(f"unique papers found: {len(papers_by_id)}")

        # Informational only here -- dry-run reporting stays fully
        # available and read-only regardless of this session's curation
        # state. The ENFORCED refusal is the freshly-re-checked gate
        # immediately before save_curation_session() below, not this one.
        if _is_active_curation(session.stage, args.session_id, checkpointer):
            print(
                f"note: session stage is {session.stage!r} with unresolved curation work -- "
                "--apply will be refused for this session (see below).",
            )

        if not changed_paper_ids:
            print("no changes: recomputed keywords already match stored keywords for every paper. Nothing to save.")
            return 0

        print(f"papers with changed keywords: {len(changed_paper_ids)}")
        for paper_id in sorted(changed_paper_ids):
            old_keywords = papers_by_id[paper_id].keywords
            print(f"  {paper_id}")
            print(f"    old: {old_keywords}")
            print(f"    new: {keyword_map[paper_id]}")

        # Verify every non-keyword field is still deep-equal to the
        # original load before ever considering a save -- a mismatch here
        # means this script itself has a bug and must refuse to write.
        original_session.reserve = [(replace(p, keywords=[]), s) for p, s in original_session.reserve]
        original_session.selected_papers = [replace(p, keywords=[]) for p in original_session.selected_papers]
        original_session.turn_history = [
            {**e, "batch": [[{**pd, "keywords": []}, s] for pd, s in e["batch"]]} for e in original_session.turn_history
        ]
        comparable_session = deepcopy(session)
        comparable_session.reserve = [(replace(p, keywords=[]), s) for p, s in comparable_session.reserve]
        comparable_session.selected_papers = [replace(p, keywords=[]) for p in comparable_session.selected_papers]
        comparable_session.turn_history = [
            {**e, "batch": [[{**pd, "keywords": []}, s] for pd, s in e["batch"]]} for e in comparable_session.turn_history
        ]
        if comparable_session != original_session:
            print(
                "error: a field other than keywords would have changed -- refusing to save. This is a bug in this script.",
                file=sys.stderr,
            )
            return 2

        if not args.apply:
            print("dry-run: pass --apply to write these changes.")
            return 0

        # ENFORCED gate, re-checked fresh here -- immediately before the
        # one and only save_curation_session() call in this script, not
        # reused from the informational check above. This script's own
        # read-diff-verify sequence above takes real (if brief) wall-clock
        # time, and this is the actual point past which a write happens,
        # so this is where "immediately before writing" has to mean.
        # Never prints paper titles/abstracts/keywords -- only the
        # session_id and stage already used elsewhere in this script's own
        # output.
        current_session = load_curation_session(args.session_id, checkpointer)
        if current_session is None:
            print(f"error: session {args.session_id!r} no longer exists -- refusing to save.", file=sys.stderr)
            return 1
        if _is_active_curation(current_session.stage, args.session_id, checkpointer):
            print(
                f"refused: session {args.session_id!r} has active/interrupted curation "
                f"(stage={current_session.stage!r}) and cannot be updated by this command. "
                "Nothing was changed. This command never writes to a session with unresolved "
                "curation work -- see this script's own module docstring for why.",
                file=sys.stderr,
            )
            return 3

        save_curation_session(session, args.session_id, checkpointer)
        print("saved.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
