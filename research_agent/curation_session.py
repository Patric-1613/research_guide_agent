"""curation-checkpointer Phase 2: activates the SQLite checkpointer
built-but-inactive during the qa.py LangGraph conversion, scoped
narrowly to the new literature-review curation flow only.

This is a genuinely new, separate entry point — not a change to
qa.py's ask() path at all. ask() still runs on _DEFAULT_GRAPH
(checkpointer=None), exactly as stateless as it always was; nothing in
this module is imported by qa.py or api.py's existing /chat endpoint.

Reuses qa.py's own sqlite_checkpointer()/QA_CHECKPOINT_DB_PATH (the
same physical data/qa_checkpoints.sqlite file, decided during the
qa.py conversion's own Phase 0) rather than creating a second
checkpoint database — LangGraph's checkpoint tables are keyed by
thread_id, so a distinct thread_id namespace ("curation-session:...")
is what actually separates this flow's rows from any future chat
thread's, not a separate file.

Checkpointed state is a plain JSON-native dict, not the PaperPoolSession
dataclass/Paper objects directly — confirmed by direct testing that
checkpointing the dataclass as-is DOES work today, but LangGraph logs
"Deserializing unregistered type ... This will be blocked in a future
version" for both Paper and PaperPoolSession, since its serializer's
pickle-fallback path for arbitrary custom types is being deprecated.
Converting to/from a plain dict at the save/load boundary (see
_session_to_dict/_dict_to_session) avoids that fallback entirely —
str/int/list/dict all serialize natively, no deprecation risk.

The graph itself is deliberately minimal for this phase: one no-op
node whose only job is giving the checkpointer something to persist
state around. The real search/refill/interrupt business logic is
curation-pool-foundation's next phase (Phase 3, the interrupt-based
curation loop) — this phase only proves persistence works.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from research_agent.query_expansion import PaperPoolSession
from research_agent.report import ANALYTICAL_SECTION_NAMES
from research_agent.schema import Paper, WebArticle

# Distinct prefix so curation-session rows in the shared
# data/qa_checkpoints.sqlite file are easy to identify/scope separately
# from any future chat thread_id, even though thread_id-based isolation
# alone is already sufficient (see this module's own docstring).
THREAD_ID_PREFIX = "curation-session:"


def curation_thread_id(session_id: str) -> str:
    return f"{THREAD_ID_PREFIX}{session_id}"


_LEGACY_REPORT_SECTION_NAMES = ("findings", "limitations", "future_scope")
# report-quality Phase R2C serialization fix: a fresh (post-R2B) report
# dict carries BOTH the 3 legacy compatibility keys (findings/
# limitations/future_scope, projected copies -- see report.py's
# _project_legacy_fields) AND all 8 real Analytical/template section
# keys as top-level entries simultaneously. The two tuples never
# overlap in key spelling (e.g. "findings" vs. "thematic_findings"), so
# this is a plain concatenation, not a dedup. Previously only
# _LEGACY_REPORT_SECTION_NAMES was serialized/deserialized here, which
# meant the 5 non-legacy-mapped keys (executive_summary,
# introduction_scope, methodology_landscape, gap_analysis, conclusion)
# silently vanished as top-level dict keys on every save/load round trip
# -- harmless for _report_to_out (which only ever reads the 3 legacy
# keys plus the already-plain "sections" list), but it meant
# report.py's own citation-preservation helpers
# (_resolve_prior_citations_for_regeneration/_restore_dropped_citations)
# could never find prior citations for those 5 sections on ANY
# regeneration reached through the real HTTP API (every request reloads
# the session fresh from the checkpointer -- see load_curation_session's
# own docstring, there is no in-process cache). Iterating over the union
# below, filtered to keys the report dict actually has, fixes this while
# staying fully backward compatible: an old report with only the 3
# legacy keys present still round-trips exactly as it always did.
_ALL_REPORT_SECTION_NAMES = _LEGACY_REPORT_SECTION_NAMES + ANALYTICAL_SECTION_NAMES


def _serialize_report(report: dict | None) -> dict | None:
    """report.py's generate_report() return shape nests raw Paper objects
    in each section's cited_papers and in the top-level skipped_papers --
    not JSON-native, so (same reasoning as every other field here) they
    must become plain dicts before a checkpointer sees them. Same for
    each section's cited_web_articles (raw WebArticle objects).

    report-quality Phase R1 bug fix: cited_web_articles used to be
    dropped here entirely -- only content/cited_papers ever survived a
    save/load round trip, silently losing every web citation a section
    had on reload. Fixed by serializing it the same way cited_papers
    already was. reference_numbers (plain ints, already JSON-native) and
    the top-level references list (plain dicts, already JSON-native) are
    passed through with .get(..., default) -- a report generated before
    this phase has neither key at all, and that ABSENCE (not an empty
    list) is exactly the signal the API serializer
    (api_app/serializers.py's _report_to_out) uses to know to derive
    them fresh via report.py's derive_legacy_references() instead of
    trusting a stored (nonexistent) value.

    report-quality Phase R2C: loops over _ALL_REPORT_SECTION_NAMES
    (legacy + every Analytical/template key), not just the 3 legacy
    names -- see that tuple's own comment for why. `if name in report`
    means an old, pre-R2B report dict (only the 3 legacy keys) still
    round-trips exactly as before; a fresh report (all 11 keys present)
    now fully round-trips every section, not just 3 of them.
    """
    if report is None:
        return None
    serialized = {
        name: {
            "content": report[name]["content"],
            "cited_papers": [p.to_dict() for p in report[name]["cited_papers"]],
            "cited_web_articles": [a.to_dict() for a in report[name].get("cited_web_articles", [])],
            "reference_numbers": report[name].get("reference_numbers", []),
        }
        for name in _ALL_REPORT_SECTION_NAMES if name in report
    }
    serialized["skipped_papers"] = [p.to_dict() for p in report["skipped_papers"]]
    if "references" in report:
        serialized["references"] = report["references"]
    # report-quality Phase R2A: same opaque pass-through convention as
    # references above -- `sections` is already a list of plain dicts
    # (no Paper/WebArticle objects nested in it), so there's nothing to
    # convert, only a presence check.
    if "sections" in report:
        serialized["sections"] = report["sections"]
    # report-quality Phase R2C: report_template is a plain string,
    # opaque pass-through same as references/sections above -- absent
    # for a report generated before this phase, which is exactly the
    # signal api_app/serializers.py's _report_to_out uses to default it
    # to "analytical" at read time rather than assuming one here.
    if "report_template" in report:
        serialized["report_template"] = report["report_template"]
    return serialized


def _deserialize_report(d: dict | None) -> dict | None:
    if d is None:
        return None
    deserialized = {
        name: {
            "content": d[name]["content"],
            "cited_papers": [Paper(**p) for p in d[name]["cited_papers"]],
            "cited_web_articles": [WebArticle(**a) for a in d[name].get("cited_web_articles", [])],
            "reference_numbers": d[name].get("reference_numbers", []),
        }
        for name in _ALL_REPORT_SECTION_NAMES if name in d
    }
    deserialized["skipped_papers"] = [Paper(**p) for p in d["skipped_papers"]]
    if "references" in d:
        deserialized["references"] = d["references"]
    if "sections" in d:
        deserialized["sections"] = d["sections"]
    if "report_template" in d:
        deserialized["report_template"] = d["report_template"]
    return deserialized


def _session_to_dict(session: PaperPoolSession) -> dict:
    return {
        "topic": session.topic,
        "display_title": session.display_title,
        "reserve": [[paper.to_dict(), score] for paper, score in session.reserve],
        "cursor": session.cursor,
        "seen_paper_ids": list(session.seen_paper_ids),
        "seen_titles": list(session.seen_titles),
        "stage": session.stage,
        "target_count": session.target_count,
        "selected_paper_ids": list(session.selected_paper_ids),
        "selected_papers": [paper.to_dict() for paper in session.selected_papers],
        "report": _serialize_report(session.report),
        "chat_history": list(session.chat_history),
        "web_articles_added": [a.to_dict() for a in session.web_articles_added],
        "pending_web_offer": session.pending_web_offer,
        "pending_report_update": session.pending_report_update,
        "refinement_notes": list(session.refinement_notes),
        "report_covered_web_article_count": session.report_covered_web_article_count,
        # curation-turn-history Phase 9b: entries are already JSON-native
        # ({"turn_number": int, "batch": [[paper_dict, score], ...],
        # "refilled": bool} -- paper_dict is a plain dict, never a real
        # Paper object here), so this passes through as-is, same
        # convention as chat_history above -- no hydration step needed.
        "turn_history": list(session.turn_history),
        "stop_reason": session.stop_reason,
        "report_approved_web_article_urls": list(session.report_approved_web_article_urls),
        "revoked_web_article_urls": list(session.revoked_web_article_urls),
    }


def _dict_to_session(d: dict) -> PaperPoolSession:
    return PaperPoolSession(
        topic=d["topic"],
        # Older sessions saved before Phase 8's display_title field existed
        # fall back to their own raw topic -- a sensible default, not a
        # crash, matching every other backward-compat .get() in this
        # function.
        display_title=d.get("display_title") or d["topic"],
        reserve=[(Paper(**paper_dict), score) for paper_dict, score in d["reserve"]],
        cursor=d["cursor"],
        seen_paper_ids=set(d["seen_paper_ids"]),
        seen_titles=set(d["seen_titles"]),
        stage=d["stage"],
        target_count=d.get("target_count", 10),
        selected_paper_ids=list(d.get("selected_paper_ids", [])),
        selected_papers=[Paper(**paper_dict) for paper_dict in d.get("selected_papers", [])],
        report=_deserialize_report(d.get("report")),
        chat_history=list(d.get("chat_history", [])),
        web_articles_added=[WebArticle(**a) for a in d.get("web_articles_added", [])],
        pending_web_offer=d.get("pending_web_offer"),
        pending_report_update=d.get("pending_report_update"),
        refinement_notes=list(d.get("refinement_notes", [])),
        report_covered_web_article_count=d.get("report_covered_web_article_count", 0),
        # Older sessions saved before Phase 9b's turn_history/stop_reason
        # fields existed simply have neither -- an empty history and "no
        # reason recorded" are both sensible, non-crashing defaults, same
        # backward-compat convention as display_title above.
        turn_history=list(d.get("turn_history", [])),
        stop_reason=d.get("stop_reason"),
        # Older sessions saved before curation-chat-add-to-report Phase 4
        # simply have no approved web sources yet -- an empty set is the
        # correct default (nothing was ever explicitly approved), not just
        # a non-crashing fallback, same backward-compat convention as
        # every other field above.
        report_approved_web_article_urls=set(d.get("report_approved_web_article_urls", [])),
        # report-quality Phase R2D: older sessions predating this fix have
        # no revoked URLs recorded yet -- an empty set is correct (nothing
        # was tracked as revoked before this field existed), same
        # backward-compat convention as report_approved_web_article_urls
        # immediately above.
        revoked_web_article_urls=set(d.get("revoked_web_article_urls", [])),
    )


class CurationSessionState(TypedDict):
    session: dict  # JSON-native serialization of PaperPoolSession — see _session_to_dict/_dict_to_session


def _sync_node(state: CurationSessionState) -> dict:
    """No-op: passes the session dict straight through. Exists only so
    the graph has a node for the checkpointer to snapshot state around —
    Phase 3 replaces this with the actual search/present/interrupt loop."""
    return {"session": state["session"]}


def build_curation_graph(checkpointer: BaseCheckpointSaver):
    """Unlike qa.py's build_qa_graph(), there's no checkpointer=None
    default here — a curation session's whole point is being persisted,
    so a caller must explicitly provide one (typically via qa.py's own
    sqlite_checkpointer() context manager)."""
    graph = StateGraph(CurationSessionState)
    graph.add_node("sync", _sync_node)
    graph.add_edge(START, "sync")
    graph.add_edge("sync", END)
    return graph.compile(checkpointer=checkpointer)


def save_curation_session(session: PaperPoolSession, session_id: str, checkpointer: BaseCheckpointSaver) -> None:
    """Persists `session` under `session_id` via the checkpointer. Safe to
    call repeatedly for the same session_id — each call is a new
    checkpoint for that thread_id, and get_state() below always returns
    the latest one."""
    graph = build_curation_graph(checkpointer)
    config = {"configurable": {"thread_id": curation_thread_id(session_id)}}
    graph.invoke({"session": _session_to_dict(session)}, config=config)


def load_curation_session(session_id: str, checkpointer: BaseCheckpointSaver) -> PaperPoolSession | None:
    """Returns the most recently saved PaperPoolSession for session_id, or
    None if nothing has ever been saved under that id — a clean, expected
    result (Phase 2d), not an error."""
    graph = build_curation_graph(checkpointer)
    config = {"configurable": {"thread_id": curation_thread_id(session_id)}}
    state = graph.get_state(config)
    if not state.values:
        return None
    return _dict_to_session(state.values["session"])


def delete_curation_session(session_id: str, checkpointer: BaseCheckpointSaver) -> None:
    """curation-review-management Phase 8, item 1: permanently removes every
    checkpoint and write ever recorded for this session_id -- delete_thread()
    is a real, built-in BaseCheckpointSaver method (confirmed directly
    against the installed langgraph-checkpoint-sqlite package, not assumed),
    already scoped to exactly this thread_id via curation_thread_id()'s
    prefix, same as every other function in this module. Safe to call even
    if session_id was never saved -- delete_thread() is a plain DELETE...
    WHERE thread_id = ?, which matches zero rows without erroring."""
    checkpointer.delete_thread(curation_thread_id(session_id))


def _find_paper_in_turn_history(session: PaperPoolSession, paper_id: str) -> dict | None:
    for entry in session.turn_history:
        for paper_dict, _ in entry["batch"]:
            if paper_dict["paper_id"] == paper_id:
                return paper_dict
    return None


def select_paper_from_history(session: PaperPoolSession, paper_id: str) -> None:
    """curation-turn-history Phase 9c: adds a previously-served-but-not-
    yet-selected paper directly to selected_paper_ids/selected_papers.
    Mutates `session` in place -- caller is responsible for
    save_curation_session() afterward, same convention report.py/
    curation_chat.py's mutating functions already use.

    SYNTHESIZE-STAGE ONLY, enforced here, not just documented in a
    comment. Reason, traced precisely (not just cited as a rule): while
    stage == "curate", a real LangGraph interrupt is pending on
    curation_loop.py's OWN graph -- a different compiled graph object
    than this module's. Writing a new checkpoint onto the same thread_id
    via THIS module's simpler sync-only graph (as save_curation_session()
    does) would become the "latest" checkpoint for that thread, but it
    carries no pending-task information at all -- the next
    resume_curation_turn() call would find nothing to resume against.
    This is the exact corruption curation-refinement-and-auto-offer
    Phase 6f already discovered once (which is why refinement text rides
    inside Command(resume=...) instead of a separate out-of-band write),
    not a new concern invented for this feature. Picking from history
    WHILE stage == "curate" must go through /curation/{id}/picks instead
    (curation_loop.py's widened pick-validation, Phase 9d) -- the only
    channel safe to use while an interrupt is pending.

    Silently a no-op if paper_id is already selected -- same duplicate-
    pick tolerance curation_loop.py's own present_and_apply_node already
    has (`if pid not in session.selected_paper_ids`), not an error here
    either.

    Raises ValueError (api.py converts to 400, same convention as
    report.py/curation_chat.py's own "not ready" checks) if stage isn't
    "synthesize" yet, if paper_id was never actually served in any past
    turn of THIS session, or (curation-editable-until-locked bug fix)
    if the review is already locked -- a report has been generated
    and/or chat has started. That second check didn't exist when this
    function was first written (Phase 9c), because reaching
    stage=="synthesize" USED to always mean curation had just finished
    and nothing more could be added; curation-editable-until-locked
    (Phase 10) later redefined "locked" to mean has_report/has_chat
    specifically, not stage alone (reopen_curation_session() already
    gates on exactly that), but this function was never updated to
    match -- a real, exploitable gap: the turn-history browser was
    still letting a paper be added here even after a report/chat
    already existed for a DIFFERENT selection, silently orphaning them
    from what the report/chat actually reflects.
    """
    if session.stage != "synthesize":
        raise ValueError(
            f"Session is not ready to select from history (stage={session.stage!r}, expected 'synthesize') -- "
            "curation must finish first; while still curating, submit this paper_id via /curation/{session_id}/picks instead."
        )
    if session.report is not None:
        raise ValueError("Cannot select from history: a report has already been generated for this review.")
    if session.chat_history:
        raise ValueError("Cannot select from history: chat has already started for this review.")
    if paper_id in session.selected_paper_ids:
        return
    paper_dict = _find_paper_in_turn_history(session, paper_id)
    if paper_dict is None:
        raise ValueError(f"paper_id {paper_id!r} was not found in this session's turn history")
    session.selected_paper_ids.append(paper_id)
    session.selected_papers.append(Paper(**paper_dict))


def reopen_curation_session(session: PaperPoolSession) -> None:
    """curation-editable-until-locked Phase 10c: lets a review that WAS
    explicitly stopped (stage=="synthesize") go back to active curation,
    as long as chat/report hasn't started yet -- the ONLY gate on
    editability as of Phase 10b (target_count/exhaustion no longer lock
    anything on their own).

    Mutates `session` in place: resets stage back to "curate" and clears
    stop_reason. Deliberately does NOT itself re-invoke
    start_curation_turn() on the checkpointer -- this module can't
    import curation_loop.py (curation_loop.py already imports FROM this
    module; importing back the other way would be circular), so the
    caller (api.py, which already imports both) is responsible for that
    afterward, exactly the same "mutate here, save/resume there"
    division of labor select_paper_from_history() above and
    save_curation_session() already use.

    Empirically verified before implementing this: a fresh
    start_curation_turn() call on a thread_id whose prior invocation
    already reached END (via an explicit stop) works cleanly as a
    reopen mechanism -- it's a plain graph.invoke(), not
    Command(resume=...), so it needs no pending task to exist. It just
    needs a session dict with stage reset to "curate", and correctly
    resumes serving from wherever cursor/selected_paper_ids left off,
    never restarting.

    Raises ValueError (api.py converts to 400, same convention as
    select_paper_from_history above) if there's nothing to reopen
    (still curating) or if the review is genuinely locked (report or
    chat already started).
    """
    if session.stage != "synthesize":
        raise ValueError("Session is still curating -- there is nothing to reopen.")
    if session.report is not None:
        raise ValueError("Cannot reopen: a report has already been generated for this review.")
    if session.chat_history:
        raise ValueError("Cannot reopen: chat has already started for this review.")
    session.stage = "curate"
    session.stop_reason = None


def list_curation_sessions(checkpointer: BaseCheckpointSaver) -> list[dict]:
    """curation-api-and-ui Phase 6b/6c: powers the frontend's "reviews
    list" panel — every session ever saved to this checkpointer, as a
    lightweight summary (not a full PaperPoolSession reconstruction;
    listing potentially many sessions doesn't need every selected
    Paper's full data, just enough for a picker UI).

    Verified directly (not assumed) that checkpointer.list(None)
    enumerates checkpoints across EVERY thread_id in the store, ordered
    newest-first globally (confirmed against langgraph's real SqliteSaver,
    not just its docstring) — so deduping to the first-seen row per
    thread_id gives that thread's latest checkpoint (confirmed against a
    real resumed session, not just a single-checkpoint one), AND that
    first-seen order across distinct thread_ids is itself already sorted
    by each thread's own most-recent activity, descending — no separate
    sort needed, this is a property of the global newest-first iteration
    order, not an assumption.

    Returns [] if nothing has ever been saved -- not an error.
    """
    seen: dict[str, dict] = {}
    for tup in checkpointer.list(None):
        thread_id = tup.config["configurable"]["thread_id"]
        if not thread_id.startswith(THREAD_ID_PREFIX) or thread_id in seen:
            continue
        session_dict = tup.checkpoint.get("channel_values", {}).get("session")
        if session_dict is not None:
            seen[thread_id] = session_dict

    return [
        {
            "session_id": thread_id[len(THREAD_ID_PREFIX):],
            "topic": session_dict["topic"],
            "display_title": session_dict.get("display_title") or session_dict["topic"],
            "stage": session_dict["stage"],
            "selected_count": len(session_dict.get("selected_paper_ids", [])),
            "target_count": session_dict.get("target_count", 10),
            "has_report": session_dict.get("report") is not None,
            "has_chat": len(session_dict.get("chat_history", [])) > 0,
        }
        for thread_id, session_dict in seen.items()
    ]
