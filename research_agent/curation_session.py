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
from research_agent.schema import Paper, WebArticle

# Distinct prefix so curation-session rows in the shared
# data/qa_checkpoints.sqlite file are easy to identify/scope separately
# from any future chat thread_id, even though thread_id-based isolation
# alone is already sufficient (see this module's own docstring).
THREAD_ID_PREFIX = "curation-session:"


def curation_thread_id(session_id: str) -> str:
    return f"{THREAD_ID_PREFIX}{session_id}"


_REPORT_SECTION_NAMES = ("findings", "limitations", "future_scope")


def _serialize_report(report: dict | None) -> dict | None:
    """report.py's generate_report() return shape nests raw Paper objects
    in each section's cited_papers and in the top-level skipped_papers --
    not JSON-native, so (same reasoning as every other field here) they
    must become plain dicts before a checkpointer sees them."""
    if report is None:
        return None
    return {
        **{
            name: {
                "content": report[name]["content"],
                "cited_papers": [p.to_dict() for p in report[name]["cited_papers"]],
            }
            for name in _REPORT_SECTION_NAMES
        },
        "skipped_papers": [p.to_dict() for p in report["skipped_papers"]],
    }


def _deserialize_report(d: dict | None) -> dict | None:
    if d is None:
        return None
    return {
        **{
            name: {
                "content": d[name]["content"],
                "cited_papers": [Paper(**p) for p in d[name]["cited_papers"]],
            }
            for name in _REPORT_SECTION_NAMES
        },
        "skipped_papers": [Paper(**p) for p in d["skipped_papers"]],
    }


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
