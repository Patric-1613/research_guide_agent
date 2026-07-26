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
from research_agent.schema import Paper

# Distinct prefix so curation-session rows in the shared
# data/qa_checkpoints.sqlite file are easy to identify/scope separately
# from any future chat thread_id, even though thread_id-based isolation
# alone is already sufficient (see this module's own docstring).
THREAD_ID_PREFIX = "curation-session:"


def curation_thread_id(session_id: str) -> str:
    return f"{THREAD_ID_PREFIX}{session_id}"


def _session_to_dict(session: PaperPoolSession) -> dict:
    return {
        "topic": session.topic,
        "reserve": [[paper.to_dict(), score] for paper, score in session.reserve],
        "cursor": session.cursor,
        "seen_paper_ids": list(session.seen_paper_ids),
        "seen_titles": list(session.seen_titles),
        "stage": session.stage,
        "target_count": session.target_count,
        "selected_paper_ids": list(session.selected_paper_ids),
    }


def _dict_to_session(d: dict) -> PaperPoolSession:
    return PaperPoolSession(
        topic=d["topic"],
        reserve=[(Paper(**paper_dict), score) for paper_dict, score in d["reserve"]],
        cursor=d["cursor"],
        seen_paper_ids=set(d["seen_paper_ids"]),
        seen_titles=set(d["seen_titles"]),
        stage=d["stage"],
        target_count=d.get("target_count", 10),
        selected_paper_ids=list(d.get("selected_paper_ids", [])),
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
