"""curation-interrupt-loop Phase 3: the actual interactive curation loop
— present a batch, pause via interrupt(), wait for the user's picks,
resume, update state, loop or refill or stop.

Reuses Phase 1's serve_next_batch()/refill_pool()/needs_refill() and
Phase 2's session serialization/checkpointer exactly — this module is
loop CONTROL FLOW only, no new pool/ranking/refill logic. Graph shape,
approved before implementation:

    START --> check_pool (real no-op node -- a conditional edge cannot
                |          target START itself, verified directly: LangGraph
                |          raises "unknown target '__start__'" at compile
                |          time, so this node exists purely as the loop-back
                |          anchor)
                v
          route_entry --"refill"--> refill_node --> serve_batch
                |                                        |
                +--"serve"-------------------------------+
                                                           v
                                                   route_after_serve --"exhausted"--> stop_exhausted (END)
                                                           |
                                                     "present"
                                                           v
                                                present_and_apply  <-- interrupt() lives here, ALONE
                                                           |
                                                           v
                                                  route_after_picks
                                                     +-"target_met"----> stop_target_met (END)
                                                     +-"user_stopped"--> stop_user_requested (END)
                                                     +-"continue"------> check_pool (loop back)

Verified directly against the installed langgraph==1.2.9 before designing
this (not assumed from general knowledge):
  - interrupt(value) halts the node; graph.invoke() returns normally with
    a "__interrupt__" key in the result, not a raised exception to the
    caller.
  - Resuming graph.invoke(Command(resume=<value>), config=same_config)
    resumes from the START of the node, RE-EXECUTING everything in it —
    confirmed empirically (a print() before interrupt() ran twice; code
    after interrupt() in the same node ran exactly once). This is why
    present_and_apply() has nothing before its interrupt() call.
  - A non-persisted per-invocation dependency (the OpenAI client
    refill_pool() needs) passes via config={"configurable": {...}},
    read in a (state, config)-signature node — verified this does NOT
    touch the checkpointer's serializer at all, unlike putting it in
    state would.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from research_agent.curation_session import (
    _dict_to_session,
    _session_to_dict,
    curation_thread_id,
)
from research_agent.query_expansion import BATCH_SIZE, refill_pool, serve_next_batch
from research_agent.schema import Paper


class CurationLoopState(TypedDict):
    session: dict  # JSON-native PaperPoolSession serialization — see curation_session.py
    current_batch: list  # this turn's presented batch: [[paper_dict, score], ...]
    stop_reason: str | None  # None while looping; set by whichever stop_* node ends the graph
    should_stop: bool  # transient: set by present_and_apply, read once by route_after_picks
    # curation-api-and-ui Phase 6c: transient like should_stop above -- reset
    # to False by _check_pool_node at the START of every turn (the loop-back
    # anchor every turn passes through, including the very first), then set
    # True only if _refill_node actually runs that same turn. Lets a client
    # distinguish "this turn triggered a fresh search" from "served from the
    # existing pool" without inferring it from reserve-size deltas across
    # separate requests.
    refilled: bool


def _route_entry(state: CurationLoopState) -> str:
    session = _dict_to_session(state["session"])
    return "refill" if session.needs_refill(batch_size=BATCH_SIZE) else "serve"


def _refill_node(state: CurationLoopState, config) -> dict:
    session = _dict_to_session(state["session"])
    configurable = config["configurable"]
    refill_pool(
        session,
        s2_api_key=configurable.get("s2_api_key"),
        client=configurable["client"],
        use_openalex_fallback=configurable.get("use_openalex_fallback", False),
        openalex_mailto=configurable.get("openalex_mailto"),
    )
    return {"session": _session_to_dict(session), "refilled": True}


def _serve_batch_node(state: CurationLoopState) -> dict:
    session = _dict_to_session(state["session"])
    batch = serve_next_batch(session, batch_size=BATCH_SIZE)
    serialized_batch = [[paper.to_dict(), score] for paper, score in batch]
    return {"session": _session_to_dict(session), "current_batch": serialized_batch}


def _route_after_serve(state: CurationLoopState) -> str:
    return "exhausted" if not state["current_batch"] else "present"


def _present_and_apply_node(state: CurationLoopState) -> dict:
    """The ONLY node touching interrupt() — nothing runs before it, so
    re-execution on resume is safe (see module docstring). Everything
    after the interrupt() call runs exactly once, on the resume pass.
    """
    resume = interrupt({
        "batch": state["current_batch"],
        "selected_count": len(state["session"]["selected_paper_ids"]),
        "target_count": state["session"]["target_count"],
    })

    picked_paper_ids = resume.get("picked_paper_ids", [])
    stop = bool(resume.get("stop", False))

    # Validate against the ACTUALLY-presented batch — never trust the
    # resume payload blindly. Anything not in current_batch is silently
    # dropped, not applied and not an error (a stale/malformed client
    # payload shouldn't corrupt session state).
    presented_by_id = {paper_dict["paper_id"]: paper_dict for paper_dict, _ in state["current_batch"]}
    valid_picks = [pid for pid in picked_paper_ids if pid in presented_by_id]

    session = _dict_to_session(state["session"])
    for pid in valid_picks:
        if pid not in session.selected_paper_ids:
            # Both lists updated together, in the same node, from the
            # same current_batch data -- kept in sync by construction, and
            # verified explicitly (not just assumed) by
            # tests/test_curation_loop.py's sync-invariant test across a
            # session that includes a refill.
            session.selected_paper_ids.append(pid)
            session.selected_papers.append(Paper(**presented_by_id[pid]))

    return {"session": _session_to_dict(session), "should_stop": stop}


def _route_after_picks(state: CurationLoopState) -> str:
    if state.get("should_stop"):
        return "user_stopped"
    session = _dict_to_session(state["session"])
    if len(session.selected_paper_ids) >= session.target_count:
        return "target_met"
    return "continue"


def _make_stop_node(reason: str):
    def _stop_node(state: CurationLoopState) -> dict:
        session = _dict_to_session(state["session"])
        session.stage = "synthesize"
        # refilled must not be left carrying a stale value from whichever
        # turn last passed through check_pool -- a stop reached via
        # present_and_apply's "target_met"/"user_stopped" routes straight
        # here WITHOUT re-visiting check_pool (interrupt-resume restarts
        # execution AT present_and_apply, not from check_pool; confirmed
        # by hitting exactly this stale-True case in testing), and there
        # is no current batch left for "was it a fresh search" to
        # describe once curation has stopped anyway.
        return {"session": _session_to_dict(session), "stop_reason": reason, "refilled": False}
    return _stop_node


def _check_pool_node(state: CurationLoopState) -> dict:
    """Loop-back anchor — exists because a conditional edge cannot target
    START itself (verified directly, not assumed: LangGraph raises
    "unknown target '__start__'" at compile time). No longer a pure
    no-op as of Phase 6c: also resets `refilled` to False here, since
    this is the one node every turn passes through before the
    refill-or-serve routing decision, first turn included — so a stale
    True from a PRIOR turn can never leak into this turn's result.
    """
    return {"refilled": False}


def build_curation_loop_graph(checkpointer: BaseCheckpointSaver):
    graph = StateGraph(CurationLoopState)
    graph.add_node("check_pool", _check_pool_node)
    graph.add_node("refill", _refill_node)
    graph.add_node("serve_batch", _serve_batch_node)
    graph.add_node("present_and_apply", _present_and_apply_node)
    graph.add_node("stop_exhausted", _make_stop_node("exhausted"))
    graph.add_node("stop_target_met", _make_stop_node("target_met"))
    graph.add_node("stop_user_requested", _make_stop_node("user_stopped"))

    graph.add_edge(START, "check_pool")
    graph.add_conditional_edges("check_pool", _route_entry, {"refill": "refill", "serve": "serve_batch"})
    graph.add_edge("refill", "serve_batch")
    graph.add_conditional_edges("serve_batch", _route_after_serve, {
        "exhausted": "stop_exhausted",
        "present": "present_and_apply",
    })
    graph.add_conditional_edges("present_and_apply", _route_after_picks, {
        "target_met": "stop_target_met",
        "user_stopped": "stop_user_requested",
        "continue": "check_pool",
    })
    graph.add_edge("stop_exhausted", END)
    graph.add_edge("stop_target_met", END)
    graph.add_edge("stop_user_requested", END)

    return graph.compile(checkpointer=checkpointer)


def start_curation_turn(session_id: str, checkpointer: BaseCheckpointSaver, initial_session_dict: dict, config: dict | None = None):
    """Starts (or restarts, if never actually reached an interrupt) a
    curation session's loop. Returns the raw invoke() result — check for
    a "__interrupt__" key to know if it's paused waiting for picks, or
    inspect result["stop_reason"] if it ended.
    """
    graph = build_curation_loop_graph(checkpointer)
    thread_config = {"configurable": {"thread_id": curation_thread_id(session_id), **(config or {})}}
    return graph.invoke(
        {"session": initial_session_dict, "current_batch": [], "stop_reason": None, "should_stop": False},
        config=thread_config,
    )


def resume_curation_turn(
    session_id: str, checkpointer: BaseCheckpointSaver,
    picked_paper_ids: list[str], stop: bool = False, config: dict | None = None,
):
    """Resumes a paused curation session with the user's picks. Returns
    the raw invoke() result, same convention as start_curation_turn()."""
    graph = build_curation_loop_graph(checkpointer)
    thread_config = {"configurable": {"thread_id": curation_thread_id(session_id), **(config or {})}}
    return graph.invoke(
        Command(resume={"picked_paper_ids": picked_paper_ids, "stop": stop}),
        config=thread_config,
    )


def get_curation_state(session_id: str, checkpointer: BaseCheckpointSaver) -> dict | None:
    """curation-api-and-ui Phase 6a: the read path the GET session-state
    endpoint needs, and the one curation_session.py's own
    load_curation_session() genuinely cannot provide -- a batch that's
    currently pending (presented, not yet picked) lives in the
    interrupt's own payload, not in PaperPoolSession's serialized
    fields, so recovering it after e.g. a page refresh requires this
    graph's own get_state(), not the smaller sync-only graph
    curation_session.py compiles.

    Verified directly (not assumed) that this graph's get_state() stays
    correct to read from even after curation_chat.py/report.py's own
    save_curation_session() writes onto the same thread_id via that
    smaller graph in between (see scratch verification during Phase 6a
    design) -- the two graphs' checkpoints interoperate safely because
    LangGraph checkpoints are keyed by thread_id, not by which compiled
    graph object wrote them.

    Returns None if session_id was never started. Otherwise
    {"session": PaperPoolSession, "pending_batch": [[paper_dict, score],
    ...] | None, "refilled": bool} -- pending_batch is None once
    curation has finished (session.stage == "synthesize") or hasn't
    started serving yet. refilled (Phase 6c) reflects whether the turn
    that produced the CURRENT pending_batch triggered a fresh search --
    reliably present here too, not just on start/resume's own raw
    result, since only curation_loop.py's own graph ever writes to a
    thread_id while curation is still active (chat/report don't touch a
    session until stage == "synthesize", past the point pending_batch
    exists at all).
    """
    graph = build_curation_loop_graph(checkpointer)
    config = {"configurable": {"thread_id": curation_thread_id(session_id)}}
    snap = graph.get_state(config)
    if not snap.values:
        return None

    session = _dict_to_session(snap.values["session"])
    pending_batch = None
    if snap.next and snap.tasks and snap.tasks[0].interrupts:
        pending_batch = snap.tasks[0].interrupts[0].value["batch"]
    return {"session": session, "pending_batch": pending_batch, "refilled": snap.values.get("refilled", False)}
