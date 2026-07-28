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
                                                     serve_batch always
                                                     routes to "present"
                                                     -- even an EMPTY batch
                                                     (curation-editable-
                                                     until-locked Phase 10b:
                                                     there is no more
                                                     "exhausted" stop --
                                                     see below)
                                                           v
                                                present_and_apply  <-- interrupt() lives here, ALONE
                                                           |
                                                           v
                                                  route_after_picks
                                                     +-"user_stopped"--> stop_user_requested (END)
                                                     +-"continue"------> check_pool (loop back)

    curation-editable-until-locked Phase 10b: reaching target_count or a
    truly exhausted search no longer ends the graph. Both used to route
    to their own stop_*/END node (stop_exhausted, stop_target_met),
    which forced session.stage="synthesize" the moment either happened
    -- locking the review out of further curation even if the user
    never asked to stop. Per explicit user decision, editability is now
    gated ONLY on whether chat/report has started, never on
    target_count or pool exhaustion: an empty batch or a target-reached
    batch is just presented to the user (via the SAME present_and_apply
    interrupt, with current_batch possibly empty) with a message,
    letting them refine/search again or explicitly stop. The ONLY way
    to reach stage="synthesize" now is the explicit stop=True resume
    payload -- stop_user_requested is the one remaining stop node.

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
    # curation-refinement-and-auto-offer Phase 6f: unlike refilled above,
    # deliberately NOT reset by _check_pool_node -- present_and_apply sets
    # this to an explicit True/False on EVERY resume (never stale/carried
    # over), and check_pool runs strictly BETWEEN present_and_apply's
    # write and _route_entry's read in the same invoke (the "continue"
    # loop-back), so resetting it there would erase the signal before
    # _route_entry ever sees it -- confirmed by tracing the actual node
    # order, not assumed from refilled's superficially similar shape.
    # Only the stop nodes reset it (for the same "never revisited"
    # reason as refilled), since nothing reads it once curation ends.
    force_refill: bool


def _route_entry(state: CurationLoopState) -> str:
    session = _dict_to_session(state["session"])
    # curation-turn-history Phase 9d: auto-refill ONLY at true exhaustion
    # (remaining()==0), not whenever remaining()<BATCH_SIZE (this used
    # needs_refill() before -- that method itself is unchanged and still
    # correctly means "< batch_size" for whatever else might use it, this
    # routing decision just no longer uses it). A partial batch (e.g. 6-7)
    # now serves as-is instead of forcing a fresh, possibly-disappointing
    # search the user never asked for -- request_refill (below, via
    # force_refill) is the explicit, user-controlled way to top up before
    # that point. refinement's own force_refill trigger is unaffected --
    # still forces a refill on demand regardless of remaining count.
    if session.remaining() == 0 or state.get("force_refill"):
        return "refill"
    return "serve"


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
    # curation-turn-history Phase 9b: this node has no interrupt() inside
    # it, so (unlike present_and_apply below) it only ever runs ONCE per
    # real turn -- never re-executed on resume -- safe to append here
    # without double-recording. An empty batch isn't a real turn a user
    # could browse back to (no papers were shown), so it's not logged --
    # still true under Phase 10b's "present even if empty" routing.
    if serialized_batch:
        session.turn_history.append({
            "turn_number": len(session.turn_history) + 1,
            "batch": serialized_batch,
            "refilled": state.get("refilled", False),
        })
    return {"session": _session_to_dict(session), "current_batch": serialized_batch}


def _route_after_serve(state: CurationLoopState) -> str:
    # curation-editable-until-locked Phase 10b: always present, even an
    # empty batch -- there is no more "exhausted" stop (see module
    # docstring). A degenerate empty-batch interrupt lets the frontend
    # show a clear "no new candidates" message while leaving the user in
    # full control (refine and search again, or explicitly stop).
    return "present"


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
    refinement = resume.get("refinement")
    # curation-turn-history Phase 9d: the explicit, user-controlled way
    # to top up the pool on demand -- reuses the SAME force_refill
    # mechanism refinement already triggers, not a second one.
    request_refill = bool(resume.get("request_refill", False))

    session = _dict_to_session(state["session"])

    # Validate against every paper EVER served this session (session.
    # turn_history), not just this turn's current_batch -- this is what
    # makes it safe to pick a paper from an earlier turn while
    # stage=="curate" (a real interrupt is pending here; the OTHER way to
    # pick from history, select_paper_from_history(), is only safe once
    # stage=="synthesize" -- see its own docstring for exactly why an
    # out-of-band write would corrupt this pending interrupt's
    # bookkeeping). turn_history already includes THIS turn's own batch
    # by the time this node runs (_serve_batch_node appends to it first,
    # same turn, before present_and_apply ever starts), so nothing from
    # current_batch is missed by validating against turn_history alone.
    # Never trust the resume payload blindly either way: anything not in
    # ANY served batch is silently dropped, not applied and not an error.
    presented_by_id: dict[str, dict] = {}
    for entry in session.turn_history:
        for paper_dict, _ in entry["batch"]:
            presented_by_id[paper_dict["paper_id"]] = paper_dict
    valid_picks = [pid for pid in picked_paper_ids if pid in presented_by_id]

    for pid in valid_picks:
        if pid not in session.selected_paper_ids:
            # Both lists updated together, in the same node, from the
            # same turn_history data -- kept in sync by construction, and
            # verified explicitly (not just assumed) by
            # tests/test_curation_loop.py's sync-invariant test across a
            # session that includes a refill.
            session.selected_paper_ids.append(pid)
            session.selected_papers.append(Paper(**presented_by_id[pid]))

    force_refill = False
    if refinement:
        session.refinement_notes.append(refinement)
        force_refill = True
    if request_refill:
        force_refill = True

    return {"session": _session_to_dict(session), "should_stop": stop, "force_refill": force_refill}


def _route_after_picks(state: CurationLoopState) -> str:
    # curation-editable-until-locked Phase 10b: reaching target_count no
    # longer forces a stop (see module docstring) -- the ONLY way to
    # reach stop_user_requested is an explicit stop=True resume payload.
    # Whether the target's been hit is still fully derivable by the
    # caller from session.selected_paper_ids vs. session.target_count;
    # it just no longer gates anything here.
    if state.get("should_stop"):
        return "user_stopped"
    return "continue"


def _make_stop_node(reason: str):
    def _stop_node(state: CurationLoopState) -> dict:
        session = _dict_to_session(state["session"])
        session.stage = "synthesize"
        # curation-turn-history Phase 9b: persisted so a reload/reopen can
        # still show WHY curation stopped -- previously this only ever
        # existed in the one HTTP response of the turn that caused it.
        # Set exactly once, same one-way semantics as `stage` itself.
        session.stop_reason = reason
        # refilled/force_refill must not be left carrying a stale value
        # from whichever turn last passed through check_pool -- a stop
        # reached via present_and_apply's "target_met"/"user_stopped"
        # routes straight here WITHOUT re-visiting check_pool
        # (interrupt-resume restarts execution AT present_and_apply, not
        # from check_pool; confirmed by hitting exactly this stale-True
        # case in testing), and there is no current batch left for
        # either flag to meaningfully describe once curation has stopped.
        return {
            "session": _session_to_dict(session), "stop_reason": reason,
            "refilled": False, "force_refill": False,
        }
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
    graph.add_node("stop_user_requested", _make_stop_node("user_stopped"))

    graph.add_edge(START, "check_pool")
    graph.add_conditional_edges("check_pool", _route_entry, {"refill": "refill", "serve": "serve_batch"})
    graph.add_edge("refill", "serve_batch")
    graph.add_conditional_edges("serve_batch", _route_after_serve, {
        "present": "present_and_apply",
    })
    graph.add_conditional_edges("present_and_apply", _route_after_picks, {
        "user_stopped": "stop_user_requested",
        "continue": "check_pool",
    })
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
    picked_paper_ids: list[str], stop: bool = False, refinement: str | None = None,
    request_refill: bool = False, config: dict | None = None,
):
    """Resumes a paused curation session with the user's picks. Returns
    the raw invoke() result, same convention as start_curation_turn().

    refinement (curation-refinement-and-auto-offer Phase 6f): optional
    free-text steering, carried in the SAME resume payload
    picked_paper_ids/stop already use -- confirmed necessary (not just
    convenient) during Phase 6f-1's design: mutating a mid-curation
    session out-of-band via curation_session.py's smaller graph while a
    real interrupt is pending corrupts that pending task's bookkeeping,
    so refinement has to flow through this exact channel, not a
    separate one.

    request_refill (curation-turn-history Phase 9d): the explicit,
    user-controlled "search for more now" action -- same force_refill
    mechanism refinement already triggers, same reason it has to ride in
    this exact payload rather than a separate call. picked_paper_ids may
    also reference a paper from an EARLIER turn (not just the batch this
    interrupt presented) -- see _present_and_apply_node's own docstring.
    """
    graph = build_curation_loop_graph(checkpointer)
    thread_config = {"configurable": {"thread_id": curation_thread_id(session_id), **(config or {})}}
    return graph.invoke(
        Command(resume={
            "picked_paper_ids": picked_paper_ids, "stop": stop,
            "refinement": refinement, "request_refill": request_refill,
        }),
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
