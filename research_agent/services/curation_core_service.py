from __future__ import annotations

import uuid

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import (
    CurationPicksRequest,
    CurationStartRequest,
    CurationTurnResponse,
    SubmittedLane,
)
from research_agent.api_app.serializers import _turn_result_to_response
from research_agent.config import get_settings
from research_agent.curation_loop import get_curation_state, resume_curation_turn, start_curation_turn
from research_agent.curation_session import _session_to_dict
from research_agent.query_expansion import PaperPoolSession
from research_agent.research_lanes import ResearchLane, new_lane_id, validate_lane_list_for_construction
from research_agent.services.curation_helpers import _curation_config
from research_agent.services.errors import ServiceError
from research_agent.usage_guard import guard_paid_action


def _frozen_lanes_from_request(submitted: list[SubmittedLane]) -> list[ResearchLane]:
    """Research Lanes (RL4): turn the client's editable lane content into
    the FROZEN, server-owned lane objects a session persists. Fresh
    opaque lane_id per lane (never a client value), origin="user" (the
    most truthful existing value -- the user authored/approved this
    content), generation_version=1. Every value is validated through the
    RL1 construction contract (1..4 lanes, >=1 enabled, non-empty
    label/query, bounded lengths, opaque non-label lane_id). Raises
    ServiceError(400) on any invalid input -- called BEFORE guard_paid_
    action opens, so a rejection does no admission / telemetry / provider
    / embedding / persistence work."""
    if not submitted:
        raise ServiceError(400, "Lane mode requires at least one submitted research lane.")
    try:
        lanes = [
            ResearchLane(
                lane_id=new_lane_id(),
                label=sl.label, question=sl.question, query=sl.query,
                enabled=sl.enabled, origin="user", generation_version=1,
            )
            for sl in submitted
        ]
        validate_lane_list_for_construction(lanes)
    except (ValueError, TypeError) as exc:
        raise ServiceError(400, f"Invalid research lane input: {exc}") from exc
    return lanes


def start_curation(req: CurationStartRequest, cp) -> CurationTurnResponse:
    # Research Lanes (RL4): the feature-flag gate and full lane validation
    # run FIRST -- before guard_paid_action opens -- so a disabled feature
    # or an invalid lane set returns a clean 4xx with zero admission /
    # telemetry / provider / embedding / persistence work. The flag gates
    # ONLY creation of a new lane session; a request with no `lanes` is
    # the exact existing single-query path regardless of the flag.
    settings = get_settings()
    frozen_lanes: list[ResearchLane] | None = None
    if req.lanes is not None:
        if not settings.research_lanes_enabled:
            raise ServiceError(403, "Research lanes are not enabled on this deployment.")
        frozen_lanes = _frozen_lanes_from_request(req.lanes)

    # session_id (attached below via telemetry.set_action_subject) doesn't
    # exist until it's minted a few lines down -- the action has to open
    # before that, since it needs to wrap the candidate-pool/ranking/
    # canonicalization work that happens first. Usage Protection M2.2A:
    # same reasoning as search_service.run_search -- no subject yet, so
    # only the coarse global admission check applies, no lease. Lane mode
    # keeps this exact ownership: retrieve_across_lanes runs INSIDE this
    # one guard, no new action type, no nested guard.
    with guard_paid_action("curation_start"):
        client = api._state["client"]

        if frozen_lanes is not None:
            retrieval = api.retrieve_across_lanes(
                req.topic, frozen_lanes,
                k_for_widening=req.target_count,
                s2_api_key=settings.semantic_scholar_api_key, client=client,
                collection=api._state["collection"],
                use_openalex_fallback=req.use_openalex_fallback, openalex_mailto=settings.openalex_mailto,
            )
            ranked = retrieval.ranked
            lanes_field: list[ResearchLane] = frozen_lanes
            paper_lane_ids = dict(retrieval.paper_lane_ids)
            lane_result_counts = dict(retrieval.lane_result_counts)
        else:
            deduped = api.build_candidate_pool(
                req.topic, req.target_count, s2_api_key=settings.semantic_scholar_api_key, client=client,
                use_openalex_fallback=req.use_openalex_fallback, openalex_mailto=settings.openalex_mailto,
            )
            ranked, _ = api.rank_full_pool(req.topic, deduped, client=client, collection=api._state["collection"])
            lanes_field = []
            paper_lane_ids = {}
            lane_result_counts = {}

        if not ranked:
            raise ServiceError(404, "No papers found for this topic.")

        # curation-review-management Phase 8, item 5: after confirming
        # papers actually exist for this topic (no point spending an LLM
        # call otherwise), produce a clean display title. Never touches
        # req.topic itself -- that's still what search/ranking/refinement
        # keep using for the rest of this session's life.
        display_title = api.canonicalize_topic(req.topic, client=client)

        session_id = uuid.uuid4().hex
        telemetry.set_action_subject("session", session_id)
        session = PaperPoolSession(
            topic=req.topic, display_title=display_title, reserve=ranked, target_count=req.target_count,
            lanes=lanes_field, paper_lane_ids=paper_lane_ids, lane_result_counts=lane_result_counts,
        )
        result = start_curation_turn(session_id, cp, _session_to_dict(session), config=_curation_config())
    return _turn_result_to_response(session_id, req.target_count, result)


def submit_picks(session_id: str, req: CurationPicksRequest, cp) -> CurationTurnResponse:
    state = get_curation_state(session_id, cp)
    if state is None:
        raise ServiceError(404, "session_id not found")
    if state["pending_batch"] is None:
        raise ServiceError(400, "Session is not awaiting picks (curation already finished).")

    target_count = state["session"].target_count
    # Usage Protection M2.2B: curation_refill only actually happens when
    # the pool needs replenishing (session.remaining()==0) or the caller
    # explicitly requested one (request_refill/refinement) -- that
    # decision is made INSIDE the graph (curation_loop.py's own
    # _route_entry), not predictable here without duplicating it, so the
    # guard now lives in curation_loop.py's _refill_node itself (the
    # exact point paid work becomes certain) rather than wrapping this
    # whole call unconditionally the way the old discard_if_empty=True
    # paid_action did.
    result = resume_curation_turn(
        session_id, cp, picked_paper_ids=req.picked_paper_ids, stop=req.stop,
        refinement=req.refinement, request_refill=req.request_refill, config=_curation_config(),
    )
    return _turn_result_to_response(session_id, target_count, result)
