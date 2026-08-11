from __future__ import annotations

import uuid

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import CurationPicksRequest, CurationStartRequest, CurationTurnResponse
from research_agent.api_app.serializers import _turn_result_to_response
from research_agent.config import get_settings
from research_agent.curation_loop import get_curation_state, resume_curation_turn, start_curation_turn
from research_agent.curation_session import _session_to_dict
from research_agent.query_expansion import PaperPoolSession
from research_agent.services.curation_helpers import _curation_config
from research_agent.services.errors import ServiceError
from research_agent.usage_guard import guard_paid_action


def start_curation(req: CurationStartRequest, cp) -> CurationTurnResponse:
    # session_id (attached below via telemetry.set_action_subject) doesn't
    # exist until it's minted a few lines down -- the action has to open
    # before that, since it needs to wrap the candidate-pool/ranking/
    # canonicalization work that happens first. Usage Protection M2.2A:
    # same reasoning as search_service.run_search -- no subject yet, so
    # only the coarse global admission check applies, no lease.
    with guard_paid_action("curation_start"):
        client = api._state["client"]
        settings = get_settings()
        deduped = api.build_candidate_pool(
            req.topic, req.target_count, s2_api_key=settings.semantic_scholar_api_key, client=client,
            use_openalex_fallback=req.use_openalex_fallback, openalex_mailto=settings.openalex_mailto,
        )
        ranked, _ = api.rank_full_pool(req.topic, deduped, client=client, collection=api._state["collection"])
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
        session = PaperPoolSession(topic=req.topic, display_title=display_title, reserve=ranked, target_count=req.target_count)
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
