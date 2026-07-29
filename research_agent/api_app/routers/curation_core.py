import os
import uuid

from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.curation_loop import get_curation_state, resume_curation_turn, start_curation_turn
from research_agent.curation_session import _session_to_dict
from research_agent.query_expansion import PaperPoolSession

router = APIRouter()


@router.post("/curation/start", response_model=api.CurationTurnResponse)
def curation_start(req: api.CurationStartRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    with api._upstream_error_guard("curation_start"):
        client = api._state["client"]
        s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
        deduped = api.build_candidate_pool(
            req.topic, req.target_count, s2_api_key=s2_key, client=client,
            use_openalex_fallback=req.use_openalex_fallback, openalex_mailto=os.getenv("OPENALEX_MAILTO") or None,
        )
        ranked, _ = api.rank_full_pool(req.topic, deduped, client=client, collection=api._state["collection"])
        if not ranked:
            raise HTTPException(status_code=404, detail="No papers found for this topic.")

        # curation-review-management Phase 8, item 5: after confirming
        # papers actually exist for this topic (no point spending an LLM
        # call otherwise), produce a clean display title. Never touches
        # req.topic itself -- that's still what search/ranking/refinement
        # keep using for the rest of this session's life.
        display_title = api.canonicalize_topic(req.topic, client=client)

        session_id = uuid.uuid4().hex
        session = PaperPoolSession(topic=req.topic, display_title=display_title, reserve=ranked, target_count=req.target_count)
        result = start_curation_turn(session_id, cp, _session_to_dict(session), config=api._curation_config())
        return api._turn_result_to_response(session_id, req.target_count, result)


@router.post("/curation/{session_id}/picks", response_model=api.CurationTurnResponse)
def curation_picks(session_id: str, req: api.CurationPicksRequest, cp=Depends(api.get_curation_checkpointer)) -> api.CurationTurnResponse:
    with api._upstream_error_guard("curation_picks"):
        state = get_curation_state(session_id, cp)
        if state is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        if state["pending_batch"] is None:
            raise HTTPException(status_code=400, detail="Session is not awaiting picks (curation already finished).")

        target_count = state["session"].target_count
        result = resume_curation_turn(
            session_id, cp, picked_paper_ids=req.picked_paper_ids, stop=req.stop,
            refinement=req.refinement, request_refill=req.request_refill, config=api._curation_config(),
        )
        return api._turn_result_to_response(session_id, target_count, result)
