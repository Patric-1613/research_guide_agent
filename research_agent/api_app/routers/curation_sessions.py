from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.services.curation_session_service import delete_session, get_state, list_reviews

router = APIRouter()


# Registered BEFORE GET /curation/{session_id} below — Starlette matches
# routes in registration order, so /curation/reviews must come first or a
# request for it would match {session_id}="reviews" instead of this route.
@router.get("/curation/reviews", response_model=list[api.CurationReviewSummary])
def curation_list_reviews(cp=Depends(api.get_curation_checkpointer)) -> list[api.CurationReviewSummary]:
    return list_reviews(cp)


@router.get("/curation/{session_id}", response_model=api.CurationStateResponse)
def curation_get_state(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationStateResponse:
    result = get_state(session_id, cp)
    if result is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    return result


@router.delete("/curation/{session_id}", response_model=api.CurationDeleteResponse)
def curation_delete(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> api.CurationDeleteResponse:
    result = delete_session(session_id, cp)
    if result is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    return result
