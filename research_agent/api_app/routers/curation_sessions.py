from fastapi import APIRouter, Depends, HTTPException

import research_agent.api as api
from research_agent.api_app.access import (
    AccessContext,
    ownership_repo,
    require_approved_access,
    require_curation_session_access,
)
from research_agent.api_app.schemas import CurationDeleteResponse, CurationReviewSummary, CurationStateResponse
from research_agent.curation_ownership import delete_owned_curation_session
from research_agent.services.curation_session_service import delete_session, get_state, list_reviews

router = APIRouter()


# Registered BEFORE GET /curation/{session_id} below — Starlette matches
# routes in registration order, so /curation/reviews must come first or a
# request for it would match {session_id}="reviews" instead of this route.
@router.get("/curation/reviews", response_model=list[CurationReviewSummary])
def curation_list_reviews(
    access: AccessContext = Depends(require_approved_access),
    cp=Depends(api.get_curation_checkpointer),
) -> list[CurationReviewSummary]:
    if access.is_firebase:
        with ownership_repo() as repo:
            owned = {row.session_id for row in repo.list_owner_sessions(access.owner_id, limit=500)}
        return list_reviews(cp, owned_session_ids=owned)
    return list_reviews(cp)


@router.get(
    "/curation/{session_id}",
    response_model=CurationStateResponse,
    dependencies=[Depends(require_curation_session_access)],
)
def curation_get_state(session_id: str, cp=Depends(api.get_curation_checkpointer)) -> CurationStateResponse:
    result = get_state(session_id, cp)
    if result is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    return result


@router.delete("/curation/{session_id}", response_model=CurationDeleteResponse)
def curation_delete(
    session_id: str,
    access: AccessContext = Depends(require_curation_session_access),
    cp=Depends(api.get_curation_checkpointer),
) -> CurationDeleteResponse:
    if access.is_firebase:
        # Ownership was confirmed by the dependency. Delete follows the
        # coordinator's order: curation_owners row first (the session is
        # unreachable the instant that succeeds), then the checkpoint.
        with ownership_repo() as repo:
            existed = delete_owned_curation_session(
                session_id=session_id, ownership_repo=repo, checkpointer=cp,
            )
        if not existed:
            raise HTTPException(status_code=404, detail="session_id not found")
        return CurationDeleteResponse(session_id=session_id, deleted=True)
    result = delete_session(session_id, cp)
    if result is None:
        raise HTTPException(status_code=404, detail="session_id not found")
    return result
