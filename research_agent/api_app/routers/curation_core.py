from fastapi import APIRouter, Depends

import research_agent.api as api
from research_agent.api_app.access import (
    AccessContext,
    record_curation_ownership,
    require_approved_access,
    require_curation_session_access,
)
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationPicksRequest, CurationStartRequest, CurationTurnResponse
from research_agent.curation_ownership import mint_session_id
from research_agent.services.curation_core_service import start_curation, submit_picks

router = APIRouter()


@router.post("/curation/start", response_model=CurationTurnResponse)
def curation_start(
    req: CurationStartRequest,
    access: AccessContext = Depends(require_approved_access),
    cp=Depends(api.get_curation_checkpointer),
) -> CurationTurnResponse:
    if access.is_firebase:
        # Ownership is recorded before the checkpoint is created, the same
        # order curation_ownership's coordinator documents: the session_id
        # is minted and its curation_owners row inserted here, then handed
        # to start_curation so its checkpoint is written under that id. A
        # later failure inside start_curation leaves a "fail-closed
        # incomplete" owner row that the reconciliation script sweeps --
        # it never leaves a reachable session without an owner.
        session_id = mint_session_id()
        record_curation_ownership(session_id, access.owner_id, topic=req.topic, stage="curate")
        with _upstream_error_guard("curation_start"):
            return start_curation(req, cp, session_id=session_id)
    with _upstream_error_guard("curation_start"):
        return start_curation(req, cp)


@router.post(
    "/curation/{session_id}/picks",
    response_model=CurationTurnResponse,
    dependencies=[Depends(require_curation_session_access)],
)
def curation_picks(session_id: str, req: CurationPicksRequest, cp=Depends(api.get_curation_checkpointer)) -> CurationTurnResponse:
    with _upstream_error_guard("curation_picks"):
        return submit_picks(session_id, req, cp)
