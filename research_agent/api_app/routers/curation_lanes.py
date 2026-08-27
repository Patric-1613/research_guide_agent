"""Research Lanes (RL2): POST /curation/lanes/suggest.

Thin shell -- identical shape to routers/curation_core.py. All
orchestration (feature-flag gate, usage guard, the provider call, safe
error mapping) lives in services/lane_suggestion_service.py. This router
only: wraps the body in _upstream_error_guard (so a raw OpenAIError
becomes a clean 503, never a leaked stack trace) and converts
ServiceError -> HTTPException with status/detail preserved exactly.

The Basic Auth middleware (outermost) already protects this route -- an
unauthorized request is rejected before FastAPI resolves this handler at
all; there is no route-specific auth here.
"""

from fastapi import APIRouter, HTTPException

from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import LaneSuggestRequest, LaneSuggestResponse
from research_agent.services.errors import ServiceError
from research_agent.services.lane_suggestion_service import suggest_lanes_for_topic

router = APIRouter()


@router.post("/curation/lanes/suggest", response_model=LaneSuggestResponse)
def curation_lanes_suggest(req: LaneSuggestRequest) -> LaneSuggestResponse:
    with _upstream_error_guard("curation_lane_suggest"):
        try:
            return suggest_lanes_for_topic(req)
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
