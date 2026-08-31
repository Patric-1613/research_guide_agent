"""Research Lanes (RL2): POST /curation/lanes/suggest.

Thin shell -- identical shape to routers/curation_core.py. All
orchestration (feature-flag gate, usage guard, the provider call, safe
error mapping) lives in services/lane_suggestion_service.py. This router
only wraps the body in _upstream_error_guard, so a raw OpenAIError
becomes a clean 503 rather than a leaked stack trace; a ServiceError
raised by the service is mapped to its HTTP response by the centralized
handler in api_app/app.py.

The Basic Auth middleware (outermost) already protects this route -- an
unauthorized request is rejected before FastAPI resolves this handler at
all; there is no route-specific auth here.
"""

from fastapi import APIRouter

from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import (
    CurationCapabilitiesResponse,
    LaneSuggestRequest,
    LaneSuggestResponse,
)
from research_agent.config import get_settings
from research_agent.services.lane_suggestion_service import suggest_lanes_for_topic

router = APIRouter()


@router.get("/curation/capabilities", response_model=CurationCapabilitiesResponse)
def curation_capabilities() -> CurationCapabilitiesResponse:
    """Research Lanes (RL5): the smallest protected, zero-provider capability
    probe. No admission, telemetry, DB, or provider work -- just a strict
    read of the RESEARCH_LANES_ENABLED flag (uncached get_settings(), same
    per-request read the suggestion service uses). Protected automatically
    by the outermost Basic Auth middleware, like every route except
    GET /health. Returns ONLY {research_lanes_enabled: bool}."""
    return CurationCapabilitiesResponse(research_lanes_enabled=get_settings().research_lanes_enabled)


@router.post("/curation/lanes/suggest", response_model=LaneSuggestResponse)
def curation_lanes_suggest(req: LaneSuggestRequest) -> LaneSuggestResponse:
    with _upstream_error_guard("curation_lane_suggest"):
        return suggest_lanes_for_topic(req)
