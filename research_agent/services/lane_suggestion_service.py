"""Research Lanes (RL2): service layer for POST /curation/lanes/suggest.

Owns the ONE thing the router must not: orchestration -- feature-flag
gate, usage-guard admission, the single provider call, and mapping a
malformed provider result to the repository's existing safe error
response. The router stays a thin request/response shell (same split as
every other services/*.py module).

Order of operations, deliberately:
  1. Feature flag. RESEARCH_LANES_ENABLED is False by default -- when off,
     this raises ServiceError(403) BEFORE guard_paid_action opens, so a
     disabled deployment does zero admission, zero telemetry, zero
     provider work. (Unauthorized requests never even reach here -- the
     Basic Auth middleware is the outermost layer.)
  2. guard_paid_action("curation_lane_suggest"). No subject (no session
     exists yet -- same as /search and /curation/start) => coarse global
     admission check only, NO session lease. Admission runs BEFORE the
     provider call; an admission rejection propagates as UsageGuardRejection
     to the centralized handler unchanged.
  3. api.suggest_lanes(topic) -- exactly one gpt-4.1-mini structured call,
     recorded as a child call on this one top-level paid action.
  4. LaneSuggestionError (malformed structured output) -> ServiceError(503,
     {"error": "..."}) -- byte-identical shape to what _upstream_error_guard
     produces for a genuine OpenAIError, and no raw provider text. A real
     OpenAIError is NOT caught here: it propagates to the router's
     _upstream_error_guard, same as /search and /curation/start.

Both failure paths still leave one paid_action row with outcome="error"
(the guard's own `except Exception` branch) -- the provider call happened
and was billable.
"""

from __future__ import annotations

import research_agent.api as api
from research_agent.api_app.schemas import LaneSuggestRequest, LaneSuggestResponse
from research_agent.config import get_settings
from research_agent.lane_suggestion import LaneSuggestionError
from research_agent.services.errors import ServiceError
from research_agent.usage_guard import guard_paid_action

# HTTP 403: the endpoint exists and the caller is authenticated, but this
# deployment has the feature switched off -- a refusal to act, not a
# missing resource (404) or a bad request (400). Chosen because there is
# no prior feature-disabled endpoint in this codebase and 403 is the
# closest fit to "understood, authenticated, but not permitted here",
# while staying clearly distinct from the auth middleware's own 401.
_FEATURE_DISABLED_STATUS = 403
_FEATURE_DISABLED_MESSAGE = "Research lanes are not enabled on this deployment."

# Same safe 503 shape _upstream_error_guard emits ({"error": "<x> service
# unavailable"}) -- so a malformed structured result and a raw provider
# exception look identical to the client, and neither exposes provider text.
_MALFORMED_STATUS = 503
_MALFORMED_DETAIL = {"error": "curation_lane_suggest service unavailable"}


def suggest_lanes_for_topic(req: LaneSuggestRequest) -> LaneSuggestResponse:
    if not get_settings().research_lanes_enabled:
        raise ServiceError(_FEATURE_DISABLED_STATUS, _FEATURE_DISABLED_MESSAGE)

    client = api._state["client"]
    try:
        with guard_paid_action("curation_lane_suggest"):
            lanes = api.suggest_lanes(req.topic, client=client)
    except LaneSuggestionError as exc:
        # No `from exc` chaining into the HTTP detail -- ServiceError's
        # detail is a fixed, safe dict; exc's (already content-free) text
        # never reaches the response.
        raise ServiceError(_MALFORMED_STATUS, _MALFORMED_DETAIL) from exc

    return LaneSuggestResponse(lanes=[api._research_lane_to_out(lane) for lane in lanes])
