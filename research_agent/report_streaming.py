"""Usage Protection M4.3A: report-generation/regeneration progress
streaming event vocabulary.

Mirrors research_agent/chat_streaming.py's own Part A shape exactly (the
same small `*Event` dataclass wrapping `research_agent/sse.py::format_
sse_event`) but is entirely INDEPENDENT of it -- a fresh, closed report-
only vocabulary, not an extension of M4.1/M4.2's chat phase set. M4.2 is
closed and this module does not import from, or modify, chat_streaming.py
or curation_chat_streaming.py at all.

**Frozen event types, matching M4.1's own SSE protocol exactly:**
`started`, `phase`, `completed`, `error`, `done`. Deliberately no
`delta` -- report generation/regeneration never streams prose, partial
sections, or any other incremental text; there is no proven non-prose
use for a report `delta` event, so none exists.

**Frozen phase vocabulary, each mapped to one real, observable execution
boundary in research_agent/curation_report_streaming.py** -- never
emitted speculatively, never a fake/invented step:
  - "generating": the one fresh generate/regenerate call
    (generate_report_for_session / regenerate_report_with_new_sources).
  - "evaluating": refine_report_if_requested's own evaluate_report call
    -- only when refinement_mode="single".
  - "revising": refine_report_if_requested's own revise_report call --
    only when refinement_mode="single" AND a revision is genuinely
    needed.
  - "saving": append_report_version + save_curation_session, the one
    persistence commit point.
No "finalizing" phase: reference/section finalization
(_build_references_and_renumber and friends) happens synchronously
INSIDE generate_report/_regenerate_report_sections_with_sources/
revise_report themselves -- it is not a separately orchestrable step,
so there is no real boundary to report progress on between "the last
content-producing call returns" and "persistence begins."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from research_agent.sse import format_sse_event

ReportStreamEventType = Literal["started", "phase", "completed", "error", "done"]

ReportStreamPhase = Literal["generating", "evaluating", "revising", "saving"]


@dataclass
class ReportStreamEvent:
    """One event in the report-streaming protocol. `data` must already
    be the complete, bounded, approved payload for `type` -- see each
    `build_*_event` helper below for the one-and-only place each event
    type's payload shape is decided."""

    type: ReportStreamEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        return format_sse_event(self.type, self.data)


def build_started_event() -> ReportStreamEvent:
    return ReportStreamEvent(type="started", data={})


def build_phase_event(phase: ReportStreamPhase) -> ReportStreamEvent:
    return ReportStreamEvent(type="phase", data={"phase": phase})


def build_completed_event(report_out_json: dict[str, Any]) -> ReportStreamEvent:
    """`report_out_json` must already be the JSON-compatible dump of a
    real `api_app.schemas.ReportOut` instance (`.model_dump(mode="json")`)
    -- built via the EXISTING serializer (`api_app.serializers._report_
    to_out`), never manually reconstructed field-by-field here. This
    module deliberately stays free of any `api_app` import (matching
    research_agent's own domain/presentation layering) -- the caller
    (research_agent/curation_report_streaming.py) is responsible for
    calling the real serializer and dumping it before reaching this
    function."""
    return ReportStreamEvent(type="completed", data=report_out_json)


def build_error_event(reason_code: str, message: str) -> ReportStreamEvent:
    """`message` must always be a small, fixed, safe string -- never
    `str(exc)`, never a provider's own error body. Matches `research_
    agent.curation_chat_streaming.HandledStreamFailure`'s own contract."""
    return ReportStreamEvent(type="error", data={"reason_code": reason_code, "message": message})


def build_done_event() -> ReportStreamEvent:
    return ReportStreamEvent(type="done", data={})
