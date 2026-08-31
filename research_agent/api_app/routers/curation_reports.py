from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse, Response

import research_agent.api as api
from research_agent.api_app.errors import _upstream_error_guard
from research_agent.api_app.schemas import CurationGenerateReportRequest, CurationRegenerateReportRequest, ReportOut
from research_agent.services.curation_report_service import (
    activate_report_version,
    export_active_report,
    get_or_create_report,
    regenerate_report,
    stream_generate_report,
    stream_regenerate_report,
)

router = APIRouter()

# report-quality Phase R5C.1/R5C.2: DOCX's media type has no short/
# plain form the way "text/markdown" does -- this is the real,
# registered OOXML WordprocessingML content type. markdown stays on
# PlainTextResponse (str content, charset appended automatically);
# docx/pdf use the base Response class below with raw bytes and no
# charset, since both are binary.
_DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_MEDIA_TYPE = "application/pdf"


@router.post("/curation/{session_id}/report", response_model=ReportOut)
def curation_report(
    session_id: str, req: CurationGenerateReportRequest = CurationGenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report"):
        return get_or_create_report(
            session_id, cp, report_template=req.report_template, refinement_mode=req.refinement_mode,
        )


@router.post("/curation/{session_id}/report/regenerate", response_model=ReportOut)
def curation_report_regenerate(
    session_id: str, req: CurationRegenerateReportRequest = CurationRegenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    with _upstream_error_guard("curation_report_regenerate"):
        return regenerate_report(
            session_id, cp, report_template=req.report_template, refinement_mode=req.refinement_mode,
        )


# Usage Protection M4.3A: the streaming counterparts -- both existing
# endpoints above are completely unchanged, still the default for any
# client that hasn't adopted streaming yet. No _upstream_error_guard on
# either: like M4.2A's own chat-stream route, nothing either of these
# routes does BEFORE returning the StreamingResponse ever calls an LLM
# synchronously (every provider call happens later, inside the streamed
# generator body, which reports its own safe error events instead -- see
# curation_report_streaming.py's own docstring).
@router.post("/curation/{session_id}/report/stream")
def curation_report_stream(
    session_id: str, req: CurationGenerateReportRequest = CurationGenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
):
    return stream_generate_report(session_id, req, cp)


@router.post("/curation/{session_id}/report/regenerate/stream")
def curation_report_regenerate_stream(
    session_id: str, req: CurationRegenerateReportRequest = CurationRegenerateReportRequest(),
    cp=Depends(api.get_curation_checkpointer),
):
    return stream_regenerate_report(session_id, req, cp)


@router.get("/curation/{session_id}/report/export")
def curation_report_export(
    session_id: str, format: str = "markdown", cp=Depends(api.get_curation_checkpointer),
) -> Response:
    """report-quality Phase R5A/R5C.1/R5C.2: exports the session's
    ACTIVE report version -- never always-the-latest, same "session.
    report always mirrors the active version" invariant every other
    report read already relies on -- as a downloadable document.

    `format` defaults to "markdown" -- the deliberate, lower-risk
    choice over requiring it explicitly: a client hitting this endpoint
    with no query string at all still gets a sensible result, rather
    than an error for omitting a parameter with a sensible default.
    "docx" and "pdf" are the other supported values as of R5C.2. Any
    other value (a typo) still 400s -- see export_active_report's own
    docstring for why that check runs before any session lookup at all.

    No response_model/response_class here -- returning a real Response
    instance directly is what lets this set a custom media_type per
    format and a Content-Disposition header for the download filename;
    FastAPI uses a returned Response object as-is. markdown keeps using
    PlainTextResponse (str content, "; charset=utf-8" appended
    automatically); docx/pdf are binary, so they use the base Response
    class with raw bytes and no charset.
    """
    with _upstream_error_guard("curation_report_export"):
        content, filename = export_active_report(session_id, cp, format)
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        if format == "docx":
            return Response(content=content, media_type=_DOCX_MEDIA_TYPE, headers=headers)
        if format == "pdf":
            return Response(content=content, media_type=_PDF_MEDIA_TYPE, headers=headers)
        return PlainTextResponse(content, media_type="text/markdown", headers=headers)


@router.post("/curation/{session_id}/reports/{version_id}/activate", response_model=ReportOut)
def curation_report_activate_version(
    session_id: str, version_id: str, cp=Depends(api.get_curation_checkpointer),
) -> ReportOut:
    """report-quality Phase R3: switches which report version is
    active/current for this session -- a pure pointer switch, never a
    regeneration, never a mutation of any version's own content. An
    unknown version_id (or one from a different session) 404s, same as
    an unknown session_id -- both mean "this URL doesn't point at
    anything real," not a client error worth a 400."""
    with _upstream_error_guard("curation_report_activate_version"):
        return activate_report_version(session_id, version_id, cp)
