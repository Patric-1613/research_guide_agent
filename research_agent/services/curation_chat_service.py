from __future__ import annotations

import research_agent.api as api
import research_agent.telemetry as telemetry
from research_agent.api_app.schemas import (
    ChatTurn,
    CitedPaperOut,
    CitedWebArticleOut,
    CurationChatAddToReportRequest,
    CurationChatAddToReportResponse,
    CurationChatDeleteRequest,
    CurationChatDeleteResponse,
    CurationChatEditRequest,
    CurationChatEditResponse,
    CurationChatRequest,
    CurationChatResponse,
)
from research_agent.api_app.serializers import _report_to_out
from research_agent.curation_session import load_curation_session, save_curation_session
from research_agent.services.errors import ServiceError


def answer_curation_chat(session_id: str, req: CurationChatRequest, cp) -> CurationChatResponse:
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    # A single "curation_chat" action covers the whole turn, including any
    # nested second ask (web-offer accept) or report-regeneration
    # (report-update-offer accept) chat_turn() triggers internally -- "first
    # active action wins" means neither ever opens its own top-level row.
    with telemetry.paid_action("curation_chat", subject_type="session", subject_id=session_id):
        try:
            result = api.chat_turn(session, req.message, client=api._state["client"])
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)

    return CurationChatResponse(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
        web_offer_made=result.get("web_offer_made", False),
        web_offer_declined=result.get("web_offer_declined", False),
        web_search_used=result.get("web_search_used", False),
        new_web_articles_found=result.get("new_web_articles_found"),
        report_update_offer_made=result.get("report_update_offer_made", False),
        report_update_declined=result.get("report_update_declined", False),
        report_updated=result.get("report_updated", False),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
    )


def delete_curation_chat_exchanges(session_id: str, req: CurationChatDeleteRequest, cp) -> CurationChatDeleteResponse:
    if not req.exchange_ids:
        raise ServiceError(400, "exchange_ids must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    deleted_exchange_ids, report_possibly_stale = api.delete_chat_exchanges(session, req.exchange_ids)
    save_curation_session(session, session_id, cp)

    return CurationChatDeleteResponse(
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        deleted_exchange_ids=deleted_exchange_ids,
        report_possibly_stale=report_possibly_stale,
    )


def add_curation_chat_exchanges_to_report(
    session_id: str, req: CurationChatAddToReportRequest, cp,
) -> CurationChatAddToReportResponse:
    """curation-chat-add-to-report Phase 4. Mutates session.report_
    approved_web_article_urls and each eligible exchange's
    added_to_report ONLY after regenerate_report_with_approved_web_
    sources succeeds -- a raised exception (ValueError from a precondition,
    or anything else from the LLM call) propagates before either mutation,
    structurally guaranteeing a failed regeneration never marks anything
    approved/added.
    """
    if not req.exchange_ids:
        raise ServiceError(400, "exchange_ids must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")
    if session.report is None:
        raise ServiceError(400, "Generate a report first before adding web sources to it.")

    eligible_ids, skipped_ids = api.select_eligible_exchanges_for_report(session, req.exchange_ids)
    if not eligible_ids:
        raise ServiceError(400, "No eligible web-backed exchanges to add to the report.")

    newly_approved_urls = api.cited_web_article_urls_for_exchanges(session, eligible_ids)
    approved_web_articles = api.resolve_approved_web_articles_for_regeneration(session, newly_approved_urls)

    with telemetry.paid_action("report_regenerate", subject_type="session", subject_id=session_id):
        try:
            new_report = api.regenerate_report_with_approved_web_sources(
                session, approved_web_articles, client=api._state["client"],
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

    # report-quality Phase R3: appended as a new version (generation_
    # reason=api.GENERATION_REASON_CHAT_ADD_TO_REPORT), not assigned to
    # session.report directly -- append_report_version keeps session.
    # report mirrored to it as a side effect, same as every other
    # report-mutation call site.
    api.append_report_version(session, new_report, api.GENERATION_REASON_CHAT_ADD_TO_REPORT)
    api.approve_web_article_urls(session, newly_approved_urls)
    api.mark_exchanges_added_to_report(session, eligible_ids)
    save_curation_session(session, session_id, cp)

    return CurationChatAddToReportResponse(
        report=_report_to_out(session.report, api.get_active_report_version(session)),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        added_exchange_ids=eligible_ids,
        skipped_exchange_ids=skipped_ids,
        source_count=len(newly_approved_urls),
    )


def edit_curation_chat_exchange(session_id: str, req: CurationChatEditRequest, cp) -> CurationChatEditResponse:
    if not req.exchange_id:
        raise ServiceError(400, "exchange_id must not be empty")
    if not req.question or not req.question.strip():
        raise ServiceError(400, "question must not be empty")
    session = load_curation_session(session_id, cp)
    if session is None:
        raise ServiceError(404, "session_id not found")

    with telemetry.paid_action("curation_chat", subject_type="session", subject_id=session_id):
        try:
            result, report_possibly_stale = api.edit_chat_exchange(
                session, req.exchange_id, req.question, client=api._state["client"],
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc
    save_curation_session(session, session_id, cp)

    return CurationChatEditResponse(
        answer=result["answer"], answerable=result["answerable"],
        cited_papers=[CitedPaperOut(paper_id=p.paper_id, title=p.title) for p in result["cited_papers"]],
        cited_web_articles=[CitedWebArticleOut(url=a.url, title=a.title) for a in result["cited_web_articles"]],
        web_offer_made=result.get("web_offer_made", False),
        web_offer_declined=result.get("web_offer_declined", False),
        web_search_used=result.get("web_search_used", False),
        new_web_articles_found=result.get("new_web_articles_found"),
        report_update_offer_made=result.get("report_update_offer_made", False),
        report_update_declined=result.get("report_update_declined", False),
        report_updated=result.get("report_updated", False),
        chat_history=[ChatTurn(**turn) for turn in session.chat_history],
        report_possibly_stale=report_possibly_stale,
    )
