"""Request/response Pydantic models for research_agent/api.py's endpoints.

Moved out of api.py (Phase 4) so the models have a real, independent home
that routers/services can import directly — api.py re-exports every name
here so `research_agent.api.<Name>` and `patch.object(api, "<Name>", ...)`
keep working unchanged for anything still reaching them that way.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from research_agent.citations import CitationStyle


class SearchRequest(BaseModel):
    topic: str
    # 3-30: a deliberate, code-enforced bound on how many results a single
    # request can ask for, independent of any particular frontend.
    top_k: int = Field(default=10, ge=3, le=30)
    # Round-2 enhancement 2: surfaces doi/citation_count metadata that's
    # already stored per-paper in Chroma (embeddings.py) — no re-indexing
    # needed. min_citation_count=0 means "no filter" per the brief.
    doi_required: bool = False
    min_citation_count: int = Field(default=0, ge=0)
    # Round-2 enhancement 5: independent of top_k — web articles are a
    # separate, smaller pool, never counted alongside the paper results.
    web_max_results: int = Field(default=4, ge=1, le=10)
    # LLM-assisted query expansion (query_expansion.py): default False so
    # existing behavior is unchanged unless explicitly opted into. When
    # True, bypasses the agent's own search/rerank entirely in favor of
    # expanded_search() — see that function's docstring for why. Web
    # article search is agent-only right now, so it's skipped (empty)
    # whenever this is True; a known, deliberate gap for this phase.
    use_query_expansion: bool = False


class PaperOut(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    url: str | None
    doi: str | None
    citation_count: int | None
    source: str
    source_urls: dict[str, str]
    score: float | None = None


class WebArticleOut(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None
    source_domain: str


class SearchResponse(BaseModel):
    search_id: int
    topic: str
    created_at: str
    papers: list[PaperOut]
    # Round-2 enhancement 5: a genuinely separate section from `papers` —
    # never interleaved with it, never counted toward top_k.
    web_articles: list[WebArticleOut] = []


class SummarizeRequest(BaseModel):
    search_id: int
    style: CitationStyle = "apa"


class PaperSummaryOut(BaseModel):
    paper_id: str
    title: str
    summary: str
    apa_citation: str
    harvard_citation: str
    bibtex: str
    citation: str  # whichever of the above matches the requested style


class ThemeOut(BaseModel):
    theme_name: str
    papers: list[PaperSummaryOut]


class WebSummaryOut(BaseModel):
    synthesis: str
    cited_articles: list[WebArticleOut]


class SummarizeResponse(BaseModel):
    search_id: int
    topic: str
    style: CitationStyle
    themes: list[ThemeOut]
    gaps_and_disagreements: str
    skipped_paper_ids: list[str]
    # Round-2 enhancement 5: its own block, never merged into the
    # paper-themes summary above. None when this search found no web
    # articles at all (nothing to summarize).
    web_summary: WebSummaryOut | None = None


class CitedPaperOut(BaseModel):
    paper_id: str
    title: str


class CitedWebArticleOut(BaseModel):
    url: str
    title: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # curation-chat-metadata Phase 1: all additive, all defaulted -- a
    # pre-Phase-1 {role, content} dict still constructs cleanly via
    # ChatTurn(**turn), with every field below at its default. Only
    # curation_chat.py's chat_turn() populates these (on curation
    # sessions); the separate one-shot pipeline's /chat (ChatRequest.history
    # below) never sets them, so its entries stay at these defaults too.
    #
    # exchange_id: shared by the user question and assistant answer of ONE
    # chat_turn() call -- None for entries that predate this phase.
    exchange_id: str | None = None
    # used_web_search / cited_web_articles / added_to_report are per-ANSWER
    # metadata -- only ever set on the assistant entry of a pair.
    used_web_search: bool = False
    cited_web_articles: list[CitedWebArticleOut] = Field(default_factory=list)
    # Always False in Phase 1 -- no code path sets this True yet (that's a
    # later phase's "Add to report" action).
    added_to_report: bool = False


class ChatRequest(BaseModel):
    search_id: int
    question: str
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    # Round-2 enhancement 5: kept as its own list (tagged [Web N] in the
    # answer text) so the UI can render it distinguishably from cited_papers
    # ([Paper N]), not merged into one generic citation list.
    cited_web_articles: list[CitedWebArticleOut]
    history: list[ChatTurn]


class LibraryItem(BaseModel):
    search_id: int
    topic: str
    created_at: str
    paper_count: int
    has_summary: bool
    web_article_count: int


class CurationStartRequest(BaseModel):
    topic: str
    # 1-30: matches report.py's own documented 30-paper cap; target_count
    # is "how many picks the user wants total," not a per-batch size.
    target_count: int = Field(default=10, ge=1, le=30)
    use_openalex_fallback: bool = False


class CurationPicksRequest(BaseModel):
    picked_paper_ids: list[str] = []
    stop: bool = False
    # curation-refinement-and-auto-offer Phase 6f: optional free-text
    # steering (e.g. "focus on more recent work"), carried into the
    # SAME resume payload picked_paper_ids/stop already use -- see
    # resume_curation_turn's own docstring for why it can't be a
    # separate out-of-band call instead.
    refinement: str | None = None
    # curation-turn-history Phase 9d: explicit "search for more now"
    # request -- reuses the SAME force_refill mechanism refinement above
    # already triggers, not a second one. False (the default) preserves
    # every existing caller's exact behavior unchanged.
    request_refill: bool = False


class CurationTurnResponse(BaseModel):
    session_id: str
    stage: str
    target_count: int
    selected_paper_ids: list[str]
    batch: list[PaperOut] = []
    stop_reason: str | None = None
    # Phase 6c: whether THIS turn's batch came from a fresh search
    # (refill_pool ran) vs. serving from the already-fetched pool —
    # meaningless once stop_reason is set (batch is always [] then), where
    # it's always False rather than a stale carry-over value.
    refilled: bool = False
    # How many already-fetched, not-yet-served candidates remain in the
    # pool after this turn's batch — PaperPoolSession.remaining(), the
    # exact number that decides whether the NEXT turn needs a fresh
    # search at all. Surfaces the pool panel's "N more candidates
    # already fetched" status line without the frontend having to infer
    # it from anything.
    reserve_remaining: int = 0
    # Phase 6f: every refinement note applied so far this session, so the
    # UI can show what's currently steering the search.
    refinement_notes: list[str] = []


class ReportSectionOut(BaseModel):
    content: str
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut] = []


class ReportOut(BaseModel):
    findings: ReportSectionOut
    limitations: ReportSectionOut
    future_scope: ReportSectionOut
    skipped_paper_ids: list[str]


class TurnHistoryEntryOut(BaseModel):
    turn_number: int
    batch: list[PaperOut]
    refilled: bool


class CurationStateResponse(BaseModel):
    session_id: str
    topic: str
    # curation-review-management Phase 8, item 5: canonicalize_topic()'s
    # restatement of `topic`, for display only -- `topic` above is
    # unchanged and still what actually drives search/ranking/refinement.
    display_title: str
    stage: str
    target_count: int
    selected_paper_ids: list[str]
    selected_papers: list[PaperOut]
    # Only non-None mid-curation, with a genuinely pending interrupt — the
    # exact property a page refresh during curation needs to recover from
    # the backend, not from anything held only in browser memory (Phase 6d).
    pending_batch: list[PaperOut] | None = None
    refilled: bool = False
    reserve_remaining: int = 0
    refinement_notes: list[str] = []
    report: ReportOut | None = None
    chat_history: list[ChatTurn] = []
    web_articles_added: list[WebArticleOut] = []
    pending_web_offer: dict | None = None
    pending_report_update: dict | None = None
    # curation-turn-history Phase 9b: every batch ever served, in order --
    # lets a client redraw ANY past turn's cards/abstracts, not just the
    # currently-pending one. Unbounded (see PaperPoolSession.turn_history).
    turn_history: list[TurnHistoryEntryOut] = []
    # Persisted so a reload/reopen can still show WHY curation stopped
    # (target_met / user_stopped / exhausted) -- None while stage=="curate".
    stop_reason: str | None = None


class CurationChatRequest(BaseModel):
    message: str


class CurationChatResponse(BaseModel):
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut]
    web_offer_made: bool = False
    web_offer_declined: bool = False
    web_search_used: bool = False
    new_web_articles_found: int | None = None
    # curation-refinement-and-auto-offer Phase 6f-3
    report_update_offer_made: bool = False
    report_update_declined: bool = False
    report_updated: bool = False
    chat_history: list[ChatTurn]


class CurationChatDeleteRequest(BaseModel):
    # curation-chat-delete Phase 3: exchange_id, not individual message ids
    # -- deleting an exchange always removes both the user question and
    # assistant answer that share it (see delete_chat_exchanges()).
    exchange_ids: list[str]


class CurationChatDeleteResponse(BaseModel):
    chat_history: list[ChatTurn]
    # The subset of the requested ids that actually matched >=1 entry and
    # were removed -- never includes an id that matched nothing (an
    # unknown id is a silent no-op, not an error; see the service).
    deleted_exchange_ids: list[str]
    # True if any REMOVED assistant entry had added_to_report=True. Phase 3
    # deliberately does not regenerate or otherwise touch session.report --
    # this is only a signal for the frontend to show a "report may be
    # stale" warning; real stale-report handling is a later phase.
    report_possibly_stale: bool = False


class CurationChatAddToReportRequest(BaseModel):
    exchange_ids: list[str]


class CurationChatAddToReportResponse(BaseModel):
    report: ReportOut
    chat_history: list[ChatTurn]
    # Exchanges whose cited web sources were newly approved and reflected
    # in `report` above -- always a subset of the request, never includes
    # an unknown/ineligible/already-added id (see skipped_exchange_ids).
    added_exchange_ids: list[str]
    skipped_exchange_ids: list[str]
    # Unique NEWLY approved cited web URLs this call -- NOT the cumulative
    # approved set (session.report_approved_web_article_urls may already
    # be larger from a previous call).
    source_count: int


class CurationReviewSummary(BaseModel):
    session_id: str
    topic: str
    # curation-review-management Phase 8, item 5: same display-only
    # restatement as CurationStateResponse.display_title -- this is what
    # the left panel's review list should actually show as each review's
    # name.
    display_title: str
    stage: str
    selected_count: int
    target_count: int
    has_report: bool
    has_chat: bool


class CurationDeleteResponse(BaseModel):
    session_id: str
    deleted: bool = True


class CurationSelectFromHistoryRequest(BaseModel):
    paper_id: str


class CurationSelectFromHistoryResponse(BaseModel):
    session_id: str
    selected_paper_ids: list[str]
