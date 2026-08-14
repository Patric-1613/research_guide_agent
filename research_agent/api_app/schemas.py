"""Request/response Pydantic models for research_agent/api.py's endpoints.

Moved out of api.py (Phase 4) so the models have a real, independent home
that routers/services can import directly — api.py re-exports every name
here so `research_agent.api.<Name>` and `patch.object(api, "<Name>", ...)`
keep working unchanged for anything still reaching them that way.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from research_agent.api_app.constrained_types import IdList, OptionalUserText, UserText
from research_agent.citations import CitationStyle
from research_agent.config import get_usage_policy
from research_agent.report import RefinementMode, ReportTemplate


class SearchRequest(BaseModel):
    topic: UserText
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
    # Paper Keywords and Filtering, K1: pure pass-through of Paper.keywords
    # (see research_agent/keywords.py) -- never computed here. Defaults to
    # [] so a legacy Paper persisted before this field existed (which
    # reconstructs with keywords=[] per schema.py's own dataclass default)
    # serializes identically to a paper this phase genuinely found no
    # keywords for -- both mean "nothing to show," not "an error."
    keywords: list[str] = Field(default_factory=list)


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
    # report-quality Phase R3.2 Chunk 1: the paper-citation counterpart to
    # cited_web_articles above, same per-answer-only/additive/defaulted
    # convention -- a turn predating this field (or predating curation-
    # chat-metadata Phase 1 entirely) simply has none, [] being the
    # correct "nothing was ever recorded" default rather than a crash.
    # Lightweight {paper_id, title} entries only, deliberately NOT full
    # Paper objects -- session.selected_papers already holds the full
    # data (url/doi/etc.) chat is scoped to for the whole session, so a
    # paper_id is enough to resolve a real hyperlink/citation later
    # without duplicating paper data onto every single chat turn.
    cited_papers: list[CitedPaperOut] = Field(default_factory=list)
    # Always False in Phase 1 -- no code path sets this True yet (that's a
    # later phase's "Add to report" action).
    added_to_report: bool = False
    # chat-web-relevance-guardrails R7C: only ever set (True or False) on
    # an assistant entry that actually cited web articles -- True means a
    # genuine answer-time relevance check ran; False means that check
    # fail-opened, so relevance is unverified. None/absent covers both "no
    # web citation to judge" and "predates R7C" -- same additive/defaulted
    # convention as every field above, and the exact signal curation_
    # chat.py's select_eligible_exchanges_for_report gates report-
    # promotion eligibility on (missing/None is treated as eligible;
    # explicit False is not).
    web_relevance_verified: bool | None = None


class ChatTurnIn(BaseModel):
    """Usage Protection M2.2C: the request-side mirror of ChatTurn above,
    used only by ChatRequest.history (client-echoed chat history for the
    stateless one-shot /chat) -- identical shape, `content` constrained.
    ChatTurn itself is also a RESPONSE field (ChatResponse.history and
    several curation response models), so it stays completely
    unconstrained; an assistant-generated answer can legitimately exceed
    the user-text length bound."""

    role: Literal["user", "assistant"]
    content: UserText
    exchange_id: str | None = None
    used_web_search: bool = False
    cited_web_articles: list[CitedWebArticleOut] = Field(default_factory=list)
    cited_papers: list[CitedPaperOut] = Field(default_factory=list)
    added_to_report: bool = False
    web_relevance_verified: bool | None = None


class ChatRequest(BaseModel):
    search_id: int
    question: UserText
    # List-size bound consistent with the 100-turn session policy (2
    # raw entries -- one user, one assistant -- per turn; see
    # research_agent/session_limits.py's own docstring for the exact
    # turn-counting rule this mirrors, used for curation's own stored
    # chat_history rather than this stateless, client-echoed history).
    history: list[ChatTurnIn] = Field(default_factory=list, max_length=get_usage_policy().max_chat_turns_per_session * 2)


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
    topic: UserText
    # 1-30: matches report.py's own documented 30-paper cap; target_count
    # is "how many picks the user wants total," not a per-batch size.
    target_count: int = Field(default=10, ge=1, le=30)
    use_openalex_fallback: bool = False


class CurationPicksRequest(BaseModel):
    picked_paper_ids: IdList = Field(default_factory=list)
    stop: bool = False
    # curation-refinement-and-auto-offer Phase 6f: optional free-text
    # steering (e.g. "focus on more recent work"), carried into the
    # SAME resume payload picked_paper_ids/stop already use -- see
    # resume_curation_turn's own docstring for why it can't be a
    # separate out-of-band call instead.
    refinement: OptionalUserText = None
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


class ReferenceEntry(BaseModel):
    """report-quality Phase R1: one entry in a report's global, report-
    wide References list -- the target of that report's inline [1],
    [2], [3]... markers. `number` is stable across the whole report
    (the same source always shows the same number, never per-section).
    `link_url` is the frontend's one hyperlink target regardless of
    kind -- a paper's DOI/arXiv/plain url (whichever format_apa_citation
    itself preferred) for a paper, or the article's own url for a web
    source -- so the UI never needs to branch on `kind` to know what to
    link to."""

    number: int
    kind: Literal["paper", "web"]
    title: str
    formatted: str
    paper_id: str | None = None
    url: str | None = None
    link_url: str | None = None


class ReportSectionOut(BaseModel):
    content: str
    # cited_papers/cited_web_articles are kept as compatibility fields --
    # the frontend no longer renders them as pills (superseded by the
    # inline markers + references below), but the API shape stays
    # additive, not a breaking change.
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut] = []
    # Which global reference numbers (see ReportOut.references below)
    # this section's content actually cites -- kept as data, not
    # rendered directly, so a future chat-reference explorer can answer
    # "where is reference N used" without re-parsing content for marker
    # positions.
    reference_numbers: list[int] = Field(default_factory=list)


class ReportSection(BaseModel):
    """report-quality Phase R2A: one entry in a report's dynamic section
    list (ReportOut.sections below) -- the future primary report body.
    For now, every report's `sections` is derived 1:1, in order, from
    the existing findings/limitations/future_scope fields (see
    report.py's derive_sections_from_legacy_report) -- real independent
    multi-section generation is a later phase, not this one. `key` is a
    stable machine id (e.g. "findings"), `title` the human-readable
    heading (e.g. "Findings") -- kept separate so a future template can
    rename/reorder headings without the frontend needing to hardcode
    them."""

    key: str
    title: str
    content: str
    reference_numbers: list[int] = Field(default_factory=list)


class ReportRefinementOut(BaseModel):
    """report-quality Phase R4.1/R4.2: refinement detail exposed via the
    API -- still deliberately smaller than report.py's own internal
    ReportEvaluation (no raw prompt-facing text). R4.2 added issues/
    revision_instructions/section_scores; all three default safely so a
    report carrying only R4.1-era metadata (enabled/rounds/
    initial_score/final_score, persisted before R4.2 shipped) still
    constructs this model cleanly via ReportRefinementOut(**report[
    "refinement"]) -- no migration needed, Pydantic's own defaults fill
    the gap.

    report-quality Phase R4.2: issues/revision_instructions/
    section_scores describe the DRAFT the one evaluation ran against,
    not necessarily the finalized report -- R4.1/R4.2 never re-evaluate
    after a revision. See refine_report_if_requested's own docstring."""

    enabled: bool
    rounds: int
    initial_score: int | None = None
    final_score: int | None = None
    issues: list[str] = Field(default_factory=list)
    revision_instructions: str = ""
    section_scores: dict[str, int] | None = None


class ReportOut(BaseModel):
    # Legacy fields -- kept as compatibility fields, not the primary
    # report body once dynamic sections (below) are actually generated
    # independently in a later phase. Still required/always populated in
    # THIS phase, since report generation itself hasn't changed yet --
    # every report, old or new, genuinely has these three.
    findings: ReportSectionOut
    limitations: ReportSectionOut
    future_scope: ReportSectionOut
    skipped_paper_ids: list[str]
    # report-quality Phase R1: the global, deduped, report-wide reference
    # list every section's inline [N] markers point into. Defaulted so an
    # old, pre-R1 ReportOut construction site (if any ever existed) still
    # builds cleanly -- in practice always populated by _report_to_out,
    # either from a freshly-generated report or derived on the fly for an
    # old persisted one (see serializers.py/report.py's
    # derive_legacy_references).
    references: list[ReferenceEntry] = Field(default_factory=list)
    # report-quality Phase R2A: the future primary report body. Additive
    # and always populated by _report_to_out (derived from
    # findings/limitations/future_scope above when a report has no
    # `sections` of its own yet, which is every report in this phase) --
    # existing consumers reading findings/limitations/future_scope
    # directly are completely unaffected.
    sections: list[ReportSection] = Field(default_factory=list)
    # report-quality Phase R2C: which reader-depth template generated (or
    # is treated as having generated) this report. Defaulted to
    # "analytical" so an old, pre-R2C persisted report -- which has no
    # report_template key on its own dict at all -- serializes cleanly
    # without _report_to_out having to special-case it further than the
    # same "absence means analytical" default report.py's own regenerate
    # functions already apply.
    report_template: ReportTemplate = "analytical"
    # report-quality Phase R3: version metadata for the specific report
    # VERSION this ReportOut represents -- all optional/defaulted to
    # None, since a construction site without version context (or an
    # old session's derived-implicit version, whose real creation time
    # was never recorded and is never fabricated -- see curation_
    # session.py's _derive_implicit_report_versions) still builds
    # cleanly. created_at is an ISO 8601 string when known, same
    # convention SearchResponse.created_at/LibraryItem.created_at
    # already use elsewhere in this file.
    version_id: str | None = None
    version_number: int | None = None
    created_at: str | None = None
    generation_reason: str | None = None
    # report-quality Phase R4.1: minimal refinement metadata -- None
    # whenever refinement was never requested for this report (the
    # overwhelming common case), never a partially-filled dict. See
    # ReportRefinementOut's own docstring for exactly what this
    # deliberately does NOT expose.
    refinement: ReportRefinementOut | None = None


class ReportVersionSummary(BaseModel):
    """report-quality Phase R3: one lightweight entry in CurationState
    Response.report_versions -- deliberately does NOT embed the full
    report body (unlike ReportOut above), so a session with several
    report versions doesn't multiply the size of every single GET
    /curation/{id} response by however many versions it has. Enough to
    render a compact version list/selector (version number, template,
    when, why, and whether it's the one currently active) -- switching
    to a version's full content is what POST /curation/{id}/reports/
    {version_id}/activate is for."""

    version_id: str
    version_number: int
    created_at: str | None = None
    report_template: ReportTemplate
    generation_reason: str
    is_active: bool


class CurationGenerateReportRequest(BaseModel):
    # report-quality Phase R2C: None/omitted means "use the default
    # template" (analytical) -- see services/curation_report_service.py's
    # get_or_create_report for the exact resolution. Every existing
    # client that POSTs {} or no body at all keeps working unchanged.
    report_template: ReportTemplate | None = None
    # report-quality Phase R4.1: None/omitted means "off" -- byte-
    # identical behavior to before this field existed, zero extra LLM
    # calls. Only matters on the fresh-generation branch, same
    # "ignored on a cache hit" precedent report_template above already
    # established -- see get_or_create_report's own docstring.
    refinement_mode: RefinementMode | None = None


class CurationRegenerateReportRequest(BaseModel):
    # report-quality Phase R2C: None/omitted means "preserve the
    # existing report's current template" -- an explicit value switches
    # it for this regeneration. See report.py's
    # _regenerate_report_sections_with_sources for the exact resolution
    # rule (the same one this field's absence maps onto).
    report_template: ReportTemplate | None = None
    # report-quality Phase R4.1: None/omitted means "off" -- byte-
    # identical behavior to before this field existed, zero extra LLM
    # calls.
    refinement_mode: RefinementMode | None = None


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
    # report-quality Phase R3: lightweight summaries of every report
    # version this session has ever produced, in order -- `report`
    # above stays exactly what it always was (the ACTIVE version's full
    # body); this is the additional data a compact version selector
    # needs to render without a second round trip. Empty for a session
    # that's never generated a report at all, same as `report` itself
    # being None in that case.
    report_versions: list[ReportVersionSummary] = []
    # Mirrors session.active_report_version_id -- None only when
    # report_versions is empty. Redundant with report_versions' own
    # is_active flags in principle, but exposed directly so a client
    # doesn't have to scan the list to find which entry is active.
    active_report_version_id: str | None = None
    # report-quality Phase R3.2 Chunk 2: the global, deduped, chat-wide
    # reference list every qualifying assistant turn's rewritten inline
    # [N] markers (in chat_history above) point into -- reuses the exact
    # same ReferenceEntry shape report.py's own references already use,
    # but this list and `report.references` are two entirely
    # independently-numbered derivations; a shared shape does not imply
    # shared numbering or shared state. Derived fresh from session.
    # chat_history on every read (curation_chat.py's derive_chat_
    # references), never persisted -- empty for a session with no
    # qualifying assistant turns yet.
    chat_references: list[ReferenceEntry] = []


class CurationChatRequest(BaseModel):
    message: UserText


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
    exchange_ids: IdList


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
    exchange_ids: IdList


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


class CurationChatEditRequest(BaseModel):
    exchange_id: str
    question: UserText


class CurationChatEditResponse(BaseModel):
    # curation-chat-edit Phase 5: mirrors CurationChatResponse's per-turn
    # fields as closely as practical (same field names/meanings), plus
    # report_possibly_stale. report_update_offer_made/declined/report_updated
    # are included for frontend-handling consistency even though this
    # path can never actually set them True -- editing always clears any
    # pending_report_update before the fresh chat_turn() call, so that
    # branch can never fire here (see edit_chat_exchange's docstring).
    answer: str
    answerable: bool
    cited_papers: list[CitedPaperOut]
    cited_web_articles: list[CitedWebArticleOut]
    web_offer_made: bool = False
    web_offer_declined: bool = False
    web_search_used: bool = False
    new_web_articles_found: int | None = None
    report_update_offer_made: bool = False
    report_update_declined: bool = False
    report_updated: bool = False
    chat_history: list[ChatTurn]
    # True if the edited exchange's old answer, or any later exchange
    # truncated away by the edit, had added_to_report=True. Phase 5 never
    # auto-regenerates the report or prunes report_approved_web_article_
    # urls -- this is only a signal for the frontend to show a warning.
    report_possibly_stale: bool = False


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
