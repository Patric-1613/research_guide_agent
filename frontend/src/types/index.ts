// Mirrors research_agent/api.py's Pydantic response/request models
// (curation-api-and-ui Phase 6a/6c) field-for-field -- kept as plain
// interfaces, not generated, since the backend has no OpenAPI-client
// codegen step wired up yet; if the two drift, the HTTP tests on both
// sides (tests/test_curation_api.py, src/lib/api/client.test.ts) are what
// catch it.

export interface PaperOut {
  paper_id: string
  title: string
  authors: string[]
  year: number | null
  venue: string | null
  abstract: string | null
  url: string | null
  doi: string | null
  citation_count: number | null
  source: string
  source_urls: Record<string, string>
  score: number | null
}

export interface WebArticleOut {
  title: string
  url: string
  snippet: string
  published_date: string | null
  source_domain: string
}

export interface CitedPaperOut {
  paper_id: string
  title: string
}

export interface CitedWebArticleOut {
  url: string
  title: string
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
  // curation-chat-metadata Phase 1: additive/defaulted on the backend --
  // absent or default on any pre-Phase-1 entry. Optional here too (rather
  // than required-with-defaults) so a client-optimistic message literal
  // (e.g. ChatModePanel's pendingMessage echo, which by definition hasn't
  // round-tripped through the backend yet) doesn't need to fake them.
  // exchange_id links the user question and assistant answer of one
  // exchange; used_web_search/cited_web_articles/added_to_report are
  // per-answer (assistant only).
  exchange_id?: string | null
  used_web_search?: boolean
  cited_web_articles?: CitedWebArticleOut[]
  added_to_report?: boolean
}

export interface CurationTurnResponse {
  session_id: string
  stage: string
  target_count: number
  selected_paper_ids: string[]
  batch: PaperOut[]
  stop_reason: string | null
  refilled: boolean
  reserve_remaining: number
  refinement_notes: string[]
}

// report-quality Phase R1: one entry in a report's global, report-wide
// References list -- the target of that report's inline [1], [2], [3]...
// markers. `number` is stable across the whole report (the same source
// always shows the same number, never per-section). `link_url` is the
// one hyperlink target regardless of kind, so the UI never needs to
// branch on `kind` to know what to link to.
export interface ReferenceEntry {
  number: number
  kind: 'paper' | 'web'
  title: string
  formatted: string
  paper_id?: string | null
  url?: string | null
  link_url?: string | null
}

export interface ReportSectionOut {
  content: string
  // Compatibility fields -- no longer rendered as pills (superseded by
  // inline markers + the References section), kept additive on the API
  // shape, not removed.
  cited_papers: CitedPaperOut[]
  cited_web_articles: CitedWebArticleOut[]
  // Which global reference numbers (see ReportOut.references) this
  // section's content actually cites. Optional/defaulted since an
  // old-shape report predating this field could in principle still
  // reach the frontend without it.
  reference_numbers?: number[]
}

// report-quality Phase R2A: one entry in a report's dynamic section list
// (ReportOut.sections) -- the future primary report body. For now, every
// report's `sections` is derived 1:1, in order, from the legacy
// findings/limitations/future_scope fields below (see the backend's
// derive_sections_from_legacy_report) -- real independent multi-section
// generation is a later phase, not this one.
export interface ReportSection {
  key: string
  title: string
  content: string
  reference_numbers?: number[]
}

export interface ReportOut {
  // Legacy fields -- kept as compatibility fields, not the primary
  // report body once dynamic sections (below) are generated
  // independently in a later phase.
  findings: ReportSectionOut
  limitations: ReportSectionOut
  future_scope: ReportSectionOut
  skipped_paper_ids: string[]
  // The global, deduped, report-wide reference list every section's
  // inline [N] markers point into. Optional/defaulted for the same
  // old-shape-report reason as reference_numbers above -- the backend
  // always derives it (fresh generation or on-the-fly for an old
  // report), but the frontend renders safely even if it's ever absent.
  references?: ReferenceEntry[]
  // report-quality Phase R2A: the future primary report body. Optional
  // for the same old-shape-report reason as references above, though in
  // practice the backend always derives it (see the backend's
  // _report_to_out) -- the frontend still defends against it being
  // absent, independent of trusting that derivation.
  sections?: ReportSection[]
}

export interface TurnHistoryEntryOut {
  turn_number: number
  batch: PaperOut[]
  refilled: boolean
}

export interface CurationStateResponse {
  session_id: string
  topic: string
  // Phase 8, item 5: a canonicalize_topic() restatement of `topic`, for
  // display only -- `topic` above is unchanged and still what actually
  // drove search/ranking/refinement when this review was created.
  display_title: string
  stage: string
  target_count: number
  selected_paper_ids: string[]
  selected_papers: PaperOut[]
  pending_batch: PaperOut[] | null
  refilled: boolean
  reserve_remaining: number
  refinement_notes: string[]
  report: ReportOut | null
  chat_history: ChatTurn[]
  web_articles_added: WebArticleOut[]
  pending_web_offer: { question: string } | null
  pending_report_update: Record<string, unknown> | null
  // Phase 9b: every batch ever served, in order -- lets the UI redraw
  // ANY past turn's cards/abstracts, not just the currently-pending one.
  turn_history: TurnHistoryEntryOut[]
  // Persisted so a reload/reopen can still show WHY curation stopped
  // (target_met / user_stopped / exhausted). None while stage=="curate".
  stop_reason: string | null
}

export interface CurationChatResponse {
  answer: string
  answerable: boolean
  cited_papers: CitedPaperOut[]
  cited_web_articles: CitedWebArticleOut[]
  web_offer_made: boolean
  web_offer_declined: boolean
  web_search_used: boolean
  new_web_articles_found: number | null
  report_update_offer_made: boolean
  report_update_declined: boolean
  report_updated: boolean
  chat_history: ChatTurn[]
}

export interface CurationChatDeleteRequest {
  exchange_ids: string[]
}

export interface CurationChatDeleteResponse {
  chat_history: ChatTurn[]
  deleted_exchange_ids: string[]
  report_possibly_stale: boolean
}

export interface CurationChatAddToReportRequest {
  exchange_ids: string[]
}

export interface CurationChatAddToReportResponse {
  report: ReportOut
  chat_history: ChatTurn[]
  added_exchange_ids: string[]
  skipped_exchange_ids: string[]
  // Unique NEWLY approved cited web URLs this call, not the cumulative
  // approved set.
  source_count: number
}

export interface CurationChatEditRequest {
  exchange_id: string
  question: string
}

export interface CurationChatEditResponse {
  answer: string
  answerable: boolean
  cited_papers: CitedPaperOut[]
  cited_web_articles: CitedWebArticleOut[]
  web_offer_made: boolean
  web_offer_declined: boolean
  web_search_used: boolean
  new_web_articles_found: number | null
  report_update_offer_made: boolean
  report_update_declined: boolean
  report_updated: boolean
  chat_history: ChatTurn[]
  report_possibly_stale: boolean
}

export interface CurationReviewSummary {
  session_id: string
  topic: string
  display_title: string
  stage: string
  selected_count: number
  target_count: number
  has_report: boolean
  has_chat: boolean
}

export interface CurationStartRequest {
  topic: string
  target_count?: number
  use_openalex_fallback?: boolean
}

export interface CurationPicksRequest {
  picked_paper_ids: string[]
  stop?: boolean
  refinement?: string | null
  // Phase 9d: the explicit "search for more now" action -- forces a
  // refill even when the reserve isn't truly exhausted yet.
  request_refill?: boolean
}

export interface CurationSelectFromHistoryRequest {
  paper_id: string
}

export interface CurationSelectFromHistoryResponse {
  session_id: string
  selected_paper_ids: string[]
}

export interface CurationChatRequest {
  message: string
}

export interface CurationDeleteResponse {
  session_id: string
  deleted: boolean
}

export interface ApiErrorBody {
  detail: string | { error: string } | unknown
}

export class ApiError extends Error {
  status: number
  body: ApiErrorBody | null

  constructor(status: number, body: ApiErrorBody | null) {
    const detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? {})
    super(`API request failed (${status}): ${detail}`)
    this.status = status
    this.body = body
  }
}
