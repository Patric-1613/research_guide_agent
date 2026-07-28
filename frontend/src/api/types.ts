// Mirrors research_agent/api.py's Pydantic response/request models
// (curation-api-and-ui Phase 6a/6c) field-for-field -- kept as plain
// interfaces, not generated, since the backend has no OpenAPI-client
// codegen step wired up yet; if the two drift, the HTTP tests on both
// sides (tests/test_curation_api.py, src/api/client.test.ts) are what
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

export interface ReportSectionOut {
  content: string
  cited_papers: CitedPaperOut[]
  cited_web_articles: CitedWebArticleOut[]
}

export interface ReportOut {
  findings: ReportSectionOut
  limitations: ReportSectionOut
  future_scope: ReportSectionOut
  skipped_paper_ids: string[]
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
