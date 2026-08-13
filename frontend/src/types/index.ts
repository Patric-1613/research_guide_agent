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
  // report-quality Phase R3.2 Chunk 1
  cited_papers?: CitedPaperOut[]
  cited_web_articles?: CitedWebArticleOut[]
  added_to_report?: boolean
  // chat-web-relevance-guardrails R7C: only ever set (true/false) on an
  // assistant entry that actually cited web articles -- true means a
  // genuine answer-time relevance check ran; false means that check
  // fail-opened, so relevance is unverified. undefined/null covers both
  // "no web citation to judge" and "predates R7C" -- same additive
  // convention as every field above. See ChatMessageRow's own
  // isEligibleForAddToReport for where this gates the "Add to report"
  // action.
  web_relevance_verified?: boolean | null
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

// report-quality Phase R2C: which reader-depth template generated (or
// is treated as having generated) a report. Same plain string-union
// convention as this file already uses elsewhere (e.g. no enum), mirrors
// the backend's own report.py ReportTemplate Literal field-for-field.
export type ReportTemplate = 'foundational' | 'analytical' | 'expert'

// report-quality Phase R4.1: whether/how the optional draft -> evaluate
// -> revise -> finalize refinement loop ran for a report. Same plain
// string-union convention as ReportTemplate above, mirrors the
// backend's own report.py RefinementMode Literal field-for-field.
export type RefinementMode = 'off' | 'single'

// report-quality Phase R5A/R5B/R5C.3: which export format GET
// /curation/{id}/report/export?format=... accepts, mirroring the
// backend's own supported set (research_agent/services/
// curation_report_service.py's _EXPORT_FORMATS) field-for-field, same
// plain string-union convention as ReportTemplate/RefinementMode above.
export type ReportExportFormat = 'markdown' | 'pdf' | 'docx'

// report-quality Phase R4.1/R4.2: refinement metadata a report carries
// when refinement was requested for it, mirrors the backend's own
// ReportRefinementOut. R4.2's issues/revision_instructions/
// section_scores describe the DRAFT the one evaluation ran against,
// not necessarily the finalized report currently shown -- R4.1/R4.2
// never re-evaluate after a revision. Typed as required (not
// optional): the backend model's own defaults guarantee these are
// always present in the response once R4.2 is deployed, even for a
// report that only ever had R4.1-era metadata persisted.
export interface ReportRefinementOut {
  enabled: boolean
  rounds: number
  initial_score: number | null
  final_score: number | null
  issues: string[]
  revision_instructions: string
  section_scores: Record<string, number> | null
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
  // report-quality Phase R2C: optional/defaulted the same way -- the
  // backend always derives it (defaults old/omitted reports to
  // "analytical"), the frontend still defends against absence.
  report_template?: ReportTemplate
  // report-quality Phase R3: version metadata for the specific report
  // VERSION this ReportOut represents. All optional -- a construction
  // site without version context (or an old session's derived-implicit
  // version, whose real creation time was never recorded) still has a
  // well-formed ReportOut with these simply absent.
  version_id?: string | null
  version_number?: number | null
  created_at?: string | null
  generation_reason?: string | null
  // report-quality Phase R4.1: absent/null whenever refinement was
  // never requested for this report (the overwhelming common case).
  refinement?: ReportRefinementOut | null
}

// report-quality Phase R3: one lightweight entry in CurationStateResponse
// .report_versions -- deliberately does NOT carry the full report body
// (that's what activating a version and reading the refreshed `report`
// is for), just enough to render a compact version selector.
export interface ReportVersionSummary {
  version_id: string
  version_number: number
  created_at: string | null
  report_template: ReportTemplate
  generation_reason: string
  is_active: boolean
}

// report-quality Phase R2C: both request bodies are fully optional --
// omitting report_template entirely preserves every existing caller's
// behavior (generate defaults to analytical, regenerate preserves the
// existing report's template), matching the backend's own
// CurationGenerateReportRequest/CurationRegenerateReportRequest.
export interface CurationGenerateReportRequest {
  report_template?: ReportTemplate
  // report-quality Phase R4.1: omitted/off means no refinement -- same
  // "absence preserves prior behavior" convention as report_template
  // above.
  refinement_mode?: RefinementMode
}

export interface CurationRegenerateReportRequest {
  report_template?: ReportTemplate
  refinement_mode?: RefinementMode
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
  // report-quality Phase R3: lightweight summaries of every report
  // version this session has ever produced, in order. `report` above
  // stays exactly what it always was (the ACTIVE version's full body).
  report_versions: ReportVersionSummary[]
  active_report_version_id: string | null
  // report-quality Phase R3.2 Chunk 2: chat's own References list,
  // derived fresh from chat_history on every read -- independent
  // numbering from report.references entirely (see
  // research_agent/curation_chat.py's derive_chat_references).
  chat_references: ReferenceEntry[]
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

// Usage Protection M4.1/M4.2A: the frozen streaming event vocabulary
// research_agent/chat_streaming.py defines and research_agent/
// curation_chat_streaming.py emits, in order `started -> phase* -> delta*
// -> completed -> done` on success, or `started -> phase* -> delta* ->
// error -> done` on a handled failure. Mirrors ChatStreamEventType/
// ChatStreamPhase field-for-field -- kept as plain interfaces, not
// generated, same convention as every other type in this file. See
// lib/api/chatStream.ts for the adapter that decodes the raw SSE
// response into these, and hooks/useCurationSession.ts for the state
// machine that enforces their ordering.
export type ChatStreamPhase =
  | 'preparing_context'
  | 'summarizing_history'
  | 'checking_relevance'
  | 'searching_web'
  | 'generating'
  | 'saving'

export type ChatStreamEventType = 'started' | 'phase' | 'delta' | 'completed' | 'error' | 'done'

export interface ChatStreamStartedEvent {
  type: 'started'
  data: Record<string, never>
}

export interface ChatStreamPhaseEvent {
  type: 'phase'
  data: { phase: ChatStreamPhase }
}

export interface ChatStreamDeltaEvent {
  type: 'delta'
  data: { text: string }
}

// Deliberately narrower than a persisted ChatTurn -- exchange_id,
// relevance metadata, and offer state are NOT here (they only ever come
// from a canonical reload, see useCurationSession's own sendChatMessage
// Streaming docstring). answer here is provisional/preview text only.
export interface ChatStreamCompletedEvent {
  type: 'completed'
  data: {
    answer: string
    answerable: boolean
    cited_papers: CitedPaperOut[]
    cited_web_articles: CitedWebArticleOut[]
  }
}

export interface ChatStreamErrorEvent {
  type: 'error'
  data: { reason_code: string; message: string }
}

export interface ChatStreamDoneEvent {
  type: 'done'
  data: Record<string, never>
}

export type ChatStreamServerEvent =
  | ChatStreamStartedEvent
  | ChatStreamPhaseEvent
  | ChatStreamDeltaEvent
  | ChatStreamCompletedEvent
  | ChatStreamErrorEvent
  | ChatStreamDoneEvent

// Usage Protection M4.3A/M4.3B: the frozen report-generation/regeneration
// progress streaming event vocabulary research_agent/report_streaming.py
// defines and research_agent/curation_report_streaming.py emits, in
// order `started -> phase* -> completed -> done` on success (with NO
// phase events at all on an initial-generation cache hit), or `started
// -> phase* -> error -> done` on a handled failure. Deliberately no
// `delta` type -- report generation never streams prose/partial
// sections. Entirely independent of the ChatStream* types above (M4.2
// is closed; these mirror its shape but share no types with it) -- see
// lib/api/reportStream.ts for the adapter that decodes the raw SSE
// response into these, and hooks/useCurationSession.ts for the state
// machine that enforces their ordering.
export type ReportStreamPhase = 'generating' | 'evaluating' | 'revising' | 'saving'

export type ReportStreamEventType = 'started' | 'phase' | 'completed' | 'error' | 'done'

export interface ReportStreamStartedEvent {
  type: 'started'
  data: Record<string, never>
}

export interface ReportStreamPhaseEvent {
  type: 'phase'
  data: { phase: ReportStreamPhase }
}

// The completed payload is the EXISTING ReportOut shape, unmodified --
// reused directly (never a narrower/parallel type) so this can never
// drift from the real API response shape the non-streaming endpoints
// already return.
export interface ReportStreamCompletedEvent {
  type: 'completed'
  data: ReportOut
}

export interface ReportStreamErrorEvent {
  type: 'error'
  data: { reason_code: string; message: string }
}

export interface ReportStreamDoneEvent {
  type: 'done'
  data: Record<string, never>
}

export type ReportStreamServerEvent =
  | ReportStreamStartedEvent
  | ReportStreamPhaseEvent
  | ReportStreamCompletedEvent
  | ReportStreamErrorEvent
  | ReportStreamDoneEvent

export interface CurationDeleteResponse {
  session_id: string
  deleted: boolean
}

export interface ApiErrorBody {
  detail: string | { error: string } | unknown
}

// Usage Protection M2.3: one field-level validation error, as safely
// derived from FastAPI's own 422 `detail` array shape
// ({type, loc, msg, ctx, input}). Deliberately keeps ONLY `loc`/`msg` --
// `input` echoes the raw submitted value back (which is exactly the
// "serialized response object"/raw user content this phase's own
// safety rules forbid displaying) and `ctx`/`type` are internal detail,
// not user-safe text.
export interface ApiValidationError {
  loc: string
  msg: string
}

const RETRY_AFTER_INTEGER_SECONDS = /^\d+$/

// Only the backend's own Retry-After format is supported: a positive
// integer number of seconds (research_agent/api_app/app.py always sends
// str(exc.retry_after_seconds), never an HTTP-date). An HTTP-date value
// (or anything else non-integer, non-positive, or missing) safely
// becomes null rather than being mis-parsed into a wrong number.
function parseRetryAfterSeconds(value: string | null | undefined): number | null {
  if (!value) return null
  const trimmed = value.trim()
  if (!RETRY_AFTER_INTEGER_SECONDS.test(trimmed)) return null
  const seconds = Number.parseInt(trimmed, 10)
  return seconds > 0 ? seconds : null
}

export class ApiError extends Error {
  status: number
  body: ApiErrorBody | null
  // Usage Protection M2.3: structured fields parsed once here, so every
  // caller (the shared error-message helper, any component that wants
  // to branch on it) reads the same already-safe data instead of each
  // re-parsing `body` itself. All null when not present/not the
  // recognized shape -- never guessed or fabricated.
  //
  // Stable backend reason code (research_agent/usage_guard.py's
  // GuardReasonCode, research_agent/session_limits.py's
  // SessionCapacityReasonCode) -- present only for the M2 guard-error
  // `{detail: {reason_code, message}}` shape.
  reasonCode: string | null
  // The backend's own already-safe, user-facing message -- present for
  // the M2 guard-error shape (detail.message) AND for the older plain-
  // string `{detail: "..."}` shape (e.g. ServiceError's 400/404s)
  // several existing endpoints already use, so a caller doesn't need to
  // know which shape it was to read a safe message.
  safeMessage: string | null
  // Non-null only when a valid Retry-After header was present.
  retryAfterSeconds: number | null
  // Present only for FastAPI's own 422 validation-error array shape.
  validationErrors: ApiValidationError[] | null

  constructor(status: number, body: ApiErrorBody | null, retryAfterHeader?: string | null) {
    const detail: unknown = body?.detail
    let reasonCode: string | null = null
    let safeMessage: string | null = null
    let validationErrors: ApiValidationError[] | null = null

    if (typeof detail === 'string') {
      safeMessage = detail
    } else if (Array.isArray(detail)) {
      validationErrors = detail
        .filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === 'object')
        .map((entry) => ({
          loc: Array.isArray(entry.loc)
            ? entry.loc.filter((part) => typeof part === 'string' || typeof part === 'number').join('.')
            : '',
          msg: typeof entry.msg === 'string' ? entry.msg : 'Invalid value.',
        }))
    } else if (detail && typeof detail === 'object') {
      const record = detail as Record<string, unknown>
      if (typeof record.reason_code === 'string') reasonCode = record.reason_code
      if (typeof record.message === 'string') safeMessage = record.message
    }

    // .message stays a short, always-safe summary -- NEVER a
    // JSON.stringify of the raw response body (the previous behavior
    // here, which could surface a serialized object, or a 422's raw
    // `input` field, straight to the UI). Existing callers that only
    // checked a plain-string detail keep seeing that exact string.
    const messageText =
      safeMessage ??
      (validationErrors
        ? `Validation failed for ${validationErrors.length} field(s).`
        : `API request failed (${status}).`)
    super(messageText)

    this.status = status
    this.body = body
    this.reasonCode = reasonCode
    this.safeMessage = safeMessage
    this.validationErrors = validationErrors
    this.retryAfterSeconds = parseRetryAfterSeconds(retryAfterHeader)
  }
}
