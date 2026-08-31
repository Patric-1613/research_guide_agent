import type {
  ApiErrorBody,
  CurationChatAddToReportRequest,
  CurationChatAddToReportResponse,
  CurationChatDeleteRequest,
  CurationChatDeleteResponse,
  CurationChatEditRequest,
  CurationChatEditResponse,
  CurationCapabilitiesResponse,
  CurationChatRequest,
  CurationChatResponse,
  CurationDeleteResponse,
  CurationPicksRequest,
  CurationReviewSummary,
  CurationSelectFromHistoryResponse,
  CurationStartRequest,
  CurationStateResponse,
  CurationTurnResponse,
  LaneSuggestResponse,
  RefinementMode,
  ReportExportFormat,
  ReportOut,
  ReportTemplate,
} from '../../types'
import { ApiError } from '../../types'

// PR3: strips any trailing slash(es) from an explicitly configured
// VITE_API_BASE_URL so `${baseUrl()}${path}` (every call site below,
// chatStream.ts, and reportStream.ts all build URLs this way, and every
// `path` they pass already starts with '/') never produces an
// accidental '//'. A same-origin '' base has nothing to strip.
function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, '')
}

// PR3: the three inputs that decide the effective base URL, factored out
// of baseUrl() as a pure function so tests can exercise every branch
// (explicit override / production same-origin / local-dev default)
// directly with plain booleans, instead of needing to stub Vite's own
// import.meta.env.PROD flag. `configured` empty/undefined means "no
// explicit override" -- an explicitly EMPTY VITE_API_BASE_URL is treated
// the same as unset, not as "explicitly same-origin", so a blank env var
// can never silently break local dev's cross-origin fetch to :8000.
export function resolveBaseUrl(configured: string | undefined, isProductionBuild: boolean): string {
  if (configured) {
    return normalizeBaseUrl(configured)
  }
  // Production builds default to same-origin (the app is served by the
  // same FastAPI process it calls, api_app/app.py's new static-frontend
  // mount) -- an empty base means every `${baseUrl()}${path}` call
  // becomes a plain relative path, resolved by the browser against the
  // page's own origin.
  if (isProductionBuild) {
    return ''
  }
  // Local Vite dev server: the backend is a separate process on its
  // default port, unchanged from before this function existed.
  return 'http://localhost:8000'
}

// Reads at call time via Vite's import.meta.env, not module load time --
// makes it possible to swap in tests without needing to reload the module.
// Exported (Usage Protection M4.2B) so lib/api/chatStream.ts and lib/
// api/reportStream.ts build the same base URL for their own fetch()
// calls, rather than re-deriving it.
export function baseUrl(): string {
  return resolveBaseUrl(import.meta.env.VITE_API_BASE_URL, import.meta.env.PROD)
}

// Usage Protection M4.2B: pulled out of request() below, unchanged
// behavior, so lib/api/chatStream.ts's own fetch() call (which needs the
// raw streaming response body on success, so it can't go through
// request()/postJson() itself) still maps a non-2xx response to the same
// ApiError shape/Retry-After handling every other endpoint already gets.
export async function throwApiErrorIfNotOk(response: Response): Promise<void> {
  if (response.ok) return
  let body: ApiErrorBody | null = null
  try {
    body = await response.json()
  } catch {
    body = null
  }
  // Usage Protection M2.3: only the Retry-After header is ever read
  // here -- response headers are not exposed generally, this is the
  // one specific header ApiError's own constructor knows how to
  // safely parse (see its own docstring on the supported format).
  throw new ApiError(response.status, body, response.headers.get('Retry-After'))
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl()}${path}`, {
    ...init,
    // H1: send the browser's stored Basic-Auth credentials on every
    // request, so a split-origin production deployment (frontend and API
    // on different origins) works behind the fail-closed auth gate. Inert
    // and harmless for the same-origin case. Paired with the backend's
    // allow_credentials=True + explicit allow_origins (never "*").
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  await throwApiErrorIfNotOk(response)
  return response.json() as Promise<T>
}

function postJson<T>(path: string, body: unknown, init?: RequestInit): Promise<T> {
  return request<T>(path, { ...init, method: 'POST', body: JSON.stringify(body) })
}

function deleteRequest<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'DELETE' })
}

export const curationApi = {
  listReviews: (): Promise<CurationReviewSummary[]> => request('/curation/reviews'),

  // Research Lanes (RL5): the zero-provider capability probe. A rejection
  // here is handled by the caller as "lane mode unavailable", never as a
  // user-facing error -- single search must keep working regardless.
  getCurationCapabilities: (signal?: AbortSignal): Promise<CurationCapabilitiesResponse> =>
    request('/curation/capabilities', { signal }),

  // Research Lanes (RL5): one inexpensive structured LLM call server-side;
  // returns exactly three editable suggestions. The client sends only the
  // topic -- there is no client-controlled suggestion count.
  suggestResearchLanes: (topic: string, signal?: AbortSignal): Promise<LaneSuggestResponse> =>
    postJson('/curation/lanes/suggest', { topic }, { signal }),

  start: (req: CurationStartRequest): Promise<CurationTurnResponse> => postJson('/curation/start', req),

  picks: (sessionId: string, req: CurationPicksRequest): Promise<CurationTurnResponse> =>
    postJson(`/curation/${sessionId}/picks`, req),

  getState: (sessionId: string): Promise<CurationStateResponse> => request(`/curation/${sessionId}`),

  // report-quality Phase R2C: reportTemplate is optional on both --
  // omitted sends the exact same {} body every existing caller/test
  // already relies on (generate defaults to analytical server-side,
  // regenerate preserves the existing report's template server-side).
  //
  // report-quality Phase R4.1: refinementMode follows the identical
  // "omitted/off means no key in the body at all" convention -- only
  // "single" is ever sent explicitly, keeping the payload minimal and
  // every existing caller/test (which never passes it) byte-identical.
  generateReport: (sessionId: string, reportTemplate?: ReportTemplate, refinementMode?: RefinementMode): Promise<ReportOut> =>
    postJson(`/curation/${sessionId}/report`, {
      ...(reportTemplate ? { report_template: reportTemplate } : {}),
      ...(refinementMode && refinementMode !== 'off' ? { refinement_mode: refinementMode } : {}),
    }),

  regenerateReport: (sessionId: string, reportTemplate?: ReportTemplate, refinementMode?: RefinementMode): Promise<ReportOut> =>
    postJson(`/curation/${sessionId}/report/regenerate`, {
      ...(reportTemplate ? { report_template: reportTemplate } : {}),
      ...(refinementMode && refinementMode !== 'off' ? { refinement_mode: refinementMode } : {}),
    }),

  // report-quality Phase R3: version_id is a path parameter (matches the
  // backend router), no body -- same {} POST-with-no-payload convention
  // as reopen() below.
  activateReportVersion: (sessionId: string, versionId: string): Promise<ReportOut> =>
    postJson(`/curation/${sessionId}/reports/${versionId}/activate`, {}),

  // report-quality Phase R5A/R5B/R5C.3: a plain URL string, NOT a
  // request()/postJson() call -- deliberately doesn't go through the
  // JSON request helper above, since this is meant to be used as a
  // real browser download link (<a href={...} download>), not fetched
  // via JS at all. Always exports the session's currently ACTIVE
  // report version (the backend endpoint's own contract) -- there is
  // no version_id param here to get wrong. format defaults to
  // "markdown" for backward compatibility with pre-R5C.3 callers;
  // "pdf"/"docx" are now equally supported (R5C.1/R5C.2 backend work).
  getReportExportUrl: (sessionId: string, format: ReportExportFormat = 'markdown'): string =>
    `${baseUrl()}/curation/${sessionId}/report/export?format=${format}`,

  chat: (sessionId: string, req: CurationChatRequest): Promise<CurationChatResponse> =>
    postJson(`/curation/${sessionId}/chat`, req),

  // curation-chat-delete Phase 3: POST, not DELETE-with-body -- matches
  // the backend router's own choice (see curation_chat.py's router
  // comment): deleteRequest() below has no body parameter at all, and
  // every other payload-carrying mutation in this client is already POST.
  deleteChatExchanges: (sessionId: string, req: CurationChatDeleteRequest): Promise<CurationChatDeleteResponse> =>
    postJson(`/curation/${sessionId}/chat/exchanges/delete`, req),

  // curation-chat-add-to-report Phase 4: same POST convention as delete above.
  addChatExchangesToReport: (
    sessionId: string, req: CurationChatAddToReportRequest,
  ): Promise<CurationChatAddToReportResponse> => postJson(`/curation/${sessionId}/chat/exchanges/add-to-report`, req),

  // curation-chat-edit Phase 5: same POST convention as delete/add-to-report above.
  editChatExchange: (sessionId: string, req: CurationChatEditRequest): Promise<CurationChatEditResponse> =>
    postJson(`/curation/${sessionId}/chat/exchanges/edit`, req),

  deleteReview: (sessionId: string): Promise<CurationDeleteResponse> =>
    deleteRequest(`/curation/${sessionId}`),

  // Phase 9c: synthesize-stage only -- see select_paper_from_history()'s
  // docstring (research_agent/curation_session.py) for exactly why this
  // is unsafe while stage=="curate" (a real interrupt pending on the
  // OTHER graph). Picking from history while still curating goes
  // through picks() above instead (picked_paper_ids may reference any
  // paper in turn_history, not just the current batch -- Phase 9d).
  selectFromHistory: (sessionId: string, paperId: string): Promise<CurationSelectFromHistoryResponse> =>
    postJson(`/curation/${sessionId}/select-from-history`, { paper_id: paperId }),

  // curation-editable-until-locked Phase 10c/10e: only valid on a review
  // that's stopped (stage=="synthesize") but hasn't been chatted/
  // reported yet -- reopen_curation_session()'s own docstring (research_
  // agent/curation_session.py) has the full reasoning. Backend is the
  // authoritative check; the frontend just hides the action otherwise.
  reopen: (sessionId: string): Promise<CurationTurnResponse> =>
    postJson(`/curation/${sessionId}/reopen`, {}),
}
