import type {
  ApiErrorBody,
  CurationChatAddToReportRequest,
  CurationChatAddToReportResponse,
  CurationChatDeleteRequest,
  CurationChatDeleteResponse,
  CurationChatEditRequest,
  CurationChatEditResponse,
  CurationChatRequest,
  CurationChatResponse,
  CurationDeleteResponse,
  CurationPicksRequest,
  CurationReviewSummary,
  CurationSelectFromHistoryResponse,
  CurationStartRequest,
  CurationStateResponse,
  CurationTurnResponse,
  RefinementMode,
  ReportExportFormat,
  ReportOut,
  ReportTemplate,
} from '../../types'
import { ApiError } from '../../types'

// Reads at call time via Vite's import.meta.env, not module load time --
// makes it possible to swap in tests without needing to reload the module.
function baseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let body: ApiErrorBody | null = null
    try {
      body = await response.json()
    } catch {
      body = null
    }
    throw new ApiError(response.status, body)
  }
  return response.json() as Promise<T>
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}

function deleteRequest<T>(path: string): Promise<T> {
  return request<T>(path, { method: 'DELETE' })
}

export const curationApi = {
  listReviews: (): Promise<CurationReviewSummary[]> => request('/curation/reviews'),

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

  // report-quality Phase R5A/R5B: a plain URL string, NOT a request()/
  // postJson() call -- deliberately doesn't go through the JSON request
  // helper above, since this is meant to be used as a real browser
  // download link (<a href={...} download>), not fetched via JS at all.
  // Always exports the session's currently ACTIVE report version (the
  // backend endpoint's own contract) -- there is no version_id param
  // here to get wrong. format defaults to "markdown", the only value
  // R5B supports.
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
