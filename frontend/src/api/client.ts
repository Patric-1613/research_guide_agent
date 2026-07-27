import type {
  ApiErrorBody,
  CurationChatRequest,
  CurationChatResponse,
  CurationPicksRequest,
  CurationReviewSummary,
  CurationStartRequest,
  CurationStateResponse,
  CurationTurnResponse,
  ReportOut,
} from './types'
import { ApiError } from './types'

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

export const curationApi = {
  listReviews: (): Promise<CurationReviewSummary[]> => request('/curation/reviews'),

  start: (req: CurationStartRequest): Promise<CurationTurnResponse> => postJson('/curation/start', req),

  picks: (sessionId: string, req: CurationPicksRequest): Promise<CurationTurnResponse> =>
    postJson(`/curation/${sessionId}/picks`, req),

  getState: (sessionId: string): Promise<CurationStateResponse> => request(`/curation/${sessionId}`),

  generateReport: (sessionId: string): Promise<ReportOut> =>
    postJson(`/curation/${sessionId}/report`, {}),

  regenerateReport: (sessionId: string): Promise<ReportOut> =>
    postJson(`/curation/${sessionId}/report/regenerate`, {}),

  chat: (sessionId: string, req: CurationChatRequest): Promise<CurationChatResponse> =>
    postJson(`/curation/${sessionId}/chat`, req),
}
