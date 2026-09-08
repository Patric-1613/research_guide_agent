// Day 5: every protected transport carries the Firebase ID token in
// firebase mode, and none of them add an Authorization header when the
// auth bridge is not registered (disabled/basic mode).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { curationApi, downloadReportExport } from './client'
import { streamCurationChat } from './chatStream'
import { streamGenerateReport, streamRegenerateReport } from './reportStream'
import { clearAuthBridge, registerAuthBridge } from '../auth/authBridge'
import { ApiError } from '../../types'

function jsonFetch(status: number, body: unknown, headers: Record<string, string> = {}) {
  const mock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    blob: async () => new Blob(['x']),
    body: new ReadableStream({ start: (c) => c.close() }),
    headers: new Headers(headers),
  })
  vi.stubGlobal('fetch', mock)
  return mock
}

function authHeaderOf(mock: ReturnType<typeof vi.fn>): string | undefined {
  const init = mock.mock.calls[0][1] as RequestInit
  return new Headers(init.headers).get('Authorization') ?? undefined
}

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', 'http://test-api.local')
})
afterEach(() => {
  clearAuthBridge()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('no auth bridge registered (disabled / basic mode)', () => {
  it('curationApi requests carry no Authorization header', async () => {
    const mock = jsonFetch(200, [])
    await curationApi.listReviews()
    expect(authHeaderOf(mock)).toBeUndefined()
  })

  it('chat stream carries no Authorization header', async () => {
    const mock = jsonFetch(200, {})
    const it = streamCurationChat('s1', { message: 'x' }, { signal: new AbortController().signal })
    await it.next().catch(() => {})
    expect(authHeaderOf(mock)).toBeUndefined()
  })

  it('report stream carries no Authorization header', async () => {
    const mock = jsonFetch(200, {})
    const it = streamGenerateReport('s1', {}, { signal: new AbortController().signal })
    await it.next().catch(() => {})
    expect(authHeaderOf(mock)).toBeUndefined()
  })

  it('report export carries no Authorization header', async () => {
    const mock = jsonFetch(200, null, { 'Content-Disposition': 'attachment; filename="r.md"' })
    vi.stubGlobal('URL', { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} })
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    await downloadReportExport('s1', 'markdown')
    expect(authHeaderOf(mock)).toBeUndefined()
    clickSpy.mockRestore()
  })
})

describe('auth bridge registered (firebase mode)', () => {
  beforeEach(() => {
    registerAuthBridge({ getToken: async () => 'ID-TOKEN-42', onUnauthorized: () => {} })
  })

  it('curationApi GET requests carry Authorization: Bearer <token>', async () => {
    const mock = jsonFetch(200, [])
    await curationApi.listReviews()
    expect(authHeaderOf(mock)).toBe('Bearer ID-TOKEN-42')
  })

  it('curationApi POST requests carry the Bearer token', async () => {
    const mock = jsonFetch(200, { session_id: 's1', stage: 'curate', target_count: 1, selected_paper_ids: [], batch: [], stop_reason: null, refilled: false })
    await curationApi.start({ topic: 't' })
    expect(authHeaderOf(mock)).toBe('Bearer ID-TOKEN-42')
  })

  it('the research-lane suggestion call carries the Bearer token', async () => {
    const mock = jsonFetch(200, { lanes: [] })
    await curationApi.suggestResearchLanes('topic')
    expect(authHeaderOf(mock)).toBe('Bearer ID-TOKEN-42')
  })

  it('chat streaming carries the Bearer token', async () => {
    const mock = jsonFetch(200, {})
    const it = streamCurationChat('s1', { message: 'x' }, { signal: new AbortController().signal })
    await it.next().catch(() => {})
    expect(authHeaderOf(mock)).toBe('Bearer ID-TOKEN-42')
  })

  it('report streaming (generate and regenerate) carries the Bearer token', async () => {
    const gen = jsonFetch(200, {})
    await streamGenerateReport('s1', {}, { signal: new AbortController().signal }).next().catch(() => {})
    expect(authHeaderOf(gen)).toBe('Bearer ID-TOKEN-42')

    const regen = jsonFetch(200, {})
    await streamRegenerateReport('s1', {}, { signal: new AbortController().signal }).next().catch(() => {})
    expect(authHeaderOf(regen)).toBe('Bearer ID-TOKEN-42')
  })

  it('the report export download carries the Bearer token and triggers a blob download', async () => {
    const mock = jsonFetch(200, null, { 'Content-Disposition': 'attachment; filename="review.pdf"' })
    const createObjectURL = vi.fn(() => 'blob:abc')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

    await downloadReportExport('s1', 'pdf')

    expect(authHeaderOf(mock)).toBe('Bearer ID-TOKEN-42')
    expect(mock.mock.calls[0][0]).toBe('http://test-api.local/curation/s1/report/export?format=pdf')
    expect(createObjectURL).toHaveBeenCalled()
    expect(clickSpy).toHaveBeenCalled()
    expect(revokeObjectURL).toHaveBeenCalled()
    clickSpy.mockRestore()
  })

  it('a 401 notifies the auth layer but still rejects (no silent retry)', async () => {
    const onUnauthorized = vi.fn()
    registerAuthBridge({ getToken: async () => 't', onUnauthorized })
    const mock = jsonFetch(401, { detail: { reason_code: 'unauthorized', message: 'x' } })

    await expect(curationApi.start({ topic: 't' })).rejects.toBeInstanceOf(ApiError)
    expect(onUnauthorized).toHaveBeenCalledTimes(1)
    expect(mock).toHaveBeenCalledTimes(1) // the request was not retried
  })

  it('the ID token never appears in the request URL', async () => {
    const mock = jsonFetch(200, [])
    await curationApi.listReviews()
    expect(String(mock.mock.calls[0][0])).not.toContain('ID-TOKEN-42')
  })
})
