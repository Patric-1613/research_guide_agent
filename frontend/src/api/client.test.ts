import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { curationApi } from './client'
import { ApiError } from './types'

function mockFetchOnce(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', 'http://test-api.local')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('curationApi', () => {
  it('start() posts to /curation/start with the request body and returns the parsed turn response', async () => {
    const fetchMock = mockFetchOnce(200, {
      session_id: 's1', stage: 'curate', target_count: 10,
      selected_paper_ids: [], batch: [], stop_reason: null, refilled: false,
    })

    const result = await curationApi.start({ topic: 'transformers', target_count: 10 })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/start',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ topic: 'transformers', target_count: 10 }),
      }),
    )
    expect(result.session_id).toBe('s1')
    expect(result.stage).toBe('curate')
  })

  it('picks() posts to the session-scoped path', async () => {
    const fetchMock = mockFetchOnce(200, {
      session_id: 's1', stage: 'curate', target_count: 10,
      selected_paper_ids: ['p1'], batch: [], stop_reason: null, refilled: false,
    })

    await curationApi.picks('s1', { picked_paper_ids: ['p1'] })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/picks',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('getState() issues a plain GET, no body', async () => {
    const fetchMock = mockFetchOnce(200, {
      session_id: 's1', topic: 't', stage: 'curate', target_count: 10,
      selected_paper_ids: [], selected_papers: [], pending_batch: null, refilled: false,
      report: null, chat_history: [], web_articles_added: [], pending_web_offer: null, pending_report_update: null,
    })

    await curationApi.getState('s1')

    expect(fetchMock).toHaveBeenCalledWith('http://test-api.local/curation/s1', expect.not.objectContaining({ method: 'POST' }))
  })

  it('deleteReview() issues a DELETE to the session-scoped path', async () => {
    const fetchMock = mockFetchOnce(200, { session_id: 's1', deleted: true })

    const result = await curationApi.deleteReview('s1')

    expect(fetchMock).toHaveBeenCalledWith('http://test-api.local/curation/s1', expect.objectContaining({ method: 'DELETE' }))
    expect(result).toEqual({ session_id: 's1', deleted: true })
  })

  it('throws an ApiError carrying the status and parsed detail on a non-2xx response', async () => {
    mockFetchOnce(404, { detail: 'session_id not found' })

    await expect(curationApi.getState('does-not-exist')).rejects.toMatchObject({
      status: 404,
      body: { detail: 'session_id not found' },
    })
  })

  it('ApiError message includes the string detail for a clean error surface', async () => {
    mockFetchOnce(400, { detail: 'Session is not awaiting picks (curation already finished).' })

    try {
      await curationApi.picks('s1', { picked_paper_ids: [] })
      expect.fail('expected picks() to throw')
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).message).toContain('Session is not awaiting picks')
    }
  })
})
