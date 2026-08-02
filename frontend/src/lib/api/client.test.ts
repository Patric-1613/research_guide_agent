import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { curationApi } from './client'
import { ApiError } from '../../types'

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

  it('deleteChatExchanges() posts (not DELETE) to the exchanges/delete path with the exchange_ids body', async () => {
    const fetchMock = mockFetchOnce(200, { chat_history: [], deleted_exchange_ids: ['ex-1'], report_possibly_stale: false })

    const result = await curationApi.deleteChatExchanges('s1', { exchange_ids: ['ex-1'] })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/chat/exchanges/delete',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ exchange_ids: ['ex-1'] }) }),
    )
    expect(result.deleted_exchange_ids).toEqual(['ex-1'])
  })

  it('addChatExchangesToReport() posts to the add-to-report path with the exchange_ids body', async () => {
    const fetchMock = mockFetchOnce(200, {
      report: { findings: { content: '', cited_papers: [], cited_web_articles: [] }, limitations: { content: '', cited_papers: [] }, future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [] },
      chat_history: [], added_exchange_ids: ['ex-1'], skipped_exchange_ids: [], source_count: 1,
    })

    const result = await curationApi.addChatExchangesToReport('s1', { exchange_ids: ['ex-1'] })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/chat/exchanges/add-to-report',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ exchange_ids: ['ex-1'] }) }),
    )
    expect(result.added_exchange_ids).toEqual(['ex-1'])
    expect(result.source_count).toBe(1)
  })

  it('editChatExchange() posts to the exchanges/edit path with the exchange_id + question body', async () => {
    const fetchMock = mockFetchOnce(200, {
      answer: 'fresh answer', answerable: true, cited_papers: [], cited_web_articles: [],
      web_offer_made: false, web_offer_declined: false, web_search_used: false, new_web_articles_found: null,
      report_update_offer_made: false, report_update_declined: false, report_updated: false,
      chat_history: [], report_possibly_stale: false,
    })

    const result = await curationApi.editChatExchange('s1', { exchange_id: 'ex-1', question: 'edited question' })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/chat/exchanges/edit',
      expect.objectContaining({
        method: 'POST', body: JSON.stringify({ exchange_id: 'ex-1', question: 'edited question' }),
      }),
    )
    expect(result.answer).toBe('fresh answer')
    expect(result.report_possibly_stale).toBe(false)
  })

  it('reopen() posts to the session-scoped /reopen path with no body payload beyond {}', async () => {
    const fetchMock = mockFetchOnce(200, {
      session_id: 's1', stage: 'curate', target_count: 10,
      selected_paper_ids: ['p1'], batch: [], stop_reason: null, refilled: false,
    })

    const result = await curationApi.reopen('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/reopen',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
    expect(result.stage).toBe('curate')
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
