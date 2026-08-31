import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { baseUrl, curationApi, resolveBaseUrl } from './client'
import { ApiError } from '../../types'

// Usage Protection M2.3: headers defaults to an empty Headers so every
// EXISTING call site (none of which pass a 3rd argument) keeps working
// unchanged -- response.headers.get(...) (ApiError's own Retry-After
// read in client.ts) resolves to null rather than throwing.
function mockFetchOnce(status: number, body: unknown, headers: Record<string, string> = {}) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    headers: new Headers(headers),
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

// PR3: resolveBaseUrl() is the pure function baseUrl() delegates to --
// tested directly with plain booleans rather than stubbing Vite's own
// import.meta.env.PROD flag (vi.stubEnv only reliably stubs string-typed
// env vars, not PROD/DEV's boolean flags).
describe('resolveBaseUrl', () => {
  it('an explicit configured URL wins over both defaults', () => {
    expect(resolveBaseUrl('https://deployed.example.org', false)).toBe('https://deployed.example.org')
    expect(resolveBaseUrl('https://deployed.example.org', true)).toBe('https://deployed.example.org')
  })

  it('strips a trailing slash from an explicitly configured URL', () => {
    expect(resolveBaseUrl('https://deployed.example.org/', false)).toBe('https://deployed.example.org')
  })

  it('strips multiple trailing slashes from an explicitly configured URL', () => {
    expect(resolveBaseUrl('https://deployed.example.org///', false)).toBe('https://deployed.example.org')
  })

  it('a configured URL with no trailing slash is left unchanged', () => {
    expect(resolveBaseUrl('https://deployed.example.org', false)).toBe('https://deployed.example.org')
  })

  it('production build with no override defaults to same-origin (empty base)', () => {
    expect(resolveBaseUrl(undefined, true)).toBe('')
  })

  it('an explicitly empty configured URL is treated as unset, not as same-origin', () => {
    expect(resolveBaseUrl('', true)).toBe('')
    expect(resolveBaseUrl('', false)).toBe('http://localhost:8000')
  })

  it('local development build with no override defaults to http://localhost:8000', () => {
    expect(resolveBaseUrl(undefined, false)).toBe('http://localhost:8000')
  })
})

describe('baseUrl', () => {
  it('reads the explicit VITE_API_BASE_URL override at call time', () => {
    // beforeEach below already stubs VITE_API_BASE_URL to
    // 'http://test-api.local' for the whole file (shared with every
    // curationApi test) -- this proves baseUrl() itself, not just
    // resolveBaseUrl(), wires that value through.
    expect(baseUrl()).toBe('http://test-api.local')
  })
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

  // H1: every JSON API request sends the browser's stored Basic-Auth
  // credentials so a split-origin production deployment works behind the
  // fail-closed auth gate.
  it('requests are sent with credentials: "include" (GET and POST alike)', async () => {
    const getMock = mockFetchOnce(200, [])
    await curationApi.listReviews()
    expect(getMock).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ credentials: 'include' }))

    const postMock = mockFetchOnce(200, {
      session_id: 's1', stage: 'curate', target_count: 10,
      selected_paper_ids: [], batch: [], stop_reason: null, refilled: false,
    })
    await curationApi.start({ topic: 't', target_count: 10 })
    expect(postMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ method: 'POST', credentials: 'include' }),
    )
  })

  // report-quality Phase R2C
  it('generateReport() posts {} when no template is given', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.generateReport('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
  })

  it('generateReport() posts { report_template } when a template is given', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.generateReport('s1', 'foundational')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ report_template: 'foundational' }) }),
    )
  })

  it('regenerateReport() posts {} when no template is given', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.regenerateReport('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
  })

  it('regenerateReport() posts { report_template } when a template is given', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.regenerateReport('s1', 'expert')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ report_template: 'expert' }) }),
    )
  })

  // report-quality Phase R4.1
  it('generateReport() posts { refinement_mode: "single" } when refinementMode is "single"', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.generateReport('s1', undefined, 'single')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ refinement_mode: 'single' }) }),
    )
  })

  it('generateReport() posts both report_template and refinement_mode when both are given', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.generateReport('s1', 'expert', 'single')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report',
      expect.objectContaining({
        method: 'POST', body: JSON.stringify({ report_template: 'expert', refinement_mode: 'single' }),
      }),
    )
  })

  it('generateReport() posts {} when refinementMode is "off" -- same minimal payload as omitted', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.generateReport('s1', undefined, 'off')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
  })

  it('regenerateReport() posts { refinement_mode: "single" } when refinementMode is "single"', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.regenerateReport('s1', undefined, 'single')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ refinement_mode: 'single' }) }),
    )
  })

  it('regenerateReport() posts {} when refinementMode is omitted -- existing behavior unchanged', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.regenerateReport('s1')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
  })

  // report-quality Phase R3
  it('activateReportVersion() posts {} to the version-scoped activate path', async () => {
    const fetchMock = mockFetchOnce(200, {
      findings: { content: '', cited_papers: [] }, limitations: { content: '', cited_papers: [] },
      future_scope: { content: '', cited_papers: [] }, skipped_paper_ids: [],
    })

    await curationApi.activateReportVersion('s1', 'v2')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/reports/v2/activate',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    )
  })

  // report-quality Phase R5B
  it('getReportExportUrl() builds the markdown export URL for the given session', () => {
    const url = curationApi.getReportExportUrl('s1', 'markdown')

    expect(url).toBe('http://test-api.local/curation/s1/report/export?format=markdown')
  })

  it('getReportExportUrl() defaults to format=markdown when omitted', () => {
    const url = curationApi.getReportExportUrl('s1')

    expect(url).toBe('http://test-api.local/curation/s1/report/export?format=markdown')
  })

  // report-quality Phase R5C.3: pdf/docx follow the exact same URL
  // shape as markdown -- only the format query value changes.
  it('getReportExportUrl() builds the pdf export URL for the given session', () => {
    const url = curationApi.getReportExportUrl('s1', 'pdf')

    expect(url).toBe('http://test-api.local/curation/s1/report/export?format=pdf')
  })

  it('getReportExportUrl() builds the docx export URL for the given session', () => {
    const url = curationApi.getReportExportUrl('s1', 'docx')

    expect(url).toBe('http://test-api.local/curation/s1/report/export?format=docx')
  })

  it('getReportExportUrl() never calls fetch -- it is a plain URL string for a browser download link', () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    curationApi.getReportExportUrl('s1', 'markdown')

    expect(fetchMock).not.toHaveBeenCalled()
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

  // Research Lanes (RL5)
  it('getCurationCapabilities() issues a plain GET to /curation/capabilities', async () => {
    const fetchMock = mockFetchOnce(200, { research_lanes_enabled: true })

    const result = await curationApi.getCurationCapabilities()

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/capabilities',
      expect.not.objectContaining({ method: 'POST' }),
    )
    expect(result).toEqual({ research_lanes_enabled: true })
  })

  it('suggestResearchLanes() posts only the topic to /curation/lanes/suggest', async () => {
    const fetchMock = mockFetchOnce(200, { lanes: [] })

    await curationApi.suggestResearchLanes('reducing hallucination in RAG')

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/lanes/suggest',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ topic: 'reducing hallucination in RAG' }) }),
    )
  })

  it('start() forwards a lanes payload when one is given', async () => {
    const fetchMock = mockFetchOnce(200, {
      session_id: 's1', stage: 'curate', target_count: 10,
      selected_paper_ids: [], batch: [], stop_reason: null, refilled: false,
    })

    await curationApi.start({
      topic: 't', target_count: 5,
      lanes: [{ label: 'L', question: 'q', query: 'qq', enabled: true }],
    })

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/start',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          topic: 't', target_count: 5,
          lanes: [{ label: 'L', question: 'q', query: 'qq', enabled: true }],
        }),
      }),
    )
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
