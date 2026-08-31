import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { streamGenerateReport, streamRegenerateReport, ReportStreamTransportError } from './reportStream'
import { ApiError } from '../../types'

beforeEach(() => {
  vi.stubEnv('VITE_API_BASE_URL', 'http://test-api.local')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

function sseBody(frames: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(frames)
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes)
      controller.close()
    },
  })
}

function mockStreamingFetch(status: number, body: ReadableStream<Uint8Array> | null, headers: Record<string, string> = {}) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    body,
    json: async () => ({ detail: 'rejected' }),
    headers: new Headers(headers),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const FULL_REPORT_JSON =
  '{"findings":{"content":"f","cited_papers":[],"cited_web_articles":[]},'
  + '"limitations":{"content":"l","cited_papers":[],"cited_web_articles":[]},'
  + '"future_scope":{"content":"fs","cited_papers":[],"cited_web_articles":[]},'
  + '"skipped_paper_ids":[]}'

async function collectGenerate(sessionId: string, signal = new AbortController().signal) {
  const events = []
  for await (const event of streamGenerateReport(sessionId, {}, { signal })) {
    events.push(event)
  }
  return events
}

async function collectRegenerate(sessionId: string, signal = new AbortController().signal) {
  const events = []
  for await (const event of streamRegenerateReport(sessionId, {}, { signal })) {
    events.push(event)
  }
  return events
}

describe('streamGenerateReport', () => {
  it('posts to /curation/{session_id}/report/stream with the correct method, headers, and body', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody(`event: started\ndata: {}\n\nevent: completed\ndata: ${FULL_REPORT_JSON}\n\nevent: done\ndata: {}\n\n`))
    const controller = new AbortController()

    await collectGenerate('s1', controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/stream',
      expect.objectContaining({
        method: 'POST',
        // H1: same credentialed-CORS contract as the JSON client.
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        signal: controller.signal,
      }),
    )
  })

  it('sends report_template and refinement_mode in the body, same shape as the non-streaming endpoint', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))

    const events = []
    for await (const event of streamGenerateReport(
      's1', { report_template: 'expert', refinement_mode: 'single' }, { signal: new AbortController().signal },
    )) {
      events.push(event)
    }

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/stream',
      expect.objectContaining({ body: JSON.stringify({ report_template: 'expert', refinement_mode: 'single' }) }),
    )
  })

  it('yields decoded frames as typed ReportStreamServerEvent objects, in order', async () => {
    mockStreamingFetch(
      200,
      sseBody(
        'event: started\ndata: {}\n\n'
        + 'event: phase\ndata: {"phase":"generating"}\n\n'
        + 'event: phase\ndata: {"phase":"saving"}\n\n'
        + `event: completed\ndata: ${FULL_REPORT_JSON}\n\n`
        + 'event: done\ndata: {}\n\n',
      ),
    )

    const events = await collectGenerate('s1')

    expect(events.map((e) => e.type)).toEqual(['started', 'phase', 'phase', 'completed', 'done'])
    expect(events[1].data).toEqual({ phase: 'generating' })
    expect(events[3].data).toMatchObject({ findings: { content: 'f' } })
  })

  it('non-2xx responses are mapped to the existing ApiError shape, never reading the body as a stream', async () => {
    mockStreamingFetch(404, null)

    await expect(collectGenerate('missing-session')).rejects.toBeInstanceOf(ApiError)
  })

  it('a 429 preserves Retry-After via the same ApiError mapping every other endpoint uses', async () => {
    mockStreamingFetch(429, null, { 'Retry-After': '30' })

    try {
      await collectGenerate('s1')
      throw new Error('expected a rejection')
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).retryAfterSeconds).toBe(30)
    }
  })

  it('forwards the given AbortSignal to fetch', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))
    const controller = new AbortController()

    await collectGenerate('s1', controller.signal)

    const [, init] = fetchMock.mock.calls[0]
    expect(init.signal).toBe(controller.signal)
  })

  it('a missing response body fails safely with ReportStreamTransportError, not a crash', async () => {
    mockStreamingFetch(200, null)

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })

  it('a malformed SSE frame fails safely with ReportStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('this is not valid SSE framing at all\n\n'))

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })

  it('malformed JSON inside a data: line fails safely with ReportStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: started\ndata: {not valid json\n\n'))

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })

  it('an unrecognized event name fails safely with ReportStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: not_a_real_event\ndata: {}\n\n'))

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })

  it('a stream that ends without its final frame\'s terminator fails safely with ReportStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}'))

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })

  it('an invalid/incomplete completed payload fails safely with ReportStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: completed\ndata: {"not_a_report": true}\n\n'))

    await expect(collectGenerate('s1')).rejects.toBeInstanceOf(ReportStreamTransportError)
  })
})

describe('streamRegenerateReport', () => {
  it('posts to /curation/{session_id}/report/regenerate/stream with the correct method, headers, and body', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))
    const controller = new AbortController()

    await collectRegenerate('s1', controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate/stream',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}), signal: controller.signal }),
    )
  })

  it('sends report_template and refinement_mode in the body, same shape as the non-streaming endpoint', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))

    const events = []
    for await (const event of streamRegenerateReport(
      's1', { report_template: 'foundational', refinement_mode: 'single' }, { signal: new AbortController().signal },
    )) {
      events.push(event)
    }

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/report/regenerate/stream',
      expect.objectContaining({ body: JSON.stringify({ report_template: 'foundational', refinement_mode: 'single' }) }),
    )
  })

  it('non-2xx responses are mapped to the existing ApiError shape', async () => {
    mockStreamingFetch(400, null)

    await expect(collectRegenerate('s1')).rejects.toBeInstanceOf(ApiError)
  })
})
