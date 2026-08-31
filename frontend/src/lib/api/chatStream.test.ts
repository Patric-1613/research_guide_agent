import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { streamCurationChat, ChatStreamTransportError } from './chatStream'
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

async function collect(sessionId: string, message: string, signal = new AbortController().signal) {
  const events = []
  for await (const event of streamCurationChat(sessionId, { message }, { signal })) {
    events.push(event)
  }
  return events
}

describe('streamCurationChat', () => {
  it('posts to /curation/{session_id}/chat/stream with the correct method, headers, and body', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))
    const controller = new AbortController()

    await collect('s1', 'hello', controller.signal)

    expect(fetchMock).toHaveBeenCalledWith(
      'http://test-api.local/curation/s1/chat/stream',
      expect.objectContaining({
        method: 'POST',
        // H1: same credentialed-CORS contract as the JSON client.
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'hello' }),
        signal: controller.signal,
      }),
    )
  })

  it('yields decoded frames as typed ChatStreamServerEvent objects, in order', async () => {
    mockStreamingFetch(
      200,
      sseBody(
        'event: started\ndata: {}\n\n'
        + 'event: phase\ndata: {"phase":"generating"}\n\n'
        + 'event: delta\ndata: {"text":"Hi"}\n\n'
        + 'event: completed\ndata: {"answer":"Hi","answerable":true,"cited_papers":[],"cited_web_articles":[]}\n\n'
        + 'event: done\ndata: {}\n\n',
      ),
    )

    const events = await collect('s1', 'hi')

    expect(events.map((e) => e.type)).toEqual(['started', 'phase', 'delta', 'completed', 'done'])
    expect(events[1].data).toEqual({ phase: 'generating' })
    expect(events[2].data).toEqual({ text: 'Hi' })
  })

  it('correctly passes through response-body chunks split across multiple reader reads (the existing M4.1 decoder handles the buffering)', async () => {
    const frame1 = 'event: started\ndata: {}\n\n'
    const frame2 = 'event: done\ndata: {}\n\n'
    const bytes1 = new TextEncoder().encode(frame1.slice(0, 10))
    const bytes2 = new TextEncoder().encode(frame1.slice(10) + frame2)
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes1)
        controller.enqueue(bytes2)
        controller.close()
      },
    })
    mockStreamingFetch(200, body)

    const events = await collect('s1', 'hi')

    expect(events.map((e) => e.type)).toEqual(['started', 'done'])
  })

  it('non-2xx responses are mapped to the existing ApiError shape, never reading the body as a stream', async () => {
    mockStreamingFetch(404, null)

    await expect(collect('missing-session', 'hi')).rejects.toBeInstanceOf(ApiError)
  })

  it('a 429 preserves Retry-After via the same ApiError mapping every other endpoint uses', async () => {
    mockStreamingFetch(429, null, { 'Retry-After': '30' })

    try {
      await collect('s1', 'hi')
      throw new Error('expected a rejection')
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).retryAfterSeconds).toBe(30)
    }
  })

  it('forwards the given AbortSignal to fetch', async () => {
    const fetchMock = mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}\n\n'))
    const controller = new AbortController()

    await collect('s1', 'hi', controller.signal)

    const [, init] = fetchMock.mock.calls[0]
    expect(init.signal).toBe(controller.signal)
  })

  it('a missing response body fails safely with ChatStreamTransportError, not a crash', async () => {
    mockStreamingFetch(200, null)

    await expect(collect('s1', 'hi')).rejects.toBeInstanceOf(ChatStreamTransportError)
  })

  it('a malformed SSE frame fails safely with ChatStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('this is not valid SSE framing at all\n\n'))

    await expect(collect('s1', 'hi')).rejects.toBeInstanceOf(ChatStreamTransportError)
  })

  it('an unrecognized event name fails safely with ChatStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: not_a_real_event\ndata: {}\n\n'))

    await expect(collect('s1', 'hi')).rejects.toBeInstanceOf(ChatStreamTransportError)
  })

  it('a stream that ends without its final frame\'s terminator fails safely with ChatStreamTransportError', async () => {
    mockStreamingFetch(200, sseBody('event: started\ndata: {}\n\nevent: done\ndata: {}'))

    await expect(collect('s1', 'hi')).rejects.toBeInstanceOf(ChatStreamTransportError)
  })
})
